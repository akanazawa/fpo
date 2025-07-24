from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import numpy as np
import time
import wandb

from .ppo import PPO  # Assuming you keep shared utils in PPO base

@dataclass
class FpoActionInfo:
    x_t_path: torch.Tensor         # (*, flow_steps, action_dim)
    loss_eps: torch.Tensor         # (*, sample_dim, action_dim)
    loss_t: torch.Tensor           # (*, sample_dim, 1)
    initial_cfm_loss: torch.Tensor # (*,)

class FPO(PPO):
    def __init__(self, actor_class, critic_class, env, **hyperparameters):
        self.actor_class = actor_class
        self.critic_class = critic_class
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_hyperparameters(hyperparameters)

        self.obs_dim_actor = env.observation_space.shape[0] + env.action_space.shape[0] + 1
        self.obs_dim_critic = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Actor: diffusion policy
        self.actor = actor_class(self.obs_dim_actor, self.act_dim, **getattr(self, 'actor_kwargs', {})).to(self.device)
        self.num_train_samples = hyperparameters['num_fpo_samples']
        self.positive_advantage = hyperparameters.get('positive_advantage', False)
        
        print(f"training FPO with {self.num_train_samples} samples")
        print(f"positive_advantage = {self.positive_advantage}")

        # Model files saved as standard fpo_actor.pth, fpo_critic.pth

        # Critic: regular feedforward
        self.critic = critic_class(self.obs_dim_critic, 1).to(self.device)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Just for compatibility, this is not used by FPO's actor
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rews': [],
            'actor_losses': [],
        }
    
    def get_action(self, obs):
        """
        For FPO: returns pred_action + CFM loss info.
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action, x_t_path, eps, t, initial_cfm_loss = self.actor.sample_action_with_info(obs_tensor, self.num_train_samples)
            
        action_info = FpoActionInfo(
            x_t_path=x_t_path,
            loss_eps=eps,
            loss_t=t,
            initial_cfm_loss=initial_cfm_loss,
        )
        # dummy log_prob for API compatibility
        log_prob = torch.tensor(0.0)
        return action.squeeze().cpu().numpy(), log_prob, action_info

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []
        batch_action_info = []
        
        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        ep_vals = []
        ep_dones = []
        
        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode
            ep_vals = [] # state values collected per episode
            ep_dones = [] # done flag collected per episode
            
            # Reset the environment. sNote that obs is short for observation. 
            obs, _ = self.env.reset()
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps            
            for ep_t in range(self.max_timesteps_per_episode):
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()

                ep_dones.append(done)
                t += 1
                # Track observations in this batch                
                batch_obs.append(obs)
                # calculate action and make a step
                action, log_prob, action_info = self.get_action(obs)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                val = self.critic(obs_tensor).detach()
                
                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Track recent reward, action, and action log probability                
                ep_rews.append(rew)
                ep_vals.append(val.item())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                batch_action_info.append(action_info)
                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

            # Log the episodic returns and episodic lengths in this batch.
            self.logger['batch_rews'] = batch_rews
            self.logger['batch_lens'] = batch_lens
            

        return (torch.tensor(np.array(batch_obs), dtype=torch.float),
                torch.tensor(np.array(batch_acts), dtype=torch.float),
                torch.tensor(np.array(batch_log_probs), dtype=torch.float),
                batch_rews, batch_lens, batch_vals, batch_dones, batch_action_info)

    def learn(self, total_timesteps):
        t_so_far, i_so_far = 0, 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones, batch_action_info = self.rollout()
            batch_obs = batch_obs.to(self.device)
            batch_acts = batch_acts.to(self.device)

            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # === GAE + reward-to-go ===            
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones).to(self.device)
            with torch.no_grad():
                V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V
            if self.positive_advantage:
                A_k = F.softplus(A_k)
            else:
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)


            # This is the loop where we update our network for some n epochs
            # === Minibatching ===            
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []

            for _ in range(self.n_updates_per_iteration):
                # === Learning rate annealing ===
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = max(self.lr * (1.0 - frac), 0.0)
                self.actor_optim.param_groups[0]['lr'] = new_lr
                self.critic_optim.param_groups[0]['lr'] = new_lr
                self.logger['lr'] = new_lr

                np.random.shuffle(inds)
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    if len(idx) == 0:
                        continue
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_adv = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]
                    mini_infos = [batch_action_info[i] for i in idx]

                    # Compute losses
                    loss_eps = torch.stack([info.loss_eps for info in mini_infos]).to(self.device)
                    loss_t = torch.stack([info.loss_t for info in mini_infos]).to(self.device)
                    initial_cfm_loss = torch.stack([info.initial_cfm_loss for info in mini_infos]).to(self.device)
                    V = self.critic(mini_obs).squeeze(-1)
                    entropy = torch.tensor(0.0).to(self.device)  # Or pull from actor if defined

                    # Flatten B x N -> BN
                    B, N, D = loss_eps.shape
                    flat_obs = mini_obs.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)       # [B*N, D_s]
                    flat_acts = mini_acts.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)     # [B*N, D_a]
                    flat_eps = loss_eps.reshape(B * N, D)                                      # [B*N, D_a]
                    flat_t = loss_t.reshape(B * N, 1)                                          # [B*N, 1]
                    flat_init_loss = initial_cfm_loss.reshape(B * N)                           # [B*N]

                    cfm_loss = self.actor.compute_cfm_loss(flat_obs, flat_acts, flat_eps, flat_t)
                    cfm_difference = flat_init_loss - cfm_loss
                    # Convert back to [B, N] to take average
                    cfm_difference = cfm_difference.view(B, N)
                    cfm_difference = torch.clamp(cfm_difference, -3, 3)
                    rho_s = torch.exp(torch.clamp(cfm_difference.mean(dim=1), -3, 3))
                    # === Losses ===                    
                    surr1 = rho_s * mini_adv
                    surr2 = torch.clamp(rho_s, 1 - self.clip, 1 + self.clip) * mini_adv
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    # Discount entropy loss by given coefficient                    
                    actor_loss -= self.ent_coef * entropy.mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)
        
                    # === Backward + step ===                    
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # Gradient Clipping with given threshold                                                            
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network                    
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())

                    
                    # === Logging metrics ===
                    metrics = {
                        "clipped_ratio_mean": (torch.abs(rho_s - 1.0) > self.clip).float().mean().item(),
                        "cfm_difference": cfm_difference.mean().item(),
                        "policy_ratio_mean": rho_s.mean().item(),
                        "policy_ratio_min": rho_s.min().item(),
                        "policy_ratio_max": rho_s.max().item(),
                        "policy_loss": (-torch.min(surr1, surr2)).mean().item(),
                        "adv": mini_adv.mean().item(),
                        "surrogate_loss1_mean": surr1.mean().item(),
                        "surrogate_loss2_mean": surr2.mean().item(),
                        "action_min": mini_acts.min().item(),
                        "action_max": mini_acts.max().item(),
                    }
                    wandb.log(metrics)


            avg_loss = sum(loss) / len(loss)
            self.logger['actor_losses'].append(avg_loss)
            if self.logger['i_so_far'] % 10 == 0:
                wandb.log({'advantage_hist': wandb.Histogram(A_k.cpu().numpy())})
            self._log_summary()

            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './fpo_actor.pth')
                torch.save(self.critic.state_dict(), './fpo_critic.pth')
                wandb.save(f"{self.run_name}_actor_iter{i_so_far}.pth")
                wandb.save(f"{self.run_name}_critic_iter{i_so_far}.pth")
