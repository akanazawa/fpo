import torch
import math
from torch import nn

from pufferlib.pytorch import layer_init
import pufferlib.models

from puffer_phc.flow_matching_utils.solver import ODESolver
from puffer_phc.flow_matching_utils.path import CondOTProbPath
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES

class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

        # Point to the original policy's methods
        self.set_deterministic_action = self.policy.set_deterministic_action
        self.discriminate = self.policy.discriminate
        self.update_obs_rms = self.policy.update_obs_rms
        self.update_amp_obs_rms = self.policy.update_amp_obs_rms

    @property
    def mean_bound_loss(self):
        return self.policy.mean_bound_loss


class PolicyWithDiscriminator(nn.Module):
    def __init__(self, env, hidden_size=512):
        super().__init__()
        self.is_continuous = True
        self._deterministic_action = False

        self.input_size = env.single_observation_space.shape[0]
        self.action_size = env.single_action_space.shape[0]

        # Assume the action space is symmetric (low=-high)
        self.soft_bound = 0.9 * env.single_action_space.high[0]

        self.obs_norm = torch.jit.script(RunningNorm(self.input_size))

        ### Actor
        self.actor_mlp = None
        self.mu = nn.Sequential(
            layer_init(nn.Linear(hidden_size, self.action_size), std=0.01),
        )

        # NOTE: Original PHC uses a constant std. Something to experiment?
        self.sigma = nn.Parameter(
            torch.zeros(self.action_size, requires_grad=False, dtype=torch.float32),
            requires_grad=False,
        )
        nn.init.constant_(self.sigma, -2.9)

        ### Critic
        self.critic_mlp = None

        ### Discriminator
        self.use_amp_obs = env.amp_observation_space is not None
        self.amp_obs_norm = None

        if self.use_amp_obs:
            amp_obs_size = env.amp_observation_space.shape[0]
            self.amp_obs_norm = torch.jit.script(RunningNorm(amp_obs_size))

            self._disc_mlp = nn.Sequential(
                layer_init(nn.Linear(amp_obs_size, 1024)),
                nn.ReLU(),
                layer_init(nn.Linear(1024, hidden_size)),
                nn.ReLU(),
            )
            self._disc_logits = layer_init(torch.nn.Linear(hidden_size, 1))

        self.obs_pointer = None
        self.mean_bound_loss = None

    def create_relative_pose_dropout_mask(self, include_action=True, root_track=False, hand_track=False):
        # self observation dropout mask: height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos 
        # = 1 + 24 * 15 - 3 = 358
        state_mask = torch.ones(1, 358)
        left_hand_id = SMPL_MUJOCO_NAMES.index("L_Wrist")
        right_hand_id = SMPL_MUJOCO_NAMES.index("R_Wrist")
        assert len(SMPL_MUJOCO_NAMES) == 24, "SMPL_MUJOCO_NAMES should have 24 joints"

        # imitation observation dropout mask
        # diff_local_body_pos_flat: # 24 * 3
        diff_local_body_pos_flat_mask = torch.zeros(1, 24 * 3)
        diff_local_body_pos_flat_mask[:, :3] = 1 # don't drop root_pos
        if hand_track:
            diff_local_body_pos_flat_mask[:, left_hand_id*3:(left_hand_id+1)*3] = 1 # don't drop left_hand_pos
            diff_local_body_pos_flat_mask[:, right_hand_id*3:(right_hand_id+1)*3] = 1 # don't drop right_hand_pos
        # diff_local_body_rot_flat: # 24 * 6
        diff_local_body_rot_flat_mask = torch.zeros(1, 24 * 6)
        if not root_track and not hand_track:
            diff_local_body_rot_flat_mask[:, :6] = 1
        # diff_local_vel: # 24 * 3
        diff_local_vel_mask = torch.zeros(1, 24 * 3)
        if not root_track and not hand_track:
            diff_local_vel_mask[:, :3] = 1
        # diff_local_ang_vel: # 24 * 3
        diff_local_ang_vel_mask = torch.zeros(1, 24 * 3)
        if not root_track and not hand_track:
            diff_local_ang_vel_mask[:, :3] = 1
        # local_ref_body_pos: # 24 * 3
        local_ref_body_pos_mask = torch.zeros(1, 24 * 3)
        local_ref_body_pos_mask[:, :3] = 1
        if hand_track:
            local_ref_body_pos_mask[:, left_hand_id*3:(left_hand_id+1)*3] = 1
            local_ref_body_pos_mask[:, right_hand_id*3:(right_hand_id+1)*3] = 1

        # local_ref_body_rot: # 24 * 6
        local_ref_body_rot_mask = torch.zeros(1, 24 * 6)
        if not root_track and not hand_track:
            local_ref_body_rot_mask[:, :6] = 1 # don't drop root_pos

        if include_action:
            # noised action dropout mask, should always be 1
            noised_action_mask = torch.ones(1, self.action_size)
            # concat all masks
            relative_pose_dropout_mask = torch.cat([noised_action_mask, state_mask, diff_local_body_pos_flat_mask, diff_local_body_rot_flat_mask, diff_local_vel_mask,
                                                        diff_local_ang_vel_mask, local_ref_body_pos_mask, local_ref_body_rot_mask], dim=-1)
            assert relative_pose_dropout_mask.shape[-1] == self.input_size + self.action_size
        else:
            relative_pose_dropout_mask = torch.cat([state_mask, diff_local_body_pos_flat_mask, diff_local_body_rot_flat_mask, diff_local_vel_mask,
                                                        diff_local_ang_vel_mask, local_ref_body_pos_mask, local_ref_body_rot_mask], dim=-1)
            assert relative_pose_dropout_mask.shape[-1] == self.input_size
        return relative_pose_dropout_mask

    def generate_soft_dropout_mask(self, batch_size: int, device: torch.device, include_action: bool):
        # Hardcoded numbers from create_relative_pose_dropout_mask
        # TODO: Consider making num_bodies configurable from env if possible
        num_bodies = 24 
        body_feat_size_state = 15 # pos, vel, rot, ang_vel for state representation

        full_mask_components = []

        if include_action:
            action_mask_segment = torch.ones(batch_size, self.action_size, device=device)
            full_mask_components.append(action_mask_segment)

        # Structure of self.input_size for observation part:
        # It's a concatenation of state features and various imitation signal features.
        # Based on create_relative_pose_dropout_mask:
        # 1. State Mask part (size 358 in typical humanoid config)
        # This is: height (1) + root_vel (3) + root_rot (6) + root_ang_vel (3) = 13 root-related state features.
        # Plus (num_bodies - 1) * body_feat_size_state for other bodies' states.
        state_features_size = 1 + num_bodies * body_feat_size_state - 3 # e.g., 358
        
        # State features part: all state features (root and non-root) are NOT dropped when soft_dropout is active.
        state_features_mask_segment = torch.ones(batch_size, state_features_size, device=device)
        full_mask_components.append(state_features_mask_segment)
        
        # 2. Imitation signals (diffs, refs for local body features) - non-root parts are soft-dropped
        imitation_feature_defs = [
            { "name": "diff_local_body_pos_flat", "total_size_fn": lambda nb: nb * 3, "root_size": 3 },
            { "name": "diff_local_body_rot_flat", "total_size_fn": lambda nb: nb * 6, "root_size": 6 },
            { "name": "diff_local_vel",           "total_size_fn": lambda nb: nb * 3, "root_size": 3 },
            { "name": "diff_local_ang_vel",       "total_size_fn": lambda nb: nb * 3, "root_size": 3 },
            { "name": "local_ref_body_pos",       "total_size_fn": lambda nb: nb * 3, "root_size": 3 },
            { "name": "local_ref_body_rot",       "total_size_fn": lambda nb: nb * 6, "root_size": 6 },
        ]

        current_total_obs_size_check = state_features_size

        # Generate per-joint dropout decisions for non-root joints
        # (num_bodies - 1) because root joint's imitation signals are always kept.
        # Probability of keeping a joint's imitation signals
        joint_keep_prob = 1.0 - self.condition_drop_ratio
        joint_keep_probs_tensor = torch.full((batch_size, num_bodies - 1), joint_keep_prob, device=device)
        # joint_dropout_decisions[b, j] = 1 if (j+1)-th body's (non-root) imitation signals are KEPT for batch b.
        joint_dropout_decisions = torch.bernoulli(joint_keep_probs_tensor)

        for feat_def in imitation_feature_defs:
            total_size_for_feat = feat_def["total_size_fn"](num_bodies)
            root_size_for_feat = feat_def["root_size"]
            
            # Append root part mask (always kept for imitation signals)
            root_part_segment = torch.ones(batch_size, root_size_for_feat, device=device)
            full_mask_components.append(root_part_segment)

            non_root_overall_size_for_feat = total_size_for_feat - root_size_for_feat

            if non_root_overall_size_for_feat > 0:
                assert num_bodies > 1, "Non-root features found, but num_bodies <= 1, which is inconsistent."
                # Calculate feature size per non-root body for this specific feature type
                # Assumes features are evenly distributed among non-root bodies.
                assert non_root_overall_size_for_feat % (num_bodies - 1) == 0, \
                    f"Feature '{feat_def['name']}' size {non_root_overall_size_for_feat} is not divisible by number of non-root bodies {num_bodies - 1}"
                feat_size_per_non_root_body = non_root_overall_size_for_feat // (num_bodies - 1)

                non_root_segments_for_this_feature = []
                for i in range(num_bodies - 1): # Iterate through each non-root body index (0 to num_bodies-2)
                    # Get the dropout decision for the i-th non-root body (overall body index i+1)
                    # decision_for_this_joint is (batch_size, 1), 1.0 for keep, 0.0 for drop
                    decision_for_this_joint = joint_dropout_decisions[:, i].unsqueeze(-1)
                    
                    # Create mask for this specific non-root body's part of the current feature type
                    # All features for this body (of this type) are either kept or dropped together.
                    joint_feature_mask_segment = torch.ones(batch_size, feat_size_per_non_root_body, device=device) * decision_for_this_joint
                    non_root_segments_for_this_feature.append(joint_feature_mask_segment)
                
                # Concatenate masks for all non-root bodies for this feature type
                if non_root_segments_for_this_feature: # Should always be true if non_root_overall_size_for_feat > 0
                    non_root_part_segment = torch.cat(non_root_segments_for_this_feature, dim=-1)
                    full_mask_components.append(non_root_part_segment)
            
            current_total_obs_size_check += total_size_for_feat
            
        # Sanity check that the constructed observation mask matches self.input_size
        assert current_total_obs_size_check == self.input_size, \
            f"Constructed observation feature size {current_total_obs_size_check} does not match self.input_size {self.input_size}"

        final_mask = torch.cat(full_mask_components, dim=-1)
        
        expected_total_size = (self.action_size if include_action else 0) + self.input_size
        assert final_mask.shape[0] == batch_size
        assert final_mask.shape[-1] == expected_total_size, \
            f"Final mask size {final_mask.shape[-1]} does not match expected total size {expected_total_size}"
        
        return final_mask

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, obs):
        raise NotImplementedError

    def decode_actions(self, hidden, lookup=None):
        raise NotImplementedError

    def set_deterministic_action(self, value):
        self._deterministic_action = value

    def discriminate(self, amp_obs):
        if not self.use_amp_obs:
            return None

        norm_amp_obs = self.amp_obs_norm(amp_obs)
        disc_mlp_out = self._disc_mlp(norm_amp_obs)
        disc_logits = self._disc_logits(disc_mlp_out)
        return disc_logits

    # NOTE: Used for network weight regularization
    # def disc_logit_weights(self):
    #     return torch.flatten(self._disc_logits.weight)

    # def disc_weights(self):
    #     weights = []
    #     for m in self._disc_mlp.modules():
    #         if isinstance(m, nn.Linear):
    #             weights.append(torch.flatten(m.weight))

    #     weights.append(torch.flatten(self._disc_logits.weight))
    #     return weights

    def update_obs_rms(self, obs):
        self.obs_norm.update(obs)

    def update_amp_obs_rms(self, amp_obs):
        if not self.use_amp_obs:
            return

        self.amp_obs_norm.update(amp_obs)

    def bound_loss(self, mu):
        mu_loss = torch.zeros_like(mu)
        mu_loss = torch.where(mu > self.soft_bound, (mu - self.soft_bound) ** 2, mu_loss)
        mu_loss = torch.where(mu < -self.soft_bound, (mu + self.soft_bound) ** 2, mu_loss)
        return mu_loss.mean()


# NOTE: The PHC implementation, which has no LSTM. 17.0M params
class PHCPolicy(PolicyWithDiscriminator):
    def __init__(self, env, hidden_size=512, condition_drop_ratio=0.0, num_envs=4096, soft_dropout=False, root_track=False, hand_track=False):
        super().__init__(env, hidden_size)
        self.condition_drop_ratio = condition_drop_ratio
        self.relative_pose_dropout_mask = self.create_relative_pose_dropout_mask(include_action=False, root_track=root_track, hand_track=hand_track)
        self.sample_mask = torch.bernoulli(torch.ones(num_envs, 1) * self.condition_drop_ratio) * torch.ones(num_envs, self.input_size)
        self.soft_dropout = soft_dropout
        # NOTE: Original PHC network + LayerNorm
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        # NOTE: Original PHC network + LayerNorm
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

    def encode_observations(self, obs, sample_mask=None):
        # Remember the normed obs to use in the critic
        self.obs_pointer = self.obs_norm(obs)
        if self.condition_drop_ratio > 0:
            B = obs.shape[0]
            if self.soft_dropout:
                self.soft_condition_mask = sample_mask.to(obs.device) if sample_mask is not None else self.generate_soft_dropout_mask(B, obs.device, include_action=False)
                condition_mask_to_apply = self.soft_condition_mask
            else: # Original hard dropout logic
                relative_pose_dropout_mask = self.relative_pose_dropout_mask.to(obs.device) # Shape [1, D_obs]
                batch_bernoulli_mask = sample_mask.to(obs.device) if sample_mask is not None else self.sample_mask.to(obs.device)
                condition_mask_to_apply = batch_bernoulli_mask * relative_pose_dropout_mask + (1 - batch_bernoulli_mask) * torch.ones_like(relative_pose_dropout_mask)
            
            self.obs_pointer = self.obs_pointer * condition_mask_to_apply
        return self.actor_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.mu(hidden)
        std = torch.exp(self.sigma).expand_as(mu)

        if self._deterministic_action is True:
            std = torch.clamp(std, max=1e-6)

        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: Separate critic network takes input directly
        value = self.critic_mlp(self.obs_pointer)
        return probs, value


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class adaLN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm1(x) * (1+scale) + shift
        return x


# NOTE: The PHC implementation, which has no LSTM. 17.0M params
class FlowMatchingPolicy(PolicyWithDiscriminator):
    def __init__(self, env, hidden_size=512, parameterization="velocity", solver_step_size=0.1, perturb_action_std=0.0,
                 prior_noise_std=1.0, zero_action_input=False, condition_drop_ratio=0.0, num_sampled_t=1, num_envs=4096,
                 sample_t_strategy="uniform", p_mean=-1.2, p_std=1.2, soft_dropout=False, root_track=False, hand_track=False, **kwargs):
        super().__init__(env, hidden_size)
        print(f"Flow matching policy with parameterization: {parameterization}")
        self.zero_action_input = zero_action_input
        if self.zero_action_input:
            print("!!! WARNING: Zeroing action input for FlowMatchingPolicy !!!")

        # NOTE: Original PHC network + LayerNorm
        self.actor_mlp = nn.Sequential(
            # layer_init(nn.Linear(self.input_size, 2048)),
            layer_init(nn.Linear(self.input_size+self.action_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
        )
        self.actor_norm = adaLN(hidden_size)
        self.post_adaln_non_linearity = nn.SiLU()
        nn.init.constant_(self.actor_norm.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.actor_norm.adaLN_modulation[-1].bias, 0)

        # NOTE: Original PHC network + LayerNorm
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

        # Noise embedder.
        self.noise_emb = TimestepEmbedder(hidden_size)

        # Flow matching stuff.
        self.solver = ODESolver()
        self.path = CondOTProbPath()
        self.parameterization = parameterization
        self.solver_step_size = solver_step_size
        self.perturb_action_std = torch.tensor(perturb_action_std)
        self.prior_noise_std = torch.tensor(prior_noise_std)
        self.condition_drop_ratio = condition_drop_ratio
        self.relative_pose_dropout_mask = self.create_relative_pose_dropout_mask(include_action=True, root_track=root_track, hand_track=hand_track)
        self.sample_mask = torch.bernoulli(torch.ones(num_envs, 1) * self.condition_drop_ratio) * torch.ones(num_envs, self.input_size+self.action_size)
        self.sample_t_strategy = sample_t_strategy
        self.p_mean = p_mean
        self.p_std = p_std
        self.soft_dropout = soft_dropout

    def sample_noise(self, noise_shape, device):
        noise = torch.randn(noise_shape, dtype=torch.float32, device=device)

        # NOTE: scale the prior noise if using velocity parameterization
        # if self.parameterization == "velocity":
        noise = noise * self.prior_noise_std.to(device)

        return noise

    def sample_ts(self, B, device):
        if self.sample_t_strategy == "uniform":
            return torch.rand(B, device=device)
        elif self.sample_t_strategy == "lognormal":
            rnd_normal = torch.randn((B,), device=device)
            sigma = (rnd_normal * self.p_std + self.p_mean).exp()
            time = 1 / (1 + sigma)
            time = torch.clip(time, min=0.0001, max=1.0)
            return time

    def sample_actions(self, obs):

        assert not torch.is_grad_enabled(), "Autograd should not be enabled during the sampling chain!"

        B = obs.shape[0]
        x_0 = self.sample_noise([B, self.action_size], obs.device)
        time_grid = torch.tensor([0.0, 1.0], device=obs.device)

        if self.condition_drop_ratio > 0:
            if self.soft_dropout:
                self.soft_condition_mask = self.generate_soft_dropout_mask(B, obs.device, include_action=True)
                active_condition_mask = self.soft_condition_mask
            else:
                condition_drop_ratio = self.relative_pose_dropout_mask.to(obs.device) # Shape [1, D_full]
                batch_bernoulli_mask = self.sample_mask.to(obs.device)
                active_condition_mask = batch_bernoulli_mask * condition_drop_ratio + (1 - batch_bernoulli_mask) * torch.ones_like(condition_drop_ratio) # [B or num_envs, D_full]
        else:
            active_condition_mask = None

        def velocity_fn(x, t, obs, condition_mask=None):
            # Remember the normed obs to use in the critic
            obs_pointer = self.obs_norm(obs)
            # Zero out action input if configured
            x_eff = torch.zeros_like(x) if self.zero_action_input else x
            # Concatenate noised action and normed obs
            x_inp = torch.cat([x_eff, obs_pointer], dim=1)
            # x_inp = obs_pointer
            if condition_mask is not None:
                x_inp = x_inp * condition_mask

            # Get noise embedding for the current timestep
            t_batch = torch.ones([B], device=obs.device) * t
            noise_emb = self.noise_emb(t_batch * (0.0 if self.zero_action_input else 1.0))
            hidden = self.actor_mlp(x_inp)  # Get features before adaLN
            hidden = self.actor_norm(hidden, noise_emb)
            hidden = self.post_adaln_non_linearity(hidden)

            if self.parameterization == "velocity":
                velocity = self.mu(hidden)
            elif self.parameterization == "data":
                x1 = self.mu(hidden)
                velocity = self.path.target_to_velocity(x_1=x1, x_t=x, t=t_batch.unsqueeze(-1))
            return velocity

        x_1 = self.solver.sample(
            velocity_fn,
            time_grid=time_grid,
            x_init=x_0,
            method="euler",
            return_intermediates=False,
            atol=1e-5,
            rtol=1e-5,
            step_size=self.solver_step_size,
            obs=obs,
            condition_mask=active_condition_mask,
        )

        if self.perturb_action_std > 0:
            std = self.perturb_action_std
            std = torch.clamp(std, max=1e-6) if self._deterministic_action is True else std
            x_1 = x_1 + torch.randn_like(x_1) * std

        return x_1

    def flow_matching_loss(self, actions, observations, t=None, noise=None, return_noise_t=False, sample_mask=None):
        noise = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        t = self.sample_ts(actions.shape[0], actions.device) if t is None else t
        path_sample = self.path.sample(t=t, x_0=noise, x_1=actions)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t

        if self.condition_drop_ratio > 0:
            if self.soft_dropout:
                active_condition_mask = sample_mask.to(observations.device)
            else:
                batch_bernoulli_mask = sample_mask.to(observations.device)
                relative_pose_dropout_mask = self.relative_pose_dropout_mask.to(observations.device) # [1, D_full]
                active_condition_mask = batch_bernoulli_mask * relative_pose_dropout_mask + (1 - batch_bernoulli_mask) * torch.ones_like(relative_pose_dropout_mask) # [B, D_full]
        else:
            active_condition_mask = None

        with torch.cuda.amp.autocast():
            # model prediction
            obs_pointer = self.obs_norm(observations)
            # Zero out action input if configured
            x_t_eff = torch.zeros_like(x_t) if self.zero_action_input else x_t
            x_inp = torch.cat([x_t_eff, obs_pointer], dim=-1)

            if active_condition_mask is not None:
                 # x_inp is [B, D_full], active_condition_mask is [B, D_full]
                x_inp = x_inp * active_condition_mask.unsqueeze(1)
            
            # x_inp = obs_pointer
            noise_emb = self.noise_emb(t * (0.0 if self.zero_action_input else 1.0))
            hidden = self.actor_mlp(x_inp)  # Get features before adaLN
            hidden = self.actor_norm(hidden, noise_emb.unsqueeze(1))
            hidden = self.post_adaln_non_linearity(hidden)
            if self.parameterization == "velocity":
                velocity = self.mu(hidden)
                x1 = self.path.velocity_to_target(x_t=x_t, velocity=velocity, t=t.unsqueeze(-1).unsqueeze(-1))
                # loss  
                log_probs = -((u_t - velocity) ** 2) / (2 * 0.05 ** 2)
                loss = - log_probs.reshape(-1).mean()
                # loss = torch.pow(velocity - u_t, 2).mean(-1)
            elif self.parameterization == "data":
                x1 = self.mu(hidden)
                log_probs = -((x1 - actions) ** 2) / (2 * 0.05 ** 2)
                loss = - log_probs.reshape(-1).mean()
                # loss = torch.pow(x1 - actions, 2).mean(-1)

        # (TODO) Mean bound loss
        if self.training:
            self.mean_bound_loss = self.bound_loss(x1)


        if return_noise_t:
            return log_probs.mean(-1).reshape(-1), loss, noise, t
        else:
            value = self.critic_mlp(obs_pointer)
            return log_probs.mean(-1).reshape(-1), loss, value
    
    def forward(self, observations):
        # Remember the normed obs to use in the critic
        self.obs_pointer = self.obs_norm(observations)
        # NOTE: Separate critic network takes input directly
        value = self.critic_mlp(self.obs_pointer)
        with torch.no_grad():
            actions = self.sample_actions(observations)
        return actions, value


class LSTMCriticPolicy(PolicyWithDiscriminator):
    def __init__(self, env, hidden_size=512):
        super().__init__(env, hidden_size)

        # Actor: Original PHC network
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, self.action_size), std=0.01),
        )
        self.mu = None

        ### Critic with LSTM
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.ReLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, hidden_size)),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.ReLU(),  # handle the LSTM output
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

    def encode_observations(self, obs):
        # Remember the normed obs to use in the actor
        self.obs_pointer = self.obs_norm(obs)

        # NOTE: hidden goes through LSTM, then to the value (critic head)
        return self.critic_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.actor_mlp(self.obs_pointer)
        std = torch.exp(self.sigma).expand_as(mu)

        if self._deterministic_action is True:
            std = torch.clamp(std, max=1e-6)

        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            # mean_violation = nn.functional.relu(torch.abs(mu) - 1)  # bound hard coded to 1
            # self.mean_bound_loss = mean_violation.mean()
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: hidden from LSTM goes to the critic head
        value = self.value(hidden)
        return probs, value


# NOTE: 13.5M params, Worked for simple motions, but not capable for many, complex motions
class LSTMActorPolicy(PolicyWithDiscriminator):
    def __init__(self, env, hidden_size=512):
        super().__init__(env, hidden_size)

        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, hidden_size)),
            nn.SiLU(),
        )

        self.mu = nn.Sequential(
            nn.SiLU(),  # handle the LSTM output
            layer_init(nn.Linear(hidden_size, self.action_size), std=0.01),
        )

        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            # nn.LayerNorm(1024),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            # nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            # nn.LayerNorm(256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=0.01),
        )

    def encode_observations(self, obs):
        # Remember the obs to use in the critic
        self.obs_pointer = self.obs_norm(obs)
        return self.actor_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.mu(hidden)
        std = torch.exp(self.sigma).expand_as(mu)

        if self._deterministic_action is True:
            std = torch.clamp(std, max=1e-6)

        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            # mean_violation = nn.functional.relu(torch.abs(mu) - 1)  # bound hard coded to 1
            # self.mean_bound_loss = mean_violation.mean()
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: Separate critic network takes input directly
        value = self.critic_mlp(self.obs_pointer)
        return probs, value


class RunningNorm(nn.Module):
    def __init__(self, shape: int, epsilon=1e-5, clip=10.0):
        super().__init__()

        self.register_buffer("running_mean", torch.zeros((1, shape), dtype=torch.float32))
        self.register_buffer("running_var", torch.ones((1, shape), dtype=torch.float32))
        self.register_buffer("count", torch.ones(1, dtype=torch.float32))
        self.epsilon = epsilon
        self.clip = clip

    def forward(self, x):
        return torch.clamp(
            (x - self.running_mean.expand_as(x)) / torch.sqrt(self.running_var.expand_as(x) + self.epsilon),
            -self.clip,
            self.clip,
        )

    @torch.jit.ignore
    def update(self, x):
        # NOTE: Separated update from forward to compile the policy
        # update() must be called to update the running mean and var
        with torch.no_grad():
            x = x.float()
            assert x.dim() == 2, "x must be 2D"
            mean = x.mean(0, keepdim=True)
            var = x.var(0, unbiased=False, keepdim=True)
            weight = 1 / self.count
            self.running_mean = self.running_mean * (1 - weight) + mean * weight
            self.running_var = self.running_var * (1 - weight) + var * weight
            self.count += 1

    # NOTE: below are needed to torch.save() the model
    @torch.jit.ignore
    def __getstate__(self):
        return {
            "running_mean": self.running_mean,
            "running_var": self.running_var,
            "count": self.count,
            "epsilon": self.epsilon,
            "clip": self.clip,
        }

    @torch.jit.ignore
    def __setstate__(self, state):
        self.running_mean = state["running_mean"]
        self.running_var = state["running_var"]
        self.count = state["count"]
        self.epsilon = state["epsilon"]
        self.clip = state["clip"]
