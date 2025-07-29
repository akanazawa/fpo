from __future__ import annotations

from functools import partial
from typing import Literal, assert_never

import jax
import jax_dataclasses as jdc
import mujoco_playground as mjp
import optax
from jax import Array
from jax import numpy as jnp

from flow_policy.networks import MlpWeights

from . import math_utils, networks, rollouts


@jdc.pytree_dataclass
class FpoConfig:
    # Flow parameters.
    flow_steps: jdc.Static[int] = 10
    output_mode: jdc.Static[Literal["u", "u_but_supervise_as_eps"]] = (
        "u_but_supervise_as_eps"
    )
    timestep_embed_dim: jdc.Static[int] = 8
    """"Must be divisible by 2."""
    n_samples_per_action: jdc.Static[int] = 8
    average_losses_before_exp: jdc.Static[bool] = True

    discretize_t_for_training: jdc.Static[bool] = True
    feather_std: float = 0.0
    policy_mlp_output_scale: float = 0.25

    loss_mode: jdc.Static[Literal["fpo", "denoising_mdp"]] = "fpo"
    final_steps_only: jdc.Static[bool] = False

    # Fixed noise level for sampling via denoising MDP. This is used for
    # DDPO-style policy updates.
    sde_sigma: float = 0.0

    clipping_epsilon: float = 0.05

    # Based on Brax PPO config:
    batch_size: jdc.Static[int] = 1024
    discounting: float = 0.995
    episode_length: int = 1000
    learning_rate: float = 3e-4
    normalize_observations: jdc.Static[bool] = True
    num_envs: jdc.Static[int] = 2048
    num_evals: jdc.Static[int] = 10
    num_minibatches: jdc.Static[int] = 32
    num_timesteps: jdc.Static[int] = 60_000_000
    num_updates_per_batch: jdc.Static[int] = 16
    reward_scaling: float = 10.0
    unroll_length: jdc.Static[int] = 30

    gae_lambda: float = 0.95
    normalize_advantage: jdc.Static[bool] = True
    value_loss_coeff: float = 0.25

    def __post_init__(self) -> None:
        assert self.timestep_embed_dim % 2 == 0

    @property
    def iterations_per_env(self) -> int:
        """Number of iterations (=policy forward passes) per environment at the
        start of each training step."""
        return (
            self.num_minibatches * self.batch_size * self.unroll_length
        ) // self.num_envs


@jdc.pytree_dataclass
class FpoParams:
    policy: MlpWeights
    value: MlpWeights


@jdc.pytree_dataclass
class FpoActionInfo:
    loss_eps: Array  # (*, sample_dim, action_dim)
    loss_t: Array  # (*, sample_dim, 1)
    initial_cfm_loss: Array  # (*,)


@jdc.pytree_dataclass
class DenoisingMdpActionInfo:
    """For treating the denoising chain as an MDP."""

    full_x_t_path: Array  # (*, flow_steps, action_dim)
    initial_log_likelihood: Array  # (*, flow_steps)


@jdc.pytree_dataclass
class FlowSchedule:
    t_current: Array  # (*, flow_steps) - timesteps at the start of each step
    t_next: Array  # (*, flow_steps) - timesteps at the end of each step


FpoTransition = rollouts.TransitionStruct[FpoActionInfo | DenoisingMdpActionInfo]


@jdc.pytree_dataclass
class FpoState:
    """PPO agent state."""

    env: jdc.Static[mjp.MjxEnv]
    config: FpoConfig
    params: FpoParams
    obs_stats: math_utils.RunningStats

    opt: jdc.Static[optax.GradientTransformation]
    opt_state: optax.OptState

    prng: Array
    steps: Array

    @staticmethod
    @jdc.jit
    def init(prng: Array, env: jdc.Static[mjp.MjxEnv], config: FpoConfig) -> FpoState:
        obs_size = env.observation_size
        action_size = env.action_size
        assert isinstance(obs_size, int)

        prng0, prng1, prng2 = jax.random.split(prng, num=3)
        actor_net = networks.mlp_init(
            # Policy takes both observation and action as input. We'll just concatenate them!
            prng0,
            (
                obs_size + action_size + config.timestep_embed_dim,
                32,
                32,
                32,
                32,
                action_size,
            ),
        )
        critic_net = networks.mlp_init(prng1, (obs_size, 256, 256, 256, 256, 256, 1))

        network_params = FpoParams(actor_net, critic_net)

        # We'll manage learning rate ourselves!
        opt = optax.scale_by_adam()
        return FpoState(
            env=env,
            config=config,
            params=network_params,
            obs_stats=math_utils.RunningStats.init((obs_size,)),
            opt=opt,
            opt_state=opt.init(network_params),  # type: ignore
            prng=prng2,
            steps=jnp.zeros((), dtype=jnp.int32),
        )

    def get_schedule(self) -> FlowSchedule:
        full_t_path = jnp.linspace(1.0, 0.0, self.config.flow_steps + 1)
        t_current = full_t_path[:-1]
        return FlowSchedule(
            t_current=t_current,
            t_next=full_t_path[1:],
        )

    def embed_timestep(self, t: Array) -> Array:
        """Embed (*, 1) timestep into (*, timestep_embed_dim)."""
        assert t.shape[-1] == 1
        freqs = 2 ** jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t * freqs
        out = jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)
        assert out.shape == (*t.shape[:-1], self.config.timestep_embed_dim)
        return out

    def _compute_cfm_loss(
        self,
        obs_norm: Array,
        action: Array,
        eps: Array,
        t: Array,
    ) -> Array:
        """Computes from:
        - obs_norm: (*, obs_dim)
        - action: (*, action_dim)

        A CFM loss term with shape:
        - (*,)

        That is, one per obs-action pair, which is averaged across
        `n_samples_per_action` sampled eps-t pairs.
        """
        # Compute flow matching terms.
        (*batch_dims, action_dim) = action.shape
        samples_dim = self.config.n_samples_per_action
        obs_dim = self.env.observation_size
        sample_shape = (*batch_dims, samples_dim)
        assert eps.shape == (*batch_dims, samples_dim, action_dim)
        assert t.shape == (*batch_dims, samples_dim, 1)
        x_t = t * eps + (1.0 - t) * action[..., None, :]
        network_pred = (
            networks.flow_mlp_fwd(
                self.params.policy,
                jnp.broadcast_to(obs_norm[..., None, :], (*sample_shape, obs_dim)),
                x_t,
                self.embed_timestep(t),
            )
            * self.config.policy_mlp_output_scale
        )
        if self.config.output_mode == "u":
            velocity_pred = network_pred
            velocity_gt = eps - action[..., None, :]  # u = x1 - x0
            out = jnp.mean((velocity_pred - velocity_gt) ** 2, axis=-1)
        elif self.config.output_mode == "u_but_supervise_as_eps":
            # We want to compute velocity_pred => x1_pred.
            velocity_pred = network_pred  # x1 - x0
            x0_pred = x_t - t * velocity_pred
            x1_pred = x0_pred + velocity_pred
            out = jnp.mean((eps - x1_pred) ** 2, axis=-1)
        else:
            assert_never(self.config.output_mode)

        assert out.shape == (*batch_dims, samples_dim)
        return out

    def _compute_denoising_log_likelihood(
        self,
        obs_norm: Array,
        x_t_path: Array,
    ) -> Array:
        """Computes log likelihood for each Euler step in a denoising path. This is used for MDP / DPPO experiments.

        Args:
            obs_norm: (*, obs_dim) - normalized observation
            x_t_path: (*, flow_steps+1, action_dim) - states at each timestep (including final x0)

        Returns:
            log_likelihood: (*, flow_steps) - log likelihood for each step
        """
        (*batch_dims, total_states, action_dim) = x_t_path.shape

        schedule = self.get_schedule()
        flow_steps = schedule.t_current.shape[0]

        # Verify input shapes.
        assert total_states == flow_steps + 1, (
            f"Expected {flow_steps + 1} states, got {total_states}"
        )
        assert x_t_path.shape == (*batch_dims, flow_steps + 1, action_dim)
        assert schedule.t_current.shape == (flow_steps,)
        assert schedule.t_next.shape == (flow_steps,)
        assert obs_norm.shape == (*batch_dims, self.env.observation_size)

        # Extract states for all transitions.
        x_t = x_t_path[..., :-1, :]  # (*, flow_steps, action_dim) - start states
        x_t_next = x_t_path[..., 1:, :]  # (*, flow_steps, action_dim) - end states
        assert x_t.shape == (*batch_dims, flow_steps, action_dim)
        assert x_t_next.shape == (*batch_dims, flow_steps, action_dim)

        # Compute dt from the actual timestep differences.
        dt = schedule.t_next - schedule.t_current  # (flow_steps,)
        assert dt.shape == (flow_steps,)

        # Get predicted reverse velocities for all steps at once.
        velocity_pred = (
            networks.flow_mlp_fwd(
                self.params.policy,
                jnp.broadcast_to(
                    obs_norm[..., None, :],
                    (*batch_dims, flow_steps, obs_norm.shape[-1]),
                ),
                x_t,
                jnp.broadcast_to(
                    self.embed_timestep(schedule.t_current[..., None]),
                    (
                        *batch_dims,
                        self.config.flow_steps,
                        self.config.timestep_embed_dim,
                    ),
                ),
            )
            * self.config.policy_mlp_output_scale
        )

        # Simple expected next state with fixed sigma.
        assert velocity_pred.shape == x_t.shape == (*batch_dims, flow_steps, action_dim)
        expected_x_t_next = x_t + dt[None, :, None] * velocity_pred
        assert expected_x_t_next.shape == x_t_next.shape

        # Compute realized noise with fixed sigma.
        realized_noise = (x_t_next - expected_x_t_next) / (
            self.config.sde_sigma + 1e-6
        )[None, :, None]
        assert realized_noise.shape == x_t_next.shape

        # Log probability of standard normal: -0.5 * ||z||^2 - d/2 * log(2pi).
        all_log_likelihoods = (
            -0.5 * jnp.sum(realized_noise**2, axis=-1)  # (*, flow_steps)
            - 0.5 * action_dim * jnp.log(2 * jnp.pi)
        )
        assert all_log_likelihoods.shape == (*batch_dims, flow_steps)
        return all_log_likelihoods

    def sample_action(
        self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, FpoActionInfo | DenoisingMdpActionInfo]:
        """Sample an action from the policy given an observation."""
        if self.config.normalize_observations:
            obs_norm = (obs - self.obs_stats.mean) / self.obs_stats.std
        else:
            obs_norm = obs

        (*batch_dims, obs_dim) = obs.shape
        assert obs_dim == self.env.observation_size

        def euler_step(
            carry: Array, inputs: tuple[FlowSchedule, Array]
        ) -> tuple[Array, Array]:
            x_t = carry
            assert x_t.shape == (*batch_dims, self.env.action_size)
            schedule_t, noise = inputs
            assert schedule_t.t_current.shape == ()
            assert schedule_t.t_next.shape == ()
            assert noise.shape == x_t.shape

            # Compute dt as the difference between current and next timestep
            dt = schedule_t.t_next - schedule_t.t_current

            # Get velocity from flow model using the current timestep
            # This is the reverse velocity, which takes us from t=1 to t=0!
            velocity = (
                networks.flow_mlp_fwd(
                    self.params.policy,
                    obs_norm,
                    x_t,
                    jnp.broadcast_to(
                        self.embed_timestep(schedule_t.t_current[None]),
                        (*batch_dims, self.config.timestep_embed_dim),
                    ),
                )
                * self.config.policy_mlp_output_scale
            )

            # Simple SDE with fixed sigma - no time-dependent noise.
            x_t_next = x_t + dt * velocity + self.config.sde_sigma * noise
            assert x_t_next.shape == x_t.shape
            return x_t_next, x_t

        prng_sample, prng_loss, prng_feather, prng_noise = jax.random.split(prng, num=4)

        # Generate full timestep path and slice it for current/next pairs
        noise_path = jax.random.normal(
            prng_noise,
            (self.config.flow_steps, *batch_dims, self.env.action_size),
        )
        x0, x_t_path = jax.lax.scan(
            euler_step,
            init=jax.random.normal(prng_sample, (*batch_dims, self.env.action_size)),
            xs=(self.get_schedule(), noise_path),
        )

        if not deterministic:
            # Perturb the action with noise.
            perturb = (
                jax.random.normal(prng_feather, (*batch_dims, self.env.action_size))
                * self.config.feather_std
            )
            x0 = x0 + perturb

        # Create action info based on loss mode
        if self.config.loss_mode == "fpo":
            # Sample eps and t for FPO loss.
            sample_shape = (*batch_dims, self.config.n_samples_per_action)
            prng_eps, prng_t = jax.random.split(prng_loss)
            eps = jax.random.normal(prng_eps, (*sample_shape, self.env.action_size))
            if self.config.discretize_t_for_training:
                t = self.get_schedule().t_current[
                    jax.random.randint(
                        prng_t,
                        shape=(*sample_shape, 1),
                        minval=0,
                        maxval=self.config.flow_steps,
                    )
                ]
            else:
                t = jax.random.uniform(prng_t, (*sample_shape, 1))
            initial_cfm_loss = self._compute_cfm_loss(obs_norm, x0, eps=eps, t=t)

            return x0, FpoActionInfo(
                loss_eps=eps,
                loss_t=t,
                initial_cfm_loss=initial_cfm_loss,
            )
        else:  # denoising_mdp
            # x_t_path contains states at the START of each Euler step (from scan).
            # x0 is the final state at t=0.
            # For the MDP, we track all states in the denoising chain.
            assert x_t_path.shape == (
                self.config.flow_steps,
                *batch_dims,
                self.env.action_size,
            )
            assert x0.shape == (*batch_dims, self.env.action_size)

            # Move flow_steps dimension from first to second-to-last position.
            mdp_x_t_path = jnp.moveaxis(x_t_path, 0, -2)
            assert mdp_x_t_path.shape == (
                *batch_dims,
                self.config.flow_steps,
                self.env.action_size,
            )

            # Append x0 to create full path for likelihood computation.
            full_x_t_path = jnp.concatenate([mdp_x_t_path, x0[..., None, :]], axis=-2)
            assert full_x_t_path.shape == (
                *batch_dims,
                self.config.flow_steps + 1,
                self.env.action_size,
            )

            # Compute initial log likelihood for each Euler step.
            # Use full_x_t_path which has flow_steps+1 states for flow_steps transitions.
            initial_log_likelihood = self._compute_denoising_log_likelihood(
                obs_norm, full_x_t_path
            )
            assert initial_log_likelihood.shape == (*batch_dims, self.config.flow_steps)

            return x0, DenoisingMdpActionInfo(
                full_x_t_path=full_x_t_path,  # Store only flow_steps states for MDP
                initial_log_likelihood=initial_log_likelihood,
            )

    @jdc.jit
    def training_step(
        self, transitions: FpoTransition
    ) -> tuple[FpoState, dict[str, Array]]:
        # We're use a (T, B) shape convention, corresponding to a "scan of the
        # vmap" and not a "vmap of the scan".
        config = self.config
        assert transitions.reward.shape == (config.iterations_per_env, config.num_envs)

        # Update observation statistics.
        state = self
        if config.normalize_observations:
            with jdc.copy_and_mutate(state) as state:
                state.obs_stats = state.obs_stats.update(transitions.obs)
        del self

        def step_batch(state: FpoState, _):
            step_prng = jax.random.fold_in(state.prng, state.steps)
            state, metrics = jax.lax.scan(
                partial(
                    FpoState._step_minibatch, prng=jax.random.fold_in(step_prng, 0)
                ),
                init=state,
                xs=transitions.prepare_minibatches(
                    step_prng, config.num_minibatches, config.batch_size
                ),
            )
            return state, metrics

        # Do N updates over the full batch of transitions.
        state, metrics = jax.lax.scan(
            step_batch,
            init=state,
            length=config.num_updates_per_batch,
        )

        return state, metrics

    def _step_minibatch(
        self, transitions: FpoTransition, prng: Array
    ) -> tuple[FpoState, dict[str, Array]]:
        """One training step over a minibatch of transitions."""

        assert transitions.reward.shape == (
            self.config.unroll_length,
            self.config.batch_size,
        )
        (loss, metrics), grads = jax.value_and_grad(
            lambda params: FpoState._compute_fpo_loss(
                jdc.replace(self, params=params),
                transitions,
                prng,
            ),
            has_aux=True,
        )(self.params)
        assert isinstance(grads, FpoParams)
        assert isinstance(loss, Array)
        assert isinstance(metrics, dict)

        param_update, new_opt_state = self.opt.update(grads, self.opt_state)  # type: ignore
        param_update = jax.tree.map(
            lambda x: -self.config.learning_rate * x, param_update
        )
        with jdc.copy_and_mutate(self) as state:
            state.params = jax.tree.map(jnp.add, self.params, param_update)
            state.opt_state = new_opt_state
            state.steps = state.steps + 1
        return state, metrics

    def _compute_fpo_loss(
        self, transitions: FpoTransition, prng: Array
    ) -> tuple[Array, dict[str, Array]]:
        del prng  # Unused for now.

        (timesteps, batch_dim) = transitions.reward.shape
        assert transitions.obs.shape == (
            timesteps,
            batch_dim,
            self.env.observation_size,
        )
        assert transitions.action.shape == (
            timesteps,
            batch_dim,
            self.env.action_size,
        )

        metrics = dict[str, Array]()

        if self.config.normalize_observations:
            obs_norm = (transitions.obs - self.obs_stats.mean) / self.obs_stats.std
        else:
            obs_norm = transitions.obs
        value_pred = networks.value_mlp_fwd(self.params.value, obs_norm)
        assert value_pred.shape == (timesteps, batch_dim)

        bootstrap_obs_norm = (
            transitions.next_obs[-1:, :, :] - self.obs_stats.mean
        ) / self.obs_stats.std
        bootstrap_value = networks.value_mlp_fwd(self.params.value, bootstrap_obs_norm)
        assert bootstrap_value.shape == (1, batch_dim)

        gae_vs, gae_advantages = jax.lax.stop_gradient(
            rollouts.compute_gae(
                truncation=transitions.truncation,
                discount=transitions.discount * self.config.discounting,
                rewards=transitions.reward * self.config.reward_scaling,
                values=value_pred,
                bootstrap_value=bootstrap_value,
                gae_lambda=self.config.gae_lambda,
            )
        )

        # Log advantage statistics before normalization
        metrics["advantages_mean"] = jnp.mean(gae_advantages)
        metrics["advantages_std"] = jnp.std(gae_advantages)
        metrics["advantages_min"] = jnp.min(gae_advantages)
        metrics["advantages_max"] = jnp.max(gae_advantages)

        if self.config.normalize_advantage:
            gae_advantages = (gae_advantages - gae_advantages.mean()) / (
                gae_advantages.std() + 1e-8
            )

        # Compute policy ratio based on loss mode
        if self.config.loss_mode == "fpo":
            # Original FPO loss computation
            assert isinstance(transitions.action_info, FpoActionInfo)

            # Check action info shapes
            assert transitions.action_info.loss_eps.shape == (
                timesteps,
                batch_dim,
                self.config.n_samples_per_action,
                self.env.action_size,
            )
            assert transitions.action_info.loss_t.shape == (
                timesteps,
                batch_dim,
                self.config.n_samples_per_action,
                1,
            )
            assert transitions.action_info.initial_cfm_loss.shape == (
                timesteps,
                batch_dim,
                self.config.n_samples_per_action,
            )

            cfm_loss = self._compute_cfm_loss(
                obs_norm,
                transitions.action,
                eps=transitions.action_info.loss_eps,
                t=transitions.action_info.loss_t,
            )
            assert cfm_loss.shape == transitions.action_info.initial_cfm_loss.shape

            if self.config.average_losses_before_exp:
                rho_s = jnp.exp(
                    jnp.mean(
                        transitions.action_info.initial_cfm_loss,
                        axis=-1,
                        keepdims=True,
                    )
                    - jnp.mean(
                        cfm_loss,
                        axis=-1,
                        keepdims=True,
                    )
                )
                assert rho_s.shape == (
                    timesteps,
                    batch_dim,
                    1,
                )
            else:
                # Compute FPO ratio. We clip before exponentiation to prevent
                # outliers from blowing up the loss; this is optional.
                rho_s = jnp.exp(
                    jnp.clip(
                        transitions.action_info.initial_cfm_loss - cfm_loss, -3.0, 3.0
                    )
                )
                assert rho_s.shape == (
                    timesteps,
                    batch_dim,
                    self.config.n_samples_per_action,
                )
        else:  # denoising_mdp
            # Compute policy ratio for denoising MDP
            assert isinstance(transitions.action_info, DenoisingMdpActionInfo)

            # Check action info shapes
            assert transitions.action_info.full_x_t_path.shape == (
                timesteps,
                batch_dim,
                self.config.flow_steps + 1,
                self.env.action_size,
            )
            assert transitions.action_info.initial_log_likelihood.shape == (
                timesteps,
                batch_dim,
                self.config.flow_steps,
            )

            # Compute current log likelihoods for each step in the denoising chain
            current_log_likelihood = self._compute_denoising_log_likelihood(
                obs_norm, transitions.action_info.full_x_t_path
            )
            assert (
                current_log_likelihood.shape
                == transitions.action_info.initial_log_likelihood.shape
            )

            # Sum log likelihoods across flow steps to get joint probability.
            rho_s = jnp.exp(
                current_log_likelihood - transitions.action_info.initial_log_likelihood,
            )
            if self.config.final_steps_only:
                # Use only the final steps for likelihood.
                rho_s = rho_s[..., self.config.flow_steps // 2 :]

        # Shared PPO loss computation
        assert gae_advantages.shape == (timesteps, batch_dim)

        surrogate_loss1 = rho_s * gae_advantages[..., None]
        surrogate_loss2 = (
            jnp.clip(
                rho_s,
                1 - self.config.clipping_epsilon,
                1 + self.config.clipping_epsilon,
            )
            * gae_advantages[..., None]
        )

        # Check that surrogate losses have the same shape as rho_s
        assert surrogate_loss1.shape == rho_s.shape
        assert surrogate_loss2.shape == rho_s.shape

        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

        # Metrics
        metrics["clipped_ratio_mean"] = jnp.mean(
            jnp.abs(rho_s - 1.0) > self.config.clipping_epsilon
        )
        metrics["policy_ratio_mean"] = jnp.mean(rho_s)
        metrics["policy_ratio_min"] = jnp.min(rho_s)
        metrics["policy_ratio_max"] = jnp.max(rho_s)
        metrics["policy_loss"] = policy_loss
        metrics["surrogate_loss1_mean"] = jnp.mean(surrogate_loss1)
        metrics["surrogate_loss2_mean"] = jnp.mean(surrogate_loss2)

        # Log action distribution statistics
        metrics["action_min"] = jnp.min(transitions.action)
        metrics["action_max"] = jnp.max(transitions.action)

        # Don't supervise value function on truncated timesteps.
        v_error = (gae_vs - value_pred) * (1 - transitions.truncation)

        # Value function statistics
        metrics["value_mean"] = jnp.mean(value_pred)
        metrics["value_std"] = jnp.std(value_pred)
        metrics["value_min"] = jnp.min(value_pred)
        metrics["value_max"] = jnp.max(value_pred)
        metrics["value_target_mean"] = jnp.mean(gae_vs)
        metrics["value_error_mean"] = jnp.mean(v_error)
        metrics["value_error_std"] = jnp.std(v_error)

        v_loss = jnp.mean(v_error**2) * self.config.value_loss_coeff
        metrics["v_loss"] = v_loss

        # Compute the total loss that will be used for optimization
        total_loss = policy_loss + v_loss

        return total_loss, metrics
