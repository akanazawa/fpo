"""Rollout helpers for Mujoco playground."""

from __future__ import annotations

from typing import Protocol, Self

import jax
import jax_dataclasses as jdc
import mujoco_playground as mjp
import numpy as onp
import wandb
from jax import Array
from jax import numpy as jnp
from mujoco import mjx
from wandb.sdk.wandb_run import Run


@jdc.pytree_dataclass
class TransitionStruct[TActionInfo]:
    """Transitions we'll return from rollouts."""

    obs: Array  # (*, obs_dim)
    next_obs: Array  # (*, obs_dim)
    action: Array  # (*, action_dim)
    action_info: TActionInfo
    reward: Array  # (*,)
    truncation: Array  # (*,)
    """1 if the environment was truncated, future value is 0."""
    discount: Array  # (*,)
    """1 during normal transitions, 0 if the environment terminated."""

    def prepare_minibatches(
        self,
        prng: Array,
        num_minibatches: int,
        minibatch_size: int,
    ) -> Self:
        """Prepare minibatches for training.

        Assumes inputs have leading axes (T, num_envs).
        Prepares outputs with shape (num_minibatches, unroll_length, batch_size, ...).
        """
        (T, num_envs, obs_dim) = self.obs.shape
        del obs_dim
        subseq_count = num_minibatches * minibatch_size
        subseq_length = T * num_envs // subseq_count  # same as unroll_length in brax.
        shuffle_indices = jax.random.permutation(prng, subseq_count)

        def prepare_batch(x: Array) -> Array:
            """Reshape transitions to add leading (num_batches, unroll_length, batch_size) axes."""
            suffix = x.shape[2:]
            x = x.swapaxes(0, 1)  # (iters, envs) => (envs, iters)
            x = x.reshape((-1, subseq_length) + suffix)
            x = x[shuffle_indices, ...]  # (idx, time)
            x = x.reshape((num_minibatches, minibatch_size, subseq_length) + suffix)
            x = x.swapaxes(1, 2)
            assert x.shape == (num_minibatches, subseq_length, minibatch_size) + suffix
            return x

        return jax.tree.map(prepare_batch, self)


class AgentState[TActionInfo](Protocol):
    env: mjp.MjxEnv

    def sample_action(
        self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, TActionInfo]: ...


@jdc.pytree_dataclass
class EvalOutputs:
    scalar_metrics: dict[str, Array]
    histogram_metrics: dict[str, Array]
    actions: Array  # shape: (T, B, action_dim)
    action_timestep_mask: Array  # shape: (T, B), 1 if action is valid, 0 if not

    def log_to_wandb(self, run: Run, step: int) -> None:
        """Log evaluation metrics to Weights & Biases.

        Args:
            run: The wandb run to log to.
            step: The current training step.
        """
        metrics = {}
        metrics.update(
            {
                f"eval/{k}": v
                for k, v in jax.tree.map(onp.array, self.scalar_metrics).items()
            }
        )

        # Log histograms with appropriate masking
        # First, log simple histograms (rewards, steps)
        for key, values in self.histogram_metrics.items():
            metrics[f"eval/{key}"] = wandb.Histogram(onp.array(values).tolist())

        # Then log action histograms with masking
        action_dim = self.actions.shape[-1]
        flat_actions = self.actions.reshape(-1, action_dim)
        flat_mask = self.action_timestep_mask.reshape(-1)

        for i in range(action_dim):
            # Apply mask to get only valid actions
            masked_actions = flat_actions[:, i][flat_mask > 0]
            if masked_actions.size > 0:  # Only log if there are valid actions
                metrics[f"eval/action_{i}"] = wandb.Histogram(masked_actions.tolist())

        run.log(metrics, step=step)


@jdc.jit
def eval_policy(
    agent_state: AgentState,
    prng: Array,
    num_envs: jdc.Static[int],
    max_episode_length: jdc.Static[int],
) -> EvalOutputs:
    """Run policy evaluation.

    Returns:
        EvalOutputs: A dataclass containing evaluation metrics and histograms
    """
    # Initialize rollout state
    rollout_state = BatchedRolloutState.init(agent_state.env, prng, num_envs)

    # Perform rollouts without auto-resetting and with deterministic actions
    _, transitions = rollout_state.rollout(
        agent_state,
        episode_length=max_episode_length,
        iterations_per_env=max_episode_length,
        auto_reset=False,
        deterministic=True,
    )
    valid_mask = transitions.discount > 0.0

    # Calculate rewards per episode by summing all rewards.
    rewards = jnp.sum(transitions.reward, axis=0)

    # Count steps per episode based on valid mask.
    steps = jnp.sum(valid_mask, axis=0)

    # Calculate scalar metrics
    scalar_metrics = {
        "reward_mean": jnp.mean(rewards),
        "reward_min": jnp.min(rewards),
        "reward_max": jnp.max(rewards),
        "reward_std": jnp.std(rewards),
        "steps_mean": jnp.mean(steps),
        "steps_min": jnp.min(steps),
        "steps_max": jnp.max(steps),
        "steps_std": jnp.std(steps),
    }

    # Histogram data for simple metrics
    histogram_metrics = {
        "reward": rewards.flatten(),
        "steps": steps.flatten(),
    }

    # Return the EvalOutputs dataclass with all metrics and masked actions
    return EvalOutputs(
        scalar_metrics=scalar_metrics,
        histogram_metrics=histogram_metrics,
        actions=transitions.action,  # Keep the original shape (T, B, action_dim)
        action_timestep_mask=valid_mask,  # Shape (T, B)
    )


@jdc.pytree_dataclass
class BatchedRolloutState:
    """Rollout state at one step."""

    env: jdc.Static[mjp.MjxEnv]
    env_state: mjp.State
    first_obs: Array
    first_data: mjx.Data
    steps: Array
    num_envs: jdc.Static[int]
    prng: Array

    @staticmethod
    @jdc.jit
    def init(
        env: jdc.Static[mjp.MjxEnv],
        prng: Array,
        num_envs: jdc.Static[int],
    ) -> BatchedRolloutState:
        """Reset the environment."""
        prng, reset_prng = jax.random.split(prng, num=2)
        state = jax.vmap(env.reset)(jax.random.split(reset_prng, num=num_envs))
        return BatchedRolloutState(
            env=env,
            env_state=state,
            first_obs=state.obs,  # type: ignore
            first_data=state.data,
            steps=jnp.zeros_like(state.done),
            # steps=jax.random.randint(
            #     prng, shape=state.done.shape, minval=0, maxval=1000
            # ),
            num_envs=num_envs,
            prng=prng,
        )

    @jdc.jit
    def rollout[TActionInfo](
        self,
        agent_state: AgentState[TActionInfo],
        episode_length: jdc.Static[int],
        iterations_per_env: jdc.Static[int],
        auto_reset: jdc.Static[bool] = True,
        deterministic: jdc.Static[bool] = False,
    ) -> tuple["BatchedRolloutState", TransitionStruct[TActionInfo]]:
        def env_step(carry: "BatchedRolloutState", _):
            state = carry

            # Sample action.
            prng_act, prng_next = jax.random.split(state.prng)
            assert isinstance(state.env_state.obs, Array)
            action, action_info = agent_state.sample_action(
                state.env_state.obs, prng_act, deterministic=deterministic
            )

            # Environment step.
            next_env_state = jax.vmap(state.env.step)(state.env_state, jnp.tanh(action))
            assert isinstance(next_env_state.obs, Array)

            # Bookkeeping.
            next_steps = state.steps + 1
            truncation = next_steps >= episode_length  # time-limit
            done_env = next_env_state.done.astype(bool)  # true death
            done_or_tr = jnp.logical_or(done_env, truncation)
            discount = 1.0 - done_env.astype(jnp.float32)  # **keep 1 on trunc**

            # Record transition.
            transition = TransitionStruct(
                obs=state.env_state.obs,
                next_obs=next_env_state.obs,
                action=action,
                action_info=action_info,
                reward=next_env_state.reward,
                truncation=truncation.astype(jnp.float32),
                discount=discount,
            )

            # Reset environment if auto_reset is True and env is done or truncated.
            next_state = state
            if auto_reset:
                where_done = lambda x, y: jnp.where(
                    done_or_tr.reshape(
                        done_or_tr.shape + (1,) * (x.ndim - done_or_tr.ndim)
                    ),
                    x,
                    y,
                )
                next_env_state = next_env_state.replace(  # type: ignore
                    obs=jax.tree.map(
                        where_done,
                        state.first_obs,
                        next_env_state.obs,
                    ),
                    data=jax.tree.map(
                        where_done,
                        state.first_data,
                        next_env_state.data,
                    ),
                    done=jnp.zeros_like(next_env_state.done),
                )

                # Update rollout state.
                with jdc.copy_and_mutate(next_state) as next_state:
                    next_state.env_state = next_env_state
                    next_state.steps = jnp.where(done_or_tr, 0, state.steps + 1)
                    next_state.prng = prng_next
            else:
                # Just update the state without resetting
                with jdc.copy_and_mutate(next_state) as next_state:
                    next_state.env_state = next_env_state
                    next_state.steps = next_steps  # Always increment steps
                    next_state.prng = prng_next

            return next_state, transition

        final_state, traj = jax.lax.scan(env_step, self, (), length=iterations_per_env)
        return final_state, traj


def compute_gae(
    truncation: Array,  # (T, B)
    discount: Array,  # (T, B)
    rewards: Array,  # (T, B)
    values: Array,  # (T, B)
    bootstrap_value: Array,  # (1, B)
    gae_lambda: float,
):
    """Computes `(values, advantages)` via GAE."""
    trunc_mask = 1 - truncation

    values_t_plus_1 = jnp.concatenate([values[1:], bootstrap_value], axis=0)
    deltas = rewards + discount * values_t_plus_1 - values

    # We don't compute values for the "next observation" of truncated
    # timesteps, so we shouldn't compute TD errors.
    deltas = deltas * trunc_mask
    accum_scale = discount * gae_lambda * trunc_mask

    def compute_vs_minus_v_xs(carry, x):
        acc = carry
        delta_t, accum_scale_t = x

        # discount_t is typically the constant gamma. It's set to 0 for
        # "done state" episodes.
        acc = delta_t + accum_scale_t * acc
        return acc, acc  # (carry, y)

    _, vs_minus_v_xs = jax.lax.scan(
        compute_vs_minus_v_xs,
        init=jnp.zeros_like(bootstrap_value.squeeze(axis=0)),
        xs=(deltas, accum_scale),
        reverse=True,
    )

    gae_values = jnp.add(vs_minus_v_xs, values)
    gae_values_t_plus_1 = jnp.concatenate([gae_values[1:], bootstrap_value], axis=0)
    advantages = rewards + discount * gae_values_t_plus_1 - values

    # No advantage estimated for the final timestep of truncated episodes, since
    # the value estimate for the next observation is wrong.
    advantages = advantages * trunc_mask
    return gae_values, advantages
