from __future__ import annotations


import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as onp
from jax import Array


@jdc.pytree_dataclass
class NormalDistribution:
    """Normal distribution.

    Only operates elementwise.
    """

    loc: Array
    scale: Array

    def sample(self, seed: Array) -> Array:
        return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

    def log_prob(self, x: Array) -> Array:
        log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        return log_unnormalized - log_normalization

    def entropy(self) -> Array:
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        entropy = 0.5 + log_normalization
        return entropy * jnp.ones_like(self.loc)


def tanh_log_det_jacobian(x: Array) -> Array:
    return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))


@jdc.pytree_dataclass
class RunningStats:
    """Running statistics using Welford's method."""

    count: Array  # ()
    mean: Array  # (*shape,)
    var_sum: Array  # (*shape,)
    std: Array  # (*shape,)

    @staticmethod
    def init(shape: tuple[int, ...]) -> RunningStats:
        return RunningStats(
            # int32 overflows too easily, float32 should be fine.
            count=jnp.zeros((), dtype=jnp.float32),
            mean=jnp.zeros(shape),
            var_sum=jnp.zeros(shape),
            # Make sure normalization works correctly in the initial state.
            std=jnp.ones(shape),
        )

    def update(self, x: Array) -> RunningStats:
        """Update running stats with a new batch of observations."""

        batch_ndims = x.ndim - self.mean.ndim
        assert x.shape[batch_ndims:] == self.mean.shape == self.var_sum.shape

        new_count = self.count + onp.prod(x.shape[:batch_ndims])
        diff_to_old_mean = x - self.mean
        new_mean = (
            self.mean + jnp.sum(diff_to_old_mean, axis=range(batch_ndims)) / new_count
        )
        new_var_sum = jnp.sum(
            diff_to_old_mean * (x - new_mean), axis=range(batch_ndims)
        )
        var_clipped = jnp.clip(new_var_sum / new_count, 1e-12, 1e12)
        return RunningStats(
            count=new_count,
            mean=new_mean,
            var_sum=new_var_sum,
            std=jnp.sqrt(var_clipped),
        )
