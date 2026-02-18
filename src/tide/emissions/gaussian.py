"""Gaussian emission model for integration testing.

Provides a simple 2-state Gaussian HMM used in unit tests to verify
forward-backward and Viterbi correctness against a known reference implementation.
"""

import jax.numpy as jnp

from tide.types import Array


def gaussian_log_prob(
    obs: Array,
    means: Array,
    variances: Array,
) -> Array:
    """Compute log emission probabilities under Gaussian model.

    Args:
        obs: (N, T) float32 observations (raw percent, not scaled).
        means: (K,) state means.
        variances: (K,) state variances.

    Returns:
        (N, T, K) log emission probabilities.
    """
    obs_3d = obs[:, :, None]  # (N, T, 1)
    mu = means[None, None, :]  # (1, 1, K)
    var = variances[None, None, :]  # (1, 1, K)

    log_prob = -0.5 * (
        jnp.log(2.0 * jnp.pi * var) + (obs_3d - mu) ** 2 / var
    )

    return log_prob
