"""Covariate-driven dynamic transition model.

Computes per-pixel, per-timestep transition matrices via softmax
over covariate features:

    logits[i, j] = covariates @ weights[i, j, :]  (j != i)
    logits[i, i] = 0  (reference: staying)
    A[i, :] = softmax(logits[i, :])
"""

import functools

import jax
import jax.numpy as jnp

from tide.types import Array


@functools.partial(jax.jit, static_argnames=["K"])
def compute_dynamic_log_trans(
    covariates: Array,
    weights: Array,
    K: int = 6,
) -> Array:
    """Compute dynamic log transition matrices from covariates.

    Args:
        covariates: (N, T-1, D) covariate features per pixel-timestep.
        weights: (K, K-1, D) transition weights per source state.
            For source state i, weights[i] has K-1 rows for non-reference targets.

    Returns:
        log_trans: (N, T-1, K, K) log transition matrices.
    """
    N, Tm1, D = covariates.shape

    def _compute_single_row(cov_ntm1, weights_i, source_idx):
        """Compute transition probs from one source state for all pixels/times.

        Args:
            cov_ntm1: (D,) covariates for one pixel-timestep.
            weights_i: (K-1, D) weights for non-reference targets.
            source_idx: int, source state index.

        Returns:
            (K,) log transition probabilities from state source_idx.
        """
        # Logits for non-reference states
        logits_nonref = cov_ntm1 @ weights_i.T  # (K-1,)

        # Insert zero for reference (self-transition)
        logits = jnp.zeros(K)
        mask = jnp.arange(K) != source_idx
        logits = jnp.where(mask, _scatter_nonref(logits_nonref, source_idx, K), 0.0)

        return jax.nn.log_softmax(logits)

    def _scatter_nonref(vals, ref_idx, K):
        """Place K-1 values into K-length array, skipping ref_idx."""
        out = jnp.zeros(K)
        below = jnp.arange(K) < ref_idx
        above = jnp.arange(K) > ref_idx
        # Indices into vals: positions below ref_idx map to 0..ref_idx-1
        # positions above ref_idx map to ref_idx..K-2
        val_idx = jnp.where(below, jnp.arange(K), jnp.arange(K) - 1)
        val_idx = jnp.clip(val_idx, 0, K - 2)
        out = jnp.where(below | above, vals[val_idx], 0.0)
        return out

    def _compute_all_sources(cov_single):
        """Compute full K x K log transition matrix for one pixel-timestep."""
        rows = []
        for i in range(K):
            row = _compute_single_row(cov_single, weights[i], i)
            rows.append(row)
        return jnp.stack(rows)  # (K, K)

    # Vectorize over N and T-1 dimensions
    cov_flat = covariates.reshape(-1, D)  # (N*(T-1), D)
    log_trans_flat = jax.vmap(_compute_all_sources)(cov_flat)  # (N*(T-1), K, K)
    log_trans = log_trans_flat.reshape(N, Tm1, K, K)

    return log_trans


def init_transition_weights(K: int = 6, D: int = 8) -> Array:
    """Initialize transition weights to zeros (softmax gives uniform non-self probs).

    Args:
        K: Number of states.
        D: Number of covariate features.

    Returns:
        weights: (K, K-1, D) initialized weights.
    """
    return jnp.zeros((K, K - 1, D), dtype=jnp.float32)
