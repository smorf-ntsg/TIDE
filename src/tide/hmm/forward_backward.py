"""Log-space forward-backward algorithm using jax.lax.scan.

All computation in log-space for numerical stability.
Supports both static (K, K) and dynamic (N, T-1, K, K) transition matrices.
Uses jax.vmap over the pixel (N) dimension.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax

from tide.types import Array, ForwardBackwardResult


def _forward_single(
    log_emission: Array,
    log_init: Array,
    log_trans_seq: Array,
) -> tuple[Array, Array]:
    """Forward pass for a single pixel time series.

    Args:
        log_emission: (T, K) log emission probabilities.
        log_init: (K,) log initial state probabilities.
        log_trans_seq: (T-1, K, K) log transition matrices per timestep.

    Returns:
        log_alpha: (T, K) forward log-probabilities.
        log_evidence: scalar total log-likelihood.
    """
    T, K = log_emission.shape

    # Initialize: alpha_0 = init * emission_0
    log_alpha_0 = log_init + log_emission[0]

    def scan_fn(log_alpha_prev, inputs):
        lt, emit = inputs
        # sum_i alpha_prev[i] * A[i, j] for each j
        log_alpha_t = (
            jax.nn.logsumexp(log_alpha_prev[:, None] + lt, axis=0)
            + emit
        )
        return log_alpha_t, log_alpha_t

    # Scan over t = 1, ..., T-1
    _, log_alphas_rest = lax.scan(
        scan_fn, log_alpha_0, (log_trans_seq, log_emission[1:])
    )

    # Stack: (T, K)
    log_alpha = jnp.concatenate([log_alpha_0[None, :], log_alphas_rest], axis=0)

    # Total log-likelihood
    log_evidence = jax.nn.logsumexp(log_alpha[-1])

    return log_alpha, log_evidence


def _backward_single(
    log_emission: Array,
    log_trans_seq: Array,
) -> Array:
    """Backward pass for a single pixel time series.

    Args:
        log_emission: (T, K) log emission probabilities.
        log_trans_seq: (T-1, K, K) log transition matrices per timestep.

    Returns:
        log_beta: (T, K) backward log-probabilities.
    """
    T, K = log_emission.shape

    # Initialize: beta_T = 0 (log(1) = 0)
    log_beta_T = jnp.zeros(K)

    def scan_fn(log_beta_next, inputs):
        lt, emit_next = inputs
        # sum_j A[i, j] * emission[t+1, j] * beta[t+1, j] for each i
        log_beta_t = jax.nn.logsumexp(
            lt + emit_next[None, :] + log_beta_next[None, :],
            axis=1,
        )
        return log_beta_t, log_beta_t

    # Scan over t = T-2, ..., 0 (reversed)
    _, log_betas_rest = lax.scan(
        scan_fn, log_beta_T,
        (log_trans_seq[::-1], log_emission[1:][::-1]),
    )

    # Reverse to get chronological order, then append terminal
    log_beta = jnp.concatenate([log_betas_rest[::-1], log_beta_T[None, :]], axis=0)

    return log_beta


def _compute_xi_single(
    log_alpha: Array,
    log_beta: Array,
    log_emission: Array,
    log_trans_seq: Array,
    log_evidence: Array,
) -> Array:
    """Compute log pairwise marginals for a single pixel.

    log_xi[t, i, j] = log P(z_t=i, z_{t+1}=j | observations)

    Args:
        log_alpha: (T, K)
        log_beta: (T, K)
        log_emission: (T, K)
        log_trans_seq: (T-1, K, K)
        log_evidence: scalar

    Returns:
        log_xi: (T-1, K, K)
    """

    def compute_xi_t(inputs):
        alpha_t, beta_tp1, emit_tp1, lt = inputs
        # xi[t, i, j] = alpha[t, i] * A[i, j] * emission[t+1, j] * beta[t+1, j] / P(obs)
        log_xi_t = (
            alpha_t[:, None]       # (K, 1)
            + lt                   # (K, K)
            + emit_tp1[None, :]    # (1, K)
            + beta_tp1[None, :]    # (1, K)
            - log_evidence         # scalar
        )
        return log_xi_t

    log_xi = jax.vmap(compute_xi_t)((
        log_alpha[:-1],       # (T-1, K)
        log_beta[1:],         # (T-1, K)
        log_emission[1:],     # (T-1, K)
        log_trans_seq,        # (T-1, K, K)
    ))

    return log_xi


@functools.partial(jax.jit, static_argnames=["compute_xi"])
def forward_backward(
    log_emission: Array,
    log_init: Array,
    log_trans: Array,
    compute_xi: bool = True,
) -> ForwardBackwardResult:
    """Batched forward-backward algorithm.

    All computation in log-space. Uses jax.vmap over the N (pixel) dimension.

    Args:
        log_emission: (N, T, K) log emission probabilities.
        log_init: (K,) log initial state probabilities.
        log_trans: (K, K) static or (N, T-1, K, K) dynamic log transitions.
        compute_xi: Whether to compute pairwise marginals (needed for EM).

    Returns:
        ForwardBackwardResult with log_gamma, log_xi (or None), log_likelihood.
    """
    N, T, K = log_emission.shape

    if log_trans.ndim == 4:
        # Dynamic: (N, T-1, K, K) — already per-pixel
        log_trans_batched = log_trans
        trans_vmap_axis = 0
    else:
        # Static: (K, K) — broadcast to (T-1, K, K)
        log_trans_batched = jnp.broadcast_to(
            log_trans[None, :, :], (T - 1, K, K)
        )
        trans_vmap_axis = None

    log_alpha, log_evidence = jax.vmap(
        _forward_single, in_axes=(0, None, trans_vmap_axis)
    )(log_emission, log_init, log_trans_batched)

    log_beta = jax.vmap(
        _backward_single, in_axes=(0, trans_vmap_axis)
    )(log_emission, log_trans_batched)

    # Posterior: gamma[t, k] = alpha[t, k] * beta[t, k] / P(obs)
    log_gamma = log_alpha + log_beta - log_evidence[:, None, None]

    # Pairwise marginals (optional, memory-intensive at K=5)
    log_xi = None
    if compute_xi:
        log_xi = jax.vmap(
            _compute_xi_single, in_axes=(0, 0, 0, trans_vmap_axis, 0)
        )(log_alpha, log_beta, log_emission, log_trans_batched, log_evidence)

    return ForwardBackwardResult(
        log_gamma=log_gamma,
        log_xi=log_xi,
        log_likelihood=log_evidence,
    )
