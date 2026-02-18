"""Log-space Viterbi algorithm using jax.lax.scan.

Finds the most likely state sequence (MAP estimate).
Reference: existing NumPy Viterbi in hmm_processor.py:126-160.
"""

import jax
import jax.numpy as jnp
from jax import lax

from tide.types import Array, ViterbiResult


# JIT-compiled entry points for static and dynamic transition matrices.
# Separate functions avoid recompilation when switching between modes.


def _viterbi_single(
    log_emission: Array,
    log_init: Array,
    log_trans_seq: Array,
) -> tuple[Array, Array]:
    """Viterbi decoding for a single pixel time series.

    Args:
        log_emission: (T, K) log emission probabilities.
        log_init: (K,) log initial state probabilities.
        log_trans_seq: (T-1, K, K) log transition matrices per timestep.

    Returns:
        states: (T,) int32 most likely state sequence.
        log_prob: scalar log probability of best path.
    """
    T, K = log_emission.shape

    # Initialize: delta_0 = init * emission_0
    delta_0 = log_init + log_emission[0]  # (K,)

    def forward_fn(delta_prev, inputs):
        lt, emit = inputs
        # candidates[i, j] = delta_prev[i] + log_trans[i, j]
        candidates = delta_prev[:, None] + lt  # (K, K)
        # Best previous state for each current state
        psi_t = candidates.argmax(axis=0)  # (K,)
        delta_t = candidates.max(axis=0) + emit  # (K,)
        return delta_t, psi_t

    # Forward scan: t = 1, ..., T-1
    delta_final, psi_all = lax.scan(
        forward_fn, delta_0, (log_trans_seq, log_emission[1:])
    )
    # psi_all: (T-1, K)

    # Best final state
    best_last = delta_final.argmax()
    log_prob = delta_final.max()

    # Backtracking via reverse scan
    def backtrack_fn(state, psi_t):
        prev_state = psi_t[state]
        return prev_state, prev_state

    _, states_reversed = lax.scan(
        backtrack_fn,
        best_last,
        psi_all[::-1],  # Reverse: from T-2 to 0
    )

    # Assemble full state sequence
    states = jnp.concatenate([states_reversed[::-1], best_last[None]])

    return states.astype(jnp.int32), log_prob


@jax.jit
def viterbi(
    log_emission: Array,
    log_init: Array,
    log_trans: Array,
) -> ViterbiResult:
    """Batched Viterbi decoding.

    Uses jax.vmap over the N (pixel) dimension.

    Args:
        log_emission: (N, T, K) log emission probabilities.
        log_init: (K,) log initial state probabilities.
        log_trans: (K, K) static or (N, T-1, K, K) dynamic log transitions.

    Returns:
        ViterbiResult with states (N, T) and log_prob (N,).
    """
    N, T, K = log_emission.shape

    if log_trans.ndim == 4:
        # Dynamic: (N, T-1, K, K) — vmap over pixel dim
        states, log_prob = jax.vmap(
            _viterbi_single, in_axes=(0, None, 0)
        )(log_emission, log_init, log_trans)
    else:
        # Static: (K, K) — broadcast to (T-1, K, K) for uniform scan interface
        log_trans_seq = jnp.broadcast_to(log_trans[None, :, :], (T - 1, K, K))
        states, log_prob = jax.vmap(
            _viterbi_single, in_axes=(0, None, None)
        )(log_emission, log_init, log_trans_seq)

    return ViterbiResult(states=states, log_prob=log_prob)
