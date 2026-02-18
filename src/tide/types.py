"""Type aliases and named tuples for TIDE."""

from typing import NamedTuple

import jax.numpy as jnp

# Array type alias (JAX arrays)
Array = jnp.ndarray


class ZIBParams(NamedTuple):
    """Zero-Inflated Beta emission parameters per state.

    pi: (K,) probability of structural zero
    mu: (K,) Beta mean (0, 1)
    phi: (K,) Beta precision (> 0)
    """
    pi: Array
    mu: Array
    phi: Array


class HMMParams(NamedTuple):
    """Full HMM parameter set.

    log_init: (K,) log initial state probabilities
    log_trans: (K, K) or (N, T-1, K, K) log transition probabilities
    emission: ZIBParams
    transition_weights: (K, K-1, D) covariate transition weights, or None
    """
    log_init: Array
    log_trans: Array
    emission: ZIBParams
    transition_weights: Array | None = None


class ForwardBackwardResult(NamedTuple):
    """Results from forward-backward algorithm.

    log_gamma: (N, T, K) log posterior state probabilities
    log_xi: (N, T-1, K, K) log pairwise marginals (or None if not needed)
    log_likelihood: (N,) log-likelihood per sequence
    """
    log_gamma: Array
    log_xi: Array | None
    log_likelihood: Array


class ViterbiResult(NamedTuple):
    """Results from Viterbi algorithm.

    states: (N, T) int32 most likely state sequence
    log_prob: (N,) log probability of best path
    """
    states: Array
    log_prob: Array


class EMResult(NamedTuple):
    """Results from Baum-Welch EM.

    params: HMMParams final parameters
    log_likelihoods: list of per-iteration total log-likelihood
    converged: bool
    n_iter: int
    """
    params: HMMParams
    log_likelihoods: Array
    converged: bool
    n_iter: int
