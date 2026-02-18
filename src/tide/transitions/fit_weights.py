"""Post-EM transition weight estimation.

Alternates between:
  E-step: forward-backward with dynamic transitions → xi marginals
  M-step: multinomial logistic regression on xi → updated weights

Emission parameters are frozen from the preceding Baum-Welch run.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np

from tide.types import Array, HMMParams, ZIBParams
from tide.emissions.zero_inflated_beta import zib_log_prob
from tide.hmm.forward_backward import forward_backward
from tide.hmm.mstep_transitions import mstep_transitions_dynamic
from tide.transitions.dynamic import (
    compute_dynamic_log_trans,
    init_transition_weights,
)

log = logging.getLogger(__name__)


def _batched_estep(
    obs: Array,
    covariates: Array,
    emission: ZIBParams,
    log_init: Array,
    weights: Array,
    K: int,
    batch_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run E-step in batches to limit GPU memory.

    Returns numpy arrays for log_xi, log_gamma, and total log-likelihood.
    """
    N = obs.shape[0]
    all_log_xi = []
    all_log_gamma = []
    total_ll = 0.0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        obs_batch = obs[start:end]
        cov_batch = covariates[start:end]

        # Emissions for this batch
        log_em = zib_log_prob(obs_batch, emission)

        # Dynamic transitions for this batch
        dyn_log_trans = compute_dynamic_log_trans(cov_batch, weights, K=K)

        # Forward-backward
        fb = forward_backward(log_em, log_init, dyn_log_trans, compute_xi=True)

        # Copy to CPU immediately to free GPU memory
        all_log_xi.append(np.asarray(fb.log_xi))
        all_log_gamma.append(np.asarray(fb.log_gamma))
        total_ll += float(jnp.sum(fb.log_likelihood))

        del log_em, dyn_log_trans, fb

    log_xi = jnp.array(np.concatenate(all_log_xi, axis=0))
    log_gamma = jnp.array(np.concatenate(all_log_gamma, axis=0))

    return log_xi, log_gamma, total_ll


def fit_transition_weights(
    obs: Array,
    covariates: Array,
    emission: ZIBParams,
    log_init: Array,
    K: int = 6,
    max_iter: int = 20,
    tol: float = 1e-3,
    l2_reg: float = 1e-3,
    lbfgs_maxiter: int = 50,
    init_weights: Array | None = None,
    batch_size: int = 10_000,
) -> tuple[Array, list[float]]:
    """Estimate covariate transition weights with frozen emissions.

    Iterates E-step (forward-backward) and M-step (L-BFGS per source state)
    until log-likelihood converges.

    Args:
        obs: (N, T) float32 observed tree cover in [0, 1].
        covariates: (N, T-1, D) covariate features.
        emission: Frozen ZIB emission parameters.
        log_init: (K,) log initial state probabilities.
        K: Number of HMM states.
        max_iter: Maximum EM-style iterations.
        tol: Relative convergence tolerance on log-likelihood.
        l2_reg: L2 regularization strength for weight estimation.
        lbfgs_maxiter: Max inner L-BFGS iterations per M-step.
        init_weights: (K, K-1, D) initial weights, or None for zeros.
        batch_size: Pixels per GPU batch during E-step.

    Returns:
        weights: (K, K-1, D) estimated transition weights.
        log_likelihoods: Per-iteration total log-likelihood.
    """
    N, T = obs.shape
    D = covariates.shape[2]

    if init_weights is None:
        weights = init_transition_weights(K, D)
    else:
        weights = init_weights

    log_likelihoods = []
    prev_ll = -jnp.inf

    for iteration in range(max_iter):
        # E-step: batched forward-backward with dynamic transitions
        log_xi, log_gamma, total_ll = _batched_estep(
            obs, covariates, emission, log_init, weights, K,
            batch_size=batch_size,
        )

        log_likelihoods.append(total_ll)
        log.info(f"Weight estimation iter {iteration}: LL = {total_ll:.2f}")

        # Check convergence
        if iteration > 0:
            rel_change = abs(total_ll - prev_ll) / max(abs(prev_ll), 1.0)
            if rel_change < tol:
                log.info(
                    f"Converged at iteration {iteration} "
                    f"(rel_change={rel_change:.2e} < tol={tol:.2e})"
                )
                break
        prev_ll = total_ll

        # M-step: update weights via multinomial logistic regression
        weights = mstep_transitions_dynamic(
            log_xi,
            log_gamma,
            covariates,
            prev_weights=weights,
            l2_reg=l2_reg,
            maxiter=lbfgs_maxiter,
        )

    return weights, log_likelihoods
