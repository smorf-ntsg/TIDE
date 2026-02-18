"""Baum-Welch EM algorithm for ZIB-HMM parameter estimation.

Runs on a stratified sample of pixels (not all 1.85B).
Learned parameters are then frozen for full-dataset inference.
"""

import logging

import jax.numpy as jnp

from tide.types import Array, HMMParams, ZIBParams, EMResult
from tide.emissions.zero_inflated_beta import zib_log_prob
from tide.hmm.forward_backward import forward_backward
from tide.hmm.mstep_zib import mstep_emissions
from tide.hmm.mstep_transitions import mstep_transitions_static

log = logging.getLogger(__name__)


def _init_params(K: int = 6) -> HMMParams:
    """Initialize HMM parameters from ecological priors."""
    from tide.config import DEFAULT_INIT_PROBS, DEFAULT_TRANS
    from tide.emissions.zero_inflated_beta import init_default_params

    log_init = jnp.log(jnp.array(DEFAULT_INIT_PROBS, dtype=jnp.float32))
    log_trans = jnp.log(jnp.array(DEFAULT_TRANS, dtype=jnp.float32))
    emission = init_default_params(K)

    return HMMParams(log_init=log_init, log_trans=log_trans, emission=emission)


def baum_welch(
    obs: Array,
    K: int = 6,
    max_iter: int = 100,
    tol: float = 1e-4,
    lbfgs_maxiter: int = 50,
    params: HMMParams | None = None,
) -> EMResult:
    """Run Baum-Welch EM to estimate ZIB-HMM parameters.

    Args:
        obs: (N, T) float32 observations in [0, 1].
        K: Number of states.
        max_iter: Maximum EM iterations.
        tol: Relative log-likelihood convergence tolerance.
        lbfgs_maxiter: Max inner L-BFGS iterations for emission M-step.
        params: Optional initial parameters (otherwise use defaults).

    Returns:
        EMResult with learned parameters and convergence info.
    """
    if params is None:
        params = _init_params(K)

    N, T = obs.shape
    log_likelihoods = []

    for iteration in range(max_iter):
        # --- E-step ---
        log_emission = zib_log_prob(obs, params.emission)  # (N, T, K)

        fb_result = forward_backward(
            log_emission, params.log_init, params.log_trans,
            compute_xi=True,
        )

        # float() forces a single GPU→CPU sync to read the scalar. Using float()
        # rather than .item() avoids a second implicit transfer when the value is
        # used in Python arithmetic below.
        total_ll = float(jnp.sum(fb_result.log_likelihood))
        log_likelihoods.append(total_ll)
        log.info(f"EM iter {iteration}: log-likelihood = {total_ll:.2f}")

        # Check convergence
        if iteration > 0:
            prev_ll = log_likelihoods[-2]
            rel_change = abs(total_ll - prev_ll) / max(abs(prev_ll), 1.0)
            if rel_change < tol:
                log.info(
                    f"Converged at iteration {iteration} "
                    f"(rel_change={rel_change:.2e} < tol={tol:.2e})"
                )
                return EMResult(
                    params=params,
                    log_likelihoods=jnp.array(log_likelihoods),
                    converged=True,
                    n_iter=iteration + 1,
                )

        # --- M-step: initial probabilities ---
        gamma_0 = jnp.exp(fb_result.log_gamma[:, 0, :])  # (N, K)
        new_init = gamma_0.mean(axis=0)
        new_init = jnp.clip(new_init, 1e-8, None)
        new_init = new_init / new_init.sum()
        new_log_init = jnp.log(new_init)

        # --- M-step: static transitions ---
        new_log_trans = mstep_transitions_static(
            fb_result.log_xi, fb_result.log_gamma
        )

        # --- M-step: emissions ---
        new_emission, sort_idx = mstep_emissions(
            obs, fb_result.log_gamma, params.emission,
            lbfgs_maxiter=lbfgs_maxiter,
        )

        # Apply emission sort permutation to init and trans for consistency
        new_log_init = new_log_init[sort_idx]
        new_log_trans = new_log_trans[sort_idx][:, sort_idx]

        params = HMMParams(
            log_init=new_log_init,
            log_trans=new_log_trans,
            emission=new_emission,
        )

    log.warning(f"EM did not converge after {max_iter} iterations")
    return EMResult(
        params=params,
        log_likelihoods=jnp.array(log_likelihoods),
        converged=False,
        n_iter=max_iter,
    )
