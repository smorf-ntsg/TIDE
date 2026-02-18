"""M-step for Zero-Inflated Beta emission parameters.

Per-state optimization:
- pi_k: closed-form (weighted fraction of zeros)
- mu_k, phi_k: L-BFGS via jaxopt on weighted log-likelihood of non-zero obs
- Ordering constraint: mu_0 < mu_1 < ... < mu_{K-1} via cumulative sigmoid deltas
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from tide.types import Array, ZIBParams
from tide.emissions.zero_inflated_beta import constrain_params, unconstrain_params


def _weighted_zero_fraction(
    obs: Array,
    gamma: Array,
) -> Array:
    """Closed-form M-step for zero-inflation parameter pi.

    pi_k = sum_n,t gamma[n,t,k] * I(obs[n,t]==0) / sum_n,t gamma[n,t,k]

    Args:
        obs: (N, T) observations.
        gamma: (N, T, K) posterior responsibilities (not log).

    Returns:
        pi: (K,) updated zero-inflation probabilities.
    """
    is_zero = (obs == 0.0)  # (N, T)
    # Weighted count of zeros per state
    numer = jnp.sum(gamma * is_zero[:, :, None], axis=(0, 1))  # (K,)
    denom = jnp.sum(gamma, axis=(0, 1))  # (K,)
    pi = numer / jnp.maximum(denom, 1e-10)
    return jnp.clip(pi, 1e-6, 1.0 - 1e-6)


def _beta_nll_single_state(
    raw_params: Array,
    obs_nz: Array,
    weights_nz: Array,
) -> Array:
    """Negative weighted log-likelihood of Beta distribution for one state.

    Args:
        raw_params: (2,) unconstrained [raw_mu, raw_phi].
        obs_nz: (M,) non-zero observations.
        weights_nz: (M,) posterior weights for this state.

    Returns:
        Scalar negative weighted log-likelihood.
    """
    mu = jax.nn.sigmoid(raw_params[0])
    phi = jax.nn.softplus(raw_params[1]) + 1.0

    alpha = jnp.clip(mu * phi, 1e-4, 1e4)
    beta = jnp.clip((1.0 - mu) * phi, 1e-4, 1e4)

    log_pdf = (
        gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
        + (alpha - 1.0) * jnp.log(obs_nz)
        + (beta - 1.0) * jnp.log(1.0 - obs_nz)
    )

    return -jnp.sum(weights_nz * log_pdf)


def mstep_emissions(
    obs: Array,
    log_gamma: Array,
    prev_params: ZIBParams,
    lbfgs_maxiter: int = 50,
) -> tuple[ZIBParams, Array]:
    """Full M-step for ZIB emission parameters.

    1. pi_k: closed-form from weighted zero fractions
    2. mu_k, phi_k: L-BFGS optimization on non-zero observations
    3. Ordering constraint enforced via sorting mu

    Args:
        obs: (N, T) observations in [0, 1].
        log_gamma: (N, T, K) log posterior responsibilities.
        prev_params: Previous iteration parameters (warm start).
        lbfgs_maxiter: Max L-BFGS iterations.

    Returns:
        Tuple of (Updated ZIBParams, sort_idx permutation array).
    """
    import jaxopt

    K = log_gamma.shape[2]
    gamma = jnp.exp(log_gamma)  # (N, T, K)

    # Step 1: Closed-form pi update
    new_pi = _weighted_zero_fraction(obs, gamma)

    # Step 2: Optimize mu, phi per state on non-zero observations
    is_nonzero = (obs > 0.0)  # (N, T)
    obs_flat = obs.reshape(-1)  # (N*T,)
    nz_mask = is_nonzero.reshape(-1)  # (N*T,)

    # Clamp non-zero obs for Beta stability
    obs_flat_clamped = jnp.clip(obs_flat, 1e-6, 1.0 - 1e-6)

    new_mu_list = []
    new_phi_list = []

    for k in range(K):
        weights_flat = gamma[:, :, k].reshape(-1)  # (N*T,)
        # Extract non-zero observations and their weights for this state
        obs_nz = jnp.where(nz_mask, obs_flat_clamped, 0.5)  # 0.5 dummy: safe under log
        w_nz = jnp.where(nz_mask, weights_flat, 0.0)

        # Initialize from previous params (inverse sigmoid for mu, inverse softplus for phi)
        mu_k = jnp.clip(prev_params.mu[k], 1e-6, 1.0 - 1e-6)
        mu_init = jnp.log(mu_k / (1.0 - mu_k))
        # Stable inverse softplus: for large x, inv_softplus(x) ≈ x
        phi_shifted = jnp.maximum(prev_params.phi[k] - 1.0, 0.01)
        phi_init = jnp.where(
            phi_shifted > 20.0,
            phi_shifted,
            jnp.log(jnp.expm1(phi_shifted)),
        )
        raw_init = jnp.array([mu_init, phi_init])

        solver = jaxopt.LBFGS(
            fun=_beta_nll_single_state,
            maxiter=lbfgs_maxiter,
            tol=1e-5,
        )
        result = solver.run(raw_init, obs_nz=obs_nz, weights_nz=w_nz)
        raw_opt = result.params

        new_mu_list.append(jax.nn.sigmoid(raw_opt[0]))
        new_phi_list.append(jax.nn.softplus(raw_opt[1]) + 1.0)

    new_mu = jnp.array(new_mu_list)
    new_phi = jnp.array(new_phi_list)

    # Enforce monotonicity: sort mu and reorder phi accordingly
    sort_idx = jnp.argsort(new_mu)
    new_mu = new_mu[sort_idx]
    new_phi = new_phi[sort_idx]
    new_pi = new_pi[sort_idx]

    return ZIBParams(pi=new_pi, mu=new_mu, phi=new_phi), sort_idx
