"""Zero-Inflated Beta (ZIB) emission model for bounded [0,1] tree cover data.

The ZIB distribution handles the mixture of:
- Structural zeros (truly treeless pixels) via zero-inflation parameter pi
- Continuous positive values via Beta(alpha, beta) parameterized as (mu, phi)

This resolves the fundamental flaw of Gaussian emissions on bounded data,
where ~41% of probability mass falls below zero.
"""

import functools

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from tide.types import Array, ZIBParams


def _log_beta_pdf(x: Array, alpha: Array, beta: Array) -> Array:
    """Log PDF of Beta distribution.

    Uses gammaln for numerical stability:
        log B(x; a, b) = gammaln(a+b) - gammaln(a) - gammaln(b)
                        + (a-1)*log(x) + (b-1)*log(1-x)

    Args:
        x: Observations in (0, 1), shape (...)
        alpha: Shape parameter > 0, shape (...)
        beta: Shape parameter > 0, shape (...)

    Returns:
        Log PDF values, shape (...)
    """
    return (
        gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
        + (alpha - 1.0) * jnp.log(x)
        + (beta - 1.0) * jnp.log(1.0 - x)
    )


@functools.partial(jax.jit, static_argnames=["eps"])
def zib_log_prob(
    obs: Array,
    params: ZIBParams,
    eps: float = 1e-6,
) -> Array:
    """Compute log emission probabilities under Zero-Inflated Beta model.

    For each observation x and state k:
        if x == 0: log P(x|k) = log(pi_k)
        if x > 0:  log P(x|k) = log(1 - pi_k) + log_beta_pdf(x; mu_k*phi_k, (1-mu_k)*phi_k)

    Args:
        obs: (N, T) float32 observations in [0, 1]. Exact zeros are structural.
        params: ZIBParams with pi (K,), mu (K,), phi (K,).
        eps: Clamp boundary for Beta PDF evaluation.

    Returns:
        (N, T, K) log emission probabilities.
    """
    pi, mu, phi = params.pi, params.mu, params.phi
    K = pi.shape[0]

    # Clamp non-zero observations away from boundaries for Beta PDF stability
    obs_clamped = jnp.clip(obs, eps, 1.0 - eps)

    # Beta shape parameters: alpha = mu * phi, beta = (1 - mu) * phi
    alpha = mu * phi  # (K,)
    beta = (1.0 - mu) * phi  # (K,)

    # Broadcast: obs_clamped (N, T) -> (N, T, 1), alpha/beta (K,) -> (1, 1, K)
    obs_3d = obs_clamped[:, :, None]  # (N, T, 1)

    log_beta = _log_beta_pdf(obs_3d, alpha[None, None, :], beta[None, None, :])  # (N, T, K)

    # Zero indicator: (N, T)
    is_zero = (obs == 0.0)

    # Log probabilities for zero and non-zero components
    log_pi = jnp.log(jnp.clip(pi, eps, 1.0 - eps))  # (K,)
    log_1m_pi = jnp.log(jnp.clip(1.0 - pi, eps, 1.0 - eps))  # (K,)

    # For x == 0: log(pi_k)
    log_prob_zero = log_pi[None, None, :]  # (1, 1, K) broadcast to (N, T, K)

    # For x > 0: log(1 - pi_k) + log_beta_pdf(x; alpha_k, beta_k)
    log_prob_pos = log_1m_pi[None, None, :] + log_beta  # (N, T, K)

    # Select based on zero indicator
    log_prob = jnp.where(is_zero[:, :, None], log_prob_zero, log_prob_pos)

    return log_prob


def constrain_params(
    raw_pi: Array,
    raw_mu_deltas: Array,
    raw_phi: Array,
) -> ZIBParams:
    """Transform unconstrained parameters to valid ZIB parameter space.

    Enforces:
        - pi in (0, 1) via sigmoid
        - mu in (0, 1) with monotonic ordering via recursive stick-breaking
        - phi > 0 via softplus

    The monotonicity constraint (mu_0 < mu_1 < ... < mu_{K-1}) prevents
    label switching during EM optimization.

    Args:
        raw_pi: (K,) unconstrained zero-inflation logits
        raw_mu_deltas: (K,) unconstrained increment logits. 
            Each determines fraction of *remaining* interval [mu_{k-1}, 1].
        raw_phi: (K,) unconstrained precision values

    Returns:
        ZIBParams with properly constrained values.
    """
    pi = jax.nn.sigmoid(raw_pi)

    # Monotonic mu: Recursive stick-breaking
    # mu_k = mu_{k-1} + (1 - mu_{k-1}) * sigmoid(raw_delta_k)
    # This ensures 0 < mu_0 < mu_1 < ... < 1 for all K.
    
    deltas = jax.nn.sigmoid(raw_mu_deltas)
    
    def step_fn(prev_mu, delta):
        # Add small epsilon to ensures strict monotonicity
        fraction = jnp.clip(delta, 1e-6, 1.0 - 1e-6)
        new_mu = prev_mu + (1.0 - prev_mu) * fraction
        return new_mu, new_mu

    # Scan accumulates the stick-breaking process
    _, mu = jax.lax.scan(step_fn, 0.0, deltas)
    
    # Ensure mu stays safely away from 0 and 1
    mu = jnp.clip(mu, 1e-4, 0.999)

    phi = jax.nn.softplus(raw_phi) + 1.0  # Minimum precision of 1

    return ZIBParams(pi=pi, mu=mu, phi=phi)


def unconstrain_params(params: ZIBParams) -> tuple[Array, Array, Array]:
    """Invert constrain_params for initialization.

    Args:
        params: ZIBParams with constrained values.

    Returns:
        (raw_pi, raw_mu_deltas, raw_phi) unconstrained arrays.
    """
    # Inverse sigmoid: logit
    raw_pi = jnp.log(params.pi / (1.0 - params.pi))

    # Inverse recursive stick-breaking
    # mu_k = mu_{k-1} + (1 - mu_{k-1}) * fraction_k
    # fraction_k = (mu_k - mu_{k-1}) / (1 - mu_{k-1})
    
    mu_padded = jnp.concatenate([jnp.array([0.0]), params.mu])
    prev_mu = mu_padded[:-1]
    curr_mu = mu_padded[1:]
    
    # Avoid division by zero if mu is saturated
    denom = jnp.maximum(1.0 - prev_mu, 1e-6)
    fractions = (curr_mu - prev_mu) / denom
    
    # Clip to valid sigmoid range
    fractions = jnp.clip(fractions, 1e-6, 1.0 - 1e-6)
    
    raw_mu_deltas = jnp.log(fractions / (1.0 - fractions))

    # Inverse softplus: log(exp(x) - 1) where x = phi - 1
    phi_shifted = params.phi - 1.0
    raw_phi = jnp.log(jnp.exp(jnp.clip(phi_shifted, 0.01, 20.0)) - 1.0)

    return raw_pi, raw_mu_deltas, raw_phi


def init_default_params(K: int = 6) -> ZIBParams:
    """Create default ZIB parameters from ecological priors.

    Args:
        K: Number of states. Must match length of DEFAULT_INIT arrays in config.

    Returns:
        ZIBParams with ecologically-motivated initial values.
    """
    from tide.config import DEFAULT_INIT_PI, DEFAULT_INIT_MU, DEFAULT_INIT_PHI

    if K != len(DEFAULT_INIT_PI):
        raise ValueError(
            f"K={K} does not match config defaults (len={len(DEFAULT_INIT_PI)}). "
            f"Provide explicit ZIBParams for non-default K."
        )

    return ZIBParams(
        pi=jnp.array(DEFAULT_INIT_PI, dtype=jnp.float32),
        mu=jnp.array(DEFAULT_INIT_MU, dtype=jnp.float32),
        phi=jnp.array(DEFAULT_INIT_PHI, dtype=jnp.float32),
    )
