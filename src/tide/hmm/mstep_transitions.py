"""M-step for transition parameters.

Static transitions: closed-form from xi / gamma.
Dynamic transitions: weighted multinomial logistic regression per source state.
"""

import jax
import jax.numpy as jnp

from tide.types import Array


def mstep_transitions_static(
    log_xi: Array,
    log_gamma: Array,
) -> Array:
    """Closed-form M-step for static transition matrix.

    A[i, j] = sum_{n,t} xi[n,t,i,j] / sum_{n,t} gamma[n,t,i]

    Args:
        log_xi: (N, T-1, K, K) log pairwise marginals.
        log_gamma: (N, T, K) log posterior responsibilities.

    Returns:
        log_trans: (K, K) updated log transition matrix.
    """
    # Sum xi over pixels and timesteps (in log-space, exponentiate first)
    xi_sum = jnp.sum(jnp.exp(log_xi), axis=(0, 1))  # (K, K)
    # Gamma for t=0..T-2 (not the last timestep)
    gamma_sum = jnp.sum(jnp.exp(log_gamma[:, :-1, :]), axis=(0, 1))  # (K,)

    # Normalize rows
    trans = xi_sum / jnp.maximum(gamma_sum[:, None], 1e-10)
    trans = jnp.clip(trans, 1e-8, None)
    # Re-normalize to ensure proper probability
    trans = trans / trans.sum(axis=1, keepdims=True)

    return jnp.log(trans)


def mstep_transitions_dynamic(
    log_xi: Array,
    log_gamma: Array,
    covariates: Array,
    prev_weights: Array,
    l2_reg: float = 1e-3,
    maxiter: int = 50,
) -> Array:
    """L-BFGS M-step for covariate-driven transition weights.

    For each source state i, fits multinomial logistic regression:
        P(j | i, x) = softmax(x @ W[i])_j

    with state i as the reference category (weight = 0).

    Args:
        log_xi: (N, T-1, K, K) log pairwise marginals.
        log_gamma: (N, T, K) log posterior responsibilities.
        covariates: (N, T-1, D) covariate features per pixel-timestep.
        prev_weights: (K, K-1, D) previous transition weights (warm start).
        l2_reg: L2 regularization strength.
        maxiter: Max L-BFGS iterations.

    Returns:
        weights: (K, K-1, D) updated transition weights.
    """
    import jaxopt

    K = log_xi.shape[2]
    D = covariates.shape[2]

    xi = jnp.exp(log_xi)  # (N, T-1, K, K)
    # Flatten pixel and time dimensions
    xi_flat = xi.reshape(-1, K, K)  # (M, K, K) where M = N*(T-1)
    cov_flat = covariates.reshape(-1, D)  # (M, D)

    def _scatter_nonref(vals_M, ref_idx):
        """Insert (M, K-1) non-ref logits into (M, K) full logits, skipping ref_idx."""
        # Build integer column indices for the K-1 non-ref positions
        cols = jnp.arange(K)
        # Shift columns >= ref_idx to make room for the zero at ref_idx
        nonref_cols = jnp.where(cols < ref_idx, cols, cols + 1)
        nonref_cols = nonref_cols[:K - 1]  # (K-1,)
        logits = jnp.zeros((vals_M.shape[0], K))
        logits = logits.at[:, nonref_cols].set(vals_M)
        return logits

    def nll_per_source(weights_i, source_idx):
        """NLL for transitions from source state i."""
        # weights_i: (K-1, D) - weights for non-reference states
        # Reference state (i) has logit = 0
        logits_nonref = cov_flat @ weights_i.T  # (M, K-1)

        # Insert zero logit for reference state using integer scatter
        logits = _scatter_nonref(logits_nonref, source_idx)  # (M, K)

        # Softmax log-probabilities
        log_probs = jax.nn.log_softmax(logits, axis=1)  # (M, K)

        # Weighted by xi: expected count of (i -> j) transitions
        target_weights = xi_flat[:, source_idx, :]  # (M, K)

        # NLL = -sum target_weights * log_probs
        nll = -jnp.sum(target_weights * log_probs)
        # L2 regularization
        nll += l2_reg * jnp.sum(weights_i ** 2)
        return nll

    new_weights = []
    for i in range(K):
        solver = jaxopt.LBFGS(
            fun=lambda w, idx=i: nll_per_source(w, idx),
            maxiter=maxiter,
            tol=1e-5,
        )
        result = solver.run(prev_weights[i])
        new_weights.append(result.params)

    return jnp.stack(new_weights)  # (K, K-1, D)
