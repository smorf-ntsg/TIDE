"""Tests for forward-backward algorithm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tide.hmm.forward_backward import forward_backward


def _make_simple_hmm(K=2, T=10, N=5):
    """Create a simple HMM for testing."""
    rng = np.random.default_rng(42)

    # Random log emissions
    log_emission = jnp.array(
        rng.standard_normal((N, T, K)).astype(np.float32) * 2
    )

    # Uniform initial
    log_init = jnp.log(jnp.ones(K) / K)

    # Simple transition matrix
    trans = np.eye(K) * 0.8 + np.ones((K, K)) * 0.2 / K
    trans = trans / trans.sum(axis=1, keepdims=True)
    log_trans = jnp.log(jnp.array(trans, dtype=jnp.float32))

    return log_emission, log_init, log_trans


class TestForwardBackward:
    def test_output_shapes(self):
        """Check output tensor shapes."""
        log_emission, log_init, log_trans = _make_simple_hmm(K=3, T=8, N=10)
        result = forward_backward(log_emission, log_init, log_trans, compute_xi=True)

        assert result.log_gamma.shape == (10, 8, 3)
        assert result.log_xi.shape == (10, 7, 3, 3)
        assert result.log_likelihood.shape == (10,)

    def test_posteriors_sum_to_one(self):
        """Posterior probabilities should sum to 1 across states."""
        log_emission, log_init, log_trans = _make_simple_hmm(K=5, T=35, N=20)
        result = forward_backward(log_emission, log_init, log_trans)

        gamma = jnp.exp(result.log_gamma)
        sums = gamma.sum(axis=2)  # (N, T)
        assert jnp.allclose(sums, 1.0, atol=1e-4)

    def test_xi_marginalizes_to_gamma(self):
        """Sum of xi over j should equal gamma for t < T-1."""
        log_emission, log_init, log_trans = _make_simple_hmm(K=3, T=10, N=5)
        result = forward_backward(log_emission, log_init, log_trans, compute_xi=True)

        xi = jnp.exp(result.log_xi)  # (N, T-1, K, K)
        gamma = jnp.exp(result.log_gamma)  # (N, T, K)

        # Sum xi[t, i, :] should approximate gamma[t, i] for t=0..T-2
        xi_marginal = xi.sum(axis=3)  # (N, T-1, K) sum over target states
        assert jnp.allclose(xi_marginal, gamma[:, :-1, :], atol=1e-4)

    def test_log_likelihood_finite(self):
        """Log-likelihood should be finite."""
        log_emission, log_init, log_trans = _make_simple_hmm()
        result = forward_backward(log_emission, log_init, log_trans)
        assert jnp.all(jnp.isfinite(result.log_likelihood))

    def test_no_xi_mode(self):
        """Should work without computing xi."""
        log_emission, log_init, log_trans = _make_simple_hmm()
        result = forward_backward(log_emission, log_init, log_trans, compute_xi=False)
        assert result.log_xi is None
        assert result.log_gamma is not None

    def test_matches_numpy_2state(self):
        """Compare against reference NumPy implementation for 2-state model.

        Reproduces the vectorized NumPy forward-backward from 
        """
        N, T, K = 3, 5, 2
        rng = np.random.default_rng(123)

        log_emission_np = rng.standard_normal((N, T, K)).astype(np.float32)
        log_init_np = np.log(np.array([0.9, 0.1], dtype=np.float32))
        trans = np.array([[0.97, 0.03], [0.01, 0.99]], dtype=np.float32)
        log_trans_np = np.log(trans)

        # NumPy reference (adapted from )
        log_alpha = np.zeros((N, T, K), dtype=np.float32)
        log_alpha[:, 0, :] = log_init_np + log_emission_np[:, 0, :]
        for t in range(1, T):
            for j in range(K):
                log_alpha[:, t, j] = (
                    np.logaddexp(
                        log_alpha[:, t-1, 0] + log_trans_np[0, j],
                        log_alpha[:, t-1, 1] + log_trans_np[1, j],
                    )
                    + log_emission_np[:, t, j]
                )

        log_beta = np.zeros((N, T, K), dtype=np.float32)
        for t in range(T - 2, -1, -1):
            for i in range(K):
                log_beta[:, t, i] = np.logaddexp(
                    log_trans_np[i, 0] + log_emission_np[:, t+1, 0] + log_beta[:, t+1, 0],
                    log_trans_np[i, 1] + log_emission_np[:, t+1, 1] + log_beta[:, t+1, 1],
                )

        log_gamma_np = log_alpha + log_beta
        log_norm = np.logaddexp(log_gamma_np[:, :, 0], log_gamma_np[:, :, 1])
        log_gamma_np -= log_norm[:, :, None]

        # JAX implementation
        result = forward_backward(
            jnp.array(log_emission_np),
            jnp.array(log_init_np),
            jnp.array(log_trans_np),
        )

        # Compare posteriors
        assert jnp.allclose(
            result.log_gamma, jnp.array(log_gamma_np), atol=1e-4
        )

    def test_six_state(self):
        """Verify 6-state model runs correctly."""
        log_emission, log_init, log_trans = _make_simple_hmm(K=6, T=35, N=100)
        result = forward_backward(log_emission, log_init, log_trans, compute_xi=True)

        assert result.log_gamma.shape == (100, 35, 6)
        assert result.log_xi.shape == (100, 34, 6, 6)

        gamma = jnp.exp(result.log_gamma)
        assert jnp.allclose(gamma.sum(axis=2), 1.0, atol=1e-4)
