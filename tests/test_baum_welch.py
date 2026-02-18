"""Tests for Baum-Welch EM algorithm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tide.types import ZIBParams, HMMParams
from tide.emissions.zero_inflated_beta import zib_log_prob
from tide.hmm.baum_welch import baum_welch
from tide.hmm.forward_backward import forward_backward


def _generate_synthetic_zib_data(
    N: int = 1000,
    T: int = 35,
    K: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data from a known ZIB-HMM.

    Returns:
        obs: (N, T) float32 observations
        true_states: (N, T) int32 true state sequence
    """
    rng = np.random.default_rng(seed)

    # True parameters (6-state: Bare, Trace, Sparse, Open, Woodland, Forest)
    pi = np.array([0.90, 0.20, 0.05, 0.01, 0.005, 0.001])
    mu = np.array([0.005, 0.02, 0.07, 0.18, 0.38, 0.60])
    phi = np.array([100.0, 50.0, 30.0, 15.0, 10.0, 5.0])

    init_probs = np.array([0.50, 0.15, 0.12, 0.10, 0.08, 0.05])
    trans = np.array([
        [0.950, 0.030, 0.012, 0.005, 0.002, 0.001],
        [0.020, 0.930, 0.035, 0.010, 0.003, 0.002],
        [0.010, 0.020, 0.930, 0.030, 0.007, 0.003],
        [0.005, 0.008, 0.017, 0.940, 0.025, 0.005],
        [0.003, 0.004, 0.006, 0.017, 0.950, 0.020],
        [0.002, 0.003, 0.005, 0.010, 0.010, 0.970],
    ])

    states = np.zeros((N, T), dtype=np.int32)
    obs = np.zeros((N, T), dtype=np.float32)

    for n in range(N):
        states[n, 0] = rng.choice(K, p=init_probs)
        for t in range(1, T):
            states[n, t] = rng.choice(K, p=trans[states[n, t-1]])

    for n in range(N):
        for t in range(T):
            k = states[n, t]
            if rng.random() < pi[k]:
                obs[n, t] = 0.0
            else:
                a = mu[k] * phi[k]
                b = (1 - mu[k]) * phi[k]
                obs[n, t] = np.clip(rng.beta(a, b), 1e-6, 0.999)

    return obs, states


class TestBaumWelch:
    def test_monotonic_log_likelihood(self):
        """Log-likelihood should increase (or plateau) at each EM iteration."""
        obs, _ = _generate_synthetic_zib_data(N=200, T=10, K=6, seed=0)
        obs_jax = jnp.array(obs)

        result = baum_welch(obs_jax, K=6, max_iter=5, lbfgs_maxiter=10)

        lls = np.asarray(result.log_likelihoods)
        # Allow small numerical tolerance for decrease
        for i in range(1, len(lls)):
            assert lls[i] >= lls[i-1] - 1.0, (
                f"Log-likelihood decreased: iter {i-1}={lls[i-1]:.2f}, "
                f"iter {i}={lls[i]:.2f}"
            )

    def test_parameter_recovery(self):
        """EM should approximately recover generating parameters."""
        obs, _ = _generate_synthetic_zib_data(N=5000, T=35, K=6, seed=42)
        obs_jax = jnp.array(obs)

        result = baum_welch(obs_jax, K=6, max_iter=20, lbfgs_maxiter=20)

        # Check mu ordering maintained
        mu = np.asarray(result.params.emission.mu)
        for i in range(5):
            assert mu[i] < mu[i+1], f"mu not monotonic: {mu}"

        # Mu should be in right ballpark (within 50% relative error)
        true_mu = np.array([0.005, 0.02, 0.07, 0.18, 0.38, 0.60])
        for k in range(6):
            rel_err = abs(mu[k] - true_mu[k]) / true_mu[k]
            assert rel_err < 0.5, (
                f"State {k}: mu={mu[k]:.4f}, true={true_mu[k]:.4f}, "
                f"rel_err={rel_err:.2f}"
            )

    def test_returns_valid_params(self):
        """Returned parameters should be valid."""
        obs, _ = _generate_synthetic_zib_data(N=500, T=10, K=6, seed=1)
        obs_jax = jnp.array(obs)

        result = baum_welch(obs_jax, K=6, max_iter=3, lbfgs_maxiter=5)
        params = result.params

        # Initial probs should sum to ~1
        init_probs = jnp.exp(params.log_init)
        assert abs(init_probs.sum() - 1.0) < 0.01

        # Transition rows should sum to ~1
        trans = jnp.exp(params.log_trans)
        row_sums = trans.sum(axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=0.01)

        # Emission params in valid ranges
        assert jnp.all(params.emission.pi >= 0)
        assert jnp.all(params.emission.pi <= 1)
        assert jnp.all(params.emission.mu > 0)
        assert jnp.all(params.emission.mu < 1)
        assert jnp.all(params.emission.phi > 0)

    def test_no_nan(self):
        """No NaN values in any output."""
        obs, _ = _generate_synthetic_zib_data(N=300, T=10, K=6, seed=2)
        obs_jax = jnp.array(obs)

        result = baum_welch(obs_jax, K=6, max_iter=3, lbfgs_maxiter=5)

        assert jnp.all(jnp.isfinite(result.params.log_init))
        assert jnp.all(jnp.isfinite(result.params.log_trans))
        assert jnp.all(jnp.isfinite(result.params.emission.pi))
        assert jnp.all(jnp.isfinite(result.params.emission.mu))
        assert jnp.all(jnp.isfinite(result.params.emission.phi))
        assert jnp.all(jnp.isfinite(result.log_likelihoods))
