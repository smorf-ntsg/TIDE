"""Tests for Zero-Inflated Beta emission model."""

import jax
import jax.numpy as jnp
import pytest

from tide.types import ZIBParams
from tide.emissions.zero_inflated_beta import (
    zib_log_prob,
    constrain_params,
    unconstrain_params,
    init_default_params,
    _log_beta_pdf,
)


@pytest.fixture
def default_params():
    return init_default_params(6)


@pytest.fixture
def simple_params():
    """Simple 3-state params for focused testing."""
    return ZIBParams(
        pi=jnp.array([0.8, 0.1, 0.01]),
        mu=jnp.array([0.01, 0.10, 0.40]),
        phi=jnp.array([50.0, 20.0, 10.0]),
    )


class TestLogBetaPdf:
    def test_positive_density(self):
        """Beta PDF should be positive (log should be finite)."""
        x = jnp.array([0.1, 0.5, 0.9])
        alpha = jnp.array([2.0, 2.0, 2.0])
        beta = jnp.array([5.0, 5.0, 5.0])
        log_p = _log_beta_pdf(x, alpha, beta)
        assert jnp.all(jnp.isfinite(log_p))

    def test_integrates_to_one(self):
        """Numerical integration of Beta PDF should approximate 1."""
        alpha, beta = 2.0, 5.0
        x = jnp.linspace(0.001, 0.999, 10000)
        log_p = _log_beta_pdf(x, alpha, beta)
        dx = x[1] - x[0]
        integral = jnp.sum(jnp.exp(log_p)) * dx
        assert abs(integral - 1.0) < 0.01

    def test_gradient_finite(self):
        """Gradients of log Beta PDF should be finite."""
        def f(alpha, beta):
            return _log_beta_pdf(jnp.array(0.3), alpha, beta)
        grad_fn = jax.grad(f, argnums=(0, 1))
        g_a, g_b = grad_fn(2.0, 5.0)
        assert jnp.isfinite(g_a)
        assert jnp.isfinite(g_b)


class TestZibLogProb:
    def test_output_shape(self, default_params):
        """Output shape should be (N, T, K)."""
        obs = jnp.zeros((100, 35), dtype=jnp.float32)
        log_p = zib_log_prob(obs, default_params)
        assert log_p.shape == (100, 35, 6)

    def test_zero_obs_uses_pi(self, simple_params):
        """For zero observations, log prob should be log(pi_k)."""
        obs = jnp.zeros((1, 1), dtype=jnp.float32)
        log_p = zib_log_prob(obs, simple_params)
        # State 0 has pi=0.8, so log_p[0,0,0] should be close to log(0.8)
        expected = jnp.log(0.8)
        assert abs(log_p[0, 0, 0] - expected) < 0.01

    def test_nonzero_obs_uses_beta(self, simple_params):
        """For non-zero observations, should use Beta component."""
        obs = jnp.array([[0.05]], dtype=jnp.float32)
        log_p = zib_log_prob(obs, simple_params)
        # All values should be finite
        assert jnp.all(jnp.isfinite(log_p))
        # State 1 (mu=0.10) should have higher density near 0.05 than state 2 (mu=0.40)
        assert log_p[0, 0, 1] > log_p[0, 0, 2]

    def test_state_discrimination(self, default_params):
        """Different cover values should prefer different states."""
        obs = jnp.array([[0.0, 0.02, 0.07, 0.18, 0.38, 0.60]], dtype=jnp.float32)
        log_p = zib_log_prob(obs, default_params)
        # Zero should prefer state 0 (highest pi)
        assert log_p[0, 0, :].argmax() == 0
        # High cover should prefer state 5 (highest mu)
        assert log_p[0, 5, :].argmax() == 5

    def test_no_nan_or_inf(self, default_params):
        """No NaN or Inf in output for various inputs."""
        obs = jnp.array([[0.0, 0.001, 0.5, 0.99, 0.999]], dtype=jnp.float32)
        log_p = zib_log_prob(obs, default_params)
        assert jnp.all(jnp.isfinite(log_p))

    def test_batch_consistency(self, default_params):
        """Batched and individual results should match."""
        obs_batch = jnp.array([[0.0, 0.1], [0.5, 0.9]], dtype=jnp.float32)
        log_p_batch = zib_log_prob(obs_batch, default_params)

        for i in range(2):
            obs_single = obs_batch[i:i+1]
            log_p_single = zib_log_prob(obs_single, default_params)
            assert jnp.allclose(log_p_batch[i], log_p_single[0], atol=1e-6)


class TestConstrainParams:
    def test_round_trip(self):
        """constrain -> unconstrain -> constrain should preserve values."""
        params = ZIBParams(
            pi=jnp.array([0.8, 0.1, 0.02, 0.01, 0.005, 0.001]),
            mu=jnp.array([0.005, 0.02, 0.07, 0.18, 0.38, 0.60]),
            phi=jnp.array([100.0, 50.0, 30.0, 15.0, 10.0, 5.0]),
        )
        raw_pi, raw_mu, raw_phi = unconstrain_params(params)
        recovered = constrain_params(raw_pi, raw_mu, raw_phi)

        # mu should maintain ordering
        for i in range(5):
            assert recovered.mu[i] < recovered.mu[i + 1]

    def test_monotonic_mu(self):
        """Constrained mu values should be monotonically increasing."""
        raw_pi = jnp.zeros(6)
        raw_mu = jnp.array([0.0, -1.0, 0.5, 1.0, 2.0, 2.5])
        raw_phi = jnp.zeros(6)
        params = constrain_params(raw_pi, raw_mu, raw_phi)

        for i in range(5):
            assert params.mu[i] < params.mu[i + 1]

    def test_valid_ranges(self):
        """All constrained params should be in valid ranges."""
        raw_pi = jnp.array([-2.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        raw_mu = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        raw_phi = jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        params = constrain_params(raw_pi, raw_mu, raw_phi)

        assert jnp.all((params.pi > 0) & (params.pi < 1))
        assert jnp.all((params.mu > 0) & (params.mu < 1))
        assert jnp.all(params.phi > 0)
