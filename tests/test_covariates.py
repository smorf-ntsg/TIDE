"""Tests for covariate stacking, dynamic transitions, and end-to-end wiring.

All tests use synthetic data — no real rasters required.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tide.transitions.covariates import stack_covariates
from tide.transitions.dynamic import (
    compute_dynamic_log_trans,
    init_transition_weights,
)


class TestStackCovariates:
    """Shape and value contracts for stack_covariates."""

    N, Tm1, K = 100, 34, 6

    def test_all_covariates_shape(self):
        """Full feature set produces (N, T-1, 8) tensor."""
        covs = stack_covariates(
            fire_severity=jnp.ones((self.N, self.Tm1)),
            spei=jnp.ones((self.N, self.Tm1)),
            elevation=jnp.ones((self.N,)),
            chili=jnp.ones((self.N,)),
            aridity=jnp.ones((self.N,)),
            seed_distance=jnp.ones((self.N, self.Tm1)),
            focal_mean=jnp.ones((self.N, self.Tm1)),
        )
        assert covs.shape == (self.N, self.Tm1, 8)

    def test_intercept_only(self):
        """Single covariate still gets intercept → D=2."""
        covs = stack_covariates(
            fire_severity=jnp.ones((self.N, self.Tm1)),
        )
        assert covs.shape == (self.N, self.Tm1, 2)
        # First feature is intercept (all ones)
        np.testing.assert_allclose(covs[:, :, 0], 1.0)

    def test_static_broadcast(self):
        """Static (N,) covariates are broadcast to (N, T-1)."""
        elev = jnp.arange(self.N, dtype=jnp.float32)
        covs = stack_covariates(
            elevation=elev,
            spei=jnp.ones((self.N, self.Tm1)),
        )
        # D = intercept + spei + elevation = 3
        assert covs.shape == (self.N, self.Tm1, 3)
        # Elevation should be constant across time for each pixel
        for t in range(self.Tm1):
            np.testing.assert_allclose(covs[:, t, 2], elev)

    def test_no_covariates_raises(self):
        """Calling with no covariates raises ValueError."""
        with pytest.raises(ValueError, match="At least one covariate"):
            stack_covariates()

    def test_only_static_covariates(self):
        """Static-only covariates use default Tm1=34."""
        covs = stack_covariates(
            elevation=jnp.ones((self.N,)),
            chili=jnp.ones((self.N,)),
            aridity=jnp.ones((self.N,)),
        )
        assert covs.shape == (self.N, 34, 4)  # intercept + elevation + chili + aridity

    def test_feature_order(self):
        """Features appear in documented order: intercept, fire, spei, elev, chili, aridity, seed, focal."""
        covs = stack_covariates(
            fire_severity=jnp.full((self.N, self.Tm1), 1.0),
            spei=jnp.full((self.N, self.Tm1), 2.0),
            elevation=jnp.full((self.N,), 3.0),
            chili=jnp.full((self.N,), 4.0),
            aridity=jnp.full((self.N,), 5.0),
            seed_distance=jnp.full((self.N, self.Tm1), 6.0),
            focal_mean=jnp.full((self.N, self.Tm1), 7.0),
        )
        expected = [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        np.testing.assert_allclose(covs[0, 0, :], expected)

    def test_dtype_float32(self):
        """Output is float32."""
        covs = stack_covariates(
            fire_severity=jnp.ones((self.N, self.Tm1)),
        )
        assert covs.dtype == jnp.float32


class TestDynamicTransitions:
    """Shape and probability contracts for compute_dynamic_log_trans."""

    N, Tm1, K, D = 50, 34, 6, 8

    def test_output_shape(self):
        """Output is (N, T-1, K, K)."""
        covs = jnp.ones((self.N, self.Tm1, self.D))
        weights = init_transition_weights(self.K, self.D)
        log_trans = compute_dynamic_log_trans(covs, weights, K=self.K)
        assert log_trans.shape == (self.N, self.Tm1, self.K, self.K)

    def test_rows_sum_to_one(self):
        """Each row of exp(log_trans) sums to 1 (valid distribution)."""
        rng = np.random.default_rng(42)
        covs = jnp.array(rng.standard_normal((self.N, self.Tm1, self.D)), dtype=jnp.float32)
        weights = jnp.array(rng.standard_normal((self.K, self.K - 1, self.D)) * 0.1, dtype=jnp.float32)
        log_trans = compute_dynamic_log_trans(covs, weights, K=self.K)
        row_sums = jnp.exp(log_trans).sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_zero_weights_gives_uniform(self):
        """Zero weights → uniform off-diagonal (each row is log(1/K))."""
        covs = jnp.ones((self.N, self.Tm1, self.D))
        weights = init_transition_weights(self.K, self.D)
        log_trans = compute_dynamic_log_trans(covs, weights, K=self.K)
        # All entries should be log(1/K) = -log(K)
        expected = jnp.log(jnp.array(1.0 / self.K))
        np.testing.assert_allclose(log_trans, expected, atol=1e-5)

    def test_no_nan_or_inf(self):
        """No NaN or Inf in output even with non-trivial weights."""
        rng = np.random.default_rng(123)
        covs = jnp.array(rng.standard_normal((self.N, self.Tm1, self.D)), dtype=jnp.float32)
        weights = jnp.array(rng.standard_normal((self.K, self.K - 1, self.D)), dtype=jnp.float32)
        log_trans = compute_dynamic_log_trans(covs, weights, K=self.K)
        assert jnp.all(jnp.isfinite(log_trans))

    def test_all_log_probs_negative(self):
        """All log transition probabilities are ≤ 0."""
        rng = np.random.default_rng(7)
        covs = jnp.array(rng.standard_normal((self.N, self.Tm1, self.D)), dtype=jnp.float32)
        weights = jnp.array(rng.standard_normal((self.K, self.K - 1, self.D)) * 0.5, dtype=jnp.float32)
        log_trans = compute_dynamic_log_trans(covs, weights, K=self.K)
        assert jnp.all(log_trans <= 1e-6)

    def test_different_covariates_give_different_trans(self):
        """Different covariate values produce different transition matrices."""
        rng = np.random.default_rng(99)
        weights = jnp.array(rng.standard_normal((self.K, self.K - 1, self.D)), dtype=jnp.float32)
        covs_a = jnp.ones((1, 1, self.D))
        covs_b = jnp.ones((1, 1, self.D)) * 5.0
        lt_a = compute_dynamic_log_trans(covs_a, weights, K=self.K)
        lt_b = compute_dynamic_log_trans(covs_b, weights, K=self.K)
        assert not jnp.allclose(lt_a, lt_b)


class TestEndToEndCovariateInference:
    """Test that covariates wire correctly through viterbi/forward-backward."""

    N, T, K = 30, 10, 6

    @pytest.fixture
    def params_with_dynamic_trans(self):
        """Build HMMParams with transition weights for dynamic mode."""
        from tide.types import HMMParams, ZIBParams
        from tide.config import DEFAULT_INIT_PROBS, DEFAULT_TRANS

        K = self.K
        D = 3  # intercept + 2 covariates

        emission = ZIBParams(
            pi=jnp.array([0.90, 0.20, 0.05, 0.01, 0.005, 0.001]),
            mu=jnp.array([0.005, 0.02, 0.07, 0.18, 0.38, 0.60]),
            phi=jnp.array([100.0, 50.0, 30.0, 15.0, 10.0, 5.0]),
        )

        log_init = jnp.log(jnp.array(DEFAULT_INIT_PROBS))
        log_trans = jnp.log(jnp.array(DEFAULT_TRANS))
        weights = init_transition_weights(K, D)

        return HMMParams(
            log_init=log_init,
            log_trans=log_trans,
            emission=emission,
            transition_weights=weights,
        )

    def test_viterbi_with_dynamic_trans(self, params_with_dynamic_trans):
        """Viterbi runs with (N, T-1, K, K) dynamic log_trans."""
        from tide.emissions.zero_inflated_beta import zib_log_prob
        from tide.hmm.viterbi import viterbi

        params = params_with_dynamic_trans
        rng = np.random.default_rng(42)
        obs = jnp.array(rng.uniform(0, 0.5, (self.N, self.T)), dtype=jnp.float32)

        log_emission = zib_log_prob(obs, params.emission)

        # Build dynamic transitions from covariates
        covs = jnp.ones((self.N, self.T - 1, 3))  # D=3
        dyn_log_trans = compute_dynamic_log_trans(
            covs, params.transition_weights, K=self.K,
        )

        result = viterbi(log_emission, params.log_init, dyn_log_trans)
        assert result.states.shape == (self.N, self.T)
        assert jnp.all((result.states >= 0) & (result.states < self.K))

    def test_forward_backward_with_dynamic_trans(self, params_with_dynamic_trans):
        """Forward-backward runs with (N, T-1, K, K) dynamic log_trans."""
        from tide.emissions.zero_inflated_beta import zib_log_prob
        from tide.hmm.forward_backward import forward_backward

        params = params_with_dynamic_trans
        rng = np.random.default_rng(42)
        obs = jnp.array(rng.uniform(0, 0.5, (self.N, self.T)), dtype=jnp.float32)

        log_emission = zib_log_prob(obs, params.emission)

        covs = jnp.ones((self.N, self.T - 1, 3))
        dyn_log_trans = compute_dynamic_log_trans(
            covs, params.transition_weights, K=self.K,
        )

        result = forward_backward(
            log_emission, params.log_init, dyn_log_trans, compute_xi=True,
        )
        assert result.log_gamma.shape == (self.N, self.T, self.K)
        # Posteriors sum to 1
        post_sums = jnp.exp(result.log_gamma).sum(axis=-1)
        np.testing.assert_allclose(post_sums, 1.0, atol=1e-4)

    def test_hmm_params_backward_compatible(self):
        """HMMParams still works with 3-arg construction (no transition_weights)."""
        from tide.types import HMMParams, ZIBParams

        emission = ZIBParams(
            pi=jnp.array([0.5, 0.5]),
            mu=jnp.array([0.1, 0.3]),
            phi=jnp.array([10.0, 10.0]),
        )
        params = HMMParams(
            log_init=jnp.log(jnp.array([0.5, 0.5])),
            log_trans=jnp.log(jnp.array([[0.9, 0.1], [0.1, 0.9]])),
            emission=emission,
        )
        assert params.transition_weights is None


class TestFitTransitionWeights:
    """Tests for post-EM transition weight estimation."""

    N, T, K = 50, 15, 6
    D = 4  # intercept + 3 covariates

    @pytest.fixture
    def synthetic_setup(self):
        """Create synthetic obs + covariates with known emission params."""
        from tide.types import ZIBParams
        from tide.config import DEFAULT_INIT_PROBS

        rng = np.random.default_rng(42)

        emission = ZIBParams(
            pi=jnp.array([0.90, 0.20, 0.05, 0.01, 0.005, 0.001]),
            mu=jnp.array([0.005, 0.02, 0.07, 0.18, 0.38, 0.60]),
            phi=jnp.array([100.0, 50.0, 30.0, 15.0, 10.0, 5.0]),
        )
        log_init = jnp.log(jnp.array(DEFAULT_INIT_PROBS))

        # Synthetic observations (mostly low cover, some higher)
        obs = jnp.array(
            rng.beta(0.5, 5.0, (self.N, self.T)), dtype=jnp.float32
        )

        # Synthetic covariates: intercept + 3 random features
        covariates = jnp.array(
            np.column_stack([
                np.ones((self.N * (self.T - 1), 1)),
                rng.standard_normal((self.N * (self.T - 1), self.D - 1)),
            ]).reshape(self.N, self.T - 1, self.D),
            dtype=jnp.float32,
        )

        return obs, covariates, emission, log_init

    def test_returns_correct_shape(self, synthetic_setup):
        """Weights have shape (K, K-1, D)."""
        from tide.transitions.fit_weights import fit_transition_weights

        obs, covariates, emission, log_init = synthetic_setup
        weights, ll_history = fit_transition_weights(
            obs, covariates, emission, log_init,
            K=self.K, max_iter=3, lbfgs_maxiter=5,
        )
        assert weights.shape == (self.K, self.K - 1, self.D)

    def test_log_likelihood_recorded(self, synthetic_setup):
        """Log-likelihood history is non-empty and finite."""
        from tide.transitions.fit_weights import fit_transition_weights

        obs, covariates, emission, log_init = synthetic_setup
        weights, ll_history = fit_transition_weights(
            obs, covariates, emission, log_init,
            K=self.K, max_iter=3, lbfgs_maxiter=5,
        )
        assert len(ll_history) > 0
        assert all(np.isfinite(ll) for ll in ll_history)

    def test_weights_finite(self, synthetic_setup):
        """All estimated weights are finite (no NaN/Inf)."""
        from tide.transitions.fit_weights import fit_transition_weights

        obs, covariates, emission, log_init = synthetic_setup
        weights, _ = fit_transition_weights(
            obs, covariates, emission, log_init,
            K=self.K, max_iter=3, lbfgs_maxiter=5,
        )
        assert jnp.all(jnp.isfinite(weights))

    def test_produces_valid_transitions(self, synthetic_setup):
        """Estimated weights produce valid probability distributions."""
        from tide.transitions.fit_weights import fit_transition_weights

        obs, covariates, emission, log_init = synthetic_setup
        weights, _ = fit_transition_weights(
            obs, covariates, emission, log_init,
            K=self.K, max_iter=3, lbfgs_maxiter=5,
        )

        dyn_log_trans = compute_dynamic_log_trans(covariates, weights, K=self.K)
        row_sums = jnp.exp(dyn_log_trans).sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestPipelineWeightIntegration:
    """Test pipeline helpers for covariate sampling and numpy stacking."""

    N, T, K = 40, 10, 6
    D = 4  # intercept + 3

    def test_numpy_stack_covariates_full(self):
        """_numpy_stack_covariates produces correct shape with all covariates."""
        from tide.pipeline import _numpy_stack_covariates

        Tm1 = self.T - 1
        cov_dict = {
            "fire_severity": np.ones((self.N, Tm1), dtype=np.float32),
            "spei": np.ones((self.N, Tm1), dtype=np.float32),
            "elevation": np.ones((self.N,), dtype=np.float32),
            "chili": np.ones((self.N,), dtype=np.float32),
            "aridity": np.ones((self.N,), dtype=np.float32),
            "seed_distance": np.ones((self.N, Tm1), dtype=np.float32),
            "focal_mean": np.ones((self.N, Tm1), dtype=np.float32),
        }
        result = _numpy_stack_covariates(cov_dict)
        assert result.shape == (self.N, Tm1, 8)  # intercept + 7
        # First feature is intercept
        np.testing.assert_allclose(result[:, :, 0], 1.0)

    def test_numpy_stack_covariates_partial(self):
        """_numpy_stack_covariates works with subset of covariates."""
        from tide.pipeline import _numpy_stack_covariates

        Tm1 = self.T - 1
        cov_dict = {
            "fire_severity": None,
            "spei": None,
            "elevation": None,
            "chili": None,
            "seed_distance": np.ones((self.N, Tm1), dtype=np.float32),
            "focal_mean": np.ones((self.N, Tm1), dtype=np.float32),
        }
        result = _numpy_stack_covariates(cov_dict)
        assert result.shape == (self.N, Tm1, 3)  # intercept + 2

    def test_numpy_stack_matches_jax_stack(self):
        """_numpy_stack_covariates matches stack_covariates output."""
        from tide.pipeline import _numpy_stack_covariates

        rng = np.random.default_rng(42)
        Tm1 = self.T - 1

        fire = rng.standard_normal((self.N, Tm1)).astype(np.float32)
        elev = rng.standard_normal((self.N,)).astype(np.float32)
        sd = rng.standard_normal((self.N, Tm1)).astype(np.float32)

        cov_dict = {
            "fire_severity": fire,
            "spei": None,
            "elevation": elev,
            "chili": None,
            "seed_distance": sd,
            "focal_mean": None,
        }
        np_result = _numpy_stack_covariates(cov_dict)

        jax_result = stack_covariates(
            fire_severity=jnp.array(fire),
            elevation=jnp.array(elev),
            seed_distance=jnp.array(sd),
        )
        np.testing.assert_allclose(np_result, np.asarray(jax_result), atol=1e-6)

    def test_end_to_end_numpy_to_fit_weights(self):
        """Covariates from _numpy_stack flow through fit_transition_weights."""
        from tide.pipeline import _numpy_stack_covariates
        from tide.transitions.fit_weights import fit_transition_weights
        from tide.types import ZIBParams
        from tide.config import DEFAULT_INIT_PROBS

        rng = np.random.default_rng(42)
        obs = rng.beta(0.5, 5.0, (self.N, self.T)).astype(np.float32)

        Tm1 = self.T - 1
        cov_dict = {
            "fire_severity": None,
            "spei": None,
            "elevation": None,
            "chili": None,
            "seed_distance": rng.standard_normal((self.N, Tm1)).astype(np.float32),
            "focal_mean": rng.standard_normal((self.N, Tm1)).astype(np.float32),
        }
        cov_np = _numpy_stack_covariates(cov_dict)  # (N, T-1, 3)

        emission = ZIBParams(
            pi=jnp.array([0.90, 0.20, 0.05, 0.01, 0.005, 0.001]),
            mu=jnp.array([0.005, 0.02, 0.07, 0.18, 0.38, 0.60]),
            phi=jnp.array([100.0, 50.0, 30.0, 15.0, 10.0, 5.0]),
        )
        log_init = jnp.log(jnp.array(DEFAULT_INIT_PROBS))

        weights, ll_history = fit_transition_weights(
            jnp.array(obs), jnp.array(cov_np), emission, log_init,
            K=self.K, max_iter=2, lbfgs_maxiter=3,
        )
        D = cov_np.shape[2]
        assert weights.shape == (self.K, self.K - 1, D)
        assert len(ll_history) > 0
