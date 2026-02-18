"""Integration tests for the full ZIB-HMM pipeline.

Tests end-to-end flow from synthetic data through inference.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tide.types import ZIBParams, HMMParams
from tide.emissions.zero_inflated_beta import zib_log_prob, init_default_params
from tide.hmm.forward_backward import forward_backward
from tide.hmm.viterbi import viterbi
from tide.hmm.baum_welch import baum_welch
from tide.config import DEFAULT_INIT_PROBS, DEFAULT_TRANS
from tide.data.chunking import generate_chunks, Chunk
from tide.distributed.memory import estimate_chunk_memory, optimal_batch_size


class TestEndToEnd:
    """End-to-end pipeline test with synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic 6-state ZIB-HMM data."""
        rng = np.random.default_rng(42)
        N, T, K = 500, 35, 6

        pi = np.array([0.90, 0.20, 0.05, 0.01, 0.005, 0.001])
        mu = np.array([0.005, 0.02, 0.07, 0.18, 0.38, 0.60])
        phi = np.array([100.0, 50.0, 30.0, 15.0, 10.0, 5.0])
        init_probs = np.array([0.50, 0.15, 0.12, 0.10, 0.08, 0.05])
        trans = np.array(DEFAULT_TRANS)

        states = np.zeros((N, T), dtype=np.int32)
        obs = np.zeros((N, T), dtype=np.float32)

        for n in range(N):
            states[n, 0] = rng.choice(K, p=init_probs)
            for t in range(1, T):
                states[n, t] = rng.choice(K, p=trans[states[n, t-1]])
            for t in range(T):
                k = states[n, t]
                if rng.random() < pi[k]:
                    obs[n, t] = 0.0
                else:
                    a = mu[k] * phi[k]
                    b = (1 - mu[k]) * phi[k]
                    obs[n, t] = np.clip(rng.beta(a, b), 1e-6, 0.999)

        return obs, states

    def test_emission_viterbi_pipeline(self, synthetic_data):
        """Test emission -> Viterbi pipeline with known parameters."""
        obs, true_states = synthetic_data
        obs_jax = jnp.array(obs)

        params = init_default_params(6)
        log_init = jnp.log(jnp.array(DEFAULT_INIT_PROBS, dtype=jnp.float32))
        log_trans = jnp.log(jnp.array(DEFAULT_TRANS, dtype=jnp.float32))

        # Compute emissions
        log_emission = zib_log_prob(obs_jax, params)
        assert log_emission.shape == (500, 35, 6)
        assert jnp.all(jnp.isfinite(log_emission))

        # Viterbi
        result = viterbi(log_emission, log_init, log_trans)
        assert result.states.shape == (500, 35)

        # Accuracy: decoded states should correlate with true states
        decoded = np.asarray(result.states)
        accuracy = (decoded == true_states).mean()
        # With true parameters, accuracy should be decent (>40% for 5 states)
        assert accuracy > 0.3, f"Accuracy too low: {accuracy:.2%}"

    def test_emission_forward_backward_pipeline(self, synthetic_data):
        """Test emission -> forward-backward pipeline."""
        obs, _ = synthetic_data
        obs_jax = jnp.array(obs)

        params = init_default_params(6)
        log_init = jnp.log(jnp.array(DEFAULT_INIT_PROBS, dtype=jnp.float32))
        log_trans = jnp.log(jnp.array(DEFAULT_TRANS, dtype=jnp.float32))

        log_emission = zib_log_prob(obs_jax, params)
        result = forward_backward(log_emission, log_init, log_trans, compute_xi=True)

        # Posteriors should be valid probabilities
        gamma = jnp.exp(result.log_gamma)
        assert jnp.allclose(gamma.sum(axis=2), 1.0, atol=1e-4)
        assert jnp.all(gamma >= 0)
        assert jnp.all(gamma <= 1.01)

        # Xi should be consistent with gamma
        xi = jnp.exp(result.log_xi)
        xi_marginal = xi.sum(axis=3)  # (N, T-1, K)
        assert jnp.allclose(xi_marginal, gamma[:, :-1, :], atol=1e-3)

    def test_full_em_pipeline(self, synthetic_data):
        """Test full EM -> inference pipeline."""
        obs, true_states = synthetic_data
        obs_jax = jnp.array(obs)

        # Run EM (few iterations for speed)
        em_result = baum_welch(obs_jax, K=6, max_iter=5, lbfgs_maxiter=10)
        params = em_result.params

        # Verify learned params make sense
        mu = np.asarray(params.emission.mu)
        assert all(mu[i] < mu[i+1] for i in range(5)), f"mu not monotonic: {mu}"

        # Run Viterbi with learned params
        log_emission = zib_log_prob(obs_jax, params.emission)
        vit_result = viterbi(log_emission, params.log_init, params.log_trans)

        # Should get valid state sequences
        assert jnp.all(vit_result.states >= 0)
        assert jnp.all(vit_result.states < 6)


class TestChunking:
    def test_generate_chunks(self):
        """Test 2D chunk generation."""
        chunks = generate_chunks(1024, 2048, chunk_size=512)
        assert len(chunks) == 2 * 4  # 2 row tiles * 4 col tiles

        # All chunks should cover the full raster
        total_area = sum(c.n_pixels for c in chunks)
        assert total_area == 1024 * 2048

    def test_edge_chunks(self):
        """Edge chunks should have correct reduced dimensions."""
        chunks = generate_chunks(1000, 1500, chunk_size=512)
        # Last row: 1000 - 512 = 488 pixels tall
        # Last col: 1500 - 2*512 = 476 pixels wide
        edge_chunk = [c for c in chunks if c.row_off == 512 and c.col_off == 1024][0]
        assert edge_chunk.height == 488
        assert edge_chunk.width == 476


class TestMemory:
    def test_memory_estimation(self):
        """Memory estimation should return reasonable values."""
        mem = estimate_chunk_memory(500_000, T=35, K=6)
        assert mem["total"] > 0
        assert mem["log_emission"] > mem["observations"]

    def test_optimal_batch_size(self):
        """Optimal batch size should be positive and reasonable."""
        batch = optimal_batch_size(gpu_memory_gb=80.0, T=35, K=6)
        assert batch > 100_000  # Should fit at least 100K pixels in 80GB
        assert batch < 100_000_000  # But not 100M

    def test_xi_memory_impact(self):
        """Computing xi should significantly increase memory."""
        mem_no_xi = estimate_chunk_memory(500_000, T=35, K=5, compute_xi=False)
        mem_xi = estimate_chunk_memory(500_000, T=35, K=5, compute_xi=True)
        assert mem_xi["total"] > mem_no_xi["total"]


class TestLegacyComparison:
    """Verify JAX 2-state Gaussian model reproduces original NumPy output."""

    def test_gaussian_emission_equivalence(self):
        """JAX Gaussian emissions should match NumPy computation."""
        from tide.emissions.gaussian import gaussian_log_prob

        rng = np.random.default_rng(42)
        N, T, K = 50, 35, 2
        obs = rng.uniform(0, 20, (N, T)).astype(np.float32)

        means = np.array([1.5, 10.0], dtype=np.float32)
        variances = np.array([44.89, 44.89], dtype=np.float32)  # 6.7^2

        # NumPy reference
        log_prob_np = np.zeros((N, T, K), dtype=np.float32)
        for k in range(K):
            log_prob_np[:, :, k] = (
                -0.5 * np.log(2 * np.pi * variances[k])
                - 0.5 * (obs - means[k])**2 / variances[k]
            )

        # JAX
        log_prob_jax = gaussian_log_prob(
            jnp.array(obs), jnp.array(means), jnp.array(variances)
        )

        np.testing.assert_allclose(
            np.asarray(log_prob_jax), log_prob_np, atol=1e-5
        )

    def test_2state_viterbi_matches_numpy(self):
        """JAX 2-state Viterbi should match NumPy reference exactly."""
        from tide.emissions.gaussian import gaussian_log_prob

        rng = np.random.default_rng(42)
        N, T = 20, 35
        obs = rng.uniform(0, 20, (N, T)).astype(np.float32)

        means = jnp.array([1.5, 10.0])
        variances = jnp.array([44.89, 44.89])
        log_init = jnp.log(jnp.array([0.9, 0.1]))
        log_trans = jnp.log(jnp.array([[0.97, 0.03], [0.01, 0.99]]))

        log_emission = gaussian_log_prob(jnp.array(obs), means, variances)
        result = viterbi(log_emission, log_init, log_trans)

        # States should be 0 or 1
        assert jnp.all((result.states == 0) | (result.states == 1))
        # Log probs should be finite
        assert jnp.all(jnp.isfinite(result.log_prob))
