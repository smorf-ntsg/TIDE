"""Tests for Viterbi algorithm."""

import jax.numpy as jnp
import numpy as np
import pytest

from tide.hmm.viterbi import viterbi


def _numpy_viterbi(log_emission, log_init, log_trans):
    """Reference NumPy Viterbi (adapted from )."""
    N, T, K = log_emission.shape
    delta = np.copy(log_init[None, :] + log_emission[:, 0, :])  # (N, K)
    psi = np.zeros((N, T, K), dtype=np.int32)

    for t in range(1, T):
        candidates = delta[:, :, None] + log_trans[None, :, :]  # (N, K, K)
        psi[:, t, :] = candidates.argmax(axis=1)
        delta = candidates.max(axis=1) + log_emission[:, t, :]

    states = np.zeros((N, T), dtype=np.int32)
    states[:, -1] = delta.argmax(axis=1)
    log_prob = delta.max(axis=1)

    for t in range(T - 2, -1, -1):
        for n in range(N):
            states[n, t] = psi[n, t + 1, states[n, t + 1]]

    return states, log_prob


class TestViterbi:
    def test_output_shapes(self):
        """Check output tensor shapes."""
        rng = np.random.default_rng(42)
        N, T, K = 10, 8, 3
        log_emission = jnp.array(rng.standard_normal((N, T, K)).astype(np.float32))
        log_init = jnp.log(jnp.ones(K) / K)
        trans = np.eye(K) * 0.9 + 0.1 / K
        log_trans = jnp.log(jnp.array(trans / trans.sum(1, keepdims=True), dtype=jnp.float32))

        result = viterbi(log_emission, log_init, log_trans)
        assert result.states.shape == (10, 8)
        assert result.log_prob.shape == (10,)

    def test_states_valid_range(self):
        """State indices should be in [0, K)."""
        rng = np.random.default_rng(42)
        N, T, K = 20, 35, 5
        log_emission = jnp.array(rng.standard_normal((N, T, K)).astype(np.float32))
        log_init = jnp.log(jnp.ones(K) / K)
        trans = np.eye(K) * 0.9 + 0.1 / K
        log_trans = jnp.log(jnp.array(trans / trans.sum(1, keepdims=True), dtype=jnp.float32))

        result = viterbi(log_emission, log_init, log_trans)
        assert jnp.all(result.states >= 0)
        assert jnp.all(result.states < K)

    def test_matches_numpy_reference(self):
        """JAX Viterbi should match NumPy reference implementation."""
        rng = np.random.default_rng(123)
        N, T, K = 5, 10, 2

        log_emission_np = rng.standard_normal((N, T, K)).astype(np.float32)
        log_init_np = np.log(np.array([0.9, 0.1], dtype=np.float32))
        trans = np.array([[0.97, 0.03], [0.01, 0.99]], dtype=np.float32)
        log_trans_np = np.log(trans)

        # NumPy reference
        states_np, log_prob_np = _numpy_viterbi(log_emission_np, log_init_np, log_trans_np)

        # JAX
        result = viterbi(
            jnp.array(log_emission_np),
            jnp.array(log_init_np),
            jnp.array(log_trans_np),
        )

        np.testing.assert_array_equal(np.asarray(result.states), states_np)
        np.testing.assert_allclose(np.asarray(result.log_prob), log_prob_np, atol=1e-4)

    def test_deterministic_sequence(self):
        """With very strong emissions, Viterbi should follow the dominant state."""
        N, T, K = 1, 5, 3
        # Make state 1 dominant for all timesteps
        log_emission = jnp.full((N, T, K), -10.0)
        log_emission = log_emission.at[:, :, 1].set(0.0)
        log_init = jnp.log(jnp.ones(K) / K)
        log_trans = jnp.log(jnp.ones((K, K)) / K)

        result = viterbi(log_emission, log_init, log_trans)
        assert jnp.all(result.states == 1)

    def test_six_state_model(self):
        """6-state Viterbi should run without errors."""
        rng = np.random.default_rng(42)
        N, T, K = 100, 35, 6
        log_emission = jnp.array(rng.standard_normal((N, T, K)).astype(np.float32))
        log_init = jnp.log(jnp.ones(K) / K)
        trans = np.eye(K) * 0.9 + 0.1 / K
        log_trans = jnp.log(jnp.array(trans / trans.sum(1, keepdims=True), dtype=jnp.float32))

        result = viterbi(log_emission, log_init, log_trans)
        assert result.states.shape == (100, 35)
        assert jnp.all(jnp.isfinite(result.log_prob))
