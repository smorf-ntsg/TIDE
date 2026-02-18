"""JAX GPU worker for chunk-level inference.

Each worker processes a spatial chunk: loads data, runs inference,
returns results. Designed to be called by Dask or directly.
"""

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tide.types import HMMParams, ViterbiResult, ForwardBackwardResult
from tide.emissions.zero_inflated_beta import zib_log_prob
from tide.hmm.forward_backward import forward_backward
from tide.hmm.viterbi import viterbi
from tide.data.chunking import Chunk
from tide.io.reader import read_chunk_data

log = logging.getLogger(__name__)


def infer_chunk(
    input_paths: list[Path],
    mask_path: Path,
    chunk: Chunk,
    params: HMMParams,
    compute_posteriors: bool = True,
    batch_pixels: int = 500_000,
    device_id: int | None = None,
    config=None,
    years: list[int] | None = None,
    ref_transform=None,
    _preloaded: tuple | None = None,
) -> dict:
    """Run inference on a single spatial chunk.

    Args:
        input_paths: List of T RAP GeoTIFF paths.
        mask_path: Path to analysis region mask.
        chunk: Spatial chunk specification.
        params: Frozen HMM parameters.
        compute_posteriors: Whether to compute posterior probabilities.
        batch_pixels: Max pixels per GPU batch.
        device: JAX device to use (None for default).
        config: PipelineConfig (optional, for dynamic transitions).
        years: List of years (required if config.enable_dynamic_transitions).
        _preloaded: Pre-loaded (obs_np, cube_float, mask, valid_indices) tuple
            from async prefetch. If provided, skips I/O for this chunk.

    Returns:
        Dict with:
            states: (N, T) int32
            posteriors: (N, T, K) float32 (or None)
            valid_indices: (N, 2) int32
            chunk: Chunk object
            n_valid: int
    """
    if _preloaded is not None:
        obs_np, cube_float, mask, valid_indices = _preloaded
    else:
        obs_np, cube_float, mask, valid_indices = read_chunk_data(
            input_paths, mask_path, chunk, ref_transform=ref_transform,
        )
    N = obs_np.shape[0]

    if N == 0:
        return {
            "states": np.empty((0, len(input_paths)), dtype=np.int32),
            "posteriors": None,
            "valid_indices": valid_indices,
            "chunk": chunk,
            "n_valid": 0,
        }

    # Build dynamic log transition matrices if enabled
    dynamic_log_trans = None
    use_dynamic = (
        config is not None
        and getattr(config, "enable_dynamic_transitions", False)
        and params.transition_weights is not None
        and years is not None
    )

    if use_dynamic:
        from tide.io.reader import read_covariate_data
        from tide.transitions.covariates import stack_covariates
        from tide.transitions.dynamic import compute_dynamic_log_trans

        valid_rows = valid_indices[:, 0]
        valid_cols = valid_indices[:, 1]

        cov_data = read_covariate_data(
            config, chunk, valid_rows, valid_cols, years, cube_float,
            ref_transform=ref_transform,
        )

        # Convert to JAX and stack
        cov_jax = {}
        for key, val in cov_data.items():
            if val is not None:
                cov_jax[key] = jnp.array(val, dtype=jnp.float32)

        covariates = stack_covariates(**cov_jax)  # (N, T-1, D)

        # Pad covariates to fixed size so compute_dynamic_log_trans compiles once
        if N < batch_pixels:
            D = covariates.shape[2]
            T_minus_1 = covariates.shape[1]
            cov_padded = jnp.zeros((batch_pixels, T_minus_1, D), dtype=jnp.float32)
            cov_padded = cov_padded.at[:N].set(covariates)
            covariates = cov_padded

        dynamic_log_trans = compute_dynamic_log_trans(
            covariates, params.transition_weights,
        )  # (PAD_N, T-1, K, K) — stays on GPU
        # Slice back to actual N (on GPU)
        dynamic_log_trans = dynamic_log_trans[:N]

    T = obs_np.shape[1]
    K = params.log_init.shape[0]

    # Process in batches to fit GPU memory.
    # Pad each batch to fixed PAD_N so JIT compiles once (avoids recompilation).
    PAD_N = batch_pixels
    all_states: list[jnp.ndarray] = []
    all_posteriors: list[jnp.ndarray] = []

    # Resolve device if specified
    device = None
    if device_id is not None:
        try:
            device = jax.devices()[device_id]
        except Exception:
            log.warning(f"Could not resolve device_id {device_id}, using default")

    for start in range(0, N, PAD_N):
        end = min(start + PAD_N, N)
        actual_n = end - start

        # Pad observations to fixed PAD_N
        obs_chunk = obs_np[start:end]
        if actual_n < PAD_N:
            obs_padded = np.zeros((PAD_N, T), dtype=np.float32)
            obs_padded[:actual_n] = obs_chunk
        else:
            obs_padded = obs_chunk

        obs_batch = jnp.array(obs_padded, dtype=jnp.float32)

        if device is not None:
            obs_batch = jax.device_put(obs_batch, device)

        # Compute emissions: (batch, T, K)
        log_emission = zib_log_prob(obs_batch, params.emission)

        # Select transition matrix for this batch
        if dynamic_log_trans is not None:
            batch_log_trans = jnp.array(dynamic_log_trans[start:end])
            # Pad dynamic trans if needed
            if actual_n < PAD_N:
                pad_trans = jnp.broadcast_to(
                    params.log_trans,
                    (PAD_N - actual_n,) + params.log_trans.shape,
                )
                batch_log_trans = jnp.concatenate(
                    [batch_log_trans, pad_trans], axis=0,
                )
        else:
            batch_log_trans = params.log_trans

        # Viterbi decoding — stay on GPU, slice back to actual_n
        vit_result = viterbi(log_emission, params.log_init, batch_log_trans)
        all_states.append(vit_result.states[:actual_n])

        # Forward-backward for posteriors — stay on GPU, slice back
        if compute_posteriors:
            fb_result = forward_backward(
                log_emission, params.log_init, batch_log_trans,
                compute_xi=False,
            )
            all_posteriors.append(jnp.exp(fb_result.log_gamma[:actual_n]))

        del obs_batch, log_emission

    # Single transfer: concatenate on GPU then move to CPU once.
    states = np.asarray(jnp.concatenate(all_states, axis=0))
    posteriors = (
        np.asarray(jnp.concatenate(all_posteriors, axis=0))
        if compute_posteriors else None
    )

    log.debug(f"Chunk {chunk.chunk_id}: {N} pixels processed")

    return {
        "states": states,
        "posteriors": posteriors,
        "valid_indices": valid_indices,
        "chunk": chunk,
        "n_valid": N,
    }


def infer_chunk_prechunked(
    obs_np: np.ndarray,
    params: HMMParams,
    covariates_np: np.ndarray | None = None,
    compute_posteriors: bool = True,
    batch_pixels: int = 500_000,
) -> dict:
    """Run inference on pre-chunked numpy arrays.

    Skips all rasterio I/O and covariate computation — data arrives
    pre-loaded from .npy files.

    Args:
        obs_np: (N, T) float32 observations.
        params: Frozen HMM parameters.
        covariates_np: (N, T-1, D) float32 stacked covariates, or None.
        compute_posteriors: Whether to compute posterior probabilities.
        batch_pixels: Max pixels per GPU batch.

    Returns:
        Dict with states (N, T) uint8 and posteriors (N, T, K) uint8 or None.
    """
    N, T = obs_np.shape
    K = params.log_init.shape[0]

    if N == 0:
        return {
            "states": np.empty((0, T), dtype=np.uint8),
            "posteriors": None,
            "n_valid": 0,
        }

    # Build dynamic log transition matrices if covariates provided.
    # Keep as JAX array on GPU to avoid GPU→CPU→GPU roundtrip.
    dynamic_log_trans = None
    if covariates_np is not None and params.transition_weights is not None:
        from tide.transitions.dynamic import compute_dynamic_log_trans

        # Pad covariates to fixed size so compute_dynamic_log_trans compiles once
        PAD_N = batch_pixels
        D = covariates_np.shape[2] if covariates_np.ndim == 3 else 0
        T_minus_1 = covariates_np.shape[1] if covariates_np.ndim == 3 else T - 1
        if N < PAD_N:
            cov_padded = np.zeros((PAD_N, T_minus_1, D), dtype=np.float32)
            cov_padded[:N] = covariates_np
        else:
            cov_padded = covariates_np
        cov_jax = jnp.array(cov_padded, dtype=jnp.float32)
        dynamic_log_trans_full = compute_dynamic_log_trans(
            cov_jax, params.transition_weights,
        )  # (PAD_N, T-1, K, K) — stays on GPU
        # Slice back to actual N (on GPU)
        dynamic_log_trans = dynamic_log_trans_full[:N]

    # Padding size — all chunks have N <= batch_pixels
    PAD_N = batch_pixels

    all_states: list[jnp.ndarray] = []
    all_posteriors: list[jnp.ndarray] = []

    for start in range(0, N, PAD_N):
        end = min(start + PAD_N, N)
        actual_n = end - start

        # Pad observations to fixed PAD_N so JIT compiles once
        obs_chunk = obs_np[start:end]
        if actual_n < PAD_N:
            obs_padded = np.zeros((PAD_N, T), dtype=np.float32)
            obs_padded[:actual_n] = obs_chunk
        else:
            obs_padded = obs_chunk

        obs_batch = jnp.array(obs_padded, dtype=jnp.float32)
        log_emission = zib_log_prob(obs_batch, params.emission)

        if dynamic_log_trans is not None:
            batch_log_trans = dynamic_log_trans[start:end]
            # Pad dynamic trans if needed
            if actual_n < PAD_N:
                pad_trans = jnp.broadcast_to(
                    params.log_trans,
                    (PAD_N - actual_n,) + params.log_trans.shape,
                )
                batch_log_trans = jnp.concatenate(
                    [batch_log_trans, pad_trans], axis=0,
                )
        else:
            batch_log_trans = params.log_trans

        vit_result = viterbi(log_emission, params.log_init, batch_log_trans)
        all_states.append(vit_result.states[:actual_n])  # slice back

        if compute_posteriors:
            fb_result = forward_backward(
                log_emission, params.log_init, batch_log_trans,
                compute_xi=False,
            )
            all_posteriors.append(jnp.exp(fb_result.log_gamma[:actual_n]))

        del obs_batch, log_emission

    states = np.asarray(jnp.concatenate(all_states, axis=0))
    posteriors = (
        np.asarray(jnp.concatenate(all_posteriors, axis=0))
        if compute_posteriors else None
    )

    # Quantize for compact .npy storage
    states_u8 = states.astype(np.uint8)
    posteriors_u8 = None
    if posteriors is not None:
        posteriors_u8 = np.clip(posteriors * 200.0, 0, 200).astype(np.uint8)

    return {
        "states": states_u8,
        "posteriors": posteriors_u8,
        "n_valid": N,
    }
