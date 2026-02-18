"""Dask-based distributed chunk dispatch.

Orchestrates processing of spatial chunks across GPUs using
dask.delayed for simple, reliable parallelism.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import numpy as np

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


def _gpu_stats_str() -> str:
    """Return a one-line GPU utilization summary, or '' if pynvml unavailable."""
    if not _NVML_OK:
        return ""
    try:
        parts = []
        n = _pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h = _pynvml.nvmlDeviceGetHandleByIndex(i)
            u = _pynvml.nvmlDeviceGetUtilizationRates(h)
            m = _pynvml.nvmlDeviceGetMemoryInfo(h)
            parts.append(
                f"GPU{i}: {u.gpu}% compute, "
                f"{m.used/1024**3:.1f}/{m.total/1024**3:.0f}GB VRAM"
            )
        return " | ".join(parts)
    except Exception:
        return ""

from tide.types import HMMParams
from tide.data.chunking import Chunk, generate_chunks, filter_masked_chunks
from tide.distributed.gpu_worker import infer_chunk, infer_chunk_prechunked
from tide.io.reader import open_input_handles, close_input_handles
from tide.io.writer import (
    create_output_rasters,
    open_output_handles,
    close_output_handles,
    write_chunk_results,
)

log = logging.getLogger(__name__)


def run_full_inference(
    input_paths: list[Path],
    mask_path: Path,
    output_dir: Path,
    params: HMMParams,
    ref_profile: dict,
    years: list[int],
    n_states: int = 5,
    chunk_size: int = 512,
    batch_pixels: int = 500_000,
    compute_posteriors: bool = True,
    use_dask: bool = False,
    n_workers: int = 1,
    config=None,
    prechunked: bool = False,
    chunks_dir: Path | None = None,
) -> dict:
    """Run inference on the full dataset.

    Two execution modes:
    - Sequential: Process chunks one at a time (simple, debuggable)
    - Dask: Dispatch chunks to workers via dask.delayed

    Args:
        input_paths: List of T RAP GeoTIFF paths.
        mask_path: Path to analysis region mask.
        output_dir: Output directory.
        params: Frozen HMM parameters.
        ref_profile: Reference georeferencing profile.
        years: List of year values.
        n_states: Number of HMM states.
        chunk_size: Spatial tile dimension.
        batch_pixels: Max pixels per GPU batch.
        compute_posteriors: Whether to compute posteriors.
        use_dask: Whether to use Dask distributed.
        n_workers: Number of Dask workers (if use_dask=True).
        config: PipelineConfig (optional, for dynamic transitions).
        prechunked: If True, read from pre-chunked .npy files.
        chunks_dir: Directory containing chunk_XXXXX/ subdirectories.

    Returns:
        Dict with processing summary.
    """
    # Generate and filter chunks
    chunks = generate_chunks(
        ref_profile["height"], ref_profile["width"], chunk_size
    )
    log.info(f"Generated {len(chunks)} total chunks")

    chunks = filter_masked_chunks(chunks, mask_path)
    log.info(f"After masking: {len(chunks)} chunks with valid pixels")

    # Create output rasters
    paths = create_output_rasters(
        output_dir, ref_profile, years, n_states
    )

    total_pixels = 0

    ref_transform = ref_profile.get("transform")

    if prechunked:
        if use_dask:
            log.warning(
                "Dask mode is not supported with --prechunked. "
                "Falling back to sequential prechunked inference."
            )
        if chunks_dir is None:
            raise ValueError("chunks_dir required when prechunked=True")
        total_pixels = _run_prechunked(
            chunks_dir, params, batch_pixels, compute_posteriors,
        )
        log.info(f"Inference complete: {total_pixels:,} total pixels processed")
        return {"total_pixels": total_pixels, "n_chunks": len(chunks)}

    if use_dask:
        total_pixels = _run_dask(
            chunks, input_paths, mask_path, params, paths, years,
            n_states, batch_pixels, compute_posteriors, n_workers,
            config=config, ref_transform=ref_transform,
        )
    else:
        total_pixels = _run_sequential(
            chunks, input_paths, mask_path, params, paths, years,
            n_states, batch_pixels, compute_posteriors,
            config=config, ref_transform=ref_transform,
        )

    log.info(f"Inference complete: {total_pixels:,} total pixels processed")
    return {"total_pixels": total_pixels, "n_chunks": len(chunks)}


def _run_sequential(
    chunks, input_paths, mask_path, params, output_paths, years,
    n_states, batch_pixels, compute_posteriors, config=None,
    ref_transform=None,
):
    """Sequential chunk processing with async I/O prefetch + async writes + persistent handles.

    Three optimizations:
    1. Persistent file handles: open all rasters once at the start rather
       than per-chunk, eliminating ~898ms of file-open overhead per chunk
       (23ms × 39 files).
    2. Stacked treeCover: if treeCover_stack.tif exists, auto-detected by
       open_input_handles(). Replaces 39 separate reads with one ~6× faster
       multi-band read.
    3. Async prefetch + async writes: I/O prefetch thread reads the next chunk
       while GPU processes the current one. A separate write thread flushes
       results to disk while the GPU handles the next chunk.

    Timeline per chunk (steady state):
        Thread:  [read N+1] ─────────────────────────
        Main:               [GPU N] [write N-1 async]
        Write:                       [write N-1] ─────
    """
    handles = open_output_handles(output_paths)
    total = 0

    from tide.io.reader import read_chunk_data
    io_handles = open_input_handles(input_paths, mask_path)

    # One-time JIT warmup at padded size — absorbs XLA compilation (~30s)
    if chunks:
        import jax.numpy as jnp
        from tide.emissions.zero_inflated_beta import zib_log_prob
        from tide.hmm.viterbi import viterbi as _viterbi
        from tide.hmm.forward_backward import forward_backward as _fb

        T = len(input_paths)
        log.info(f"JIT warmup at PAD_N={batch_pixels}, T={T}...")
        _w_obs = jnp.zeros((batch_pixels, T), dtype=jnp.float32)
        _w_em = zib_log_prob(_w_obs, params.emission)
        _ = _viterbi(_w_em, params.log_init, params.log_trans)
        if compute_posteriors:
            _ = _fb(_w_em, params.log_init, params.log_trans, compute_xi=False)
        del _w_obs, _w_em
        log.info("JIT warmup complete")

    def _load(chunk):
        return read_chunk_data(
            input_paths, mask_path, chunk, ref_transform=ref_transform,
            open_srcs=io_handles["srcs"],
            open_stack_src=io_handles["stack"],
            open_mask_src=io_handles["mask"],
        )

    def _write(result, chunk):
        if result["n_valid"] > 0:
            write_chunk_results(
                handles, chunk, result["states"], result["posteriors"],
                result["valid_indices"], years, n_states,
            )
        return result["n_valid"]

    try:
        with ThreadPoolExecutor(max_workers=1) as read_pool, \
             ThreadPoolExecutor(max_workers=1) as write_pool:

            read_future = read_pool.submit(_load, chunks[0]) if chunks else None
            write_future = None

            for i, chunk in enumerate(chunks):
                # Wait for this chunk's read to finish
                preloaded = read_future.result()

                # Start reading the next chunk immediately
                if i + 1 < len(chunks):
                    read_future = read_pool.submit(_load, chunks[i + 1])

                # Run GPU inference
                result = infer_chunk(
                    input_paths, mask_path, chunk, params,
                    compute_posteriors=compute_posteriors,
                    batch_pixels=batch_pixels,
                    config=config,
                    years=years,
                    ref_transform=ref_transform,
                    _preloaded=preloaded,
                )

                # Wait for the previous write to finish before submitting new one
                # (rasterio handles are not thread-safe; serialize via pool size=1)
                if write_future is not None:
                    total += write_future.result()
                write_future = write_pool.submit(_write, result, chunk)

                if (i + 1) % 100 == 0:
                    gpu_str = _gpu_stats_str()
                    msg = f"Processed {i + 1}/{len(chunks)} chunks, {total:,} pixels"
                    if gpu_str:
                        msg += f" | {gpu_str}"
                    log.info(msg)

            # Flush final write
            if write_future is not None:
                total += write_future.result()

    finally:
        close_output_handles(handles)
        close_input_handles(io_handles)

    return total


def _run_prechunked(
    chunks_dir, params, batch_pixels, compute_posteriors,
):
    """Prechunked inference: read .npy inputs, run GPU, write .npy results.

    Timeline per chunk (steady state):
        Read thread:   [load N+1] ──────────
        Main (GPU):            [infer N] ────
        Write thread:                [save N]
    """
    import time
    from tide.pipeline import _numpy_stack_covariates

    import jax.numpy as jnp
    from tide.emissions.zero_inflated_beta import zib_log_prob
    from tide.hmm.viterbi import viterbi
    from tide.hmm.forward_backward import forward_backward

    chunk_dirs = sorted(chunks_dir.glob("chunk_*"))
    log.info(f"Prechunked inference: {len(chunk_dirs)} chunk directories")

    # One-time JIT warmup at padded size — absorbs XLA compilation (~30s)
    # so subsequent chunks run at full speed with no recompilation.
    if chunk_dirs:
        _peek_obs = np.load(chunk_dirs[0] / "obs.npy")
        T = _peek_obs.shape[1]
        del _peek_obs

        log.info(f"JIT warmup at PAD_N={batch_pixels}, T={T}...")
        _w_obs = jnp.zeros((batch_pixels, T), dtype=jnp.float32)
        _w_em = zib_log_prob(_w_obs, params.emission)
        _ = viterbi(_w_em, params.log_init, params.log_trans)
        if compute_posteriors:
            _ = forward_backward(_w_em, params.log_init, params.log_trans, compute_xi=False)
        del _w_obs, _w_em
        log.info("JIT warmup complete")

    manifest_path = chunks_dir / "manifest.npz"
    has_dynamic = False
    if manifest_path.exists():
        m = np.load(manifest_path, allow_pickle=True)
        has_dynamic = bool(m.get("dynamic_transitions", False))

    def _load(chunk_dir):
        obs = np.load(chunk_dir / "obs.npy")
        covariates = None
        if has_dynamic and params.transition_weights is not None:
            cov_dict = {}
            for key in ["fire_severity", "spei", "elevation", "chili",
                        "aridity", "seed_distance", "focal_mean"]:
                p = chunk_dir / f"{key}.npy"
                if p.exists():
                    cov_dict[key] = np.load(p)
            if cov_dict:
                covariates = _numpy_stack_covariates(cov_dict)
        return obs, covariates

    def _save(result, chunk_dir):
        if result["n_valid"] > 0:
            np.save(chunk_dir / "result_states.npy", result["states"])
            if result["posteriors"] is not None:
                np.save(chunk_dir / "result_posteriors.npy", result["posteriors"])
        return result["n_valid"]

    total = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=1) as read_pool, \
         ThreadPoolExecutor(max_workers=1) as write_pool:

        read_future = read_pool.submit(_load, chunk_dirs[0]) if chunk_dirs else None
        write_future = None

        for i, chunk_dir in enumerate(chunk_dirs):
            obs, covariates = read_future.result()

            if i + 1 < len(chunk_dirs):
                read_future = read_pool.submit(_load, chunk_dirs[i + 1])

            result = infer_chunk_prechunked(
                obs, params,
                covariates_np=covariates,
                compute_posteriors=compute_posteriors,
                batch_pixels=batch_pixels,
            )

            if write_future is not None:
                total += write_future.result()
            write_future = write_pool.submit(_save, result, chunk_dir)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                gpu_str = _gpu_stats_str()
                msg = (
                    f"Processed {i + 1}/{len(chunk_dirs)} chunks, "
                    f"{total:,} pixels, {rate:.1f} chunks/s"
                )
                if gpu_str:
                    msg += f" | {gpu_str}"
                log.info(msg)

        if write_future is not None:
            total += write_future.result()

    return total


def _run_dask(
    chunks, input_paths, mask_path, params, output_paths, years,
    n_states, batch_pixels, compute_posteriors, n_workers, config=None,
    ref_transform=None,
):
    """Dask-distributed chunk processing."""
    import dask
    from dask.distributed import Client, as_completed

    # Assign chunks to GPUs round-robin
    devices = jax.devices()
    n_gpus = len(devices)
    log.info(f"Dask mode: {n_workers} workers, {n_gpus} GPUs")

    client = Client(n_workers=n_workers, threads_per_worker=1)

    try:
        # Submit all chunks as delayed tasks, routing each to a specific GPU
        # via round-robin assignment so all GPUs are kept busy.
        futures = []
        for i, chunk in enumerate(chunks):
            device_id = devices[i % n_gpus].id if n_gpus > 0 else None
            future = client.submit(
                infer_chunk,
                input_paths, mask_path, chunk, params,
                compute_posteriors=compute_posteriors,
                batch_pixels=batch_pixels,
                device_id=device_id,
                config=config,
                years=years,
                ref_transform=ref_transform,
                key=f"chunk-{chunk.chunk_id}",
            )
            futures.append(future)

        # Write results as they complete
        handles = open_output_handles(output_paths)
        total = 0

        try:
            for future in as_completed(futures):
                result = future.result()
                if result["n_valid"] > 0:
                    chunk = result["chunk"]
                    write_chunk_results(
                        handles, chunk, result["states"], result["posteriors"],
                        result["valid_indices"], years, n_states,
                    )
                    total += result["n_valid"]
        finally:
            close_output_handles(handles)

    finally:
        client.close()

    return total
