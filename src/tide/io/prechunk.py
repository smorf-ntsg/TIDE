"""Pre-chunk input rasters to per-chunk .npy files for fast inference.

Phase 0 of the pre-chunked pipeline. Reads all input rasters once, writes
per-chunk .npy files that can be loaded in ~2ms during inference (vs ~50ms
for GeoTIFF reads with deflate decompression).

Directory structure::

    chunks_dir/
      chunk_00000/
        obs.npy              # (N, T) float32
        valid_indices.npy    # (N, 2) int32
        meta.npy             # [chunk_id, row_off, col_off, height, width, N]
        fire_severity.npy    # (N, T-1) float32  (if dynamic transitions)
        spei.npy             # (N, T-1) float32
        elevation.npy        # (N,) float32
        chili.npy            # (N,) float32
        aridity.npy          # (N,) float32
        seed_distance.npy    # (N, T-1) float32
        focal_mean.npy       # (N, T-1) float32
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from tide.config import PipelineConfig
from tide.data.chunking import Chunk, generate_chunks, filter_masked_chunks
from tide.data.rap import get_reference_profile
from tide.io.reader import (
    open_input_handles,
    close_input_handles,
    read_chunk_data,
    read_covariate_data,
)

log = logging.getLogger(__name__)


def _chunk_dir(chunks_dir: Path, chunk: Chunk) -> Path:
    """Return the directory for a given chunk."""
    return chunks_dir / f"chunk_{chunk.chunk_id:05d}"


def _save_chunk(
    chunk: Chunk,
    chunks_dir: Path,
    obs: np.ndarray,
    valid_indices: np.ndarray,
    covariates: dict[str, np.ndarray | None] | None,
) -> int:
    """Save a single chunk's data to .npy files. Returns N (valid pixels)."""
    N = obs.shape[0]
    if N == 0:
        return 0

    d = _chunk_dir(chunks_dir, chunk)
    d.mkdir(parents=True, exist_ok=True)

    np.save(d / "obs.npy", obs)
    np.save(d / "valid_indices.npy", valid_indices.astype(np.int32))
    np.save(d / "meta.npy", np.array([
        chunk.chunk_id, chunk.row_off, chunk.col_off,
        chunk.height, chunk.width, N,
    ], dtype=np.int32))

    if covariates is not None:
        for key, arr in covariates.items():
            if arr is not None:
                np.save(d / f"{key}.npy", arr.astype(np.float32))

    return N


def prechunk_all(
    config: PipelineConfig,
    chunks_dir: Path,
    n_threads: int = 6,
    dynamic_transitions: bool = False,
) -> dict:
    """Pre-chunk all input rasters to per-chunk .npy files.

    Args:
        config: Pipeline configuration.
        chunks_dir: Output directory for chunk files.
        n_threads: Number of I/O threads for parallel processing.
        dynamic_transitions: Whether to also pre-chunk covariate data.

    Returns:
        Dict with processing summary.
    """
    chunks_dir.mkdir(parents=True, exist_ok=True)

    ref = get_reference_profile(config.mask_path)
    chunks = generate_chunks(
        ref["height"], ref["width"], config.inference.chunk_size
    )
    log.info(f"Generated {len(chunks)} total chunks")

    chunks = filter_masked_chunks(chunks, config.mask_path)
    log.info(f"After masking: {len(chunks)} chunks with valid pixels")

    ref_transform = ref.get("transform")
    years = list(config.years)

    total_pixels = 0
    t_start = time.time()

    # Process chunks in parallel using threads (I/O-bound)
    # Each thread gets its own set of file handles
    def _process_chunk(chunk: Chunk, handles: dict) -> int:
        obs, cube_float, mask, valid_indices = read_chunk_data(
            config.input_paths, config.mask_path, chunk,
            ref_transform=ref_transform,
            open_srcs=handles["srcs"],
            open_stack_src=handles["stack"],
            open_mask_src=handles["mask"],
        )

        if obs.shape[0] == 0:
            return 0

        covariates = None
        if dynamic_transitions:
            valid_rows = valid_indices[:, 0]
            valid_cols = valid_indices[:, 1]
            covariates = read_covariate_data(
                config, chunk, valid_rows, valid_cols,
                years, cube_float, ref_transform=ref_transform,
            )

        return _save_chunk(chunk, chunks_dir, obs, valid_indices, covariates)

    if n_threads <= 1:
        # Single-threaded: one set of handles
        handles = open_input_handles(config.input_paths, config.mask_path)
        try:
            for i, chunk in enumerate(chunks):
                n = _process_chunk(chunk, handles)
                total_pixels += n
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    log.info(
                        f"Pre-chunked {i + 1}/{len(chunks)} "
                        f"({total_pixels:,} pixels, {rate:.1f} chunks/s)"
                    )
        finally:
            close_input_handles(handles)
    else:
        # Multi-threaded: each thread gets its own handles
        import threading
        counter_lock = threading.Lock()
        shared = {"done": 0, "pixels": 0}

        def _worker(thread_chunks: list[Chunk]) -> int:
            handles = open_input_handles(config.input_paths, config.mask_path)
            thread_total = 0
            try:
                for chunk in thread_chunks:
                    n = _process_chunk(chunk, handles)
                    thread_total += n
                    with counter_lock:
                        shared["done"] += 1
                        shared["pixels"] += n
                        done = shared["done"]
                        pixels = shared["pixels"]
                    if done % 100 == 0:
                        elapsed = time.time() - t_start
                        rate = done / elapsed
                        log.info(
                            f"Pre-chunked {done}/{len(chunks)} "
                            f"({pixels:,} pixels, {rate:.1f} chunks/s)"
                        )
            finally:
                close_input_handles(handles)
            return thread_total

        # Split chunks across threads
        chunk_groups = [[] for _ in range(n_threads)]
        for i, chunk in enumerate(chunks):
            chunk_groups[i % n_threads].append(chunk)

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(_worker, group) for group in chunk_groups]
            for future in as_completed(futures):
                total_pixels += future.result()

    elapsed = time.time() - t_start
    log.info(
        f"Pre-chunking complete: {len(chunks)} chunks, "
        f"{total_pixels:,} pixels, {elapsed:.1f}s"
    )

    # Save manifest for downstream phases
    manifest = {
        "n_chunks": len(chunks),
        "total_pixels": total_pixels,
        "chunk_size": config.inference.chunk_size,
        "dynamic_transitions": dynamic_transitions,
        "n_years": len(years),
    }
    np.savez(chunks_dir / "manifest.npz", **manifest)

    return manifest
