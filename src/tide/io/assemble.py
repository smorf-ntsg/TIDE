"""Reassemble per-chunk .npy results into output GeoTIFFs.

Phase 2 of the pre-chunked pipeline. Reads result_states.npy and
result_posteriors.npy from each chunk directory, writes into the
standard TIDE output rasters using existing write_chunk_results().
"""

import logging
import time
from pathlib import Path

import numpy as np

from tide.data.chunking import Chunk
from tide.io.writer import (
    create_output_rasters,
    open_output_handles,
    close_output_handles,
    write_chunk_results,
)

log = logging.getLogger(__name__)


def _load_chunk_dir(chunk_dir: Path) -> dict | None:
    """Load results and metadata from a chunk directory.

    Returns None if the chunk has no results (empty or missing).
    """
    states_path = chunk_dir / "result_states.npy"
    if not states_path.exists():
        return None

    meta = np.load(chunk_dir / "meta.npy")
    # meta = [chunk_id, row_off, col_off, height, width, N]
    chunk = Chunk(
        chunk_id=int(meta[0]),
        row_off=int(meta[1]),
        col_off=int(meta[2]),
        height=int(meta[3]),
        width=int(meta[4]),
    )

    states = np.load(states_path)
    valid_indices = np.load(chunk_dir / "valid_indices.npy")

    posteriors_path = chunk_dir / "result_posteriors.npy"
    posteriors = np.load(posteriors_path) if posteriors_path.exists() else None

    return {
        "chunk": chunk,
        "states": states,
        "posteriors": posteriors,
        "valid_indices": valid_indices,
        "n_valid": states.shape[0],
    }


def assemble_outputs(
    chunks_dir: Path,
    output_dir: Path,
    ref_profile: dict,
    years: list[int],
    n_states: int = 6,
) -> dict:
    """Reassemble per-chunk .npy results into output GeoTIFFs.

    Args:
        chunks_dir: Directory containing chunk_XXXXX/ subdirectories with results.
        output_dir: Output directory for GeoTIFFs.
        ref_profile: Reference georeferencing profile (from mask).
        years: List of year values.
        n_states: Number of HMM states.

    Returns:
        Dict with processing summary.
    """
    t_start = time.time()

    # Find all chunk directories
    chunk_dirs = sorted(chunks_dir.glob("chunk_*"))
    log.info(f"Found {len(chunk_dirs)} chunk directories")

    # Create output rasters
    paths = create_output_rasters(output_dir, ref_profile, years, n_states)
    handles = open_output_handles(paths)

    total_pixels = 0
    n_written = 0

    try:
        for i, chunk_dir in enumerate(chunk_dirs):
            data = _load_chunk_dir(chunk_dir)
            if data is None or data["n_valid"] == 0:
                continue

            write_chunk_results(
                handles,
                data["chunk"],
                data["states"],
                data["posteriors"],
                data["valid_indices"],
                years,
                n_states,
            )
            total_pixels += data["n_valid"]
            n_written += 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                log.info(
                    f"Assembled {i + 1}/{len(chunk_dirs)} chunks "
                    f"({total_pixels:,} pixels, {rate:.1f} chunks/s)"
                )
    finally:
        close_output_handles(handles)

    elapsed = time.time() - t_start
    log.info(
        f"Assembly complete: {n_written} chunks, "
        f"{total_pixels:,} pixels, {elapsed:.1f}s"
    )

    return {"total_pixels": total_pixels, "n_chunks": n_written}
