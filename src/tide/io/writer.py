"""Output raster creation and writing.

Output structure
----------------
  output_dir/
    states/
      states_1986.tif  ...  states_2024.tif   # 1-band uint8 per year
    posteriors/
      posteriors_1986.tif  ...  posteriors_2024.tif  # K-band uint8 per year
    trace_onset_year.tif   # 1-band int16
    sparse_year.tif
    open_year.tif
    woodland_year.tif
    forest_year.tif
    max_state.tif          # 1-band uint8

All temporal rasters (states, posteriors) use per-year single files to keep
individual file sizes manageable (~2GB / ~12GB compressed at CONUS scale).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from tide.data.chunking import Chunk

log = logging.getLogger(__name__)

NODATA_UINT8 = 255
NODATA_INT16 = -9999


def create_output_rasters(
    output_dir: Path,
    ref_profile: dict,
    years: list[int],
    n_states: int = 6,
    compression: str = "deflate",
    tile_size: int = 256,
) -> dict[str, Path]:
    """Create all output raster files (header-only).

    Per-year files:
    - states/states_{year}.tif: 1-band uint8, state labels 0-(K-1), NoData=255
    - posteriors/posteriors_{year}.tif: K-band uint8, P(k)*200, NoData=255

    Single-band summary files:
    - trace_onset_year.tif: int16, first year state >= 1
    - sparse_year.tif: int16, first year state >= 2
    - open_year.tif: int16, first year state >= 3
    - woodland_year.tif: int16, first year state >= 4
    - forest_year.tif: int16, first year state >= 5
    - max_state.tif: uint8, maximum state ever reached

    Args:
        output_dir: Output directory.
        ref_profile: Reference georeferencing profile.
        years: List of year values (e.g. range(1986, 2025)).
        n_states: Number of HMM states (K).
        compression: Compression algorithm.
        tile_size: Internal tile size for COG.

    Returns:
        Flat dict mapping key → Path. Keys:
        - "states_{year}" for each year
        - "posteriors_{year}" for each year
        - "trace_onset_year", "sparse_year", "open_year",
          "woodland_year", "forest_year", "max_state"
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    states_dir = output_dir / "states"
    post_dir = output_dir / "posteriors"
    states_dir.mkdir(exist_ok=True)
    post_dir.mkdir(exist_ok=True)

    base_profile = {
        "driver": "GTiff",
        "crs": ref_profile["crs"],
        "transform": ref_profile["transform"],
        "width": ref_profile["width"],
        "height": ref_profile["height"],
        "compress": compression,
        "tiled": True,
        "blockxsize": tile_size,
        "blockysize": tile_size,
    }

    paths = {}

    # Per-year state rasters (1-band each)
    for year in years:
        path = states_dir / f"states_{year}.tif"
        profile = {**base_profile, "dtype": "uint8", "count": 1, "nodata": NODATA_UINT8}
        with rasterio.open(path, "w", **profile):
            pass
        paths[f"states_{year}"] = path

    # Per-year posterior rasters (K-band each: band k+1 = P(state=k))
    for year in years:
        path = post_dir / f"posteriors_{year}.tif"
        profile = {**base_profile, "dtype": "uint8", "count": n_states, "nodata": NODATA_UINT8}
        with rasterio.open(path, "w", **profile):
            pass
        paths[f"posteriors_{year}"] = path

    # Single-band transition year rasters
    for name in ["trace_onset_year", "sparse_year", "open_year", "woodland_year", "forest_year"]:
        path = output_dir / f"{name}.tif"
        profile = {**base_profile, "dtype": "int16", "count": 1, "nodata": NODATA_INT16}
        with rasterio.open(path, "w", **profile):
            pass
        paths[name] = path

    # Max state raster
    path = output_dir / "max_state.tif"
    profile = {**base_profile, "dtype": "uint8", "count": 1, "nodata": NODATA_UINT8}
    with rasterio.open(path, "w", **profile):
        pass
    paths["max_state"] = path

    log.info(f"Created {len(paths)} output rasters in {output_dir}")
    return paths


def open_output_handles(paths: dict[str, Path]) -> dict[str, rasterio.DatasetWriter]:
    """Open all output rasters for writing.

    Args:
        paths: Dict mapping product key to file path (from create_output_rasters).

    Returns:
        Dict mapping product key to open rasterio writer.
    """
    return {name: rasterio.open(path, "r+") for name, path in paths.items()}


def close_output_handles(handles: dict[str, rasterio.DatasetWriter]) -> None:
    """Close all output file handles."""
    for h in handles.values():
        h.close()


def write_chunk_results(
    handles: dict[str, rasterio.DatasetWriter],
    chunk: Chunk,
    states: np.ndarray,
    posteriors: np.ndarray,
    valid_indices: np.ndarray,
    years: list[int],
    n_states: int = 6,
) -> None:
    """Write inference results for a spatial chunk.

    Args:
        handles: Open output file handles (from open_output_handles).
        chunk: Spatial chunk specification.
        states: (N, T) int32 state labels.
        posteriors: (N, T, K) float32 posterior probabilities, or None.
        valid_indices: (N, 2) row/col indices within chunk.
        years: List of year values (length T).
        n_states: Number of states K.
    """
    N, T = states.shape
    H, W = chunk.height, chunk.width
    window = chunk.window

    # Per-year state and posterior writes
    state_grid = np.full((H, W), NODATA_UINT8, dtype=np.uint8)
    post_grid = np.full((H, W), NODATA_UINT8, dtype=np.uint8)

    for t, year in enumerate(years):
        # States: 1-band per year
        state_grid[:] = NODATA_UINT8
        state_grid[valid_indices[:, 0], valid_indices[:, 1]] = states[:, t].astype(np.uint8)
        handles[f"states_{year}"].write(state_grid, window=window, indexes=1)

        # Posteriors: K-band per year — write all bands in one call
        if posteriors is not None:
            key = f"posteriors_{year}"
            if key in handles:
                post_cube = np.full((n_states, H, W), NODATA_UINT8, dtype=np.uint8)
                p = np.clip(posteriors[:, t, :] * 200.0, 0, 200).astype(np.uint8)  # (N, K)
                post_cube[:, valid_indices[:, 0], valid_indices[:, 1]] = p.T  # (K, N)
                handles[key].write(post_cube, window=window)

    # Transition year products (single-band)
    _write_transition_years(handles, chunk, states, valid_indices, years)

    # Max state (single-band)
    max_grid = np.full((H, W), NODATA_UINT8, dtype=np.uint8)
    max_states = states.max(axis=1).astype(np.uint8)
    max_grid[valid_indices[:, 0], valid_indices[:, 1]] = max_states
    handles["max_state"].write(max_grid, window=window, indexes=1)


def _write_transition_years(
    handles: dict[str, rasterio.DatasetWriter],
    chunk: Chunk,
    states: np.ndarray,
    valid_indices: np.ndarray,
    years: list[int],
) -> None:
    """Compute and write transition year rasters."""
    H, W = chunk.height, chunk.width
    window = chunk.window
    years_arr = np.array(years, dtype=np.int16)

    thresholds = {
        "trace_onset_year": 1,   # First year state >= 1 (first tree establishment)
        "sparse_year": 2,        # First year state >= 2 (encroachment underway)
        "open_year": 3,          # First year state >= 3 (open canopy established)
        "woodland_year": 4,      # First year state >= 4 (woodland density reached)
        "forest_year": 5,        # First year state >= 5 (forest canopy closure)
    }

    for name, min_state in thresholds.items():
        grid = np.full((H, W), NODATA_INT16, dtype=np.int16)
        exceeds = states >= min_state   # (N, T) bool
        first_idx = np.argmax(exceeds, axis=1)  # (N,) — index of first True
        ever_exceeds = exceeds.any(axis=1)       # (N,) — mask out all-False rows
        transition_years = years_arr[first_idx]  # (N,)
        rows = valid_indices[ever_exceeds, 0]
        cols = valid_indices[ever_exceeds, 1]
        grid[rows, cols] = transition_years[ever_exceeds]
        handles[name].write(grid, window=window, indexes=1)
