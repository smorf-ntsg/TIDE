"""Windowed COG reading utilities."""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from tide.data.chunking import Chunk

log = logging.getLogger(__name__)


def read_chunk_data(
    input_paths: list[Path],
    mask_path: Path,
    chunk: Chunk,
    ref_transform=None,
    open_srcs: list | None = None,
    open_stack_src=None,
    open_mask_src=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read all data for a spatial chunk.

    Windows are in mask/reference-grid pixel coordinates. If tree cover
    rasters are on a different (but phase-aligned) grid, ref_transform
    is used to compute the pixel offset automatically.

    Args:
        input_paths: List of T tree cover GeoTIFF paths.
        mask_path: Path to analysis region mask.
        chunk: Spatial chunk specification (in reference-grid coordinates).
        ref_transform: Affine transform of the reference grid (mask).
        open_srcs: Optional list of T already-open rasterio Datasets for tree
            cover. Avoids ~23ms per-file open overhead (see read_rap_window).
            Must NOT be shared across threads.
        open_stack_src: Optional already-open rasterio Dataset for a 39-band
            stacked GeoTIFF. When provided, all bands are read in one call
            (~6× faster cold read). Takes priority over open_srcs.
        open_mask_src: Optional already-open rasterio Dataset for the mask.

    Returns:
        obs: (N, T) float32 observations in [0, 1] for valid pixels.
        cube_float: (T, H, W) float32 tree cover data (needed for seed distance).
        mask: (H, W) bool mask.
        valid_indices: (N, 2) int array of (row, col) indices for valid pixels.
    """
    from tide.data.rap import read_rap_window, rap_to_float
    from tide.data.mask import read_mask_window

    window = chunk.window

    # Read mask (on reference grid — window used directly)
    mask = read_mask_window(mask_path, window, open_src=open_mask_src)

    # Read tree cover (may be offset from reference grid)
    # open_stack_src (single 39-band file) takes priority over open_srcs (39 files)
    cube_raw = read_rap_window(
        input_paths, window, ref_transform=ref_transform,
        open_srcs=open_srcs, open_stack_src=open_stack_src,
    )
    cube_float = rap_to_float(cube_raw)  # (T, H, W) float32

    # Extract valid pixels
    valid_rows, valid_cols = np.where(mask)
    valid_indices = np.stack([valid_rows, valid_cols], axis=1)  # (N, 2)

    # Extract time series for valid pixels: (T, H, W) -> (N, T)
    obs = cube_float[:, valid_rows, valid_cols].T  # (N, T)

    return obs, cube_float, mask, valid_indices


def _find_stack_path(input_paths: list[Path]) -> Path | None:
    """Return path to pre-stacked 39-band GeoTIFF if it exists, else None.

    Looks for treeCover_stack.tif in the same directory as the per-year files.
    """
    if not input_paths:
        return None
    stack = input_paths[0].parent / "treeCover_stack.tif"
    return stack if stack.exists() else None


def open_input_handles(
    input_paths: list[Path],
    mask_path: Path,
) -> dict:
    """Open all input rasters and return persistent handles.

    Eliminates ~23ms per-file open overhead from read_chunk_data. Intended
    for use in the I/O prefetch thread — one set of handles per thread.

    Auto-detects a pre-stacked 39-band GeoTIFF (treeCover_stack.tif) in the
    same directory as the per-year files. If found, opens that single file
    instead of 39 separate files, enabling ~6× faster cold reads.

    Usage::

        handles = open_input_handles(input_paths, mask_path)
        try:
            read_chunk_data(..., open_srcs=handles["srcs"],
                            open_stack_src=handles["stack"],
                            open_mask_src=handles["mask"])
        finally:
            close_input_handles(handles)

    Returns:
        Dict with:
        - 'srcs': list of T open Datasets (None if stack mode)
        - 'stack': open Dataset for 39-band stack (None if per-file mode)
        - 'mask': open Dataset for mask
    """
    stack_path = _find_stack_path(input_paths)
    if stack_path is not None:
        log.info(f"Using pre-stacked treeCover file: {stack_path}")
        return {
            "srcs": None,
            "stack": rasterio.open(stack_path),
            "mask": rasterio.open(mask_path),
        }
    return {
        "srcs": [rasterio.open(p) for p in input_paths],
        "stack": None,
        "mask": rasterio.open(mask_path),
    }


def close_input_handles(handles: dict) -> None:
    """Close handles returned by open_input_handles."""
    if handles.get("srcs"):
        for src in handles["srcs"]:
            src.close()
    if handles.get("stack"):
        handles["stack"].close()
    handles["mask"].close()


def _translate_window(window: Window, ref_transform, target_path) -> Window:
    """Translate a window from reference-grid coordinates to a target raster's coordinates.

    When the reference (mask) is a spatial subset of the full grid, chunk windows
    are in the mask's local pixel space. This translates them to the target raster's
    pixel space by computing the geographic offset between the two grids.

    If both grids share the same origin, returns the window unchanged.
    """
    with rasterio.open(target_path) as src:
        target_transform = src.transform

    # Compute offset: how many pixels the reference origin is shifted from the target origin
    col_off = round((ref_transform.c - target_transform.c) / target_transform.a)
    row_off = round((ref_transform.f - target_transform.f) / target_transform.e)

    if col_off == 0 and row_off == 0:
        return window

    return Window(
        col_off=window.col_off + col_off,
        row_off=window.row_off + row_off,
        width=window.width,
        height=window.height,
    )


def read_covariate_data(
    config,
    chunk: Chunk,
    valid_rows: np.ndarray,
    valid_cols: np.ndarray,
    years: list[int],
    cube_float: np.ndarray,
    ref_transform=None,
) -> dict[str, np.ndarray | None]:
    """Read and extract covariate data for valid pixels in a chunk.

    Loads each available covariate source, extracts valid pixels, and aligns
    time-varying covariates to transitions (T timesteps -> T-1 transitions).

    Args:
        config: PipelineConfig with covariate paths.
        chunk: Spatial chunk specification.
        valid_rows: (N,) row indices of valid pixels.
        valid_cols: (N,) col indices of valid pixels.
        years: List of years.
        cube_float: (T, H, W) float32 RAP tree cover (for seed distance/focal mean).
        ref_transform: Affine transform of the reference grid (mask). Used to
            translate chunk windows when the mask is a spatial subset.

    Returns:
        Dict with keys: fire_severity, spei, elevation, chili, aridity,
        seed_distance, focal_mean. Values are (N, T-1) float32 for time-varying,
        (N,) float32 for static, or None if that covariate is unavailable.
    """
    window = chunk.window
    T = len(years)
    N = len(valid_rows)

    result = {
        "fire_severity": None,
        "spei": None,
        "elevation": None,
        "chili": None,
        "aridity": None,
        "seed_distance": None,
        "focal_mean": None,
    }

    # --- Time-varying covariates: (T, H, W) -> extract -> (N, T) -> [:, :-1] -> (N, T-1) ---

    # MTBS fire severity
    mtbs_dir = getattr(config, 'effective_mtbs_dir', config.mtbs_dir)
    spei_dir = getattr(config, 'effective_spei_dir', config.spei_dir)
    dem_path = getattr(config, 'effective_dem_path', config.dem_path)
    chili_path = getattr(config, 'effective_chili_path', config.chili_path)
    aridity_path = getattr(config, 'effective_aridity_path', config.aridity_path)

    # Helper to translate window when mask is a spatial subset of the covariate grid
    def _cov_window(cov_path):
        if ref_transform is None:
            return window
        return _translate_window(window, ref_transform, cov_path)

    if mtbs_dir is not None:
        from tide.data.mtbs import read_mtbs_window
        from pathlib import Path
        mtbs_sample = Path(mtbs_dir) / f"mtbs_severity_{years[0]}.tif"
        mtbs_win = _cov_window(mtbs_sample)
        severity = read_mtbs_window(mtbs_dir, mtbs_win, years)  # (T, H, W) uint8
        sev_pixels = severity[:, valid_rows, valid_cols].T.astype(np.float32)  # (N, T)
        result["fire_severity"] = sev_pixels[:, :-1]  # (N, T-1)

    # SPEI
    if spei_dir is not None:
        from tide.data.spei import read_spei_window
        from pathlib import Path
        spei_sample = Path(spei_dir) / f"spei_{years[0]}.tif"
        spei_win = _cov_window(spei_sample)
        spei = read_spei_window(spei_dir, spei_win, years)  # (T, H, W) float32
        spei_pixels = spei[:, valid_rows, valid_cols].T  # (N, T)
        result["spei"] = spei_pixels[:, :-1]  # (N, T-1)

    # --- Static covariates: (H, W) -> extract -> (N,) ---

    # Elevation
    if dem_path is not None:
        from tide.data.dem import read_dem_window
        dem_win = _cov_window(dem_path)
        dem = read_dem_window(dem_path, dem_win)  # (H, W) float32
        result["elevation"] = dem[valid_rows, valid_cols]  # (N,)

    # CHILI
    if chili_path is not None:
        from tide.data.dem import read_chili_window
        chili_win = _cov_window(chili_path)
        chili = read_chili_window(chili_path, chili_win)  # (H, W) float32
        result["chili"] = chili[valid_rows, valid_cols]  # (N,)

    # Aridity Index
    if aridity_path is not None:
        from tide.data.dem import read_aridity_window
        aridity_win = _cov_window(aridity_path)
        aridity = read_aridity_window(aridity_path, aridity_win)  # (H, W) float32
        result["aridity"] = aridity[valid_rows, valid_cols]  # (N,)

    # --- Derived from RAP (always available): seed distance and focal mean ---
    # These need spatial (H,W) arrays for scipy, so compute per-timestep then extract.
    from tide.data.seed_distance import compute_seed_distance, compute_focal_mean

    sd_cube = np.zeros((T, N), dtype=np.float32)
    fm_cube = np.zeros((T, N), dtype=np.float32)

    for t in range(T):
        tree_cover_t = cube_float[t]  # (H, W)
        sd_hw = compute_seed_distance(tree_cover_t)  # (H, W)
        fm_hw = compute_focal_mean(tree_cover_t)  # (H, W)
        sd_cube[t] = sd_hw[valid_rows, valid_cols]
        fm_cube[t] = fm_hw[valid_rows, valid_cols]

    result["seed_distance"] = sd_cube.T[:, :-1]  # (N, T-1)
    result["focal_mean"] = fm_cube.T[:, :-1]  # (N, T-1)

    return result
