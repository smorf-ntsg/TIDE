"""Tree cover GeoTIFF reader.

Reads annual tree cover rasters (uint8, 0-100%) and converts
to float32 [0, 1] for the ZIB emission model.

Handles grid offsets: tree cover rasters may be on a different
(but phase-aligned) grid than the mask/covariate reference grid.
Windows are specified in reference-grid coordinates and automatically
translated to tree cover pixel coordinates.
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

log = logging.getLogger(__name__)

# Cached grid offset (computed once on first read)
_grid_offset: dict[str, tuple[int, int]] | None = None


def _compute_grid_offset(tc_path: Path, ref_transform) -> tuple[int, int]:
    """Compute pixel offset to convert reference-grid coordinates to tree cover coordinates.

    Both grids must share the same pixel size and be phase-aligned.

    Returns:
        (col_offset, row_offset): subtract these from reference-grid pixel
        coordinates to get tree cover pixel coordinates.
        E.g. if treeCover origin is 853 cols right of reference origin,
        reference col 853 maps to treeCover col 0, so col_offset = 853.
        tc_col = ref_col - col_offset.
    """
    global _grid_offset
    cache_key = str(tc_path)

    if _grid_offset is not None and cache_key in _grid_offset:
        return _grid_offset[cache_key]

    with rasterio.open(tc_path) as src:
        tc_transform = src.transform

    # How many pixels the TC origin is from the ref origin
    # TC origin is to the right (+col) and below (+row) of ref origin
    col_off = round((tc_transform.c - ref_transform.c) / ref_transform.a)
    row_off = round((tc_transform.f - ref_transform.f) / ref_transform.e)

    if _grid_offset is None:
        _grid_offset = {}
    _grid_offset[cache_key] = (col_off, row_off)

    if col_off != 0 or row_off != 0:
        log.info(f"Tree cover grid offset from reference: col={col_off}, row={row_off}")

    return col_off, row_off


def read_rap_window(
    paths: list[Path],
    window: Window,
    ref_transform=None,
    open_srcs: list | None = None,
    open_stack_src=None,
) -> np.ndarray:
    """Read a spatial window from all annual tree cover rasters.

    If ref_transform is provided, the window is in reference-grid pixel
    coordinates and will be translated to tree cover coordinates. If None,
    the window is used directly (assumes grids match).

    Args:
        paths: List of T GeoTIFF paths (one per year).
        window: rasterio Window in reference-grid coordinates.
        ref_transform: Affine transform of the reference grid (mask/DEM).
        open_srcs: Optional list of T already-open rasterio Dataset objects.
            When provided, file opens are skipped entirely (~23ms saved per
            file). The caller owns the handles and must close them. The handles
            must NOT be shared across threads; use one set per thread.
        open_stack_src: Optional already-open rasterio Dataset for a
            pre-stacked 39-band GeoTIFF. When provided, all T bands are read
            with a single src.read() call (~6× faster cold read). Takes
            priority over open_srcs when both are provided.

    Returns:
        (T, H, W) uint8 array of raw tree cover percentages.
    """
    T = len(paths)
    H = window.height
    W = window.width

    if ref_transform is not None:
        # _compute_grid_offset caches its result after the first call,
        # so the file open only happens once regardless of open_srcs.
        col_off, row_off = _compute_grid_offset(paths[0], ref_transform)
    else:
        col_off, row_off = 0, 0

    # Translate window from reference-grid to tree cover pixel space
    tc_col = window.col_off - col_off
    tc_row = window.row_off - row_off

    cube = np.zeros((T, H, W), dtype=np.uint8)

    # --- Fast path: single read() call for all 39 bands ---
    if open_stack_src is not None:
        src = open_stack_src
        tc_height, tc_width = src.height, src.width

        read_col = max(0, tc_col)
        read_row = max(0, tc_row)
        read_col_end = min(tc_width, tc_col + W)
        read_row_end = min(tc_height, tc_row + H)

        if read_col < read_col_end and read_row < read_row_end:
            read_w = read_col_end - read_col
            read_h = read_row_end - read_row
            tc_window = Window(read_col, read_row, read_w, read_h)
            out_col = read_col - tc_col
            out_row = read_row - tc_row
            # Read all T bands at once — the key speedup
            bands = list(range(1, T + 1))
            cube[:, out_row:out_row + read_h, out_col:out_col + read_w] = \
                src.read(bands, window=tc_window)
        return cube

    # --- Fallback: per-file reads (with or without persistent handles) ---
    def _read_band(t, src):
        tc_height, tc_width = src.height, src.width

        # Clamp to tree cover bounds
        read_col = max(0, tc_col)
        read_row = max(0, tc_row)
        read_col_end = min(tc_width, tc_col + W)
        read_row_end = min(tc_height, tc_row + H)

        if read_col >= read_col_end or read_row >= read_row_end:
            return  # Window entirely outside tree cover extent

        read_w = read_col_end - read_col
        read_h = read_row_end - read_row
        tc_window = Window(read_col, read_row, read_w, read_h)

        # Where to place in output (handles negative tc_col/tc_row)
        out_col = read_col - tc_col
        out_row = read_row - tc_row

        cube[t, out_row:out_row + read_h, out_col:out_col + read_w] = \
            src.read(1, window=tc_window)

    if open_srcs is not None:
        for t, src in enumerate(open_srcs):
            _read_band(t, src)
    else:
        for t, path in enumerate(paths):
            with rasterio.open(path) as src:
                _read_band(t, src)

    return cube


def rap_to_float(cube: np.ndarray) -> np.ndarray:
    """Convert RAP uint8 [0-100] to float32 [0, 1].

    - Exact zeros preserved (structural zeros for ZIB)
    - Values of 100 clamped to 0.999 (Beta PDF boundary)

    Args:
        cube: (...) uint8 array with values 0-100.

    Returns:
        (...) float32 array in [0, 1].
    """
    result = cube.astype(np.float32) / 100.0
    # Clamp upper boundary for Beta stability
    result = np.clip(result, 0.0, 0.999)
    return result


def get_reference_profile(path: Path) -> dict:
    """Read georeferencing profile from a RAP raster.

    Args:
        path: Path to any RAP GeoTIFF.

    Returns:
        Dict with crs, transform, width, height.
    """
    with rasterio.open(path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
        }
