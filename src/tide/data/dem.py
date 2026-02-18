"""DEM, CHILI, and Aridity Index terrain data loader.

Loads elevation, Continuous Heat-Insolation Load Index, and Aridity Index
aligned to the RAP grid. These are static covariates (single time slice,
broadcast over years).
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

log = logging.getLogger(__name__)


def _read_static_window(path: Path, window: Window) -> np.ndarray:
    """Read a static raster window, replacing nodata with NaN.

    Returns:
        (H, W) float32 array with nodata pixels set to NaN.
    """
    with rasterio.open(path) as src:
        data = src.read(1, window=window).astype(np.float32)
        if src.nodata is not None:
            data[data == np.float32(src.nodata)] = np.nan
    return data


def read_dem_window(
    dem_path: Path,
    window: Window,
) -> np.ndarray:
    """Read DEM elevation for a spatial window.

    Args:
        dem_path: Path to aligned DEM raster.
        window: rasterio Window.

    Returns:
        (H, W) float32 elevation in meters. Nodata pixels are NaN.
    """
    return _read_static_window(dem_path, window)


def read_chili_window(
    chili_path: Path,
    window: Window,
) -> np.ndarray:
    """Read CHILI (Continuous Heat-Insolation Load Index) for a spatial window.

    CSP ERGo CHILI captures topographic shading / insolation effects.

    Args:
        chili_path: Path to aligned CHILI raster.
        window: rasterio Window.

    Returns:
        (H, W) float32 CHILI values. Nodata pixels are NaN.
    """
    return _read_static_window(chili_path, window)


def read_aridity_window(
    aridity_path: Path,
    window: Window,
) -> np.ndarray:
    """Read Aridity Index (P/PET) for a spatial window.

    UNEP Aridity Index from TerraClimate or gridMET long-term mean.
    Lower values = more arid.

    Args:
        aridity_path: Path to aligned aridity raster.
        window: rasterio Window.

    Returns:
        (H, W) float32 aridity values. Nodata pixels are NaN.
    """
    return _read_static_window(aridity_path, window)
