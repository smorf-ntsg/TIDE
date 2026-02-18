"""SPEI drought index data loader.

Loads Standardized Precipitation-Evapotranspiration Index data
aligned to the RAP grid.
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

log = logging.getLogger(__name__)


def read_spei_window(
    spei_dir: Path,
    window: Window,
    years: list[int],
) -> np.ndarray:
    """Read SPEI data for a spatial window across all years.

    Expected file naming: spei_{year}.tif (aligned to RAP grid).

    Args:
        spei_dir: Directory containing aligned SPEI rasters.
        window: rasterio Window.
        years: List of years.

    Returns:
        (T, H, W) float32 SPEI values.
    """
    T = len(years)
    H, W = window.height, window.width
    spei = np.zeros((T, H, W), dtype=np.float32)

    for t, year in enumerate(years):
        path = spei_dir / f"spei_{year}.tif"
        if path.exists():
            with rasterio.open(path) as src:
                spei[t] = src.read(1, window=window).astype(np.float32)

    return spei
