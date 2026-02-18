"""MTBS fire severity data loader.

Loads Monitoring Trends in Burn Severity data aligned to the RAP grid.
Returns ordinal severity pre-reclassified in GEE:
    0 = No Fire
    1 = Low Severity
    2 = Moderate Severity
    3 = High Severity
    4 = Competitive Reset
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

log = logging.getLogger(__name__)


def read_mtbs_window(
    mtbs_dir: Path,
    window: Window,
    years: list[int],
) -> np.ndarray:
    """Read MTBS fire severity for a spatial window across all years.

    Expected file naming: mtbs_severity_{year}.tif (aligned to RAP grid).
    Values are ordinal 0-4, pre-reclassified in GEE.

    Args:
        mtbs_dir: Directory containing aligned MTBS rasters.
        window: rasterio Window.
        years: List of years.

    Returns:
        (T, H, W) uint8 ordinal fire severity (0=no fire, 1-4=severity).
    """
    T = len(years)
    H, W = window.height, window.width
    severity = np.zeros((T, H, W), dtype=np.uint8)

    for t, year in enumerate(years):
        path = mtbs_dir / f"mtbs_severity_{year}.tif"
        if path.exists():
            with rasterio.open(path) as src:
                severity[t] = src.read(1, window=window)

    return severity
