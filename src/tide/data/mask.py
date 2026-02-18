"""Analysis region mask loader."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def read_mask_window(
    mask_path: Path,
    window: Window,
    open_src=None,
) -> np.ndarray:
    """Read a spatial window from the analysis region mask raster.

    Args:
        mask_path: Path to mask GeoTIFF.
        window: rasterio Window specifying spatial extent.
        open_src: Optional already-open rasterio Dataset. When provided,
            skips the file open (~23ms overhead). Caller owns the handle.

    Returns:
        (H, W) boolean array (True = valid pixel in analysis region).
    """
    if open_src is not None:
        return open_src.read(1, window=window) > 0
    with rasterio.open(mask_path) as src:
        return src.read(1, window=window) > 0
