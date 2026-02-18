"""Grid reprojection and alignment utilities.

Ensures all covariate datasets are aligned to the RAP reference grid.
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform

log = logging.getLogger(__name__)


def align_to_reference(
    src_path: Path,
    ref_path: Path,
    dst_path: Path,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """Reproject a raster to match the reference grid exactly.

    Args:
        src_path: Input raster to reproject.
        ref_path: Reference raster defining target CRS, transform, shape.
        dst_path: Output path for aligned raster.
        resampling: Resampling method.

    Returns:
        dst_path for chaining.
    """
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )

    log.info(f"Aligned {src_path.name} -> {dst_path.name}")
    return dst_path


def verify_alignment(paths: list[Path], ref_path: Path) -> bool:
    """Verify that all rasters match the reference grid.

    Checks CRS, transform, and dimensions.

    Args:
        paths: List of raster paths to verify.
        ref_path: Reference raster.

    Returns:
        True if all match, False otherwise.
    """
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)

    for path in paths:
        with rasterio.open(path) as src:
            if src.crs != ref_crs:
                log.error(f"{path.name}: CRS mismatch ({src.crs} != {ref_crs})")
                return False
            if src.transform != ref_transform:
                log.error(f"{path.name}: Transform mismatch")
                return False
            if (src.height, src.width) != ref_shape:
                log.error(
                    f"{path.name}: Shape mismatch "
                    f"({src.height}, {src.width}) != {ref_shape}"
                )
                return False

    log.info(f"All {len(paths)} rasters aligned to reference")
    return True
