"""Seed source proximity derivation.

Computes distance-to-nearest-tree-cover from RAP data.
Used as a covariate for encroachment probability.
"""

import logging

import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)


def compute_seed_distance(
    tree_cover: np.ndarray,
    threshold: float = 0.04,
    pixel_size: float = 30.0,
) -> np.ndarray:
    """Compute Euclidean distance to nearest tree cover pixel.

    Args:
        tree_cover: (H, W) float32 tree cover fraction [0, 1].
        threshold: Minimum cover to count as seed source.
        pixel_size: Pixel size in meters for distance scaling.

    Returns:
        (H, W) float32 distance in meters to nearest seed source.
    """
    seed_mask = tree_cover >= threshold
    if not seed_mask.any():
        # No seed sources: return large constant
        return np.full_like(tree_cover, 50000.0)

    # Distance transform from non-seed pixels to nearest seed pixel
    distance = ndimage.distance_transform_edt(~seed_mask) * pixel_size
    return distance.astype(np.float32)


def compute_focal_mean(
    tree_cover: np.ndarray,
    radius: int = 5,
) -> np.ndarray:
    """Compute focal (neighborhood) mean tree cover.

    Args:
        tree_cover: (H, W) float32 tree cover fraction.
        radius: Kernel radius in pixels.

    Returns:
        (H, W) float32 focal mean tree cover.
    """
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= kernel.sum()

    result = ndimage.convolve(tree_cover, kernel, mode="reflect")
    return result.astype(np.float32)
