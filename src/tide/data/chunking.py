"""2D spatial chunk generation for tiled processing.

Generates non-overlapping 2D tiles instead of horizontal strips,
providing more uniform memory per tile and better GPU utilization.
"""

from dataclasses import dataclass

import numpy as np
from rasterio.windows import Window


@dataclass(frozen=True)
class Chunk:
    """A 2D spatial tile."""
    row_off: int
    col_off: int
    height: int
    width: int
    chunk_id: int

    @property
    def window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)

    @property
    def n_pixels(self) -> int:
        return self.height * self.width


def generate_chunks(
    total_height: int,
    total_width: int,
    chunk_size: int = 512,
) -> list[Chunk]:
    """Generate 2D tile chunks covering the full raster.

    Args:
        total_height: Raster height in pixels.
        total_width: Raster width in pixels.
        chunk_size: Tile dimension (both height and width).

    Returns:
        List of Chunk objects.
    """
    chunks = []
    chunk_id = 0

    for row_off in range(0, total_height, chunk_size):
        height = min(chunk_size, total_height - row_off)
        for col_off in range(0, total_width, chunk_size):
            width = min(chunk_size, total_width - col_off)
            chunks.append(Chunk(
                row_off=row_off,
                col_off=col_off,
                height=height,
                width=width,
                chunk_id=chunk_id,
            ))
            chunk_id += 1

    return chunks


def filter_masked_chunks(
    chunks: list[Chunk],
    mask_path,
    min_valid_fraction: float = 0.01,
) -> list[Chunk]:
    """Filter out chunks that are fully or nearly fully masked.

    Args:
        chunks: List of chunks to filter.
        mask_path: Path to analysis region mask raster.
        min_valid_fraction: Minimum fraction of valid pixels to keep chunk.

    Returns:
        Filtered list of chunks with sufficient valid pixels.
    """
    import rasterio

    valid_chunks = []
    with rasterio.open(mask_path) as src:
        for chunk in chunks:
            mask_data = src.read(1, window=chunk.window)
            valid_frac = np.count_nonzero(mask_data) / mask_data.size
            if valid_frac >= min_valid_fraction:
                valid_chunks.append(chunk)

    return valid_chunks
