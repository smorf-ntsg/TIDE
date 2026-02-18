"""Summary statistics computation."""

import json
import logging
from pathlib import Path

import numpy as np
import rasterio

log = logging.getLogger(__name__)


def compute_summary_statistics(
    output_dir: Path,
    years: list[int],
    n_states: int = 6,
) -> dict:
    """Compute summary statistics from output rasters.

    Memory-efficient: reads in chunks.

    Args:
        output_dir: Directory containing output rasters.
        years: List of years.
        n_states: Number of HMM states.

    Returns:
        Dict of summary statistics.
    """
    stats = {"per_year": {}, "transitions": {}, "overall": {}}

    states_path = output_dir / "states.tif"
    if not states_path.exists():
        log.warning("States raster not found, skipping statistics")
        return stats

    chunk_rows = 1024

    with rasterio.open(states_path) as src:
        height, width = src.height, src.width
        n_bands = src.count

        # Per-year state counts
        year_counts = {y: np.zeros(n_states + 1, dtype=np.int64) for y in years}

        for row_off in range(0, height, chunk_rows):
            h = min(chunk_rows, height - row_off)
            window = rasterio.windows.Window(0, row_off, width, h)

            for band_idx, year in enumerate(years[:n_bands], 1):
                data = src.read(band_idx, window=window)
                for s in range(n_states):
                    year_counts[year][s] += np.count_nonzero(data == s)
                year_counts[year][n_states] += np.count_nonzero(data == 255)

        for year in years[:n_bands]:
            counts = year_counts[year]
            n_valid = counts[:n_states].sum()
            stats["per_year"][str(year)] = {
                "state_counts": {f"state_{s}": int(counts[s]) for s in range(n_states)},
                "n_valid": int(n_valid),
                "n_nodata": int(counts[n_states]),
                "state_fractions": {
                    f"state_{s}": float(counts[s] / max(n_valid, 1))
                    for s in range(n_states)
                },
            }

    # Transition year statistics
    for name in ["trace_onset_year", "sparse_year", "open_year", "woodland_year", "forest_year"]:
        path = output_dir / f"{name}.tif"
        if path.exists():
            with rasterio.open(path) as src:
                data = src.read(1)
                valid = data[data != -9999]
                if valid.size > 0:
                    stats["transitions"][name] = {
                        "n_pixels": int(valid.size),
                        "mean_year": float(valid.mean()),
                        "median_year": float(np.median(valid)),
                        "min_year": int(valid.min()),
                        "max_year": int(valid.max()),
                    }

    # Save to JSON
    stats_path = output_dir / "summary_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Summary statistics saved to {stats_path}")

    return stats
