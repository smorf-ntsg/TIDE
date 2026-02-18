"""Stack annual tree cover GeoTIFFs into a single 39-band file.

One-time preprocessing step that replaces 39 × 1-band reads per chunk
with a single 39-band read, reducing cold-read time ~6× (5ms vs 32ms).

Usage
-----
    conda activate tide
    python scripts/stack_treecover.py --data-dir data/

Output
------
    data/treeCover/treeCover_stack.tif  (39 bands, uint8, tiled/deflate)

The pipeline auto-detects this file at startup. If absent, it falls back
to reading per-year files (no change in behavior required).

Size estimate
-------------
Same compressed size as the 39 separate files (~93GB total) because the
data is identical. Slight overhead from the larger file header.

Performance
-----------
Benchmarked on H100 system (NVMe):
  Baseline (39 reads):     32ms cold,  1.0ms warm
  Stacked (1 read):         5ms cold,  0.7ms warm
  Speedup:                  6.0×       1.4×
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def stack_treecover(data_dir: Path, years: list[int], dry_run: bool = False) -> Path:
    """Stack all per-year treeCover files into a single 39-band GeoTIFF.

    Args:
        data_dir: Root data directory containing treeCover/.
        years: List of years to include (must match filenames).
        dry_run: If True, log what would be done but don't write.

    Returns:
        Path to the output stacked file.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    tc_dir = data_dir / "treeCover"
    output_path = tc_dir / "treeCover_stack.tif"

    # Verify all input files exist
    paths = []
    for y in years:
        p = tc_dir / f"treeCover_{y}.tif"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        paths.append(p)

    log.info(f"Found {len(paths)} files, years {years[0]}-{years[-1]}")

    # Read profile from first file
    with rasterio.open(paths[0]) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        transform = src.transform

    # Check all files have the same grid (warn if not)
    for p in paths[1:]:
        with rasterio.open(p) as src:
            if (src.height, src.width) != (height, width):
                log.warning(
                    f"{p.name}: shape {src.height}×{src.width} != "
                    f"{height}×{width} — using largest dimensions"
                )
                height = max(height, src.height)
                width = max(width, src.width)

    T = len(paths)
    log.info(f"Output: {height}×{width}, {T} bands, uint8")
    log.info(f"Output path: {output_path}")

    if dry_run:
        log.info("Dry run — not writing.")
        return output_path

    profile.update(
        count=T,
        dtype="uint8",
        driver="GTiff",
        compress="deflate",
        tiled=True,
        blockxsize=512,   # Match inference chunk size: 1 tile = 1 chunk → no cross-tile reads
        blockysize=512,
        BIGTIFF="YES",
        interleave="band",
    )

    tile_rows = 256  # Process one tile row at a time to limit memory
    n_tile_rows = (height + tile_rows - 1) // tile_rows
    t0 = time.perf_counter()

    # Keep all source files open for the duration
    srcs = [rasterio.open(p) for p in paths]

    try:
        with rasterio.open(output_path, "w", **profile) as dst:
            for row_idx in range(n_tile_rows):
                row_off = row_idx * tile_rows
                row_end = min(height, row_off + tile_rows)
                h = row_end - row_off
                window = Window(0, row_off, width, h)

                # Read all bands for this row strip
                strip = np.zeros((T, h, width), dtype=np.uint8)
                for t, src in enumerate(srcs):
                    src_h = min(h, src.height - row_off)
                    src_w = min(width, src.width)
                    if src_h <= 0 or src_w <= 0:
                        continue
                    src_window = Window(0, row_off, src_w, src_h)
                    strip[t, :src_h, :src_w] = src.read(1, window=src_window)

                # Write all bands for this strip
                for t in range(T):
                    dst.write(strip[t], indexes=t + 1, window=window)

                if (row_idx + 1) % 100 == 0 or row_idx == n_tile_rows - 1:
                    elapsed = time.perf_counter() - t0
                    pct = (row_idx + 1) / n_tile_rows * 100
                    log.info(
                        f"Row strip {row_idx + 1}/{n_tile_rows} ({pct:.0f}%) "
                        f"— {elapsed:.0f}s elapsed"
                    )
    finally:
        for src in srcs:
            src.close()

    elapsed = time.perf_counter() - t0
    size_gb = output_path.stat().st_size / 1e9
    log.info(f"Done: {output_path} ({size_gb:.1f} GB) in {elapsed:.0f}s")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Root data directory (default: data/)")
    parser.add_argument("--start-year", type=int, default=1986)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing")
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    output = stack_treecover(args.data_dir, years, dry_run=args.dry_run)
    log.info(f"Stack ready: {output}")


if __name__ == "__main__":
    main()
