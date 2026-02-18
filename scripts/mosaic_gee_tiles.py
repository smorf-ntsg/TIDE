#!/usr/bin/env python3
"""Mosaic GEE-exported tiles into single-file rasters per product/year.

Standalone preprocessing script — not part of the ZIB-HMM pipeline.
Uses gdalbuildvrt + gdal_translate for memory-efficient, parallelizable
mosaicking. Each mosaic job uses ~constant memory regardless of output size.

Usage:
    python scripts/mosaic_gee_tiles.py --base-dir /local-scratch/smorf/sb_treecover
    python scripts/mosaic_gee_tiles.py --products dem,chili --verbose
    python scripts/mosaic_gee_tiles.py --products mtbs --force --workers 32
"""

import logging
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
import rasterio

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Product definitions
# ---------------------------------------------------------------------------

@dataclass
class ProductConfig:
    """Configuration for a single data product."""
    name: str
    raw_dir: str           # Subdirectory under base_dir with raw tiles
    output_dir: str        # Subdirectory under base_dir/data for outputs
    tile_glob: str         # Glob pattern ({year} placeholder for yearly products)
    output_name: str       # Output filename ({year} placeholder for yearly products)
    dtype: str
    nodata: int | float
    expected_tiles: int    # Expected tile count per unit (for completeness check)
    years: list[int] = field(default_factory=list)  # Empty = static product
    zero_fill_years: list[int] = field(default_factory=list)


PRODUCTS: dict[str, ProductConfig] = {
    "mtbs": ProductConfig(
        name="mtbs",
        raw_dir="mtbs",
        output_dir="data/mtbs",
        tile_glob="mtbs_severity_{year}*.tif",
        output_name="mtbs_severity_{year}.tif",
        dtype="uint8",
        nodata=0,
        expected_tiles=4,
        years=list(range(1986, 2023)),
        zero_fill_years=[2023, 2024],
    ),
    "spei": ProductConfig(
        name="spei",
        raw_dir="spei",
        output_dir="data/spei",
        tile_glob="spei_{year}*.tif",
        output_name="spei_{year}.tif",
        dtype="float32",
        nodata=-9999.0,
        expected_tiles=12,
        years=list(range(1986, 2025)),
    ),
    "spei1y": ProductConfig(
        name="spei1y",
        raw_dir="spei1y",
        output_dir="data/spei",
        tile_glob="spei_{year}*.tif",
        output_name="spei_{year}.tif",
        dtype="float32",
        nodata=-9999.0,
        expected_tiles=12,
        years=list(range(1986, 2025)),
    ),
    "aridity": ProductConfig(
        name="aridity",
        raw_dir="terrain",
        output_dir="data/terrain",
        tile_glob="aridity*.tif",
        output_name="aridity.tif",
        dtype="float32",
        nodata=-9999.0,
        expected_tiles=30,
    ),
    "dem": ProductConfig(
        name="dem",
        raw_dir="terrain",
        output_dir="data/terrain",
        tile_glob="dem*.tif",
        output_name="dem.tif",
        dtype="float32",
        nodata=-9999.0,
        expected_tiles=12,
    ),
    "chili": ProductConfig(
        name="chili",
        raw_dir="terrain",
        output_dir="data/terrain",
        tile_glob="chili*.tif",
        output_name="chili.tif",
        dtype="float32",
        nodata=-9999.0,
        expected_tiles=12,
    ),
    "treeCover": ProductConfig(
        name="treeCover",
        raw_dir="treeCover",
        output_dir="data/treeCover",
        tile_glob="treeCover_{year}*.tif",
        output_name="treeCover_{year}.tif",
        dtype="uint8",
        nodata=255,
        expected_tiles=4,
        years=list(range(1986, 2025)),
    ),
    "mask": ProductConfig(
        name="mask",
        raw_dir="mask_drylands",
        output_dir="data/mask",
        tile_glob="dryland_nlcd_mask*.tif",
        output_name="drylands_mask.tif",
        dtype="uint8",
        nodata=255,
        expected_tiles=4,
    ),
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def discover_tiles(input_dir: Path, glob_pattern: str, year: int | None = None) -> list[Path]:
    """Find tile files matching a glob pattern."""
    if year is not None:
        pattern = glob_pattern.replace("{year}", str(year))
    else:
        pattern = glob_pattern.replace("{year}", "")
    return sorted(input_dir.glob(pattern))


def mosaic_tiles(tiles: list[Path], output_path: Path, dtype: str, nodata) -> str:
    """Mosaic tiles via gdalbuildvrt + gdal_translate to COG.

    Memory usage is bounded by GDAL's block cache (~512MB default),
    independent of output raster size.

    Returns:
        Status message string.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as tmp:
        vrt_path = tmp.name

    try:
        # Build VRT (instant, no data read)
        vrt_cmd = [
            "gdalbuildvrt",
            "-overwrite",
            vrt_path,
        ] + [str(t) for t in tiles]

        result = subprocess.run(vrt_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"FAIL gdalbuildvrt: {result.stderr.strip()}"

        # Translate VRT to COG with DEFLATE compression
        translate_cmd = [
            "gdal_translate",
            "-of", "GTiff",
            "-co", "COMPRESS=DEFLATE",
            "-co", "TILED=YES",
            "-co", "BLOCKXSIZE=256",
            "-co", "BLOCKYSIZE=256",
            "-co", "BIGTIFF=YES",
            "-ot", dtype.upper().replace("FLOAT32", "Float32").replace("UINT8", "Byte"),
            "-a_nodata", str(nodata),
            vrt_path,
            str(output_path),
        ]

        result = subprocess.run(translate_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"FAIL gdal_translate: {result.stderr.strip()}"

    finally:
        Path(vrt_path).unlink(missing_ok=True)

    return f"OK {output_path.name}"


def _dtype_to_gdal(dtype: str) -> str:
    """Convert numpy/python dtype string to GDAL type string."""
    mapping = {
        "uint8": "Byte",
        "int16": "Int16",
        "uint16": "UInt16",
        "int32": "Int32",
        "uint32": "UInt32",
        "float32": "Float32",
        "float64": "Float64",
    }
    return mapping.get(dtype, dtype)


def create_zero_raster(reference_path: Path, output_path: Path, dtype: str, nodata) -> str:
    """Create a zero-filled raster matching a reference grid.

    Used for MTBS gap years (2023-2024) where no fire data exists.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdal_dtype = _dtype_to_gdal(dtype)

    # Use gdal_calc to create a zero raster from the reference
    cmd = [
        "gdal_create",
        "-of", "GTiff",
        "-co", "COMPRESS=DEFLATE",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=256",
        "-co", "BLOCKYSIZE=256",
        "-co", "BIGTIFF=YES",
        "-ot", gdal_dtype,
        "-if", str(reference_path),
        "-burn", "0",
        "-a_nodata", str(nodata),
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback: gdal_create may not be available in older GDAL
        # Use rasterio with windowed writes for bounded memory
        return _create_zero_raster_rasterio(reference_path, output_path, dtype, nodata)

    return f"OK zero-fill {output_path.name}"


def _create_zero_raster_rasterio(
    reference_path: Path, output_path: Path, dtype: str, nodata,
) -> str:
    """Fallback zero raster creation using rasterio windowed writes."""
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()

    profile.update(
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="YES",
    )

    block_h, block_w = 256, 256
    height, width = profile["height"], profile["width"]

    with rasterio.open(output_path, "w", **profile) as dst:
        for row_off in range(0, height, block_h):
            h = min(block_h, height - row_off)
            for col_off in range(0, width, block_w):
                w = min(block_w, width - col_off)
                window = rasterio.windows.Window(col_off, row_off, w, h)
                zeros = np.zeros((h, w), dtype=dtype)
                for band_idx in range(1, profile["count"] + 1):
                    dst.write(zeros, band_idx, window=window)

    return f"OK zero-fill {output_path.name}"


def _mosaic_one(args: tuple) -> str:
    """Worker function for parallel mosaic. Takes a tuple for ProcessPoolExecutor."""
    tiles, output_path, dtype, nodata, label = args
    try:
        status = mosaic_tiles(tiles, output_path, dtype, nodata)
        return f"[{label}] {status}"
    except Exception as e:
        return f"[{label}] FAIL: {e}"


def _zero_fill_one(args: tuple) -> str:
    """Worker function for parallel zero-fill."""
    reference_path, output_path, dtype, nodata, label = args
    try:
        status = create_zero_raster(reference_path, output_path, dtype, nodata)
        return f"[{label}] {status}"
    except Exception as e:
        return f"[{label}] FAIL: {e}"


def build_work_items(
    base_dir: Path, product: ProductConfig, force: bool,
) -> tuple[list[tuple], list[tuple]]:
    """Build lists of mosaic and zero-fill work items for a product.

    Returns:
        (mosaic_items, zero_fill_items) — each item is an args tuple for the worker.
    """
    input_dir = base_dir / product.raw_dir
    output_dir_path = base_dir / product.output_dir

    if not input_dir.exists():
        log.warning(f"[{product.name}] Input directory not found: {input_dir} — skipping")
        return [], []

    mosaic_items = []
    zero_fill_items = []

    if product.years:
        last_real_output = None
        for year in product.years:
            output_name = product.output_name.replace("{year}", str(year))
            output_path = output_dir_path / output_name
            label = f"{product.name}/{year}"

            if output_path.exists() and not force:
                log.debug(f"  {output_name} exists, skipping")
                last_real_output = output_path
                continue

            tiles = discover_tiles(input_dir, product.tile_glob, year=year)
            if not tiles:
                log.warning(f"  {year}: no tiles found — skipping")
                continue

            if len(tiles) < product.expected_tiles:
                log.warning(
                    f"  {year}: found {len(tiles)}/{product.expected_tiles} tiles — "
                    f"skipping incomplete set"
                )
                continue

            mosaic_items.append((tiles, output_path, product.dtype, product.nodata, label))
            last_real_output = output_path

        for year in product.zero_fill_years:
            output_name = product.output_name.replace("{year}", str(year))
            output_path = output_dir_path / output_name
            label = f"{product.name}/{year}-zerofill"

            if output_path.exists() and not force:
                log.debug(f"  {output_name} (zero-fill) exists, skipping")
                continue

            if last_real_output is None:
                log.warning(f"  {year}: cannot zero-fill — no reference raster")
                continue

            zero_fill_items.append(
                (last_real_output, output_path, product.dtype, product.nodata, label)
            )
    else:
        output_path = output_dir_path / product.output_name
        label = product.name

        if output_path.exists() and not force:
            log.info(f"  {product.output_name} exists, skipping")
            return [], []

        tiles = discover_tiles(input_dir, product.tile_glob)
        if not tiles:
            log.warning(f"  [{product.name}] No tiles found — skipping")
            return [], []

        if len(tiles) < product.expected_tiles:
            log.warning(
                f"  [{product.name}] Found {len(tiles)}/{product.expected_tiles} tiles — "
                f"skipping incomplete set"
            )
            return [], []

        mosaic_items.append((tiles, output_path, product.dtype, product.nodata, label))

    return mosaic_items, zero_fill_items


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--base-dir", type=click.Path(exists=True), default=".",
              help="Project root containing raw tile subdirs and data/ output.")
@click.option("--products", type=str, default="all",
              help="Comma-separated product names or 'all'.")
@click.option("--force", is_flag=True, help="Re-mosaic even if output exists.")
@click.option("--workers", type=int, default=8,
              help="Number of parallel mosaic processes.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(base_dir, products, force, workers, verbose):
    """Mosaic GEE-exported tiles into single-file rasters."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    base = Path(base_dir)

    if products == "all":
        product_names = list(PRODUCTS.keys())
    else:
        product_names = [p.strip() for p in products.split(",")]

    # Collect all work items across products
    all_mosaic = []
    all_zero_fill = []

    for name in product_names:
        if name not in PRODUCTS:
            log.error(f"Unknown product: {name}. Available: {list(PRODUCTS.keys())}")
            continue
        m, z = build_work_items(base, PRODUCTS[name], force)
        all_mosaic.extend(m)
        all_zero_fill.extend(z)

    log.info(f"Mosaic jobs: {len(all_mosaic)}, zero-fill jobs: {len(all_zero_fill)}")

    # Phase 1: Run all mosaics in parallel
    if all_mosaic:
        log.info(f"Starting {len(all_mosaic)} mosaic jobs with {workers} workers")
        failed = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_mosaic_one, item): item for item in all_mosaic}
            for future in as_completed(futures):
                result = future.result()
                if "FAIL" in result:
                    log.error(result)
                    failed += 1
                else:
                    log.info(result)
        if failed:
            log.error(f"{failed} mosaic jobs failed")

    # Phase 2: Zero-fill (after mosaics so reference files exist)
    if all_zero_fill:
        log.info(f"Starting {len(all_zero_fill)} zero-fill jobs with {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_zero_fill_one, item): item for item in all_zero_fill}
            for future in as_completed(futures):
                result = future.result()
                if "FAIL" in result:
                    log.error(result)
                else:
                    log.info(result)

    log.info("Done.")


if __name__ == "__main__":
    main()
