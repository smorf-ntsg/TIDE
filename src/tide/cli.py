"""CLI entry point for TIDE pipeline."""

import os

# Prevent JAX from pre-allocating 75% of GPU memory.
# Allocate on demand instead, allowing other processes to share the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import click
from pathlib import Path


@click.group()
def main():
    """TIDE: Tree Inference for Dryland Ecosystems."""
    pass


@main.command()
@click.option("--data-dir", type=click.Path(exists=True), default="data",
              help="Root data directory containing treeCover/, mtbs/, spei/, terrain/, mask/.")
@click.option("--output-dir", type=click.Path(), default="output_v2",
              help="Output directory.")
@click.option("--n-states", type=int, default=6, help="Number of HMM states.")
@click.option("--chunk-size", type=int, default=512, help="Spatial tile dimension.")
@click.option("--batch-pixels", type=int, default=500_000, help="Max pixels per GPU batch.")
@click.option("--em-max-iter", type=int, default=100, help="Max EM iterations.")
@click.option("--em-sample", type=int, default=2_000_000, help="Pixels to sample for EM.")
@click.option("--skip-em", is_flag=True, help="Skip EM, load params from file.")
@click.option("--params-file", type=click.Path(), default=None,
              help="Path to save/load learned parameters.")
@click.option("--dask", is_flag=True, help="Use Dask distributed processing.")
@click.option("--workers", type=int, default=1, help="Number of Dask workers.")
@click.option("--no-posteriors", is_flag=True, help="Skip posterior computation.")
@click.option("--mtbs-dir", type=click.Path(), default=None,
              help="MTBS fire severity directory (default: data_dir/mtbs).")
@click.option("--spei-dir", type=click.Path(), default=None,
              help="SPEI directory (default: data_dir/spei).")
@click.option("--dem-path", type=click.Path(), default=None,
              help="DEM raster path (default: data_dir/terrain/dem.tif).")
@click.option("--chili-path", type=click.Path(), default=None,
              help="CHILI raster path (default: data_dir/terrain/chili.tif).")
@click.option("--aridity-path", type=click.Path(), default=None,
              help="Aridity Index raster path (default: data_dir/terrain/aridity.tif).")
@click.option("--dynamic-transitions", is_flag=True,
              help="Enable covariate-driven dynamic transitions.")
@click.option("--prechunked", is_flag=True,
              help="Read from pre-chunked .npy files instead of GeoTIFFs.")
@click.option("--chunks-dir", type=click.Path(), default=None,
              help="Directory containing pre-chunked .npy files (default: data_dir/chunks).")
def run(data_dir, output_dir, n_states, chunk_size, batch_pixels,
        em_max_iter, em_sample, skip_em, params_file, dask, workers,
        no_posteriors, mtbs_dir, spei_dir, dem_path, chili_path,
        aridity_path, dynamic_transitions, prechunked, chunks_dir):
    """Run the full TIDE pipeline."""
    from tide.config import (
        PipelineConfig, EmissionConfig, EMConfig, InferenceConfig,
    )
    from tide.pipeline import run_pipeline

    config = PipelineConfig(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        mtbs_dir=Path(mtbs_dir) if mtbs_dir else None,
        spei_dir=Path(spei_dir) if spei_dir else None,
        dem_path=Path(dem_path) if dem_path else None,
        chili_path=Path(chili_path) if chili_path else None,
        aridity_path=Path(aridity_path) if aridity_path else None,
        enable_dynamic_transitions=dynamic_transitions,
        emission=EmissionConfig(n_states=n_states),
        em=EMConfig(max_iter=em_max_iter, n_sample_pixels=em_sample),
        inference=InferenceConfig(
            chunk_size=chunk_size,
            batch_pixels=batch_pixels,
            compute_posteriors=not no_posteriors,
        ),
    )

    params_path = Path(params_file) if params_file else config.output_dir / "params.npz"
    chunks_path = Path(chunks_dir) if chunks_dir else config.data_dir / "chunks"

    run_pipeline(
        config=config,
        skip_em=skip_em,
        params_path=params_path,
        use_dask=dask,
        n_workers=workers,
        prechunked=prechunked,
        chunks_dir=chunks_path,
    )


@main.command()
@click.option("--data-dir", type=click.Path(exists=True), default="data",
              help="Root data directory.")
@click.option("--chunks-dir", type=click.Path(), default=None,
              help="Output directory for chunk files (default: data_dir/chunks).")
@click.option("--chunk-size", type=int, default=512, help="Spatial tile dimension.")
@click.option("--dynamic-transitions", is_flag=True,
              help="Also pre-chunk covariate data.")
@click.option("--threads", type=int, default=6, help="Number of I/O threads.")
def prechunk(data_dir, chunks_dir, chunk_size, dynamic_transitions, threads):
    """Pre-chunk input rasters to .npy files for fast inference."""
    from tide.config import PipelineConfig, InferenceConfig
    from tide.io.prechunk import prechunk_all

    config = PipelineConfig(
        data_dir=Path(data_dir),
        enable_dynamic_transitions=dynamic_transitions,
        inference=InferenceConfig(chunk_size=chunk_size),
    )
    chunks_path = Path(chunks_dir) if chunks_dir else config.data_dir / "chunks"

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    result = prechunk_all(
        config, chunks_path,
        n_threads=threads,
        dynamic_transitions=dynamic_transitions,
    )
    print(f"Pre-chunked {result['n_chunks']} chunks, {result['total_pixels']:,} pixels")


@main.command()
@click.option("--chunks-dir", type=click.Path(exists=True), required=True,
              help="Directory containing pre-chunked .npy files with results.")
@click.option("--output-dir", type=click.Path(), default="output_v2",
              help="Output directory for GeoTIFFs.")
@click.option("--data-dir", type=click.Path(exists=True), default="data",
              help="Root data directory (for reference profile).")
@click.option("--n-states", type=int, default=6, help="Number of HMM states.")
def assemble(chunks_dir, output_dir, data_dir, n_states):
    """Reassemble per-chunk .npy results into output GeoTIFFs."""
    from tide.config import PipelineConfig
    from tide.data.rap import get_reference_profile
    from tide.io.assemble import assemble_outputs

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = PipelineConfig(data_dir=Path(data_dir))
    ref_profile = get_reference_profile(config.mask_path)

    result = assemble_outputs(
        Path(chunks_dir), Path(output_dir), ref_profile,
        list(config.years), n_states,
    )
    print(f"Assembled {result['n_chunks']} chunks, {result['total_pixels']:,} pixels")


@main.command()
@click.option("--params-file", type=click.Path(exists=True), required=True,
              help="Path to learned parameters file.")
def inspect_params(params_file):
    """Inspect learned HMM parameters."""
    import numpy as np

    data = np.load(params_file)
    print("=== Learned TIDE Parameters ===\n")

    print("Initial state probabilities:")
    init = np.exp(data["log_init"])
    for i, p in enumerate(init):
        print(f"  State {i}: {p:.4f}")

    print("\nTransition matrix:")
    trans = np.exp(data["log_trans"])
    print(f"  {trans}")

    print("\nEmission parameters:")
    print(f"  pi  (zero-inflation): {data['emission_pi']}")
    print(f"  mu  (Beta mean):      {data['emission_mu']}")
    print(f"  phi (Beta precision):  {data['emission_phi']}")


@main.command()
@click.option("--output-dir", type=click.Path(exists=True), default="output_v2")
def stats(output_dir):
    """Compute summary statistics from output rasters."""
    from tide.io.summary import compute_summary_statistics
    from tide.config import PipelineConfig

    config = PipelineConfig()
    result = compute_summary_statistics(
        Path(output_dir), list(config.years), config.emission.n_states
    )
    print(f"Statistics saved to {output_dir}/summary_statistics.json")


if __name__ == "__main__":
    main()
