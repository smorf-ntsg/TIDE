"""Top-level pipeline orchestration.

Two-phase execution:
  Phase A: Baum-Welch EM on stratified sample -> learned parameters
  Phase B: Full inference with frozen parameters -> output products
"""

import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tide.config import PipelineConfig
from tide.types import HMMParams
from tide.data.rap import get_reference_profile
from tide.data.chunking import generate_chunks, filter_masked_chunks
from tide.io.reader import read_chunk_data, read_covariate_data
from tide.io.summary import compute_summary_statistics
from tide.hmm.baum_welch import baum_welch
from tide.transitions.fit_weights import fit_transition_weights
from tide.distributed.orchestrator import run_full_inference

log = logging.getLogger(__name__)


def validate_inputs(config: PipelineConfig) -> None:
    """Validate that all required input files exist."""
    if not config.mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {config.mask_path}")

    for path in config.input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input raster not found: {path}")

    if config.enable_dynamic_transitions:
        for label, path in [
            ("MTBS directory", config.effective_mtbs_dir),
            ("SPEI directory", config.effective_spei_dir),
            ("DEM raster", config.effective_dem_path),
            ("CHILI raster", config.effective_chili_path),
            ("Aridity raster", config.effective_aridity_path),
        ]:
            if path is not None and not Path(path).exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        has_external = any(p is not None for p in [
            config.effective_mtbs_dir, config.effective_spei_dir,
            config.effective_dem_path, config.effective_chili_path,
        ])
        if not has_external:
            log.warning(
                "Dynamic transitions enabled but no external covariate paths set. "
                "Only RAP-derived covariates (seed_distance, focal_mean) will be used."
            )

    log.info(f"Validated {len(config.input_paths)} input rasters + mask")


def _numpy_stack_covariates(cov_dict: dict[str, np.ndarray | None]) -> np.ndarray:
    """Stack covariate arrays into (N, T-1, D) tensor using numpy.

    Same logic as transitions.covariates.stack_covariates but avoids JAX
    overhead during the I/O-bound sampling loop.

    Feature order: intercept, fire_severity, spei, elevation, chili, aridity,
                   seed_distance, focal_mean.
    """
    features = []
    N, Tm1 = None, None

    for key in ["fire_severity", "spei", "seed_distance", "focal_mean"]:
        arr = cov_dict.get(key)
        if arr is not None and arr.ndim == 2:
            N, Tm1 = arr.shape
            break

    if N is None:
        for key in ["elevation", "chili", "aridity"]:
            arr = cov_dict.get(key)
            if arr is not None:
                N = arr.shape[0]
                Tm1 = 38
                break

    if N is None:
        raise ValueError("At least one covariate must be provided")

    features.append(np.ones((N, Tm1), dtype=np.float32))

    for key in ["fire_severity", "spei", "elevation", "chili", "aridity",
                "seed_distance", "focal_mean"]:
        arr = cov_dict.get(key)
        if arr is None:
            continue
        if arr.ndim == 1:
            features.append(np.broadcast_to(arr[:, None], (N, Tm1)).copy())
        else:
            features.append(arr.astype(np.float32))

    result = np.stack(features, axis=-1)
    # Replace NaN with 0.0 (nodata covariates contribute zero to logistic regression)
    np.nan_to_num(result, copy=False, nan=0.0)
    return result


def sample_pixels_for_em(
    config: PipelineConfig,
    n_sample: int | None = None,
    return_covariates: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Draw a stratified random sample of pixels for EM training.

    Samples from multiple spatial chunks to ensure geographic diversity.

    Args:
        config: Pipeline configuration.
        n_sample: Number of pixels to sample (default from config).
        return_covariates: Also sample and return covariates.

    Returns:
        Tuple of (obs, covariates):
          obs: (N, T) float32 array of sampled pixel time series.
          covariates: (N, T-1, D) float32 array or None if not requested.
    """
    if n_sample is None:
        n_sample = config.em.n_sample_pixels

    ref = get_reference_profile(config.mask_path)
    chunks = generate_chunks(ref["height"], ref["width"], config.inference.chunk_size)
    chunks = filter_masked_chunks(chunks, config.mask_path)

    # Sample proportionally from chunks
    rng = np.random.default_rng(42)
    n_chunks_to_sample = min(200, len(chunks))
    sampled_chunks = rng.choice(chunks, n_chunks_to_sample, replace=False)

    pixels_per_chunk = n_sample // n_chunks_to_sample + 1
    all_obs = []
    all_covs = [] if return_covariates else None

    for chunk in sampled_chunks:
        obs, cube_float, mask, valid_indices = read_chunk_data(
            config.input_paths, config.mask_path, chunk,
            ref_transform=ref["transform"],
        )
        if obs.shape[0] == 0:
            continue

        # Random subsample within chunk
        n_take = min(pixels_per_chunk, obs.shape[0])
        idx = rng.choice(obs.shape[0], n_take, replace=False)
        all_obs.append(obs[idx])

        if return_covariates:
            valid_rows = valid_indices[:, 0]
            valid_cols = valid_indices[:, 1]
            cov_dict = read_covariate_data(
                config, chunk, valid_rows, valid_cols,
                list(config.years), cube_float,
                ref_transform=ref["transform"],
            )
            cov_stacked = _numpy_stack_covariates(cov_dict)  # (N_chunk, T-1, D)
            all_covs.append(cov_stacked[idx])

    obs_sample = np.concatenate(all_obs, axis=0)
    cov_sample = np.concatenate(all_covs, axis=0) if return_covariates else None

    # Trim to exact target
    if obs_sample.shape[0] > n_sample:
        idx = rng.choice(obs_sample.shape[0], n_sample, replace=False)
        obs_sample = obs_sample[idx]
        if cov_sample is not None:
            cov_sample = cov_sample[idx]

    log.info(f"Sampled {obs_sample.shape[0]:,} pixels for EM training")
    return obs_sample, cov_sample


def run_pipeline(
    config: PipelineConfig | None = None,
    skip_em: bool = False,
    params_path: Path | None = None,
    use_dask: bool = False,
    n_workers: int = 1,
    prechunked: bool = False,
    chunks_dir: Path | None = None,
) -> None:
    """Run the full TIDE pipeline.

    Args:
        config: Pipeline configuration (default if None).
        skip_em: Skip EM training and load params from file.
        params_path: Path to saved parameters (if skip_em=True, or to save to).
        use_dask: Use Dask distributed processing.
        n_workers: Number of Dask workers.
        prechunked: If True, read from pre-chunked .npy files.
        chunks_dir: Directory containing chunk_XXXXX/ subdirectories.
    """
    if config is None:
        config = PipelineConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.output_dir / "pipeline.log"),
        ],
    )

    log.info("=" * 60)
    log.info("TIDE US Drylands Tree Cover Classification Pipeline")
    log.info("=" * 60)

    t_start = time.time()

    # Step 1: Validate inputs
    validate_inputs(config)

    # Step 2: Read reference profile (from mask — shared grid with DEM/covariates)
    ref_profile = get_reference_profile(config.mask_path)
    log.info(
        f"Reference grid: {ref_profile['width']}x{ref_profile['height']} "
        f"({ref_profile['crs']})"
    )

    # Step 3: Phase A — EM Training
    if skip_em and params_path and params_path.exists():
        log.info(f"Loading pre-trained parameters from {params_path}")
        params = _load_params(params_path)
    else:
        log.info("Phase A: Baum-Welch EM on sampled pixels")
        t_em_start = time.time()

        obs_sample, _ = sample_pixels_for_em(config)
        obs_jax = jnp.array(obs_sample, dtype=jnp.float32)

        em_result = baum_welch(
            obs_jax,
            K=config.emission.n_states,
            max_iter=config.em.max_iter,
            tol=config.em.tol,
            lbfgs_maxiter=config.em.lbfgs_maxiter,
        )

        params = em_result.params
        t_em_elapsed = time.time() - t_em_start
        log.info(
            f"EM complete: {em_result.n_iter} iterations, "
            f"converged={em_result.converged}, "
            f"{t_em_elapsed:.1f}s"
        )

        # Save parameters
        if params_path:
            _save_params(params, params_path)
            log.info(f"Parameters saved to {params_path}")

        # Free EM sample arrays from GPU
        del obs_jax, em_result
        jax.clear_caches()

    # Log learned parameters
    log.info(f"Emission pi: {np.asarray(params.emission.pi)}")
    log.info(f"Emission mu: {np.asarray(params.emission.mu)}")
    log.info(f"Emission phi: {np.asarray(params.emission.phi)}")

    # Step 3½: Phase A½ — Fit covariate transition weights
    if config.enable_dynamic_transitions:
        log.info("Phase A½: Fitting covariate transition weights")
        t_wt_start = time.time()

        obs_sample, cov_sample = sample_pixels_for_em(
            config, return_covariates=True,
        )
        obs_jax_wt = jnp.array(obs_sample, dtype=jnp.float32)
        cov_jax = jnp.array(cov_sample, dtype=jnp.float32)

        weights, ll_history = fit_transition_weights(
            obs_jax_wt,
            cov_jax,
            params.emission,
            params.log_init,
            K=config.emission.n_states,
            max_iter=config.em.weight_max_iter,
            tol=config.em.weight_tol,
            l2_reg=config.em.l2_reg,
            lbfgs_maxiter=config.em.lbfgs_maxiter,
        )

        params = HMMParams(
            log_init=params.log_init,
            log_trans=params.log_trans,
            emission=params.emission,
            transition_weights=weights,
        )

        t_wt_elapsed = time.time() - t_wt_start
        log.info(
            f"Weight estimation complete: {len(ll_history)} iterations, "
            f"final LL={ll_history[-1]:.2f}, {t_wt_elapsed:.1f}s"
        )

        if params_path:
            _save_params(params, params_path)
            log.info(f"Updated parameters (with weights) saved to {params_path}")

        # Free weight estimation arrays from GPU
        del obs_jax_wt, cov_jax
        jax.clear_caches()

    # Step 4: Phase B — Full Inference
    log.info("Phase B: Full-dataset inference with frozen parameters")
    t_inf_start = time.time()

    result = run_full_inference(
        input_paths=config.input_paths,
        mask_path=config.mask_path,
        output_dir=config.output_dir,
        params=params,
        ref_profile=ref_profile,
        years=list(config.years),
        n_states=config.emission.n_states,
        chunk_size=config.inference.chunk_size,
        batch_pixels=config.inference.batch_pixels,
        compute_posteriors=config.inference.compute_posteriors,
        use_dask=use_dask,
        n_workers=n_workers,
        config=config,
        prechunked=prechunked,
        chunks_dir=chunks_dir,
    )

    t_inf_elapsed = time.time() - t_inf_start
    log.info(
        f"Inference complete: {result['total_pixels']:,} pixels, "
        f"{result['n_chunks']} chunks, {t_inf_elapsed:.1f}s"
    )

    # Step 5: Summary statistics
    log.info("Computing summary statistics")
    compute_summary_statistics(
        config.output_dir, list(config.years), config.emission.n_states
    )

    t_total = time.time() - t_start
    log.info(f"Pipeline complete in {t_total:.1f}s")


def _save_params(params: HMMParams, path: Path) -> None:
    """Save HMM parameters to npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "log_init": np.asarray(params.log_init),
        "log_trans": np.asarray(params.log_trans),
        "emission_pi": np.asarray(params.emission.pi),
        "emission_mu": np.asarray(params.emission.mu),
        "emission_phi": np.asarray(params.emission.phi),
    }
    if params.transition_weights is not None:
        save_dict["transition_weights"] = np.asarray(params.transition_weights)
    np.savez(path, **save_dict)


def _load_params(path: Path) -> HMMParams:
    """Load HMM parameters from npz file."""
    from tide.types import ZIBParams

    data = np.load(path)
    emission = ZIBParams(
        pi=jnp.array(data["emission_pi"]),
        mu=jnp.array(data["emission_mu"]),
        phi=jnp.array(data["emission_phi"]),
    )
    transition_weights = None
    if "transition_weights" in data:
        transition_weights = jnp.array(data["transition_weights"])
    return HMMParams(
        log_init=jnp.array(data["log_init"]),
        log_trans=jnp.array(data["log_trans"]),
        emission=emission,
        transition_weights=transition_weights,
    )
