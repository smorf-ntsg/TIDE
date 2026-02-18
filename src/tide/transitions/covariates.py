"""Covariate feature stacking for dynamic transitions.

Builds the (N, T-1, D) covariate tensor from individual data sources.
Feature order: intercept, fire_severity, spei, elevation, chili, aridity,
               seed_distance, focal_mean  (D=8 max)
"""

import jax.numpy as jnp

from tide.types import Array


def stack_covariates(
    fire_severity: Array | None = None,
    spei: Array | None = None,
    elevation: Array | None = None,
    chili: Array | None = None,
    aridity: Array | None = None,
    seed_distance: Array | None = None,
    focal_mean: Array | None = None,
) -> Array:
    """Stack covariate arrays into a single feature tensor.

    Each input is (N, T-1) for time-varying or (N,) for static covariates
    (broadcast over time). Always includes an intercept term as the first feature.

    Args:
        fire_severity: Ordinal fire severity (0-4, cast to float).
        spei: SPEI-12 drought index (gridMET).
        elevation: Elevation in meters (static, broadcast).
        chili: CHILI insolation index (static, broadcast).
        aridity: Aridity Index P/PET (static, broadcast).
        seed_distance: Distance to nearest tree cover.
        focal_mean: Neighborhood focal mean tree cover.

    Returns:
        (N, T-1, D) covariate tensor.
    """
    features = []

    # Determine N and T-1 from first available time-varying covariate
    N, Tm1 = None, None
    for arr in [fire_severity, spei, seed_distance, focal_mean]:
        if arr is not None and arr.ndim == 2:
            N, Tm1 = arr.shape
            break

    if N is None:
        # Fall back to static covariates
        for arr in [elevation, chili, aridity]:
            if arr is not None:
                N = arr.shape[0]
                Tm1 = 34  # Default for 35-year series
                break

    if N is None:
        raise ValueError("At least one covariate must be provided")

    # Intercept
    features.append(jnp.ones((N, Tm1), dtype=jnp.float32))

    def _add_feature(arr):
        if arr is None:
            return
        if arr.ndim == 1:
            # Static: (N,) -> (N, T-1)
            features.append(jnp.broadcast_to(arr[:, None], (N, Tm1)))
        else:
            features.append(arr)

    _add_feature(fire_severity)
    _add_feature(spei)
    _add_feature(elevation)
    _add_feature(chili)
    _add_feature(aridity)
    _add_feature(seed_distance)
    _add_feature(focal_mean)

    # Stack: list of (N, T-1) -> (N, T-1, D)
    result = jnp.stack(features, axis=-1)
    # Replace NaN with 0.0 (nodata covariates contribute zero to logistic regression)
    return jnp.nan_to_num(result, nan=0.0)
