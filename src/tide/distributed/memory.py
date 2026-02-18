"""Memory estimation and batch sizing for GPU processing."""

import logging

log = logging.getLogger(__name__)

# Bytes per element for different arrays
FLOAT32 = 4
INT32 = 4


def estimate_chunk_memory(
    n_pixels: int,
    T: int = 35,
    K: int = 5,
    D: int = 10,
    compute_xi: bool = False,
) -> dict[str, float]:
    """Estimate GPU memory requirements for a chunk.

    Args:
        n_pixels: Number of valid pixels in chunk.
        T: Number of timesteps.
        K: Number of states.
        D: Number of covariate features.
        compute_xi: Whether xi tensor will be materialized.

    Returns:
        Dict mapping component name to memory in MB.
    """
    mb = 1024 * 1024
    mem = {}

    mem["observations"] = n_pixels * T * FLOAT32 / mb
    mem["log_emission"] = n_pixels * T * K * FLOAT32 / mb
    mem["log_alpha"] = n_pixels * T * K * FLOAT32 / mb
    mem["log_beta"] = n_pixels * T * K * FLOAT32 / mb
    mem["log_gamma"] = n_pixels * T * K * FLOAT32 / mb

    if compute_xi:
        mem["log_xi"] = n_pixels * (T - 1) * K * K * FLOAT32 / mb

    mem["covariates"] = n_pixels * (T - 1) * D * FLOAT32 / mb
    mem["states_output"] = n_pixels * T * INT32 / mb
    mem["posteriors_output"] = n_pixels * T * K * FLOAT32 / mb

    mem["total"] = sum(mem.values())
    return mem


def optimal_batch_size(
    gpu_memory_gb: float = 80.0,
    T: int = 35,
    K: int = 5,
    D: int = 10,
    compute_xi: bool = False,
    safety_factor: float = 0.7,
) -> int:
    """Calculate optimal batch size (pixels) for available GPU memory.

    Args:
        gpu_memory_gb: Available GPU memory in GB.
        T: Number of timesteps.
        K: Number of states.
        D: Number of covariate features.
        compute_xi: Whether xi will be materialized.
        safety_factor: Fraction of GPU memory to use (leave room for JIT, etc).

    Returns:
        Maximum number of pixels per batch.
    """
    available_mb = gpu_memory_gb * 1024 * safety_factor

    # Memory per pixel in MB
    per_pixel = estimate_chunk_memory(1, T, K, D, compute_xi)
    per_pixel_mb = per_pixel["total"]

    batch_size = int(available_mb / per_pixel_mb)
    log.info(
        f"Optimal batch size: {batch_size:,} pixels "
        f"({per_pixel_mb * batch_size:.0f} MB / {available_mb:.0f} MB available)"
    )
    return batch_size
