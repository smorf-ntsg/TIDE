"""Static (fixed) transition matrix."""

import jax.numpy as jnp

from tide.types import Array


def make_static_log_trans(trans_probs: list[list[float]] | None = None) -> Array:
    """Create a static log transition matrix.

    Args:
        trans_probs: K x K transition probability matrix.
            If None, uses default 5-state ecological priors.

    Returns:
        (K, K) log transition matrix.
    """
    if trans_probs is None:
        from tide.config import DEFAULT_TRANS
        trans_probs = DEFAULT_TRANS

    trans = jnp.array(trans_probs, dtype=jnp.float32)
    return jnp.log(trans)
