# TIDE Engineering Documentation

## Executive Summary

This document describes the technical implementation of the Zero-Inflated Beta Hidden Markov Model (ZIB-HMM) for drylands tree cover classification. The system is built in **Python 3.12+** using **JAX** for GPU-accelerated HMM algorithms, **Rasterio** for geospatial I/O, and **Dask** for distributed processing.

**Architecture highlights**:
- Modular design: 29 source files organized into 9 packages
- 62 unit tests with pytest (58 passing; 4 GPU OOM)
- Pre-chunked .npy I/O pipeline for 6x faster inference
- GPU-accelerated inference via JAX (I/O-bound; stacked inputs + async prefetch + async writes)
- Scalable to billions of pixels via spatial chunking
- Cloud-Optimized GeoTIFF outputs

---

## 1. System Architecture

### 1.1 Directory Structure

```
sb_treecover/
├── src/tide/                    # Source package
│   ├── __init__.py
│   ├── types.py                # NamedTuples for type safety
│   ├── config.py               # Dataclass configs with defaults
│   ├── pipeline.py             # Top-level orchestration
│   ├── cli.py                  # Click CLI entry point
│   │
│   ├── emissions/              # Emission distributions
│   │   ├── zero_inflated_beta.py
│   │   └── gaussian.py         # Legacy 2-state for comparison
│   │
│   ├── hmm/                    # Core HMM algorithms (JAX)
│   │   ├── forward_backward.py
│   │   ├── viterbi.py
│   │   ├── baum_welch.py
│   │   ├── mstep_zib.py
│   │   └── mstep_transitions.py
│   │
│   ├── transitions/            # Transition modeling
│   │   ├── static.py
│   │   ├── dynamic.py
│   │   ├── covariates.py
│   │   └── fit_weights.py
│   │
│   ├── data/                   # Geospatial data I/O
│   │   ├── rap.py              # RAP tree cover reader
│   │   ├── mask.py             # Analysis region mask
│   │   ├── mtbs.py             # MTBS fire severity
│   │   ├── spei.py             # SPEI drought index
│   │   ├── dem.py              # DEM + CHILI terrain
│   │   ├── seed_distance.py    # Derived dispersal metrics
│   │   ├── alignment.py        # Grid reprojection
│   │   └── chunking.py         # Spatial tiling
│   │
│   ├── distributed/            # Parallelization
│   │   ├── memory.py           # Memory estimation
│   │   ├── gpu_worker.py       # JAX inference worker
│   │   └── orchestrator.py    # Dask dispatch
│   │
│   └── io/                     # I/O layer
│       ├── reader.py           # Windowed chunk reading
│       ├── writer.py           # COG creation
│       ├── prechunk.py         # Pre-chunk inputs to .npy (Phase 0)
│       ├── assemble.py         # Reassemble .npy results to GeoTIFF (Phase 2)
│       └── summary.py          # Statistics computation
│
├── tests/                      # Unit tests (pytest)
│   ├── conftest.py             # Shared fixtures
│   ├── test_zib_emission.py
│   ├── test_forward_backward.py
│   ├── test_viterbi.py
│   ├── test_baum_welch.py
│   ├── test_integration.py
│   └── test_covariates.py
│
├── scripts/
│   ├── mosaic_gee_tiles.py     # GEE tile mosaicking (standalone)
│   └── stack_treecover.py      # One-time: stack 39 yearly files → 1 multi-band file
│
├── notebooks/                  # Jupyter exploration (optional)
│
├── pyproject.toml              # Package config + dependencies
├── SCIENCE.md                  # Scientific methodology
├── ENGINEERING.md              # This document
└── PROGRESS.md                 # Development log
```

### 1.2 Core Dependencies

```toml
[project.dependencies]
jax[cuda12] >= 0.4.30          # GPU-accelerated array ops
jaxlib >= 0.4.30               # JAX backend
jaxopt >= 0.8                  # L-BFGS optimizer
numpy >= 1.24                  # Array ops
rasterio >= 1.3                # Geospatial I/O
dask[distributed] >= 2024.1    # Parallel dispatch
click >= 8.0                   # CLI framework

[project.optional-dependencies]
dev = [
    "pytest >= 7.0",
    "pytest-xdist",            # Parallel test execution
]
```

**Installation**:
```bash
cd sb_treecover
pip install -e .               # Editable install
pip install -e .[dev]          # Include test dependencies
```

**CLI registration**: `pyproject.toml` registers `tide` as a console script pointing to `tide.cli:main`.

---

## 2. Type System and Configuration

### 2.1 Type Definitions (`types.py`)

The codebase uses **NamedTuples** for immutable parameter containers:

```python
from typing import NamedTuple
import jax.numpy as jnp

class ZIBParams(NamedTuple):
    """Zero-Inflated Beta emission parameters."""
    pi: jnp.ndarray   # shape: (K,), zero-inflation probabilities
    mu: jnp.ndarray   # shape: (K,), Beta means
    phi: jnp.ndarray  # shape: (K,), Beta precisions

class HMMParams(NamedTuple):
    """Complete HMM parameters."""
    log_init: jnp.ndarray            # shape: (K,), log initial state probs
    log_trans: jnp.ndarray           # shape: (K, K), log transition matrix
    emission: ZIBParams              # Emission parameters
    transition_weights: jnp.ndarray | None = None  # shape: (K, D), optional

class ForwardBackwardResult(NamedTuple):
    """Forward-backward algorithm outputs."""
    gamma: jnp.ndarray  # shape: (N, T, K), state posteriors
    xi: jnp.ndarray     # shape: (N, T-1, K, K), pairwise transition probs
    log_likelihood: float

class ViterbiResult(NamedTuple):
    """Viterbi algorithm outputs."""
    states: jnp.ndarray      # shape: (N, T), most likely state sequence
    log_likelihood: float

class EMResult(NamedTuple):
    """Baum-Welch EM outputs."""
    params: HMMParams
    log_likelihood: float
    n_iter: int
    converged: bool
```

**Benefits**:
- Type hints enable IDE autocomplete and static analysis
- Immutability prevents accidental parameter modification
- JAX compatibility (NamedTuples work with jit/vmap)

### 2.2 Configuration (`config.py`)

Dataclasses define hierarchical configuration with sensible defaults:

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class EmissionConfig:
    """Emission distribution settings."""
    n_states: int = 6

@dataclass
class EMConfig:
    """EM algorithm settings."""
    max_iter: int = 100
    tol: float = 0.01
    lbfgs_maxiter: int = 100
    n_sample_pixels: int = 2_000_000
    max_missing: int = 5
    weight_max_iter: int = 20      # For dynamic transition weight fitting
    weight_tol: float = 1e-3
    l2_reg: float = 0.01

@dataclass
class InferenceConfig:
    """Full-dataset inference settings."""
    chunk_size: int = 512
    batch_pixels: int = 500_000
    compute_posteriors: bool = True

@dataclass
class IOConfig:
    """I/O paths and settings."""
    compression: str = "DEFLATE"
    predictor: int = 2  # Horizontal differencing for better compression

@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    data_dir: Path = Path("data")
    output_dir: Path = Path("output_v2")

    # Optional covariate paths (None = use defaults)
    mtbs_dir: Path | None = None
    spei_dir: Path | None = None
    dem_path: Path | None = None
    chili_path: Path | None = None

    enable_dynamic_transitions: bool = False

    # Nested configs
    emission: EmissionConfig = field(default_factory=EmissionConfig)
    em: EMConfig = field(default_factory=EMConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    io: IOConfig = field(default_factory=IOConfig)

    @property
    def years(self) -> range:
        """1986-2024 inclusive."""
        return range(1986, 2025)

    @property
    def mask_path(self) -> Path:
        """Derived: data_dir/mask/drylands_mask.tif"""
        return self.data_dir / "mask" / "drylands_mask.tif"

    @property
    def input_paths(self) -> list[Path]:
        """Derived: data_dir/treeCover/treeCover_{year}.tif"""
        return [self.data_dir / "treeCover" / f"treeCover_{y}.tif"
                for y in self.years]

    @property
    def effective_mtbs_dir(self) -> Path:
        """MTBS directory, defaulting to data_dir/mtbs if not set."""
        return self.mtbs_dir if self.mtbs_dir else self.data_dir / "mtbs"

    # Similar properties for effective_spei_dir, effective_dem_path, effective_chili_path
```

**Design rationale**:
- **Dataclasses** provide free `__init__`, `__repr__`, and type checking
- **Derived properties** reduce redundancy (user specifies `data_dir`, paths are computed)
- **Nested configs** organize related settings hierarchically
- **Default factories** prevent mutable default arguments

---

## 3. JAX Implementation Details

### 3.1 Why JAX?

**JAX** is a NumPy-compatible array library with automatic differentiation and JIT compilation:

1. **GPU acceleration**: `jax.numpy` operations run on GPU with no code changes
2. **Automatic differentiation**: `jax.grad` computes gradients for EM M-step optimization
3. **JIT compilation**: `@jax.jit` decorator compiles Python → XLA → GPU kernels
4. **Functional purity**: JAX requires pure functions (no side effects), enabling aggressive optimization

**Performance**: JAX HMM inference is **~100-1000× faster** than pure NumPy/Python loops.

### 3.2 Vectorization: `jax.vmap`

**Problem**: Process N pixels simultaneously, each with its own time series.

**Solution**: `jax.vmap` vectorizes a function over a leading batch dimension:

```python
def viterbi_single(log_obs, log_init, log_trans):
    """Viterbi for a single pixel.

    Args:
        log_obs: (T, K) log observation probabilities
        log_init: (K,) log initial state probabilities
        log_trans: (K, K) or (T-1, K, K) log transition matrix

    Returns:
        states: (T,) most likely state sequence
    """
    # ... implementation details ...
    return states

# Vectorize over N pixels
viterbi_batch = jax.vmap(viterbi_single)

# Call on batch
log_obs_batch = jnp.array(..., shape=(N, T, K))
log_init_batch = jnp.broadcast_to(log_init, (N, K))  # Same for all pixels
log_trans_batch = jnp.broadcast_to(log_trans, (N, *log_trans.shape))

states_batch = viterbi_batch(log_obs_batch, log_init_batch, log_trans_batch)
# Result shape: (N, T)
```

**Key insight**: Write the algorithm for a single pixel, then `vmap` over the batch. JAX handles parallelization automatically.

### 3.3 Sequential Loops: `jax.lax.scan`

**Problem**: HMM algorithms require sequential loops over time steps (can't parallelize over T).

**Solution**: `jax.lax.scan` is a JIT-friendly replacement for Python `for` loops:

```python
def forward_step(carry, inputs):
    """Single forward step.

    Args:
        carry: Previous alpha_t (K,)
        inputs: Tuple of (log_obs_t, log_trans_t)
            log_obs_t: (K,) log observation probs at time t
            log_trans_t: (K, K) log transition matrix

    Returns:
        carry: Updated alpha_t (K,)
        output: Same alpha_t (for stacking)
    """
    alpha_prev = carry
    log_obs_t, log_trans_t = inputs

    # Forward recursion: alpha_t(k) = sum_j alpha_{t-1}(j) * A_{jk} * B_k(obs_t)
    alpha_t = jax.scipy.special.logsumexp(
        alpha_prev[:, None] + log_trans_t, axis=0
    ) + log_obs_t

    return alpha_t, alpha_t  # (new carry, output)

# Initialize
alpha_0 = log_init + log_obs[0]  # (K,)

# Scan over T-1 time steps
carry_init = alpha_0
xs = (log_obs[1:], log_trans_seq)  # Each is (T-1, ...)

carry_final, alphas_stacked = jax.lax.scan(forward_step, carry_init, xs)
# alphas_stacked shape: (T-1, K)
```

**Why not Python `for`?**
- Python loops can't be JIT-compiled (interpreted overhead)
- `scan` enables XLA optimization (loop fusion, vectorization)
- `scan` works with `vmap` (vectorize over pixels, scan over time)

### 3.4 Log-Space Arithmetic

**Problem**: Multiplying many small probabilities (e.g., P(obs|state) over 39 years) causes underflow.

**Solution**: Work in log-space, use `logsumexp` for addition:

```python
# Linear space (WRONG - underflows):
prob = p1 * p2 * p3  # Can become 0 if any p < 1e-300

# Log space (CORRECT):
log_prob = log_p1 + log_p2 + log_p3  # Safe for any values

# Sum in linear space, computed in log space:
# result = log(exp(log_a) + exp(log_b))
result = jax.scipy.special.logsumexp(jnp.array([log_a, log_b]))
```

**Implementation**: All HMM algorithms (`forward_backward.py`, `viterbi.py`) operate on `log_init`, `log_trans`, `log_obs` and return log-probabilities.

### 3.5 Dynamic Transitions: Passing Matrices via `scan`

**Challenge**: JAX `jit` requires all branches to have the same shape. This breaks with conditional logic:

```python
# WRONG - shape mismatch in jit
if use_dynamic:
    log_trans_t = dynamic_trans[t]  # (K, K)
else:
    log_trans_t = static_trans      # (K, K)
```

Even though both branches have shape `(K, K)`, JAX's tracer sees `dynamic_trans[t]` as a dynamic index (trace-time unknown), while `static_trans` is concrete. This causes `ConcretizationError`.

**Solution**: Always pass `(T-1, K, K)` transitions via scan `xs`:

```python
# Static mode: Broadcast to (T-1, K, K)
if log_trans.ndim == 2:
    log_trans_seq = jnp.broadcast_to(log_trans, (T-1, K, K))
else:
    log_trans_seq = log_trans  # Already (T-1, K, K)

# Now scan can access log_trans_seq[t] uniformly
def step_fn(carry, inputs):
    log_obs_t, log_trans_t = inputs  # log_trans_t is always (K, K)
    # ... use log_trans_t ...

xs = (log_obs[1:], log_trans_seq)
carry_final, outputs = jax.lax.scan(step_fn, carry_init, xs)
```

---

## 4. HMM Core Algorithms

### 4.1 Forward-Backward (`hmm/forward_backward.py`)

**Purpose**: Compute state posteriors γ_t(k) = P(state=k | all observations).

**Algorithm**:
1. **Forward pass**: Compute α_t(k) = P(obs_{1:t}, state_t=k)
2. **Backward pass**: Compute β_t(k) = P(obs_{t+1:T} | state_t=k)
3. **Combine**: γ_t(k) = (α_t(k) × β_t(k)) / Σ_j (α_t(j) × β_t(j))

**Signature**:
```python
def forward_backward(
    log_obs: Array,        # (N, T, K) log observation probs
    log_init: Array,       # (K,) log initial probs
    log_trans: Array,      # (K, K) or (N, T-1, K, K) log transitions
) -> ForwardBackwardResult:
    """
    Returns:
        gamma: (N, T, K) state posteriors
        xi: (N, T-1, K, K) pairwise transition posteriors
        log_likelihood: scalar
    """
```

**Key optimizations**:
- `jax.vmap` over N pixels
- `jax.lax.scan` over T time steps
- Log-space throughout (no underflow)

**Test coverage**: `test_forward_backward.py` verifies:
- Output shapes match expectations
- Posteriors sum to 1.0 (within tolerance)
- Xi marginalizes to gamma
- 2-state special case matches NumPy reference
- 6-state smoke test runs without errors

### 4.2 Viterbi (`hmm/viterbi.py`)

**Purpose**: Find most likely state sequence: argmax_s P(s | obs).

**Algorithm**:
1. **Forward pass**: Compute δ_t(k) = max probability of path ending in state k at time t
2. **Backward pass**: Backtrack pointers to recover the path

**Signature**:
```python
def viterbi(
    log_obs: Array,        # (N, T, K)
    log_init: Array,       # (K,)
    log_trans: Array,      # (K, K) or (N, T-1, K, K)
) -> ViterbiResult:
    """
    Returns:
        states: (N, T) most likely state indices (0 to K-1)
        log_likelihood: scalar
    """
```

**Implementation details**:
- Uses `jnp.argmax` to find best predecessor at each step
- Stores pointers in auxiliary array for backtracking
- Backward scan reverses the sequence to reconstruct the path

**Test coverage**: `test_viterbi.py` verifies:
- Output shapes and dtype (uint8)
- States in valid range [0, K-1]
- Deterministic sequences decoded correctly
- NumPy reference match for 2-state case

### 4.3 Baum-Welch EM (`hmm/baum_welch.py`)

**Purpose**: Learn HMM parameters from observations.

**Algorithm**:
```
Initialize params (π, A, ZIB)
Repeat until convergence:
    E-step: Run forward-backward → compute γ, ξ
    M-step: Update params to maximize Q(params | γ, ξ)
```

**M-step details**:
- **Initial probs**: π_k = γ_{t=0}(k)
- **Transitions**: A_{ij} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
- **ZIB emissions**: `mstep_zib.py` handles per-state optimization
  - **π_k** (zero-inflation): Closed-form = #(y=0) / #(total) weighted by γ
  - **(μ_k, φ_k)**: L-BFGS optimization with constrain/unconstrain transforms

**Signature**:
```python
def baum_welch(
    obs: Array,            # (N, T) observations in [0, 1]
    K: int,                # Number of states
    max_iter: int = 100,
    tol: float = 0.01,
    lbfgs_maxiter: int = 100,
) -> EMResult:
    """
    Returns:
        EMResult with learned params, final LL, n_iter, converged flag
    """
```

**Monotonicity enforcement**: `mstep_zib.py` uses cumulative sigmoid:
```python
def unconstrain_mu(mu_sorted):
    """Map sorted mu to unconstrained space."""
    # mu = cumsum(sigmoid(unconstrained))
    # Invert: unconstrained = logit(diff(mu))
    diffs = jnp.diff(jnp.concatenate([jnp.array([0.0]), mu_sorted]))
    return jnp.log(diffs / (1 - jnp.cumsum(diffs)[:-1] + 1e-8))

def constrain_mu(unconstrained):
    """Map unconstrained → sorted mu."""
    # Recursive stick-breaking
    # mu_k = mu_{k-1} + (1 - mu_{k-1}) * sigmoid(delta_k)
    return mu
```

This ensures μ_0 < μ_1 < ... < μ_{K-1} < 1 even during optimization, without arbitrary upper bounds.

**Test coverage**: `test_baum_welch.py` verifies:
- LL increases monotonically
- Learned params recover synthetic data
- All params finite and in valid ranges
- No NaN or Inf

### 4.4 Transition Weight Fitting (`transitions/fit_weights.py`)

**Purpose**: Learn covariate coefficients for dynamic transitions (Phase A½).

**Algorithm**:
```
Initialize weights β (K, D)
Repeat until convergence:
    E-step: Forward-backward with dynamic trans → compute ξ
    M-step: For each source state i, optimize β_i:
        max Σ_{t,j} ξ_t(i,j) × log(softmax(β_i · cov_t)_j)
```

**Signature**:
```python
def fit_transition_weights(
    obs: Array,            # (N, T) observations
    covariates: Array,     # (N, T-1, D) covariate features
    emission: ZIBParams,   # Frozen emission params
    log_init: Array,       # (K,) frozen initial probs
    K: int,
    max_iter: int = 20,
    tol: float = 1e-3,
    l2_reg: float = 0.01,
    lbfgs_maxiter: int = 100,
) -> tuple[Array, list[float]]:
    """
    Returns:
        weights: (K, D) learned coefficients
        ll_history: Log-likelihoods per iteration
    """
```

**Per-source-state optimization** (`_fit_weights_for_source`):
- Extract ξ for transitions from state i
- Define negative log-likelihood:
  ```python
  def nll(beta_i):
      logits = covariates @ beta_i  # (N, T-1, K)
      log_probs = jax.nn.log_softmax(logits, axis=-1)

      # Weighted cross-entropy
      loss = -jnp.sum(xi_from_i * log_probs)

      # L2 regularization (exclude intercept)
      loss += l2_reg * jnp.sum(beta_i[1:]**2)

      return loss
  ```
- Run `jaxopt.LBFGS` to minimize

**Test coverage**: `test_covariates.py` verifies:
- Output shape (K, D)
- All weights finite
- Softmax produces valid distributions
- End-to-end integration with pipeline

---

## 5. Data Pipeline

### 5.1 Chunking Strategy (`data/chunking.py`)

**Problem**: Full raster (118k × 97k pixels) doesn't fit in GPU memory.

**Solution**: Divide into spatial tiles (default 512×512):

```python
def generate_chunks(
    height: int,
    width: int,
    chunk_size: int = 512,
) -> list[dict]:
    """Generate non-overlapping tiles.

    Returns:
        List of dicts with keys: row_off, col_off, height, width
    """
    chunks = []
    for row in range(0, height, chunk_size):
        for col in range(0, width, chunk_size):
            chunks.append({
                'row_off': row,
                'col_off': col,
                'height': min(chunk_size, height - row),
                'width': min(chunk_size, width - col),
            })
    return chunks
```

**Mask filtering**:
```python
def filter_masked_chunks(
    chunks: list[dict],
    mask_path: Path,
) -> list[dict]:
    """Remove chunks with no valid pixels."""
    valid_chunks = []
    with rasterio.open(mask_path) as mask_src:
        for chunk in chunks:
            window = rasterio.windows.Window(**chunk)
            mask_data = mask_src.read(1, window=window)
            if np.any(mask_data == 1):
                valid_chunks.append(chunk)
    return valid_chunks
```

**Result**: 44,080 total chunks → 25,130 chunks with valid pixels (CONUS drylands mask)

### 5.2 Windowed Reading (`io/reader.py`)

**Rasterio windows** enable efficient reading of subsets:

```python
def read_chunk_data(
    input_paths: list[Path],
    mask_path: Path,
    chunk: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Read tree cover time series for a chunk.

    Returns:
        cube: (N, T) tree cover for N valid pixels
        coords: (N, 2) pixel (row, col) within chunk
    """
    window = rasterio.windows.Window(**chunk)

    # Read mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1, window=window)

    # Extract valid pixel coordinates
    valid_mask = mask == 1
    coords = np.argwhere(valid_mask)  # (N, 2)
    N = len(coords)
    T = len(input_paths)

    # Allocate cube
    cube = np.full((N, T), 255, dtype=np.uint8)

    # Read each year
    for t, path in enumerate(input_paths):
        with rasterio.open(path) as src:
            data = src.read(1, window=window)
            cube[:, t] = data[valid_mask]

    return cube, coords
```

**Key optimizations**:
- Only read pixels within the current chunk window (not the full raster)
- Only extract valid pixels (mask == 1), discarding NoData
- Return compact `(N, T)` array rather than sparse `(H, W, T)`

### 5.3 Grid Reprojection (`data/alignment.py`)

**Challenge**: Input rasters are on different grids:
- RAP tree cover: 117889×91086 (EPSG:4326, irregular pixel size)
- DEM/CHILI/SPEI/mask: 118742×96888 (EPSG:4326, different bounds)
- MTBS: 118743×96889 (off by 1 pixel)

**Solution**: Reproject on-the-fly using rasterio's `WarpedVRT`:

```python
from rasterio.vrt import WarpedVRT

def read_aligned_window(
    src_path: Path,
    ref_profile: dict,
    window: rasterio.windows.Window,
) -> np.ndarray:
    """Read a window reprojected to reference grid."""
    with rasterio.open(src_path) as src:
        with WarpedVRT(src, **ref_profile) as vrt:
            data = vrt.read(1, window=window)
    return data
```

**Reference grid**: Use the mask CRS/bounds/resolution as the target. All inputs are warped to match.

**Trade-off**: Adds ~10-20% runtime overhead, but ensures pixel-perfect alignment.

### 5.4 Covariate Reading (`io/reader.py`)

**Challenge**: Covariates are 2D (elevation, CHILI) or 3D (fire, drought), but inference needs `(N, T-1, D)` per chunk.

**Solution**: `read_covariate_data` handles spatial→pixel extraction:

```python
def read_covariate_data(
    chunk: dict,
    coords: np.ndarray,      # (N, 2) pixel coordinates within chunk
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    """Read and align covariates for a chunk.

    Returns:
        Dict with keys: fire_severity, spei, elevation, chili,
                        seed_distance, focal_mean
        Each array is (N,) for static or (N, T-1) for temporal
    """
    window = rasterio.windows.Window(**chunk)
    N = len(coords)
    T = len(list(config.years))

    result = {}

    # Temporal covariates: (H, W, T-1) → (N, T-1)
    if config.effective_mtbs_dir:
        fire_cube = read_mtbs_window(config.effective_mtbs_dir, window, config.years)
        # fire_cube shape: (H, W, T)
        # Slice to T-1 (year t affects transition t→t+1)
        result['fire_severity'] = fire_cube[coords[:, 0], coords[:, 1], :-1]  # (N, T-1)

    # Similar for SPEI...

    # Static covariates: (H, W) → (N,) broadcast to (N, T-1)
    if config.effective_dem_path:
        dem = read_aligned_window(config.effective_dem_path, ref_profile, window)
        result['elevation'] = dem[coords[:, 0], coords[:, 1]]  # (N,)

    # Derived covariates (seed distance, focal mean) computed from tree cover cube
    # ...

    return result
```

**Temporal alignment**: Fire in year *t* is used for transition *t→t+1*, so we slice `fire_cube[:, :, :-1]` to get `(T-1,)` transitions.

---

## 6. Distributed Processing

### 6.1 Architecture

Two execution modes:

1. **Sequential** (`use_dask=False`): Single-threaded loop over chunks
   - Good for debugging
   - Lower memory footprint
   - Simpler logging

2. **Dask distributed** (`use_dask=True`): Parallel chunk processing
   - Scales to multiple GPUs/nodes
   - Requires Dask scheduler
   - Higher throughput

### 6.2 GPU Worker (`distributed/gpu_worker.py`)

**Entry point**:
```python
def infer_chunk(
    chunk: dict,
    input_paths: list[Path],
    mask_path: Path,
    params: HMMParams,
    ref_profile: dict,
    years: list[int],
    n_states: int,
    batch_pixels: int,
    compute_posteriors: bool,
    config: PipelineConfig,
) -> dict:
    """Process a single chunk on GPU.

    Returns:
        Dict with keys: states, posteriors (optional), coords
    """
```

**Steps**:
1. Read chunk data (tree cover + covariates if dynamic)
2. Convert uint8 → float32, normalize to [0, 1]
3. If dynamic transitions: compute `(N, T-1, K, K)` log_trans_seq per pixel
4. Split into batches of `batch_pixels` (default 500k)
5. For each batch:
   - Run Viterbi → states
   - Run forward-backward → posteriors (if enabled)
6. Concatenate batch results
7. Return to orchestrator for writing

**Memory management**:
```python
def estimate_memory_mb(N, T, K, compute_posteriors):
    """Estimate GPU memory for a batch."""
    obs_mb = (N * T * K * 4) / 1e6        # Log-obs: (N, T, K) float32
    states_mb = (N * T * 1) / 1e6         # States: (N, T) uint8
    posteriors_mb = (N * T * K * 4) / 1e6 if compute_posteriors else 0

    # Forward-backward intermediates (~2x obs size)
    workspace_mb = 2 * obs_mb if compute_posteriors else 0

    return obs_mb + states_mb + posteriors_mb + workspace_mb

# Adjust batch size to fit GPU
gpu_memory_gb = 24  # e.g., RTX 3090
safe_batch_mb = gpu_memory_gb * 1000 * 0.8  # 80% utilization

batch_pixels = max(10_000, safe_batch_mb / estimate_memory_mb(1, T, K, True) * 1000)
```

### 6.3 Orchestrator (`distributed/orchestrator.py`)

**Sequential mode**:
```python
def run_sequential(chunks, ...):
    results = []
    for chunk in tqdm(chunks):
        result = infer_chunk(chunk, ...)
        write_chunk_results(output_dir, chunk, result, ...)
        results.append(result)
    return aggregate_results(results)
```

**Dask mode**:
```python
from dask.distributed import Client, as_completed

def run_dask(chunks, n_workers, ...):
    client = Client(n_workers=n_workers, threads_per_worker=1)

    # Pass device_id (int) instead of jax.Device object
    futures = [
        client.submit(infer_chunk, chunk, ..., device_id=device.id)
        for chunk in chunks
    ]

    for future in as_completed(futures):
        result = future.result()
        write_chunk_results(...)

    client.close()
```

**Dask considerations**:
- `jax.Device` objects cannot be pickled; pass integer device IDs instead
- Each worker needs GPU access (set `CUDA_VISIBLE_DEVICES`)
- Large params (`HMMParams`) are serialized once via Dask scatter

---

## 7. Output Writing

### 7.1 Output Structure (`io/writer.py`)

Per-year files keep individual file sizes manageable at CONUS scale (~2GB/year for states, ~12GB/year for posteriors):

```
output/
  states/
    states_1986.tif ... states_2024.tif   # 1-band uint8 per year, NoData=255
  posteriors/
    posteriors_1986.tif ... posteriors_2024.tif  # K-band uint8 per year
      # band k+1 = P(state=k) × 200, stored as uint8
  trace_onset_year.tif   # int16, first year state ≥ 1 (Trace)
  sparse_year.tif        # int16, first year state ≥ 2 (Sparse)
  open_year.tif          # int16, first year state ≥ 3 (Open)
  woodland_year.tif      # int16, first year state ≥ 4 (Woodland)
  forest_year.tif        # int16, first year state ≥ 5 (Forest)
  max_state.tif          # uint8, max state ever reached
```

All files: DEFLATE-compressed, 256×256 internal tiles, EPSG:4326.

**Handle management**: `create_output_rasters()` creates all files (headers only). `open_output_handles()` opens ~85 handles (39 states + 39 posteriors + 7 summary) once at pipeline start and reuses them across chunks. `close_output_handles()` flushes all in `finally`.

### 7.2 Chunk Writing

For each chunk, `write_chunk_results()` writes:
1. **States**: per-year 1-band write — `handles["states_YYYY"].write(state_grid, indexes=1)`
2. **Posteriors**: per-year K-band write — `handles["posteriors_YYYY"].write(post_cube, window=window)` (all K bands in one call)
3. **Transition years**: vectorized argmax — `np.argmax(states >= threshold, axis=1)` → int16 year

The write thread (`ThreadPoolExecutor(max_workers=1)`) runs writes asynchronously so the GPU never waits for disk.

### 7.3 Transition Year Computation

**Vectorized implementation**:
```python
def _write_transition_years(handles, states, coords, chunk, years):
    """Compute and write transition years."""
    N, T = states.shape

    # For each threshold, find first year >= threshold
    thresholds = {
        'trace_onset_year': 1,    # State ≥ 1
        'sparse_year': 2,         # State ≥ 2
        'open_year': 3,           # State ≥ 3
        'woodland_year': 4,       # State ≥ 4
        'forest_year': 5,         # State ≥ 5
    }

    for name, thresh in thresholds.items():
        # Boolean mask: (N, T) where states >= thresh
        mask = states >= thresh

        # Find first True along axis=1 (time)
        # Use argmax (returns first max, and True > False)
        first_occurrence = np.argmax(mask, axis=1)  # (N,)

        # Check if any True exists (if not, argmax returns 0 spuriously)
        any_transition = np.any(mask, axis=1)  # (N,)

        # Map to year (0-indexed → year)
        transition_years = np.where(
            any_transition,
            years[0] + first_occurrence,
            0  # NoData
        ).astype(np.int16)

        # Write
        handles[name].write(transition_years, indexes=1, ...)
```

---

## 8. Testing Framework

### 8.1 Pytest Structure

Tests are organized by module:

```
tests/
├── conftest.py                 # Shared fixtures
├── test_zib_emission.py        # ZIB distribution tests
├── test_forward_backward.py    # Forward-backward algorithm
├── test_viterbi.py             # Viterbi algorithm
├── test_baum_welch.py          # EM training
├── test_integration.py         # End-to-end workflows
└── test_covariates.py          # Dynamic transitions
```

**Run tests**:
```bash
pytest                          # Run all
pytest -v                       # Verbose
pytest tests/test_viterbi.py    # Single module
pytest -n auto                  # Parallel (requires pytest-xdist)
pytest -k "test_emission"       # Match pattern
```

### 8.2 Fixtures (`conftest.py`)

```python
import pytest
import jax.numpy as jnp

@pytest.fixture
def synthetic_obs():
    """Generate synthetic observations for testing."""
    N, T, K = 1000, 39, 6
    # Create observations biased toward certain states
    states_true = jnp.array([...], dtype=jnp.uint8)  # (N, T)
    obs = jnp.array([...], dtype=jnp.float32)  # (N, T)
    return obs, states_true

@pytest.fixture
def default_params():
    """Default HMM parameters for testing."""
    from tide.emissions.zero_inflated_beta import init_zib_params
    from tide.config import EmissionConfig

    config = EmissionConfig(n_states=6)
    emission = init_zib_params(K=6)
    log_init = jnp.log(jnp.ones(6) / 6)
    log_trans = jnp.log(jnp.eye(6) * 0.9 + 0.02)  # Diagonal-dominant

    return HMMParams(log_init, log_trans, emission, None)
```

### 8.3 Test Categories

**Unit tests** (fast, isolated):
- Emission log-probabilities match expected values
- Constrain/unconstrain round-trips
- Monotonicity of mu via recursive stick-breaking
- Shape contracts for all functions

**Integration tests** (slower, end-to-end):
- Full EM → Viterbi pipeline
- Chunking → inference → writing
- Covariate stacking → weight fitting → dynamic inference

**Property tests** (invariants):
- Transition matrix rows sum to 1.0
- Posteriors are valid probabilities
- States are in valid range
- No NaN/Inf in outputs

**Regression tests** (NumPy reference):
- Forward-backward marginals match direct computation

### 8.4 Coverage

Current test suite:
- **62 tests total** (58 passing; 4 fail with pre-existing GPU OOM in `test_baum_welch`)
- **Coverage**: ~85% of source lines (excludes CLI, some I/O branches)

**Run with coverage**:
```bash
pytest --cov=tide --cov-report=html
# Open htmlcov/index.html
```

---

## 9. Performance and Scalability

### 9.1 Throughput Benchmarks

**GPU specs**: 2× NVIDIA H100 NVL (sequential mode uses device 0 only)

**Critical finding**: The pipeline is I/O-bound, not GPU-bound.

| Component | Profiled time | Share |
|-----------|--------------|-------|
| 39 × file open overhead | 898ms | **94%** of I/O |
| deflate read + decompress | 51ms | 5% |
| GPU Viterbi + ZIB emissions | 1.4ms | <1% |

**I/O optimizations implemented (in order of impact):**

1. **Persistent file handles** — open all rasters once at pipeline start; reuse across all chunks. Eliminates 898ms of file-open overhead per chunk (23ms × 39 files). Handles owned by prefetch thread.
2. **Async I/O prefetch** — `ThreadPoolExecutor(max_workers=1)` reads next chunk while GPU processes current chunk.
3. **Async writes** — second `ThreadPoolExecutor(max_workers=1)` flushes results to disk while GPU handles next chunk.
4. **Stacked treeCover** (`scripts/stack_treecover.py`) — one-time preprocessing: consolidates 39 per-year files into `data/treeCover/treeCover_stack.tif` (39 bands, 512×512 tiles). Auto-detected at startup. Replaces 156 GDAL decompress calls per chunk (39 files × 4 tiles each) with 1 call. ~6× cold-read speedup.
5. **Chunk size** — default 512×512 amortizes per-chunk setup over 4× more pixels vs 256×256.

**Profiled bottleneck breakdown** (per chunk, post-persistent-handles):

| Component | Time | Notes |
|---|---|---|
| treeCover cold read (39 files) | 43ms | 4 tile reads/file × 39 files |
| treeCover cold read (stacked) | ~5ms | 1 read for all 39 bands |
| Write (states + posteriors) | ~112ms | async — overlapped with GPU |
| GPU Viterbi + ZIB emissions | 1.4ms | always idle waiting for I/O |

**Measured throughput** (data_test, 23M px, 1 H100):

| Configuration | Time | Throughput | CONUS ETA |
|---|---|---|---|
| chunk=256, sequential (baseline) | 771s | 30K px/s | ~16h |
| chunk=512, async prefetch | 153s | 152K px/s | ~3.4h |
| +persistent handles | 116s | 201K px/s | ~2.6h |
| +stacked treeCover (est.) | ~84s | ~280K px/s | **~1.9h** |
| +async writes (est.) | ~73s | ~320K px/s | **~1.6h** |

**GPU utilization note**: Low GPU utilization is expected and irreducible — the GPU finishes in 1.4ms and is always idle waiting for I/O. The H100 is not the bottleneck.

### 9.2 Memory Requirements

**Per-worker GPU memory** (batch_pixels=500k, T=39, K=6):
- Observations: 500k × 39 × 6 × 4B = ~456 MB
- Intermediate buffers (alphas, betas): ~500 MB
- Model parameters: ~1 MB (negligible)
- **Total**: ~1 GB per batch

**Safe batch size**:
- 16GB GPU: 500k pixels (default)
- 40GB+ GPU (H100): 1M+ pixels safe
- 40GB GPU (A100): 2M pixels

**CPU memory** (sequential mode):
- Input chunk (512×512 × 39 years): ~10 MB
- Covariate chunk: ~20 MB
- Minimal (~100 MB total)

### 9.3 Optimization Strategies

**Completed**:
- ✅ Vectorized NumPy → JAX (100-1000× speedup)
- ✅ Log-space arithmetic (prevents underflow)
- ✅ Persistent file handles (eliminates 898ms open overhead/chunk)
- ✅ Async I/O prefetch (read overlaps GPU)
- ✅ Async writes (write overlaps GPU)
- ✅ Stacked treeCover input (6× cold-read speedup)
- ✅ Per-year output files (manageable file sizes at CONUS scale)
- ✅ Chunk-level parallelism (linear scaling to ~8 workers)
- ✅ Pre-chunked .npy I/O (6× inference speedup: ~200ms → ~30ms per chunk)

### 9.4 Pre-chunked I/O Pipeline

For production CONUS runs, a 3-phase pipeline eliminates GeoTIFF I/O from the inference hot loop:

**Phase 0** (`tide prechunk`): Reads all input rasters once with multi-threaded I/O, writes per-chunk `.npy` files. Each chunk directory contains `obs.npy` (N,T), `valid_indices.npy` (N,2), `meta.npy`, and optional covariate arrays.

**Phase 1** (`tide run --prechunked`): `_run_prechunked()` in `orchestrator.py` reads `.npy` files (~2ms), runs GPU inference via `infer_chunk_prechunked()` (~6ms), writes `result_states.npy` and `result_posteriors.npy` (~25ms). Same async prefetch + async write pattern as single-pass mode.

**Phase 2** (`tide assemble`): Reads per-chunk result `.npy` files, writes into standard GeoTIFF output products via existing `write_chunk_results()`.

**Validated on CONUS** (2026-02-16): Pre-chunking completed in 846s (14 min, 6 threads, 25,130 chunks, 4.9B pixels).

**TODO**:
- ⬜ Multi-GPU: Split chunks across GPUs (requires Dask GPU cluster)
- ⬜ COG overviews: Add pyramid levels for faster visualization

---

## 10. Command-Line Interface

### 10.1 CLI Structure

**Framework**: Click (decorator-based)

**Entry point**: `tide` (registered in `pyproject.toml`)

**Commands**:
```bash
tide --help                     # Show available commands
tide run --help                 # Show run options
tide prechunk --help            # Show pre-chunk options
tide assemble --help            # Show reassembly options
tide inspect-params --help      # Show inspect options
tide stats --help               # Show stats options
```

### 10.2 `run` Command

**Full signature**:
```bash
tide run \
  --data-dir data \                         # Root data directory
  --output-dir output_v2 \                  # Output directory
  --n-states 6 \                            # Number of states (default: 6)
  --chunk-size 512 \                        # Tile size (pixels)
  --batch-pixels 500000 \                   # GPU batch size
  --em-max-iter 100 \                       # EM iterations
  --em-sample 2000000 \                     # Pixels to sample for EM
  --skip-em \                               # Skip EM, load from file
  --params-file output_v2/params.npz \      # Params save/load path
  --dask \                                  # Use Dask distributed
  --workers 8 \                             # Number of workers
  --no-posteriors \                         # Skip posterior computation
  --dynamic-transitions \                   # Enable covariate-driven transitions
  --mtbs-dir data/mtbs \                    # MTBS directory (optional)
  --spei-dir data/spei \                    # SPEI directory (optional)
  --dem-path data/terrain/dem.tif \         # DEM path (optional)
  --chili-path data/terrain/chili.tif \     # CHILI path (optional)
  --aridity-path data/terrain/aridity.tif   # Aridity index path (optional)
```

**Common use cases**:
```bash
# Fresh run with EM
tide run --data-dir data

# Skip EM, use pre-trained params
tide run --data-dir data --skip-em --params-file params.npz

# Enable dynamic transitions
tide run --data-dir data --dynamic-transitions

# Fast test (small sample, no posteriors)
tide run --data-dir data --em-sample 100000 --no-posteriors

# Production run with pre-chunked I/O (6x faster)
tide run --data-dir data --skip-em --params-file params.npz \
    --prechunked --chunks-dir data/chunks

# Dask multi-GPU
tide run --data-dir data --dask --workers 2
```

### 10.3 `prechunk` Command

**Purpose**: Pre-chunk input rasters to `.npy` files for fast inference.

```bash
tide prechunk \
  --data-dir data \                   # Root data directory
  --chunks-dir data/chunks \          # Output directory (default: data_dir/chunks)
  --chunk-size 512 \                  # Tile size (pixels)
  --dynamic-transitions \             # Also pre-chunk covariate data
  --threads 6                         # I/O threads
```

### 10.4 `assemble` Command

**Purpose**: Reassemble per-chunk `.npy` results into output GeoTIFFs.

```bash
tide assemble \
  --chunks-dir data/chunks \          # Directory with chunk results
  --output-dir output \               # Output directory for GeoTIFFs
  --data-dir data \                   # For reference profile
  --n-states 6                        # Number of HMM states
```

### 10.5 `inspect-params` Command

**Purpose**: Inspect learned parameters from a saved .npz file.

```bash
tide inspect-params --params-file output_v2/params.npz
```

**Output**:
```
=== Learned ZIB-HMM Parameters ===

Initial state probabilities:
  State 0: 0.4432
  State 1: 0.2098
  State 2: 0.1297
  State 3: 0.1064
  State 4: 0.1109

Transition matrix:
  [[0.9647 0.0303 0.0037 0.0007 0.0004]
   [0.0831 0.8728 0.0407 0.0023 0.0012]
   [0.0183 0.0404 0.9038 0.0350 0.0025]
   [0.0051 0.0044 0.0261 0.9370 0.0273]
   [0.0031 0.0022 0.0042 0.0127 0.9778]]

Emission parameters:
  pi  (zero-inflation): [0.809  0.154  0.023  0.002  0.0001]
  mu  (Beta mean):      [0.0101 0.0188 0.0624 0.2144 0.5923]
  phi (Beta precision):  [4798.  220.   48.   21.    4.]
```

### 10.6 `stats` Command

**Purpose**: Compute summary statistics from output rasters.

```bash
tide stats --output-dir output_v2
```

**Output**: Writes `output_v2/summary_statistics.json` with:
- Per-year state counts and percentages
- Transition year histograms
- Total pixels processed

---

## 11. Deployment and Environment

### 11.1 System Requirements

**Minimum**:
- Python 3.12+
- 16 GB RAM
- 8 GB GPU (NVIDIA with CUDA 12.x)
- 500 GB storage (for inputs + outputs)

**Recommended**:
- Python 3.12 (tested)
- 64 GB RAM (for large EM samples)
- 24 GB GPU (RTX 3090, 4090, or A5000)
- 2 TB SSD storage

**Production cluster**:
- 8× nodes with 24GB GPUs
- Dask scheduler + workers
- Shared filesystem (NFS or parallel FS)

### 11.2 Installation

```bash
# Clone repository
git clone <repo_url>
cd sb_treecover

# Activate conda environment (recommended)
source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate zibhmm

# Or create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Verify
tide --help
```

**CUDA setup**:
```bash
# Check CUDA version
nvcc --version  # Should be 12.x

# Verify JAX sees GPU
python -c "import jax; print(jax.devices())"
# Expected: [cuda(id=0)]
```

### 11.3 Data Preparation

**Step 1: Download RAP tree cover** (not included in repo):
```bash
# Download from USGS or GEE
# Expected structure:
data/treeCover/treeCover_1986.tif
data/treeCover/treeCover_1987.tif
...
data/treeCover/treeCover_2024.tif
```

**Step 2: Create mask**:
```bash
# Generate drylands mask from NLCD or custom source
# Place in:
data/mask/drylands_mask.tif
```

**Step 3: (Optional) Download covariates**:
```bash
# If using dynamic transitions, download:
data/mtbs/mtbs_1986.tif ... mtbs_2024.tif  # Fire severity
data/spei/spei_1986.tif ... spei_2024.tif  # Drought
data/terrain/dem.tif                        # Elevation
data/terrain/chili.tif                      # Insolation
```

**Step 4: (Optional) Mosaic GEE tiles**:
```bash
# If data is in tiles from GEE export:
python scripts/mosaic_gee_tiles.py --base-dir ../..
```

### 11.4 Running the Pipeline

**Test run** (quick validation):
```bash
tide run \
  --data-dir data \
  --output-dir output_test \
  --em-sample 100000 \
  --chunk-size 256 \
  --no-posteriors
```

**Production run** (full dataset):
```bash
tide run \
  --data-dir data \
  --output-dir output_v2_full \
  --dask \
  --workers 8 \
  --dynamic-transitions
```

**Monitor progress**:
```bash
tail -f output_v2_full/pipeline.log
```

---

## 12. Troubleshooting

### 12.1 Common Issues

**Out of memory (GPU)**:
```
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED
```
**Solution**: Reduce `--batch-pixels`:
```bash
tide run --batch-pixels 250000  # Half the default
```

**Out of memory (CPU)**:
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce `--em-sample` or `--chunk-size`:
```bash
tide run --em-sample 1000000 --chunk-size 256
```

**JAX not finding GPU**:
```python
print(jax.devices())  # [cpu(id=0)]
```
**Solution**: Reinstall jax with CUDA:
```bash
pip uninstall jax jaxlib
pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Rasterio can't open file**:
```
rasterio.errors.RasterioIOError: data/treeCover/treeCover_1986.tif: No such file
```
**Solution**: Check paths, ensure data is downloaded and `--data-dir` points to parent of `treeCover/`.

**Dask workers not connecting**:
```
distributed.comm.core.CommClosedError: Connection refused
```
**Solution**: Check firewall, ensure workers can reach scheduler. Try sequential mode first:
```bash
tide run --workers 1  # No Dask, sequential
```

### 12.2 Debugging

**Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Run without JIT** (slower but easier to debug):
```python
import jax
jax.config.update('jax_disable_jit', True)
```

**Profile GPU usage**:
```bash
nvidia-smi dmon  # Real-time monitoring
```

**Profile CPU/memory**:
```bash
python -m cProfile -o profile.stats -m tide.cli run ...
python -m pstats profile.stats
```

### 12.3 Known Limitations

1. **No spatial autocorrelation**: Pixels are independent. Could add MRF priors.
2. **Fixed state count**: K=6 is the default. Could generalize to arbitrary K more easily.
3. **No missing data imputation**: Gaps are marginalized, not filled.
4. **Rasterio GIL**: Rasterio holds the GIL during reads, limiting per-thread parallelism (mitigated by pre-chunked .npy pipeline).

---

## 13. Future Development

### 13.1 Short-term (< 1 month)

- [x] Pre-chunked .npy I/O pipeline (`tide prechunk`, `tide run --prechunked`, `tide assemble`)
- [x] JIT warmup before main inference loop
- [ ] Add COG overview pyramids to outputs
- [ ] Create Jupyter notebook tutorial

### 13.2 Medium-term (1-3 months)

- [ ] Support variable state counts (K=3,4,5,6)
- [ ] Add spatial MRF smoothing (optional)
- [ ] Multi-GPU parallelization
- [ ] Benchmark against other HMM libraries (hmmlearn, pomegranate)

### 13.3 Long-term (3+ months)

- [ ] Semi-Markov extension (state duration modeling)
- [ ] Multi-species extension (joint tree + sagebrush HMM)
- [ ] Web-based visualization dashboard
- [ ] Cloud deployment (AWS/GCP with auto-scaling)

---

## 14. References and Resources

**JAX documentation**:
- https://jax.readthedocs.io/en/latest/
- https://jax.readthedocs.io/en/latest/notebooks/quickstart.html

**Rasterio**:
- https://rasterio.readthedocs.io/en/latest/

**HMM theory**:
- Rabiner, L. R. (1989). A tutorial on hidden Markov models. Proceedings of the IEEE, 77(2), 257-286.

**Zero-Inflated Beta regression**:
- Ospina, R., & Ferrari, S. L. (2012). A general class of zero-or-one inflated beta regression models. Computational Statistics & Data Analysis, 56(6), 1609-1623.

**Baum-Welch EM**:
- Bilmes, J. A. (1998). A gentle tutorial of the EM algorithm and its application to parameter estimation for Gaussian mixture and hidden Markov models. International Computer Science Institute, 4(510), 126.

---

**Document version**: 2.0
**Last updated**: 2026-02-16
**Authors**: SLM
