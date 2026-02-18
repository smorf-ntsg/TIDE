"""Configuration dataclasses for TIDE pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EmissionConfig:
    """ZIB emission configuration."""
    n_states: int = 6
    eps: float = 1e-6  # Clamp boundary for Beta PDF


@dataclass(frozen=True)
class EMConfig:
    """Baum-Welch EM configuration."""
    max_iter: int = 100
    tol: float = 1e-4  # Relative log-likelihood convergence
    n_sample_pixels: int = 2_000_000  # Stratified sample for EM
    lbfgs_maxiter: int = 50  # Inner L-BFGS iterations for M-step
    l2_reg: float = 1e-3  # L2 regularization for transition weights
    weight_max_iter: int = 20  # Max iterations for weight estimation
    weight_tol: float = 1e-3  # Convergence tolerance for weight estimation


@dataclass(frozen=True)
class InferenceConfig:
    """Full-dataset inference configuration."""
    chunk_size: int = 512  # Spatial tile dimension (512x512)
    batch_pixels: int = 500_000  # Max pixels per GPU batch
    compute_posteriors: bool = True
    compute_xi: bool = False  # Only needed during EM


@dataclass(frozen=True)
class IOConfig:
    """I/O configuration."""
    compression: str = "deflate"
    tile_size: int = 256
    cog: bool = True  # Cloud-Optimized GeoTIFF
    multi_band: bool = True  # One file with T bands vs T files


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""
    data_dir: Path = Path("data")
    output_dir: Path = Path("output_v2")
    years: tuple[int, ...] = tuple(range(1986, 2025))

    # Covariate paths (None → derived from data_dir when dynamic transitions enabled)
    mtbs_dir: Path | None = None
    spei_dir: Path | None = None
    dem_path: Path | None = None
    chili_path: Path | None = None
    aridity_path: Path | None = None
    enable_dynamic_transitions: bool = False

    emission: EmissionConfig = field(default_factory=EmissionConfig)
    em: EMConfig = field(default_factory=EMConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    io: IOConfig = field(default_factory=IOConfig)

    @property
    def mask_path(self) -> Path:
        return self.data_dir / "mask" / "drylands_mask.tif"

    @property
    def n_years(self) -> int:
        return len(self.years)

    @property
    def input_paths(self) -> list[Path]:
        return [
            self.data_dir / "treeCover" / f"treeCover_{y}.tif"
            for y in self.years
        ]

    @property
    def effective_mtbs_dir(self) -> Path | None:
        if self.mtbs_dir is not None:
            return self.mtbs_dir
        if self.enable_dynamic_transitions:
            return self.data_dir / "mtbs"
        return None

    @property
    def effective_spei_dir(self) -> Path | None:
        if self.spei_dir is not None:
            return self.spei_dir
        if self.enable_dynamic_transitions:
            return self.data_dir / "spei"
        return None

    @property
    def effective_dem_path(self) -> Path | None:
        if self.dem_path is not None:
            return self.dem_path
        if self.enable_dynamic_transitions:
            return self.data_dir / "terrain" / "dem.tif"
        return None

    @property
    def effective_chili_path(self) -> Path | None:
        if self.chili_path is not None:
            return self.chili_path
        if self.enable_dynamic_transitions:
            return self.data_dir / "terrain" / "chili.tif"
        return None

    @property
    def effective_aridity_path(self) -> Path | None:
        if self.aridity_path is not None:
            return self.aridity_path
        if self.enable_dynamic_transitions:
            return self.data_dir / "terrain" / "aridity.tif"
        return None


# Default 6-state initial parameters (ecological priors for bidirectional drylands model)
# States: Bare (~0%), Trace (0-4%), Sparse (4-10%), Open (10-25%), Woodland (25-50%), Forest (50%+)
DEFAULT_INIT_PI = [0.90, 0.20, 0.05, 0.01, 0.005, 0.001]
DEFAULT_INIT_MU = [0.005, 0.02, 0.07, 0.18, 0.38, 0.60]
DEFAULT_INIT_PHI = [100.0, 50.0, 30.0, 15.0, 10.0, 5.0]

# Initial state distribution (most pixels start low-cover, but meaningful forest fraction in 1986)
DEFAULT_INIT_PROBS = [0.50, 0.15, 0.12, 0.10, 0.08, 0.05]

# Default static transition matrix (6x6, bidirectional with high self-persistence)
# Allows both upward (encroachment) and downward (fire/dieback) transitions
DEFAULT_TRANS = [
    [0.950, 0.030, 0.012, 0.005, 0.002, 0.001],  # Bare
    [0.020, 0.930, 0.035, 0.010, 0.003, 0.002],  # Trace
    [0.010, 0.020, 0.930, 0.030, 0.007, 0.003],  # Sparse
    [0.005, 0.008, 0.017, 0.940, 0.025, 0.005],  # Open
    [0.003, 0.004, 0.006, 0.017, 0.950, 0.020],  # Woodland
    [0.002, 0.003, 0.005, 0.010, 0.010, 0.970],  # Forest
]
