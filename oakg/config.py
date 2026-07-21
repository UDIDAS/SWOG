"""Global experiment configuration for OAKG.

All thresholds, seeds, and paths that must be fixed for reproducibility live
here. The guidelines require every table/figure to be regenerable from the saved
configuration, so nothing experiment-defining should be hard-coded elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Anatomical vocabulary used across masking, phenotype support, and graphs.
ORGANS: tuple[str, ...] = (
    "liver",
    "pancreas",
    "spleen",
    "left_kidney",
    "right_kidney",
)

# Default top-k for all ranking metrics.
TOP_K = 10


@dataclass
class Config:
    """Fixed, serializable experiment configuration.

    Instances are JSON-serializable via :meth:`to_dict` so the exact settings
    behind any result can be archived alongside the output CSVs.
    """

    seed: int = 2027
    top_k: int = TOP_K
    organs: tuple[str, ...] = ORGANS

    # Data source. When ``use_demo_data`` is True the reproducible synthetic
    # generator is used; otherwise CSVs are loaded from ``data_dir``.
    use_demo_data: bool = True
    n_demo_cases: int = 120

    # Optional baselines.
    run_wl_baseline: bool = True
    run_neural_baselines: bool = False  # Enable once graph/CT embeddings exist.

    # Bootstrap repetitions. ``None`` -> auto (fast for demo, 10k for the study).
    n_bootstrap: int | None = None

    # Paths (relative to the repository root by default).
    data_dir: Path = field(default_factory=lambda: Path("data"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    embeddings_dir: Path = field(default_factory=lambda: Path("embeddings"))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        self.embeddings_dir = Path(self.embeddings_dir)
        if self.n_bootstrap is None:
            self.n_bootstrap = 500 if self.use_demo_data else 10_000

    # -- convenience -----------------------------------------------------
    @property
    def query_level_dir(self) -> Path:
        return self.results_dir / "query_level"

    @property
    def tables_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    def ensure_dirs(self) -> None:
        for d in (self.query_level_dir, self.tables_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "top_k": self.top_k,
            "organs": list(self.organs),
            "use_demo_data": self.use_demo_data,
            "n_demo_cases": self.n_demo_cases,
            "run_wl_baseline": self.run_wl_baseline,
            "run_neural_baselines": self.run_neural_baselines,
            "n_bootstrap": self.n_bootstrap,
            "data_dir": str(self.data_dir),
            "results_dir": str(self.results_dir),
            "embeddings_dir": str(self.embeddings_dir),
        }
