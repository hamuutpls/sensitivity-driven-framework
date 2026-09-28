"""Configuration objects: the 5-D search space, one candidate, deployment targets, experiment settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Union

GroupSize = Union[int, str]  # 32, 64, 128 or "per-channel"

PER_CHANNEL = "per-channel"


@dataclass(frozen=True)
class SearchSpace:
    """Bounds of the 5-D search space shared by every searcher."""

    sensitivity_threshold: tuple[float, float] = (0.1, 0.9)
    pruning_ratio: tuple[float, float] = (0.0, 0.6)
    group_sizes: tuple[GroupSize, ...] = (32, 64, 128, PER_CHANNEL)
    migration_strength: tuple[float, float] = (0.5, 0.95)
    cache_bits: tuple[int, ...] = (2, 4, 8)

    def validate(self, cfg: "CandidateConfig") -> None:
        """Raise ValueError if `cfg` lies outside this space."""
        errors = []
        if not _in_range(cfg.sensitivity_threshold, self.sensitivity_threshold):
            errors.append(f"sensitivity_threshold={cfg.sensitivity_threshold} not in {self.sensitivity_threshold}")
        if not _in_range(cfg.pruning_ratio, self.pruning_ratio):
            errors.append(f"pruning_ratio={cfg.pruning_ratio} not in {self.pruning_ratio}")
        if cfg.group_size not in self.group_sizes:
            errors.append(f"group_size={cfg.group_size!r} not in {self.group_sizes}")
        if not _in_range(cfg.migration_strength, self.migration_strength):
            errors.append(f"migration_strength={cfg.migration_strength} not in {self.migration_strength}")
        if cfg.cache_bits not in self.cache_bits:
            errors.append(f"cache_bits={cfg.cache_bits} not in {self.cache_bits}")
        if errors:
            raise ValueError("candidate outside search space: " + "; ".join(errors))


DEFAULT_SEARCH_SPACE = SearchSpace()


@dataclass(frozen=True)
class CandidateConfig:
    """One point in the search space, i.e. what a searcher proposes for a trial.

    Stage 0 consumes sensitivity_threshold and pruning_ratio (the robust-layer pruning ratio),
    Stage 1 group_size, Stage 2 migration_strength and Stage 3 cache_bits.
    """

    sensitivity_threshold: float
    pruning_ratio: float
    group_size: GroupSize = 128
    migration_strength: float = 0.5
    cache_bits: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CandidateConfig":
        return cls(**d)


@dataclass(frozen=True)
class DeploymentTargets:
    """Targets the final Pareto config is checked against. None means no target."""

    latency_ms_per_token: float | None = None
    memory_gb: float | None = None
    accuracy: float | None = None
    kv_memory_budget_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration data used by Stage 0 profiling."""

    dataset: str = "wikitext2"  # wikitext2 | c4 | pile10k
    n_batches: int = 16
    batch_size: int = 1
    seq_len: int = 512
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    targets: DeploymentTargets = field(default_factory=DeploymentTargets)
    search_space: SearchSpace = field(default_factory=SearchSpace)
    device: str = "auto"  # auto | cpu | cuda
    dtype: str = "float32"  # float32 | bfloat16 | float16


def _in_range(x: float, bounds: tuple[float, float]) -> bool:
    lo, hi = bounds
    return lo <= x <= hi
