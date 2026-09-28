"""Stage 0b: turn a sensitivity profile into a compression plan, plus the Stage 0 report.

Layer sensitivity >= threshold  -> protected: 8-bit, no pruning.
Otherwise                       -> compressed: 4-bit, pruned at the robust-layer pruning ratio.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sdf.stage0.sensitivity import SensitivityProfile

PROTECTED_BITS = 8
COMPRESSED_BITS = 4


@dataclass(frozen=True)
class LayerPlan:
    layer: int
    bit_width: int
    pruning_ratio: float
    protected: bool
    sensitivity: float


@dataclass(frozen=True)
class CompressionPlan:
    layers: tuple[LayerPlan, ...]
    sensitivity_threshold: float
    robust_pruning_ratio: float

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, i: int) -> LayerPlan:
        return self.layers[i]

    @property
    def protected_layers(self) -> list[int]:
        return [lp.layer for lp in self.layers if lp.protected]

    @property
    def compressed_layers(self) -> list[int]:
        return [lp.layer for lp in self.layers if not lp.protected]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensitivity_threshold": self.sensitivity_threshold,
            "robust_pruning_ratio": self.robust_pruning_ratio,
            "layers": [asdict(lp) for lp in self.layers],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompressionPlan":
        return cls(
            layers=tuple(LayerPlan(**lp) for lp in d["layers"]),
            sensitivity_threshold=d["sensitivity_threshold"],
            robust_pruning_ratio=d["robust_pruning_ratio"],
        )


def plan_compression(
    profile: SensitivityProfile, sensitivity_threshold: float, pruning_ratio: float
) -> CompressionPlan:
    if not 0.0 <= pruning_ratio < 1.0:
        raise ValueError(f"pruning_ratio must be in [0, 1), got {pruning_ratio}")
    layers = []
    for i, s in enumerate(profile.scores):
        protected = s >= sensitivity_threshold
        layers.append(
            LayerPlan(
                layer=i,
                bit_width=PROTECTED_BITS if protected else COMPRESSED_BITS,
                pruning_ratio=0.0 if protected else pruning_ratio,
                protected=protected,
                sensitivity=s,
            )
        )
    return CompressionPlan(
        layers=tuple(layers),
        sensitivity_threshold=sensitivity_threshold,
        robust_pruning_ratio=pruning_ratio,
    )


def stage0_report(profile: SensitivityProfile, plan: CompressionPlan) -> dict[str, Any]:
    """Stage 0 report: profile, protected vs compressed, bit / prune distributions, profiling cost."""
    n = len(plan)
    n_protected = len(plan.protected_layers)
    return {
        "sensitivity_method": profile.method,
        "sensitivity_profile": profile.scores,
        "num_layers": n,
        "protected": {"count": n_protected, "percent": 100.0 * n_protected / n if n else 0.0,
                      "layers": plan.protected_layers},
        "compressed": {"count": n - n_protected, "percent": 100.0 * (n - n_protected) / n if n else 0.0,
                       "layers": plan.compressed_layers},
        "bit_width_distribution": {str(k): v for k, v in sorted(Counter(lp.bit_width for lp in plan.layers).items())},
        "pruning_ratio_distribution": {str(k): v for k, v in sorted(Counter(lp.pruning_ratio for lp in plan.layers).items())},
        "profiling_cost": profile.cost,
    }
