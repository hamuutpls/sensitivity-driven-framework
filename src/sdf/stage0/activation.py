"""Stage 0 activation plan for Stage 2 (SmoothQuant, QuaRot, RPTQ, SpinQuant): bits per decoder layer for the
numbers flowing into that layer's Linear weights.

Nothing is measured here. The plan reuses the weight plan: a layer the weight plan protects, or that the pruning
guard marks as critical, keeps `protected_bits` activations; every other layer drops to `compressed_bits`. This
assumes a layer fragile for weights is fragile for activations too, which Stage 2 has to check against a
measured per-layer activation sensitivity before the thesis relies on it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sdf.stage0.planner import CompressionPlan


@dataclass(frozen=True)
class ActivationLayerPlan:
    layer: int
    act_bits: int
    protected: bool


@dataclass(frozen=True)
class ActivationPlan:
    layers: tuple[ActivationLayerPlan, ...]
    kind: str  # "uniform" | "from_weight_plan"

    @property
    def avg_bits(self) -> float:
        # Decoder layers of one model share their activation shapes, so the plain mean is the average.
        return sum(lp.act_bits for lp in self.layers) / len(self.layers)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "avg_bits": self.avg_bits, "layers": [asdict(lp) for lp in self.layers]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActivationPlan":
        return cls(tuple(ActivationLayerPlan(**lp) for lp in d["layers"]), d["kind"])

    @classmethod
    def load(cls, path: str | Path) -> "ActivationPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def activation_plan(weights: CompressionPlan, protected_bits: int, compressed_bits: int) -> ActivationPlan:
    return ActivationPlan(tuple(
        ActivationLayerPlan(lp.layer, protected_bits if lp.protected or lp.guarded else compressed_bits,
                            lp.protected or lp.guarded)
        for lp in weights.layers), "from_weight_plan")


def uniform_activation_plan(num_layers: int, bits: int) -> ActivationPlan:
    return ActivationPlan(tuple(ActivationLayerPlan(i, bits, False) for i in range(num_layers)), "uniform")
