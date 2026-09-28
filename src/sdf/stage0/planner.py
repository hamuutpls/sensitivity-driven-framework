"""Stage 0b: turn a sensitivity profile into a compression plan, and predict what the plan costs.

Layer sensitivity >= sensitive_threshold -> protected: `protected_bits`, no pruning.
Otherwise                                -> compressed: `compressed_bits`, pruned at prune_ratio_aggressive.

The "original method" allocation used for comparison is uniform: every layer gets the same bits and pruning.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sdf.search_space import PER_CHANNEL
from sdf.stage0.sensitivity import SensitivityProfile


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
    kind: str  # "sensitivity" | "uniform"
    sensitive_threshold: float | None
    prune_ratio_aggressive: float

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
            "kind": self.kind,
            "sensitive_threshold": self.sensitive_threshold,
            "prune_ratio_aggressive": self.prune_ratio_aggressive,
            "layers": [asdict(lp) for lp in self.layers],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompressionPlan":
        return cls(
            layers=tuple(LayerPlan(**lp) for lp in d["layers"]),
            kind=d["kind"],
            sensitive_threshold=d["sensitive_threshold"],
            prune_ratio_aggressive=d["prune_ratio_aggressive"],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def plan_compression(
    profile: SensitivityProfile,
    sensitive_threshold: float,
    prune_ratio_aggressive: float,
    protected_bits: int,
    compressed_bits: int,
) -> CompressionPlan:
    _check_ratio(prune_ratio_aggressive)
    layers = []
    for i, s in enumerate(profile.scores):
        protected = s >= sensitive_threshold
        layers.append(LayerPlan(
            layer=i,
            bit_width=protected_bits if protected else compressed_bits,
            pruning_ratio=0.0 if protected else prune_ratio_aggressive,
            protected=protected,
            sensitivity=s,
        ))
    return CompressionPlan(tuple(layers), "sensitivity", sensitive_threshold, prune_ratio_aggressive)


def uniform_plan(profile: SensitivityProfile, bits: int, prune_ratio: float) -> CompressionPlan:
    _check_ratio(prune_ratio)
    layers = tuple(LayerPlan(layer=i, bit_width=bits, pruning_ratio=prune_ratio, protected=False, sensitivity=s)
                   for i, s in enumerate(profile.scores))
    return CompressionPlan(layers, "uniform", None, prune_ratio)


@dataclass(frozen=True)
class PlanCost:
    """Predicted cost of a plan, before any compression is actually run."""

    weight_memory_gb: float
    avg_bits_per_weight: float  # over every parameter, including the ones kept at baseline precision
    sparsity: float  # share of all parameters pruned
    sensitivity_exposure: float  # sum_l s_l * (1 - eff_bits_l / baseline) / sum_l s_l
    per_layer_mb: tuple[float, ...]


def predict_cost(
    plan: CompressionPlan,
    profile: SensitivityProfile,
    group_size: int,
    group_overhead_bits: int,
    baseline_bits: int,
) -> PlanCost:
    """Ideal storage of a plan.

    Per compressed layer: kept weights x bits, plus one scale/zero pair (`group_overhead_bits`) per quantisation
    group (group_size weights, or one per output channel when group_size == PER_CHANNEL). Pruned weights are
    assumed to be stored for free, which holds for structured pruning; unstructured sparsity needs a sparse
    format to realise it. Parameters outside the decoder Linear weights stay at `baseline_bits`.
    """
    if not profile.layer_numel:
        raise ValueError("profile has no layer sizes; re-run profiling with this version")
    per_layer_bits = []
    exposure_num = 0.0
    total_pruned = 0.0
    for lp, numel, rows in zip(plan.layers, profile.layer_numel, profile.layer_rows):
        kept = numel * (1.0 - lp.pruning_ratio)
        bits = kept * lp.bit_width
        if lp.bit_width < baseline_bits:
            groups = rows if group_size == PER_CHANNEL else numel / group_size
            bits += groups * group_overhead_bits
        per_layer_bits.append(bits)
        total_pruned += numel - kept
        eff_bits = lp.bit_width * (1.0 - lp.pruning_ratio)
        exposure_num += lp.sensitivity * (1.0 - eff_bits / baseline_bits)

    total_numel = sum(profile.layer_numel) + profile.other_numel
    total_bits = sum(per_layer_bits) + profile.other_numel * baseline_bits
    sens_total = sum(lp.sensitivity for lp in plan.layers)
    return PlanCost(
        weight_memory_gb=total_bits / 8 / 1e9,
        avg_bits_per_weight=total_bits / total_numel,
        sparsity=total_pruned / total_numel,
        sensitivity_exposure=exposure_num / sens_total if sens_total > 0 else 0.0,
        per_layer_mb=tuple(b / 8 / 1e6 for b in per_layer_bits),
    )


def baseline_cost(profile: SensitivityProfile, baseline_bits: int) -> PlanCost:
    total_numel = sum(profile.layer_numel) + profile.other_numel
    return PlanCost(
        weight_memory_gb=total_numel * baseline_bits / 8 / 1e9,
        avg_bits_per_weight=float(baseline_bits),
        sparsity=0.0,
        sensitivity_exposure=0.0,
        per_layer_mb=tuple(n * baseline_bits / 8 / 1e6 for n in profile.layer_numel),
    )


def _check_ratio(r: float) -> None:
    if not 0.0 <= r < 1.0:
        raise ValueError(f"pruning ratio must be in [0, 1), got {r}")
