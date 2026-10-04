"""Stage 0b: turn a sensitivity profile into a compression plan, and predict what the plan costs.

Layer sensitivity >= sensitive_threshold -> protected: `protected_bits`, no pruning.
Otherwise                                -> compressed: `compressed_bits`, pruned at prune_ratio_aggressive.
Guarded layers (the ones whose removal breaks the model, see `guarded_layers`) are never pruned, whatever
their sensitivity score says; they keep the bits the plan gives them.

The "original method" allocation used for comparison is uniform: every layer gets the same bits and pruning.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from sdf.search_space import PER_CHANNEL
from sdf.stage0.sensitivity import SensitivityProfile


@dataclass(frozen=True)
class LayerPlan:
    layer: int
    bit_width: int
    pruning_ratio: float
    protected: bool
    sensitivity: float
    guarded: bool = False  # never pruned: removing this layer breaks the model



@dataclass(frozen=True)
class CompressionPlan:
    layers: tuple[LayerPlan, ...]
    kind: str  # "sensitivity" | "budget" | "uniform" | "same_size_pruning"
    sensitive_threshold: float | None
    prune_ratio_aggressive: float

    @property
    def protected_layers(self) -> list[int]:
        return [lp.layer for lp in self.layers if lp.protected]

    @property
    def compressed_layers(self) -> list[int]:
        return [lp.layer for lp in self.layers if not lp.protected]

    @property
    def guarded_layers(self) -> list[int]:
        return [lp.layer for lp in self.layers if lp.guarded]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sensitive_threshold": self.sensitive_threshold,
            "prune_ratio_aggressive": self.prune_ratio_aggressive,
            "layers": [asdict(lp) for lp in self.layers],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompressionPlan":
        return cls(tuple(LayerPlan(**lp) for lp in d["layers"]), d["kind"], d["sensitive_threshold"],
                   d["prune_ratio_aggressive"])

    @classmethod
    def load(cls, path: str | Path) -> "CompressionPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def guarded_layers(removal_scores: list[float], top_k: int) -> frozenset[int]:
    """The `top_k` layers whose removal raises perplexity most. Pruning them risks breaking the model (on
    TinyLlama, skipping layer 0 takes perplexity from 14 to ~1190), so no plan prunes them. Takes raw
    layer-removal scores, so it works for any model; top_k = 0 turns the guard off."""
    if top_k < 0:
        raise ValueError(f"guard top_k must be >= 0, got {top_k}")
    ranked = sorted(range(len(removal_scores)), key=lambda i: removal_scores[i], reverse=True)
    return frozenset(ranked[:top_k])


def _layer(i: int, s: float, protected: bool, protected_bits: int, compressed_bits: int, prune_ratio: float,
           guarded: frozenset[int]) -> LayerPlan:
    return LayerPlan(layer=i, bit_width=protected_bits if protected else compressed_bits,
                     pruning_ratio=0.0 if protected or i in guarded else prune_ratio,
                     protected=protected, sensitivity=s, guarded=i in guarded)


def plan_compression(
    scores: list[float],
    sensitive_threshold: float,
    prune_ratio_aggressive: float,
    protected_bits: int,
    compressed_bits: int,
    guarded: frozenset[int] = frozenset(),
) -> CompressionPlan:
    _check_ratio(prune_ratio_aggressive)
    layers = tuple(_layer(i, s, s >= sensitive_threshold, protected_bits, compressed_bits, prune_ratio_aggressive,
                          guarded) for i, s in enumerate(scores))
    return CompressionPlan(layers, "sensitivity", sensitive_threshold, prune_ratio_aggressive)


def uniform_plan(scores: list[float], bits: int, prune_ratio: float) -> CompressionPlan:
    _check_ratio(prune_ratio)
    layers = tuple(LayerPlan(layer=i, bit_width=bits, pruning_ratio=prune_ratio, protected=False, sensitivity=s)
                   for i, s in enumerate(scores))
    return CompressionPlan(layers, "uniform", None, prune_ratio)


def budget_matched_plan(
    scores: list[float],
    budget_gb: float,
    prune_ratio_aggressive: float,
    protected_bits: int,
    compressed_bits: int,
    cost: "Callable[[CompressionPlan], PlanCost]",
    guarded: frozenset[int] = frozenset(),
) -> CompressionPlan:
    """The sensitivity plan that fits in `budget_gb` (normally the uniform plan's predicted size).

    Protects the k most sensitive layers, with k as large as the budget allows; every other layer is
    compressed and pruned as usual (guarded layers are not pruned, and that is counted in the budget). This makes the framework-vs-original comparison size-for-size fair: any
    accuracy difference then comes from *where* the bits go, not from spending more of them.
    """
    _check_ratio(prune_ratio_aggressive)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    best = None
    for k in range(len(ranked) + 1):
        protected = set(ranked[:k])
        layers = tuple(_layer(i, s, i in protected, protected_bits, compressed_bits, prune_ratio_aggressive,
                              guarded) for i, s in enumerate(scores))
        plan = CompressionPlan(layers, "budget", None, prune_ratio_aggressive)
        if cost(plan).weight_memory_gb > budget_gb * (1 + 1e-9):
            break
        best = plan
    if best is None:  # even protecting nothing is over budget: return the k = 0 plan, caller flags it
        best = CompressionPlan(tuple(_layer(i, s, False, protected_bits, compressed_bits, prune_ratio_aggressive,
                                            guarded) for i, s in enumerate(scores)),
                               "budget", None, prune_ratio_aggressive)
    return best


def same_size_pruning_plan(
    scores: list[float],
    bits: int,
    prune_ratio: float,
    layer_numel: list[int],
    guarded: frozenset[int] = frozenset(),
) -> CompressionPlan:
    """The fair pruning test: every layer at `bits` like `uniform_plan(scores, bits, prune_ratio)`, and the same
    total number of weights removed, but placed by sensitivity instead of evenly.

    Layer l is pruned at min(1, c * (1 - s_l + 1/(n-1))), guarded layers not at all, with c chosen so that the
    weights removed equal prune_ratio * all decoder weights. Same bits and same weights kept means the same
    predicted size and the same share removed as the uniform plan, so any accuracy difference comes only from
    where the pruning goes. When the unguarded layers can't absorb it all (high ratios), they are emptied and
    the plan stays bigger than uniform; the caller sees that in the predicted cost.
    """
    _check_ratio(prune_ratio)
    n = len(scores)
    weight = [0.0 if i in guarded else 1.0 - s + 1.0 / max(n - 1, 1) for i, s in enumerate(scores)]
    target = prune_ratio * sum(layer_numel)
    removed = lambda c: sum(min(1.0, c * w) * m for w, m in zip(weight, layer_numel))  # noqa: E731
    lo, hi = 0.0, 1.0
    while removed(hi) < target and hi < 1e6:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if removed(mid) < target else (lo, mid)
    layers = tuple(LayerPlan(layer=i, bit_width=bits, pruning_ratio=min(1.0, hi * w), protected=False,
                             sensitivity=s, guarded=i in guarded) for i, (s, w) in enumerate(zip(scores, weight)))
    return CompressionPlan(layers, "same_size_pruning", None, prune_ratio)


@dataclass(frozen=True)
class PlanCost:
    """Predicted cost of a plan, before any compression is actually run."""

    weight_memory_gb: float
    avg_bits_per_weight: float  # over every parameter, including the ones kept at baseline precision
    sparsity: float  # share of all parameters pruned
    sensitivity_exposure: float  # sum_l s_l * (1 - eff_bits_l / baseline) / sum_l s_l
    per_layer_mb: tuple[float, ...]
    fixed_memory_gb: float = 0.0  # parts kept at baseline precision in every plan (embeddings, LM head, norms)


def predict_cost(
    plan: CompressionPlan,
    profile: SensitivityProfile,
    group_size: int,
    group_overhead_bits: int,
    baseline_bits: int,
) -> PlanCost:
    """Ideal storage of a plan.

    Per compressed layer: kept weights x bits, plus one scale/zero pair (`group_overhead_bits`) per quantisation
    group of kept weights (group_size weights, or one per output channel when group_size == PER_CHANNEL). Pruned weights are
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
            # Scales for kept weights only: pruned rows/columns are gone (structured), so their groups are too.
            groups = rows if group_size == PER_CHANNEL else numel / group_size
            bits += groups * (1.0 - lp.pruning_ratio) * group_overhead_bits
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
        fixed_memory_gb=profile.other_numel * baseline_bits / 8 / 1e9,
    )


def baseline_cost(profile: SensitivityProfile, baseline_bits: int) -> PlanCost:
    total_numel = sum(profile.layer_numel) + profile.other_numel
    return PlanCost(
        weight_memory_gb=total_numel * baseline_bits / 8 / 1e9,
        avg_bits_per_weight=float(baseline_bits),
        sparsity=0.0,
        sensitivity_exposure=0.0,
        per_layer_mb=tuple(n * baseline_bits / 8 / 1e6 for n in profile.layer_numel),
        fixed_memory_gb=profile.other_numel * baseline_bits / 8 / 1e9,
    )


def _check_ratio(r: float) -> None:
    if not 0.0 <= r <= 1.0:  # 1.0 empties the layer's Linear weights: the layer only passes its input on
        raise ValueError(f"pruning ratio must be in [0, 1], got {r}")
