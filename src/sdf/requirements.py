"""DeploymentRequirement: the targets a compressed model must meet, and the shortfall when it doesn't."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

# requirement field -> metric names it is checked against, in order of preference.
# All targets are upper bounds (lower is better for every one of these metrics).
_CHECKS: dict[str, tuple[str, ...]] = {
    "target_latency_ms": ("decode_ms_per_token_mean",),
    "target_memory_gb": ("model_size_gb", "predicted_weight_memory_gb"),
    "target_ppl": ("ppl_val",),
    "kv_budget_gb": ("kv_cache_gb",),
}


@dataclass
class RequirementCheck:
    met: bool | None  # None = nothing failed but at least one target could not be measured yet
    shortfall: dict[str, float] = field(default_factory=dict)  # target -> amount over the target
    checked: dict[str, str] = field(default_factory=dict)  # target -> metric it was checked against
    unmeasured: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentRequirement:
    target_latency_ms: float | None = None  # decode latency per token
    target_memory_gb: float | None = None
    target_ppl: float | None = None  # accuracy target, as a maximum validation-half perplexity
    kv_budget_gb: float | None = None
    hardware_profile: str = "unspecified"  # e.g. "colab-T4", "A100-40GB"

    def check(self, metrics: Mapping[str, Any]) -> RequirementCheck:
        result = RequirementCheck(met=True)
        for target_name, metric_names in _CHECKS.items():
            target = getattr(self, target_name)
            if target is None:
                continue
            metric = next((m for m in metric_names if _is_number(metrics.get(m))), None)
            if metric is None:
                result.unmeasured.append(target_name)
                continue
            result.checked[target_name] = metric
            over = float(metrics[metric]) - float(target)
            if over > 0:
                result.shortfall[target_name] = over
        if result.shortfall:
            result.met = False
        elif result.unmeasured:
            result.met = None
        return result


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x  # x == x rejects NaN
