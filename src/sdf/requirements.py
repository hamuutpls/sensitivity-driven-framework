"""DeploymentRequirement: the targets a compressed model must meet, and the shortfall when it doesn't."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from sdf.reporting.metrics import is_number

# requirement field -> metric names it is checked against, in order of preference.
# All targets are upper bounds (lower is better for every one of these metrics).
_CHECKS: dict[str, tuple[str, ...]] = {
    "target_latency_ms": ("decode_ms_per_token_mean",),
    "target_memory_gb": ("model_size_gb", "predicted_weight_memory_gb"),
    "target_ppl": ("ppl_val",),
    "target_vllm_ppl": ("vllm_ppl_val",),
    "kv_budget_gb": ("kv_cache_gb", "predicted_kv_memory_gb"),
}


LATENCY = _CHECKS["target_latency_ms"][0]  # the Hugging Face decode latency the limit is checked on by default
PPL = _CHECKS["target_ppl"][0]  # the Hugging Face validation perplexity the limit is checked on by default


def with_limit_metrics(metrics: Mapping[str, Any], latency_metric: str = LATENCY,
                       ppl_metric: str = PPL) -> dict[str, Any]:
    """`metrics` with the latency and perplexity the limits are checked on taken from `latency_metric` and `ppl_metric`
    (search.latency_metric, search.ppl_metric, e.g. "vllm_decode_ms_per_token", "vllm_ppl_val"): a row without one
    leaves that limit unmeasured, never the HF value instead (2026-10-29: the vLLM limit of 2.5 ms per token against an
    HF decode of about 30 ms)."""
    out = dict(metrics)
    for default, metric in ((LATENCY, latency_metric), (PPL, ppl_metric)):
        if metric != default:
            out[default] = metrics.get(metric)
    return out


@dataclass
class RequirementCheck:
    met: bool | None  # None = no target set, or nothing failed but a target could not be measured yet
    targets_set: bool = True
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
    # the same on the file vLLM runs, on top of target_ppl: both must hold; without the number it is unmeasured
    target_vllm_ppl: float | None = None
    kv_budget_gb: float | None = None
    hardware_profile: str = "unspecified"  # e.g. "RTX-5070-Ti", "A100-40GB"

    def check(self, metrics: Mapping[str, Any]) -> RequirementCheck:
        result = RequirementCheck(met=True)
        if all(getattr(self, t) is None for t in _CHECKS):
            return RequirementCheck(met=None, targets_set=False)
        for target_name, metric_names in _CHECKS.items():
            target = getattr(self, target_name)
            if target is None:
                continue
            metric = next((m for m in metric_names if is_number(metrics.get(m))), None)
            broken = next((m for m in metric_names
                           if isinstance(metrics.get(m), float) and (math.isnan(metrics[m]) or metrics[m] == math.inf)),
                          None)
            if metric is None and broken is not None:  # a diverged run (NaN/inf ppl) fails, it is not "unmeasured"
                result.checked[target_name] = broken
                result.shortfall[target_name] = math.inf
                continue
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
