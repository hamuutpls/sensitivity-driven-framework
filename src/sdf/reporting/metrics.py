"""Metric registry: display label, unit and which direction is better. One schema for every stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MetricSpec:
    label: str
    unit: str = ""
    better: Literal["lower", "higher"] | None = None  # None: informational, no win/lose


METRICS: dict[str, MetricSpec] = {
    # accuracy
    "ppl_val": MetricSpec("Perplexity (validation half)", "", "lower"),
    "ppl_heldout": MetricSpec("Perplexity (held-out half)", "", "lower"),
    # memory
    "model_size_gb": MetricSpec("Model size on disk", "GB", "lower"),
    "predicted_weight_memory_gb": MetricSpec("Predicted weight memory", "GB", "lower"),
    "peak_memory_gb": MetricSpec("Peak GPU memory", "GB", "lower"),
    "kv_cache_gb": MetricSpec("KV-cache memory", "GB", "lower"),
    # latency
    "prefill_ms_mean": MetricSpec("Prefill latency (mean)", "ms", "lower"),
    "prefill_ms_std": MetricSpec("Prefill latency (std)", "ms"),
    "decode_ms_per_token_mean": MetricSpec("Decode latency per token (mean)", "ms", "lower"),
    "decode_ms_per_token_std": MetricSpec("Decode latency per token (std)", "ms"),
    "prefill_tokens_per_s": MetricSpec("Prefill throughput", "tok/s", "higher"),
    "decode_tokens_per_s": MetricSpec("Decode throughput", "tok/s", "higher"),
    # build cost
    "build_time_s": MetricSpec("Build time", "s", "lower"),
    # Stage 0 allocation
    "avg_bits_per_weight": MetricSpec("Average bits per weight", "bits", "lower"),
    "sparsity": MetricSpec("Weight sparsity", "", None),
    "protected_layers": MetricSpec("Protected layers", "", None),
    # Sensitivity-weighted compression: sum_l s_l * (1 - effective_bits_l / 16) / sum_l s_l, where
    # effective_bits = bits * (1 - prune ratio). Lower = less compression lands on sensitive layers.
    "sensitivity_exposure": MetricSpec("Sensitivity exposure", "", "lower"),
}


def is_better(metric: str, delta: float) -> bool | None:
    spec = METRICS.get(metric)
    if spec is None or spec.better is None or delta == 0:
        return None
    return delta < 0 if spec.better == "lower" else delta > 0
