"""Metric registry: label, unit, which direction is better, and a plain-language explanation.

One schema for every stage. `plain` and `meaning` are written for a reader with no AI background; the
report uses them to explain each number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


# One-off costs (paid once, before the model is used) do not decide whether the framework beat the original;
# reports state them separately so a framework row is not called a trade-off just for needing profiling.
ONE_OFF_COSTS = frozenset({"build_time_s"})


@dataclass(frozen=True)
class MetricSpec:
    label: str  # technical name, used in tables
    unit: str = ""
    better: Literal["lower", "higher"] | None = None  # None: informational, no win/lose
    plain: str = ""  # short everyday name, e.g. "memory needed to store the model"
    meaning: str = ""  # what it measures and why it matters, in everyday words


_PPL_MEANING = (
    "Perplexity measures how well the model predicts the next word of real text it has not been trained on. "
    "Think of it as how many words the model is \"torn between\" at each step: lower means more confident, "
    "correct predictions, so a more accurate model. Compression usually raises it a little; a small rise is "
    "acceptable, a large rise means the model got noticeably worse at language."
)

METRICS: dict[str, MetricSpec] = {
    # accuracy
    "ppl_val": MetricSpec(
        "Perplexity (validation half)", "", "lower", "prediction error on test text (validation half)",
        _PPL_MEANING + " The validation half is the part of the test text that is allowed to guide tuning."),
    "ppl_heldout": MetricSpec(
        "Perplexity (held-out half)", "", "lower", "prediction error on unseen test text (held-out half)",
        _PPL_MEANING + " The held-out half is never used for tuning, so it is the honest check that results "
        "were not tuned to one particular piece of text."),
    "downstream_acc_mean": MetricSpec(
        "Downstream accuracy (mean)", "", "higher", "share of test questions answered correctly",
        "The average, over the multiple-choice tests listed in the report, of the share of questions the model "
        "answers correctly (0 to 1). These tests check common sense and reasoning rather than raw word prediction. "
        "Higher is better."),
    # memory
    "model_size_gb": MetricSpec(
        "Model size on disk", "GB", "lower", "file size of the model",
        "How much storage the model's numbers take up, in gigabytes. Smaller means it fits on cheaper devices."),
    "predicted_weight_memory_gb": MetricSpec(
        "Predicted weight memory", "GB", "lower", "memory needed to store the model",
        "How much memory the model would need after compression, in gigabytes, calculated from the plan "
        "before anything is actually compressed. Smaller means it fits on cheaper devices, such as laptops "
        "or phones."),
    "peak_memory_gb": MetricSpec(
        "Peak GPU memory", "GB", "lower", "highest graphics-card memory used while running",
        "The most graphics-card memory the model needed at any moment while it was being tested."),
    "predicted_kv_memory_gb": MetricSpec(
        "Predicted KV-cache memory", "GB", "lower", "short-term memory needed for one long text",
        "While writing, the model keeps notes on every earlier word (the KV cache). This is how much memory "
        "those notes would need for the text length set in the report, calculated from the plan."),
    "avg_kv_bits": MetricSpec(
        "Average bits per cached number", "bits", "lower", "storage used per number in the notes",
        "How many bits each number in the model's notes is stored with, on average (16 uncompressed)."),
    "avg_activation_bits": MetricSpec(
        "Average activation bits", "bits", "lower", "storage used per number passed between layers",
        "How many bits the numbers flowing between the model's layers are rounded to, on average over the "
        "layers (16 uncompressed). Planned here, applied in Stage 1."),
    "predicted_act_ppl_rise": MetricSpec(
        "Predicted perplexity rise from activation rounding", "", "lower",
        "expected accuracy loss from rounding the numbers passed between layers",
        "Stage 0 rounded each layer's incoming numbers on its own and measured how much the prediction error "
        "(perplexity) rose. This adds those rises up for the plan's bits. Smaller is better."),
    "kv_kept_share": MetricSpec(
        "Share of past tokens kept", "", None, "share of earlier words the notes keep",
        "The plan can let some layers forget earlier words that get almost no attention. 1 means nothing is "
        "forgotten; 0.5 means half of the notes are dropped."),
    "predicted_kv_ppl_rise": MetricSpec(
        "Predicted perplexity rise from cache rounding", "", "lower", "expected accuracy loss from the notes",
        "The measured rise in prediction error when each layer's notes are rounded as planned, added up over "
        "layers. It is a prediction; Stage 3 measures the real value."),
    "kv_attention_kept": MetricSpec(
        "Attention on kept tokens", "", "higher", "share of the model's focus that survives forgetting",
        "When a layer forgets earlier words, this is the share of its attention that fell on the words it "
        "kept, averaged over layers. 1 means nothing it looked at was dropped."),
    # latency
    "prefill_ms_mean": MetricSpec(
        "Prefill latency (mean)", "ms", "lower", "time to read the prompt",
        "How long the model takes to read the user's question before it starts answering, in milliseconds."),
    "prefill_ms_std": MetricSpec("Prefill latency (std)", "ms", None, "spread of prompt-reading times",
                                 "How much the prompt-reading time varied between repeated runs."),
    "decode_ms_per_token_mean": MetricSpec(
        "Decode latency per token (mean)", "ms", "lower", "time to write each word",
        "How long the model takes to produce each word (strictly, each word-piece) of its answer, in "
        "milliseconds. Lower means faster replies."),
    "decode_ms_per_token_std": MetricSpec("Decode latency per token (std)", "ms", None,
                                          "spread of per-word times",
                                          "How much the per-word time varied between repeated runs."),
    "prefill_tokens_per_s": MetricSpec("Prefill throughput", "tok/s", "higher", "prompt-reading speed",
                                       "How many word-pieces of the prompt the model reads per second."),
    "decode_tokens_per_s": MetricSpec("Decode throughput", "tok/s", "higher", "writing speed",
                                      "How many word-pieces of its answer the model writes per second."),
    # build cost
    "build_time_s": MetricSpec(
        "Build time", "s", "lower", "time to prepare the compressed model",
        "How long it took, in seconds, to produce this version of the model. This is a one-off cost paid "
        "before the model is used."),
    # Stage 0 allocation
    "avg_bits_per_weight": MetricSpec(
        "Average bits per weight", "bits", "lower", "storage used per number in the model",
        "A model is made of a huge number of numbers (see Original model). The original stores each one with 16 bits (binary "
        "digits); compression stores most of them with fewer bits, like rounding prices to the nearest "
        "dollar instead of the nearest cent. Fewer bits means a smaller model, but too few loses detail."),
    "sparsity": MetricSpec(
        "Weight sparsity", "", None, "share of the model's numbers removed",
        "Pruning deletes numbers that matter least. This is the share of all the model's numbers that the "
        "plan deletes (0.1 means 10%)."),
    "protected_layers": MetricSpec(
        "Protected layers", "", None, "number of protected layers",
        "How many of the model's layers the plan leaves at high precision because they are sensitive."),
    # Sensitivity-weighted compression: sum_l s_l * (1 - effective_bits_l / 16) / sum_l s_l, where
    # effective_bits = bits * (1 - prune ratio). Lower = less compression lands on sensitive layers.
    "sensitivity_exposure": MetricSpec(
        "Sensitivity exposure", "", "lower", "how much compression hits the fragile parts",
        "A score from 0 to 1 for how much of the compression lands on the model's most fragile layers. "
        "0 means the fragile layers are untouched; higher means they are squeezed harder. Lower is better "
        "because squeezing fragile layers is what damages accuracy."),
}


def label(metric: str) -> str:
    return METRICS[metric].label if metric in METRICS else metric


def is_number(x: object) -> bool:
    """A real, finite number (bools and NaN/inf excluded)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def is_better(metric: str, delta: float) -> bool | None:
    spec = METRICS.get(metric)
    if spec is None or spec.better is None or delta == 0:
        return None
    return delta < 0 if spec.better == "lower" else delta > 0
