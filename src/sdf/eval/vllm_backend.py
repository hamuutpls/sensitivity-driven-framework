"""vLLM backend: real file size, weight memory, speed and perplexity of a Stage 1 weight or WxAy plan.

A weight plan's model is saved as a GPTQ-format checkpoint at the plan's per-layer bits (`gptq_export`); a WxAy
plan's (weights and activations per layer) as a compressed-tensors checkpoint that also tells vLLM each layer's
activation bits (`ct_export`), run on vLLM's Humming kernels, which round the activations for real, without
compiling the model (vLLM 0.31 fails to compile 4-bit activation rounding), so WxAy speeds are comparable only
with each other. Both are
measured by
`vllm_worker` in the Python that has vLLM (`eval.vllm_python`; vLLM runs on Linux only, so run the whole pipeline
inside WSL2 or Colab). vLLM re-rounds nothing further, but the export re-rounds the weights onto grids of their own,
so `vllm_ppl_*` is comparable between rows, not with the HF perplexity. The uncompressed row (plan None) is saved
as plain FP16.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sdf.config import EvalConfig
from sdf.eval import ct_export, gptq_export
from sdf.eval.worker import run_worker
from sdf.reporting.metrics import METRICS, MetricSpec
from sdf.stage0.activation import ActivationPlan
from sdf.stage0.planner import CompressionPlan
from sdf.utils.logging import get_logger

log = get_logger(__name__)

# Bump when a change alters the measured numbers, or turns cached error rows into measurable ones, so cached vLLM
# results are not reused.
VERSION = 5  # 5: timings report their std across repeats; 4: prefix caching off, so prefill is a real prefill; 3: start-up memory share follows the free memory (rows failed vLLM's 0.92 check in 2)

_NOTE = " Measured in vLLM, a program for running models on servers, after saving the model at the plan's bits."
METRICS.update({
    "vllm_size_gb": MetricSpec("vLLM file size", "GB", "lower", "file size of the model in vLLM's format",
                               "The size of the model file, in gigabytes." + _NOTE),
    "vllm_weight_memory_gb": MetricSpec("vLLM weight memory", "GB", "lower", "memory the weights take on the graphics card",
                                        "Memory taken by the model's numbers once loaded." + _NOTE),
    "vllm_peak_memory_gb": MetricSpec("vLLM peak memory", "GB", "lower", "most graphics-card memory used",
                                      "Most graphics-card memory the program had allocated, including its reserved "
                                      "space for remembering the text so far (fixed at 1 GB for every row)." + _NOTE),
    "vllm_kv_cache_reserved_gb": MetricSpec("vLLM KV cache reserved", "GB", "lower", "space reserved for the text so far",
                                            "Memory vLLM sets aside up front for remembering the text so far; the same "
                                            "for every row, and included in the peak." + _NOTE),
    "vllm_prefill_ms": MetricSpec("vLLM prefill time", "ms", "lower", "time to read the prompt (vLLM)",
                                  "Time to read a prompt and write one word-piece." + _NOTE),
    "vllm_prefill_ms_std": MetricSpec("vLLM prefill time (std)", "ms", None,
                                      "spread of prompt-reading times (vLLM)",
                                      "How much the prefill time varied between repeated runs."),
    "vllm_decode_ms_per_token": MetricSpec("vLLM decode time per token", "ms", "lower", "time per written word-piece (vLLM)",
                                           "Time to write one word-piece." + _NOTE),
    "vllm_decode_ms_per_token_std": MetricSpec("vLLM decode time per token (std)", "ms", None,
                                               "spread of times per written word-piece (vLLM)",
                                               "How much the decode time per token varied between repeated runs."),
    "vllm_ppl_val": MetricSpec("vLLM perplexity of the re-rounded file (validation half)", "", "lower", "prediction error of the re-rounded vLLM file (validation half)",
                               "Prediction error of the saved model on the validation half of the test text (lower is "
                               "better). The weights are re-rounded when saved, so compare between rows only."),
    "vllm_ppl_heldout": MetricSpec("vLLM perplexity of the re-rounded file (held-out half)", "", "lower", "prediction error of the re-rounded vLLM file (held-out half)",
                                   "As the validation-half value, on the held-out half of the test text (lower is better)."),
})
MAIN_METRICS = ["vllm_size_gb", "vllm_decode_ms_per_token", "vllm_ppl_val"]
ACTIVATIONS = True  # runs a WxAy plan's activation bits too (the other backends get its weights only)


class VLLMError(RuntimeError):
    """vLLM could not load or run a checkpoint (e.g. a bit width its GPTQ kernels do not support)."""


def measure(model: nn.Module, tokenizer: Any, plan: CompressionPlan | None, val: torch.Tensor, held: torch.Tensor,
            cfg: EvalConfig, baseline_bits: int, group_size: int, acts: ActivationPlan | None = None,
            act_group_size: int = 128) -> dict[str, float]:
    """Every vLLM metric for one model; `plan=None` keeps every layer at FP16 (the uncompressed row). `acts`: the
    activation half of a WxAy plan (`plan` its weight half)."""
    with tempfile.TemporaryDirectory(prefix="sdf-vllm-") as tmp:
        d = Path(tmp)
        backend = None
        if plan is not None and acts is not None:  # a WxAy row, even one with 16-bit activations everywhere, so
            # every WxAy row runs on the same format and kernels and differs from another only in its bits
            try:
                size = ct_export.export(model, tokenizer, plan, acts, group_size, act_group_size, d / "model",
                                        baseline_bits)
            except ValueError as e:  # a pair, bit width or shape the format cannot hold
                raise VLLMError(f"compressed-tensors export failed: {e}") from e
            backend = "humming"  # the only vLLM kernels that round the activations (see ct_export)
        elif plan is not None and any(lp.bit_width < baseline_bits for lp in plan.layers):
            try:
                size = gptq_export.export(model, tokenizer, plan, group_size, d / "model")
            except ValueError as e:  # a bit width or shape the GPTQ format cannot hold
                raise VLLMError(f"GPTQ export failed: {e}") from e
        else:
            model.save_pretrained(d / "model", safe_serialization=True)
            tokenizer.save_pretrained(d / "model")
            size = sum(f.stat().st_size for f in (d / "model").glob("*.safetensors")) / 1e9
        args = {"model_dir": str(d / "model"), "seq_len": cfg.seq_len, "val": val.tolist(), "heldout": held.tolist(),
                "warmup": cfg.latency_warmup, "repeats": cfg.latency_repeats, "prompt_len": cfg.latency_prompt_len,
                "decode_tokens": cfg.latency_decode_tokens, "linear_backend": backend,
                "gpu_memory_utilization": cfg.vllm_gpu_memory_utilization,
                "act_layers": [] if acts is None else [lp.layer for lp, w in zip(acts.layers, plan.layers)
                                                       if lp.act_bits < baseline_bits and w.bit_width < baseline_bits]}
        out = {"vllm_size_gb": size, **run_worker(cfg.vllm_python, "vllm_worker.py", args, d, VLLMError)}
    log.info("vLLM: %s", ", ".join(f"{k}={v:.4g}" for k, v in out.items() if isinstance(v, float)))
    return out
