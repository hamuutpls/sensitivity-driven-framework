"""TensorRT-LLM backend: engine size, GPU memory, speed and perplexity of a Stage 1 weight plan.

The same GPTQ-format checkpoint as the vLLM backend (`gptq_export`, symmetric, per-layer bits) is built into a
TensorRT engine by `trtllm_worker` in the Python that has TensorRT-LLM (`eval.trtllm_python`; Linux only).
TensorRT-LLM 1.2 does not read the HF `quantization_config` of a GPTQ checkpoint ("Unsupported
quantization_config"), only its own `hf_quant_config.json`, so for a plan with one bit width that file is written
next to the checkpoint (W4A16_GPTQ / W8A16_GPTQ). Its LLM API cannot load MIXED_PRECISION from that file (it keeps a
plain QuantConfig, and building fails with "'QuantMode' object is not callable"), so a plan with several bit widths
writes `MIXED_FILE` instead: one W*A16_GPTQ entry per decoder layer, under TensorRT-LLM's own module names
(`transformer.layers.<i>.attention.*` / `.mlp.*`). The worker turns it into a TensorRT-LLM checkpoint with a
per-layer quantization config (`quant_cfg.json`), the format TensorRT-LLM's mixed-precision build reads.
A plan TensorRT-LLM cannot build raises TRTLLMError with its own reason; the runner keeps the row. The uncompressed
row (plan None) is built from plain FP16.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sdf.config import EvalConfig
from sdf.eval import gptq_export
from sdf.eval.worker import run_worker
from sdf.reporting.metrics import METRICS, MetricSpec
from sdf.stage0.planner import CompressionPlan
from sdf.stage0.sensitivity import find_decoder_layers
from sdf.utils.logging import get_logger

log = get_logger(__name__)

# Bump when a change alters the measured numbers, so cached TensorRT-LLM results are not reused.
VERSION = 3  # 3: KV block reuse off, so prefill is a real prefill; 2: mixed-bit plans build through a per-layer checkpoint (were errors in 1)

_NOTE = " Measured in TensorRT-LLM, NVIDIA's program for running models fast, after building the model at the plan's bits."
METRICS.update({
    "trtllm_engine_gb": MetricSpec("TensorRT-LLM engine size", "GB", "lower", "size of the built engine",
                                   "The size of the TensorRT engine file, in gigabytes." + _NOTE),
    "trtllm_gpu_memory_loaded_gb": MetricSpec("TensorRT-LLM memory once loaded", "GB", "lower",
                                              "graphics-card memory in use after loading",
                                              "Graphics-card memory in use after the engine is loaded, including its "
                                              "fixed space for remembering the text so far (about 1 GB)." + _NOTE),
    "trtllm_gpu_memory_after_run_gb": MetricSpec("TensorRT-LLM memory after the run", "GB", "lower",
                                                 "graphics-card memory in use after running",
                                                 "Graphics-card memory in use after all measurements." + _NOTE),
    "trtllm_build_s": MetricSpec("TensorRT-LLM build time", "s", "lower", "time to build the engine",
                                 "Time to turn the saved model into a TensorRT engine." + _NOTE),
    "trtllm_prefill_ms": MetricSpec("TensorRT-LLM prefill time", "ms", "lower", "time to read the prompt (TensorRT-LLM)",
                                    "Time to read a prompt and write one word-piece." + _NOTE),
    "trtllm_decode_ms_per_token": MetricSpec("TensorRT-LLM decode time per token", "ms", "lower",
                                             "time per written word-piece (TensorRT-LLM)",
                                             "Time to write one word-piece." + _NOTE),
    "trtllm_ppl_val": MetricSpec("TensorRT-LLM perplexity (validation half)", "", "lower",
                                 "prediction error in TensorRT-LLM (validation half)",
                                 "Prediction error of the built engine on the validation half of the test text (lower "
                                 "is better). The weights are re-rounded when saved, so compare between rows only."),
    "trtllm_ppl_heldout": MetricSpec("TensorRT-LLM perplexity (held-out half)", "", "lower",
                                     "prediction error in TensorRT-LLM (held-out half)",
                                     "As the validation-half value, on the held-out half of the test text."),
})
MAIN_METRICS = ["trtllm_engine_gb", "trtllm_decode_ms_per_token", "trtllm_ppl_val"]

_ALGO = {4: "W4A16_GPTQ", 8: "W8A16_GPTQ"}
MIXED_FILE = "trtllm_mixed_precision.json"  # read by trtllm_worker; not hf_quant_config.json (see the docstring)


class TRTLLMError(RuntimeError):
    """TensorRT-LLM could not build or run a checkpoint (e.g. a bit width it has no GPTQ kernel for)."""


def quant_file(model: nn.Module, plan: CompressionPlan, group_size: int) -> dict[str, Any]:
    """TensorRT-LLM's quantization config for `plan`: one algorithm, or MIXED_PRECISION with one entry per decoder
    layer below 16 bits, keyed by TensorRT-LLM's module names (16-bit layers and the output head stay FP16)."""
    bits = {lp.layer: lp.bit_width for lp in plan.layers}
    if len(set(bits.values())) == 1:
        b = next(iter(bits.values()))
        return {"quantization": {"quant_algo": _ALGO.get(b, f"W{b}A16_GPTQ"), "group_size": group_size,
                                 "has_zero_point": True}}
    if len(bits) != len(find_decoder_layers(model)):
        raise ValueError(f"the plan has {len(bits)} layers, the model {len(find_decoder_layers(model))}")
    per = {f"transformer.layers.{i}.{part}.*": {"quant_algo": _ALGO.get(b, f"W{b}A16_GPTQ"), "group_size": group_size,
                                                "has_zero_point": True}
           for i, b in sorted(bits.items()) if b < 16 for part in ("attention", "mlp")}
    return {"quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": per}}


def measure(model: nn.Module, tokenizer: Any, plan: CompressionPlan | None, val: torch.Tensor, held: torch.Tensor,
            cfg: EvalConfig, baseline_bits: int, group_size: int) -> dict[str, float]:
    """Every TensorRT-LLM metric for one model; `plan=None` keeps every layer at FP16 (the uncompressed row)."""
    with tempfile.TemporaryDirectory(prefix="sdf-trtllm-") as tmp:
        d = Path(tmp)
        if plan is not None and any(lp.bit_width < baseline_bits for lp in plan.layers):
            try:
                gptq_export.export(model, tokenizer, plan, group_size, d / "model")
            except ValueError as e:  # a bit width or shape the GPTQ format cannot hold
                raise TRTLLMError(f"GPTQ export failed: {e}") from e
            q = quant_file(model, plan, group_size)
            mixed = q["quantization"]["quant_algo"] == "MIXED_PRECISION"
            (d / "model" / (MIXED_FILE if mixed else "hf_quant_config.json")).write_text(json.dumps(q))
        else:
            model.save_pretrained(d / "model", safe_serialization=True)
            tokenizer.save_pretrained(d / "model")
        args = {"model_dir": str(d / "model"), "engine_dir": str(d / "engine"), "seq_len": cfg.seq_len,
                "val": val.tolist(), "heldout": held.tolist(), "warmup": cfg.latency_warmup,
                "repeats": cfg.latency_repeats, "prompt_len": cfg.latency_prompt_len,
                "decode_tokens": cfg.latency_decode_tokens}
        out = run_worker(cfg.trtllm_python, "trtllm_worker.py", args, d, TRTLLMError)
    log.info("TensorRT-LLM: %s", ", ".join(f"{k}={v:.4g}" for k, v in out.items() if isinstance(v, float)))
    return out
