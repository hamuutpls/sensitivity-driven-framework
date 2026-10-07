"""vLLM backend: real file size, weight memory, speed and perplexity of a Stage 1 weight plan.

The model is saved as a GPTQ-format checkpoint at the plan's per-layer bits (`gptq_export`) and measured by
`vllm_worker` in the Python that has vLLM (`eval.vllm_python`; vLLM runs on Linux only, so run the whole pipeline
inside WSL2 or Colab). vLLM re-rounds nothing further, but the export re-rounds the weights onto grids of their own,
so `vllm_ppl_*` is comparable between rows, not with the HF perplexity. The uncompressed row (plan None) is saved
as plain FP16.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sdf.config import EvalConfig
from sdf.eval import gptq_export
from sdf.reporting.metrics import METRICS, MetricSpec
from sdf.stage0.planner import CompressionPlan
from sdf.utils.logging import get_logger

log = get_logger(__name__)

# Bump when a change alters the measured numbers, so cached vLLM results are not reused.
VERSION = 2

_NOTE = " Measured in vLLM, a program for running models on servers, after saving the model at the plan's bits."
METRICS.update({
    "vllm_size_gb": MetricSpec("vLLM file size", "GB", "lower", "real file size of the compressed model",
                               "The size of the model file, in gigabytes." + _NOTE),
    "vllm_weight_memory_gb": MetricSpec("vLLM weight memory", "GB", "lower", "memory the weights take on the graphics card",
                                        "Memory taken by the model's numbers once loaded." + _NOTE),
    "vllm_peak_memory_gb": MetricSpec("vLLM peak memory", "GB", "lower", "most graphics-card memory used",
                                      "Most graphics-card memory the program had allocated, including its reserved "
                                      "space for remembering the text so far." + _NOTE),
    "vllm_prefill_ms": MetricSpec("vLLM prefill time", "ms", "lower", "time to read the prompt (vLLM)",
                                  "Time to read a prompt and write one word-piece." + _NOTE),
    "vllm_decode_ms_per_token": MetricSpec("vLLM decode time per token", "ms", "lower", "time per written word-piece (vLLM)",
                                           "Time to write one word-piece." + _NOTE),
    "vllm_ppl_val": MetricSpec("vLLM perplexity (validation half)", "", "lower", "prediction error in vLLM (validation half)",
                               "Prediction error of the saved model on the validation half of the test text (lower is "
                               "better). The weights are re-rounded when saved, so compare between rows only."),
    "vllm_ppl_heldout": MetricSpec("vLLM perplexity (held-out half)", "", "lower", "prediction error in vLLM (held-out half)",
                                   "As the validation-half value, on the held-out half of the test text (lower is better)."),
})
MAIN_METRICS = ["vllm_size_gb", "vllm_decode_ms_per_token", "vllm_ppl_val"]


def measure(model: nn.Module, tokenizer: Any, plan: CompressionPlan | None, val: torch.Tensor, held: torch.Tensor,
            cfg: EvalConfig, baseline_bits: int, group_size: int) -> dict[str, float]:
    """Every vLLM metric for one model; `plan=None` keeps every layer at FP16 (the uncompressed row)."""
    with tempfile.TemporaryDirectory(prefix="sdf-vllm-") as tmp:
        d = Path(tmp)
        if plan is not None and any(lp.bit_width < baseline_bits for lp in plan.layers):
            size = gptq_export.export(model, tokenizer, plan, group_size, d / "model")
        else:
            model.save_pretrained(d / "model", safe_serialization=True)
            tokenizer.save_pretrained(d / "model")
            size = sum(f.stat().st_size for f in (d / "model").glob("*.safetensors")) / 1e9
        args = {"model_dir": str(d / "model"), "seq_len": cfg.seq_len, "val": val.tolist(), "heldout": held.tolist(),
                "warmup": cfg.latency_warmup, "repeats": cfg.latency_repeats, "prompt_len": cfg.latency_prompt_len,
                "decode_tokens": cfg.latency_decode_tokens}
        (d / "args.json").write_text(json.dumps(args))
        cmd = [cfg.vllm_python, str(Path(__file__).with_name("vllm_worker.py")), str(d / "args.json")]
        log.info("vLLM: %s", " ".join(cmd))
        # vLLM compiles kernels with tools installed next to its Python (ninja); a venv's bin/ is not on PATH
        # unless the venv is activated, so put it there.
        env = {**os.environ, "PATH": os.pathsep.join([str(Path(cfg.vllm_python).parent), os.environ.get("PATH", "")])}
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        if r.returncode:
            raise RuntimeError(f"vllm_worker failed ({r.returncode}): {(r.stderr or r.stdout)[-2000:]}")
        out = {"vllm_size_gb": size, **json.loads(r.stdout.strip().splitlines()[-1])}
    log.info("vLLM: %s", ", ".join(f"{k}={v:.4g}" for k, v in out.items() if isinstance(v, float)))
    return out
