"""llama.cpp backend: real file size, speed and perplexity of a Stage 1 weight plan.

The model (weights as the method left them, stored in FP16) is converted to GGUF and quantized with llama.cpp's
own formats, each decoder layer at its planned bits (BITS_TO_GGUF); embeddings and the output head stay F16, as
in the plan. Size and speed are therefore what a deployed model with this bit layout gets. llama.cpp re-rounds the
weights on its own grid, and its perplexity tool scores only the second half of each window, so `llamacpp_ppl_*`
is a separate, labelled measurement, comparable between rows but not with the HF perplexity. GGUF stores pruned
weights as zeros, so unstructured pruning saves nothing here.

Needs the llama.cpp release programs (`eval.llamacpp_dir`) and its `convert_hf_to_gguf.py` (`eval.llamacpp_convert`).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sdf.config import EvalConfig
from sdf.reporting.metrics import METRICS, MetricSpec
from sdf.stage0.planner import CompressionPlan
from sdf.utils.logging import get_logger

log = get_logger(__name__)

# Bump when a change alters the measured numbers, so cached llama.cpp results are not reused.
# 2: the GGUF keeps the SentencePiece tokenizer (version 1 had flat token scores and wrong perplexity).
VERSION = 2

# Plan bits -> llama.cpp tensor type. K-quants (256-weight blocks, scale and minimum) are closest to the plan's
# asymmetric groups; 8 bits has no K-quant.
BITS_TO_GGUF = {16: "f16", 8: "q8_0", 6: "q6_k", 5: "q5_k", 4: "q4_k", 3: "q3_k", 2: "q2_k"}

_NOTE = (" Measured in llama.cpp, the program people use to run models on their own computers, after saving the "
         "model in its file format at the plan's bits.")
METRICS.update({
    "llamacpp_size_gb": MetricSpec(
        "llama.cpp file size", "GB", "lower", "real file size of the compressed model",
        "The size of the model file, in gigabytes." + _NOTE),
    "llamacpp_prefill_tokens_per_s": MetricSpec(
        "llama.cpp prefill throughput", "tok/s", "higher", "prompt-reading speed (llama.cpp)",
        "How many word-pieces of the prompt the model reads per second." + _NOTE),
    "llamacpp_decode_tokens_per_s": MetricSpec(
        "llama.cpp decode throughput", "tok/s", "higher", "writing speed (llama.cpp)",
        "How many word-pieces of its answer the model writes per second." + _NOTE),
    "llamacpp_decode_tokens_per_s_std": MetricSpec(
        "llama.cpp decode throughput (std)", "tok/s", None, "spread of writing speeds (llama.cpp)",
        "How much the writing speed varied between repeated runs."),
    "llamacpp_ppl_val": MetricSpec(
        "llama.cpp perplexity (validation half)", "", "lower", "prediction error in llama.cpp (validation half)",
        "Prediction error of the llama.cpp file on the validation half of the test text (lower is better). "
        "llama.cpp rounds the numbers its own way and scores only the second half of each window, so compare it "
        "between rows, not with the other perplexity columns."),
    "llamacpp_ppl_heldout": MetricSpec(
        "llama.cpp perplexity (held-out half)", "", "lower", "prediction error in llama.cpp (held-out half)",
        "As the validation-half value, on the held-out half of the test text (lower is better)."),
})
MAIN_METRICS = ["llamacpp_size_gb", "llamacpp_decode_tokens_per_s", "llamacpp_ppl_val"]


def quantize_args(plan: CompressionPlan, baseline_bits: int) -> tuple[list[str], str]:
    """(llama-quantize options, file type): the most common layer bit width is the file type, every other layer
    an override; `--pure` stops llama.cpp from raising some tensors to more bits on its own."""
    bits = [lp.bit_width for lp in plan.layers]
    common = Counter(bits).most_common(1)[0][0]
    opts = ["--pure", "--token-embedding-type", BITS_TO_GGUF[baseline_bits],
            "--output-tensor-type", BITS_TO_GGUF[baseline_bits]]
    for lp in plan.layers:
        if lp.bit_width != common:
            opts += ["--tensor-type", rf"blk\.{lp.layer}\.={BITS_TO_GGUF[lp.bit_width]}"]
    return opts, BITS_TO_GGUF[common].upper()


def _save_tokenizer(tokenizer: Any, out: Path) -> None:
    """save_pretrained plus the model's own SentencePiece `tokenizer.model`. Transformers 5 writes only
    tokenizer.json; without tokenizer.model the converter stores flat token scores, llama.cpp then splits text
    differently, and perplexity is wrong (TinyLlama FP16: 17.8 instead of 8.1)."""
    tokenizer.save_pretrained(out)
    src = Path(tokenizer.name_or_path)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(tokenizer.name_or_path, allow_patterns=["tokenizer.model"]))
    if (src / "tokenizer.model").exists():
        shutil.copy2(src / "tokenizer.model", out / "tokenizer.model")


def _run(cmd: list[Any]) -> str:
    cmd = [str(c) for c in cmd]
    log.info("llama.cpp: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise RuntimeError(f"{Path(cmd[0]).name} failed ({r.returncode}): {(r.stderr or r.stdout)[-2000:]}")
    return r.stdout + r.stderr


def _exe(cfg: EvalConfig, name: str) -> Path:
    d = Path(cfg.llamacpp_dir)
    return next((p for p in (d / f"{name}.exe", d / name) if p.exists()), d / name)


def _bench(cfg: EvalConfig, gguf: Path) -> dict[str, float]:
    """llama-bench (one warm-up run of its own, then `latency_repeats` timed runs)."""
    out = _run([_exe(cfg, "llama-bench"), "-m", gguf, "-p", cfg.latency_prompt_len, "-n", cfg.latency_decode_tokens,
                "-r", cfg.latency_repeats, "-ngl", 99, "-o", "json"])
    rows = json.loads(out[out.index("["):out.rindex("]") + 1])
    pre = next(r for r in rows if r["n_gen"] == 0)
    dec = next(r for r in rows if r["n_prompt"] == 0)
    return {"llamacpp_prefill_tokens_per_s": pre["avg_ts"], "llamacpp_decode_tokens_per_s": dec["avg_ts"],
            "llamacpp_decode_tokens_per_s_std": dec["stddev_ts"]}


def _perplexity(cfg: EvalConfig, gguf: Path, text: str, n_windows: int, path: Path) -> float:
    path.write_text(text, encoding="utf-8")
    out = _run([_exe(cfg, "llama-perplexity"), "-m", gguf, "-f", path, "-c", cfg.seq_len, "--chunks", n_windows,
                "-ngl", 99])
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", out)
    if m is None:
        raise RuntimeError(f"no perplexity in llama-perplexity output: {out[-1000:]}")
    return float(m.group(1))


def measure(model: nn.Module, tokenizer: Any, plan: CompressionPlan | None, val: torch.Tensor, held: torch.Tensor,
            cfg: EvalConfig, baseline_bits: int) -> dict[str, float]:
    """Every llama.cpp metric for one model; `plan=None` keeps every layer at F16 (the uncompressed row)."""
    with tempfile.TemporaryDirectory(prefix="sdf-gguf-") as tmp:
        d = Path(tmp)
        model.save_pretrained(d / "hf")
        _save_tokenizer(tokenizer, d / "hf")
        gguf = d / "f16.gguf"
        _run([sys.executable, cfg.llamacpp_convert, d / "hf", "--outtype", "f16", "--outfile", gguf])
        shutil.rmtree(d / "hf")  # each copy is model-sized; keep at most two on disk
        if plan is not None and any(lp.bit_width < baseline_bits for lp in plan.layers):
            opts, ftype = quantize_args(plan, baseline_bits)
            _run([_exe(cfg, "llama-quantize"), *opts, gguf, d / "q.gguf", ftype])
            gguf.unlink()
            gguf = d / "q.gguf"
        out = {"llamacpp_size_gb": gguf.stat().st_size / 1e9, **_bench(cfg, gguf)}
        for name, w in (("val", val), ("heldout", held)):
            out[f"llamacpp_ppl_{name}"] = _perplexity(cfg, gguf, tokenizer.decode(w.flatten()), len(w),
                                                      d / f"{name}.txt")
    log.info("llama.cpp: %s", ", ".join(f"{k}={v:.4g}" for k, v in out.items()))
    return out
