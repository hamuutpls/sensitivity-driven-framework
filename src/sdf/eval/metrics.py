"""Measured metrics shared by every stage: perplexity (both halves), memory, latency.

Latency is measured on the real model with the current backend (HF Transformers here), with warmup runs
and repeated timings; the raw repeats are returned so reports can show every measurement.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any

import torch
from torch import nn

from sdf.config import EvalConfig
from sdf.utils.logging import get_logger

log = get_logger(__name__)


@torch.no_grad()
def perplexity(model: nn.Module, windows: torch.Tensor, device: torch.device) -> tuple[float, list[float]]:
    """exp(mean token NLL) over `windows` (n, seq_len). Returns (ppl, per-window mean NLL)."""
    model.eval()
    nlls = []
    for w in windows:
        ids = w.unsqueeze(0).to(device)
        nlls.append(model(input_ids=ids, labels=ids).loss.float().item())
    ppl = math.exp(statistics.fmean(nlls))
    if not math.isfinite(ppl):
        raise FloatingPointError(f"perplexity is not finite ({ppl}); the model output is broken")
    return ppl, nlls


def model_size_gb(model: nn.Module) -> float:
    """Bytes of every parameter and buffer, i.e. the safetensors payload written by save_pretrained."""
    seen = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() in seen:  # tied weights (e.g. embeddings / lm_head) are stored once
            continue
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total / 1e9


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_latency(model: nn.Module, cfg: EvalConfig, device: torch.device, vocab_size: int,
                    seed: int = 0) -> list[dict[str, Any]]:
    """Time prefill of a `latency_prompt_len` prompt and `latency_decode_tokens` greedy decode steps.

    Returns one raw record per timed repeat (warmup runs are not recorded).
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, vocab_size, (1, cfg.latency_prompt_len), generator=g).to(device)
    records = []
    for i in range(cfg.latency_warmup + cfg.latency_repeats):
        _sync(device)
        t0 = time.perf_counter()
        out = model(input_ids=prompt, use_cache=True)
        _sync(device)
        prefill_s = time.perf_counter() - t0

        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        t0 = time.perf_counter()
        for _ in range(cfg.latency_decode_tokens):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        _sync(device)
        decode_s = time.perf_counter() - t0

        if i >= cfg.latency_warmup:
            records.append({
                "measurement": "latency",
                "repeat": i - cfg.latency_warmup,
                "prefill_ms": prefill_s * 1e3,
                "decode_ms_per_token": decode_s * 1e3 / cfg.latency_decode_tokens,
            })
    return records


def measure_model(model: nn.Module, validation: torch.Tensor, held_out: torch.Tensor, cfg: EvalConfig,
                  device: torch.device, seed: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Every measured metric for one model. Returns (aggregated metrics, raw measurements)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    ppl_val, nll_val = perplexity(model, validation, device)
    ppl_held, nll_held = perplexity(model, held_out, device)
    lat = measure_latency(model, cfg, device, model.config.vocab_size, seed=seed)

    prefill = [r["prefill_ms"] for r in lat]
    decode = [r["decode_ms_per_token"] for r in lat]
    metrics = {
        "ppl_val": ppl_val,
        "ppl_heldout": ppl_held,
        "model_size_gb": model_size_gb(model),
        "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
        "prefill_ms_mean": statistics.fmean(prefill),
        "prefill_ms_std": statistics.stdev(prefill) if len(prefill) > 1 else 0.0,
        "decode_ms_per_token_mean": statistics.fmean(decode),
        "decode_ms_per_token_std": statistics.stdev(decode) if len(decode) > 1 else 0.0,
        "prefill_tokens_per_s": cfg.latency_prompt_len / (statistics.fmean(prefill) / 1e3),
        "decode_tokens_per_s": 1e3 / statistics.fmean(decode),
    }
    raw = ([{"measurement": "nll_val", "window": i, "value": v} for i, v in enumerate(nll_val)]
           + [{"measurement": "nll_heldout", "window": i, "value": v} for i, v in enumerate(nll_held)]
           + lat)
    log.info("measured: ppl_val=%.3f ppl_heldout=%.3f size=%.3fGB decode=%.2fms/tok",
             ppl_val, ppl_held, metrics["model_size_gb"], metrics["decode_ms_per_token_mean"])
    return metrics, raw
