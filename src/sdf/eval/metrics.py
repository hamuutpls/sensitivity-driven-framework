"""Measured metrics shared by every stage: perplexity (both halves), memory, latency.

Latency is measured on the real model with the current backend (HF Transformers here), with warmup runs
and repeated timings; the raw repeats are returned so reports can show every measurement.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Callable

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


def _static_cache(model: nn.Module, max_len: int):
    """A fixed-size KV cache (no tensor grows between steps, so a decode step can be replayed as a CUDA graph)."""
    from transformers import StaticCache

    p = next(model.parameters())
    try:
        return StaticCache(config=model.config, max_cache_len=max_len)  # transformers >= 4.56: allocated lazily
    except TypeError:
        return StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_len, device=p.device,
                           dtype=p.dtype)


def _graphed(fn: Callable[[], torch.Tensor], reset: Callable[[], None]) -> Callable[[], torch.Tensor]:
    """Capture `fn` (fixed input tensors, updated in place) as a CUDA graph; calling the result replays it.
    `reset` empties the KV cache before each warm-up run, so warm-up never writes past its end.
    Replaying launches the whole forward pass at once, so Python and kernel-launch overhead drop out of the time."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # warm up off the default stream, as torch.cuda.graph requires
        for _ in range(2):
            reset()
            fn()
    torch.cuda.current_stream().wait_stream(s)
    reset()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()

    def replay() -> torch.Tensor:
        g.replay()
        return out
    return replay


def measure_latency(model: nn.Module, cfg: EvalConfig, device: torch.device, vocab_size: int,
                    seed: int = 0) -> list[dict[str, Any]]:
    """`_time_latency` with deterministic algorithms off: the run's deterministic mode (sdf.utils.seed) picks slower
    kernels (+0.9 ms/token decode on the host PC) and timing needs no bit-reproducibility."""
    on, warn_only = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        return _time_latency(model, cfg, device, vocab_size, seed)
    finally:
        torch.use_deterministic_algorithms(on, warn_only=warn_only)


@torch.no_grad()
def _time_latency(model: nn.Module, cfg: EvalConfig, device: torch.device, vocab_size: int,
                  seed: int = 0) -> list[dict[str, Any]]:
    """Time prefill of a `latency_prompt_len` prompt and `latency_decode_tokens` greedy decode steps.

    With `latency_mode="cuda_graph"` on a GPU, prefill and one decode step are each captured once as a CUDA graph
    and replayed, so the time is the GPU's work, not Python's (eager HF decode of a 1B model is mostly launch
    overhead). If capture fails (e.g. a method's hook syncs with the CPU), the same loop runs eagerly; each record
    says which mode ran. Returns one raw record per timed repeat (warmup runs are not recorded).
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    n_prompt, n_decode = cfg.latency_prompt_len, cfg.latency_decode_tokens
    prompt = torch.randint(0, vocab_size, (1, n_prompt), generator=g).to(device)
    cache = _static_cache(model, n_prompt + n_decode)
    prompt_pos = torch.arange(n_prompt, device=device)
    step_pos = torch.tensor([n_prompt], device=device)
    tok = torch.zeros(1, 1, dtype=torch.long, device=device)
    # An explicit all-ones mask: without one, transformers 5.18 decides whether to skip the causal mask with a
    # GPU-to-CPU read, which CUDA graph capture forbids (capture failed on Colab; 5.17 skips the check).
    mask = torch.ones(1, n_prompt + n_decode, dtype=torch.long, device=device)

    def prefill() -> torch.Tensor:
        return model(input_ids=prompt, attention_mask=mask, past_key_values=cache, cache_position=prompt_pos,
                     use_cache=True).logits

    def step() -> torch.Tensor:
        return model(input_ids=tok, attention_mask=mask, past_key_values=cache, cache_position=step_pos,
                     use_cache=True).logits

    mode = "eager"
    if cfg.latency_mode == "cuda_graph" and device.type == "cuda":
        try:
            prefill, step, mode = _graphed(prefill, cache.reset), _graphed(step, cache.reset), "cuda_graph"
        except Exception as e:  # noqa: BLE001 - any capture failure falls back to eager timing, recorded
            log.warning("CUDA graph capture failed (%s: %s); timing eagerly", type(e).__name__, e)

    records = []
    for i in range(cfg.latency_warmup + cfg.latency_repeats):
        cache.reset()  # in place: newer transformers track the fill level inside the cache
        _sync(device)
        t0 = time.perf_counter()
        logits = prefill()
        _sync(device)
        prefill_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        step_pos.fill_(n_prompt)
        tok.copy_(logits[:, -1:].argmax(-1))
        for _ in range(n_decode):
            tok.copy_(step()[:, -1:].argmax(-1))
            step_pos.add_(1)
        _sync(device)
        decode_s = time.perf_counter() - t0

        if i >= cfg.latency_warmup:
            records.append({
                "measurement": "latency",
                "mode": mode,
                "repeat": i - cfg.latency_warmup,
                "prefill_ms": prefill_s * 1e3,
                "decode_ms_per_token": decode_s * 1e3 / n_decode,
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
