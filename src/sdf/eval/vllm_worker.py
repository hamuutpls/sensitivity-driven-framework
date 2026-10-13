"""Runs inside the Python that has vLLM installed (Linux; see eval.vllm_python): perplexity, speed and weight memory
of one GPTQ-format checkpoint. Usage: python vllm_worker.py <args.json>; prints one JSON line."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # the model runs in this process, so memory is readable


def memory_share(free: float, total: float, cap: float = 0.92, margin: float = 0.02) -> float:
    """The share of the card vLLM may claim when it starts: what is free now minus a margin, at most vLLM's default
    0.92. vLLM refuses to start when less than that share is free, and the framework process, other backends or
    Stage 0 may still hold some of the card (on a 40 GB A100 0.92 left 3 GB for everything else). The KV cache has
    a fixed size (kv_cache_memory_bytes), so the share changes no measured number."""
    share = min(cap, free / total - margin)
    if share <= 0:
        raise RuntimeError(f"only {free / 1e9:.2f} of {total / 1e9:.2f} GB of GPU memory free; vLLM cannot start")
    return share


def main(args: dict) -> dict:
    import torch
    from vllm import LLM, SamplingParams

    # A fixed KV-cache size instead of a share of the card: vLLM reserves the whole KV cache up front, so with
    # gpu_memory_utilization the peak memory measured the card (33 GB on an A100 at 0.4), not the model.
    kv_bytes = args.get("kv_cache_memory_bytes", 1 << 30)
    # WxAy rows: Humming kernels, and no torch.compile / CUDA graphs (enforce_eager): compiling a model with
    # 4-bit activation rounding fails inside Humming (RecursionError in humming/device.py, vLLM 0.31). So these
    # rows' speed is not comparable with the compiled weight-only rows; their perplexity is.
    extra = {"linear_backend": args["linear_backend"], "enforce_eager": True} if args.get("linear_backend") else {}
    share = args.get("gpu_memory_utilization")
    share = memory_share(*torch.cuda.mem_get_info()) if share is None else share
    # long enough for the perplexity windows and the latency prompt + new tokens; prefix caching off, or every timed
    # prefill after the first reads the repeated prompt from the cache and measures about one decode step
    n = max(args["seq_len"], args["prompt_len"] + args["decode_tokens"]) + 8
    llm = LLM(model=args["model_dir"], dtype="float16", max_model_len=n, seed=0, enable_prefix_caching=False,
              kv_cache_memory_bytes=kv_bytes, gpu_memory_utilization=share, **extra)
    out: dict = {}
    if extra:  # a layer whose activations should be rounded must run on a kernel that does it, or the row is wrong
        def rounding(m):
            return sorted((n, type(getattr(mod.scheme, "kernel", None)).__name__) for n, mod in m.named_modules()
                          if hasattr(mod, "_humming_input_quant_config"))
        found = sum(llm.apply_model(rounding), [])
        wrong = [f"{n}: {k}" for n, k in found if "Humming" not in k]
        if wrong:
            raise RuntimeError(f"activation rounding not on a Humming kernel: {', '.join(wrong[:5])}")
        layers = sorted({int(re.search(r"layers\.(\d+)\.", n).group(1)) for n, _ in found})
        if layers != sorted(args["act_layers"]):  # vLLM read the config as weight-only somewhere
            raise RuntimeError(f"vLLM rounds the activations of layers {layers}, the plan of {args['act_layers']}")
    try:
        out["vllm_weight_memory_gb"] = sum(llm.apply_model(
            lambda m: sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers())))) / 1e9
    except Exception as e:  # apply_model differs between vLLM versions
        out["vllm_weight_memory_error"] = repr(e)

    def nll(windows: list[list[int]]) -> float:
        res = llm.generate([{"prompt_token_ids": w} for w in windows],
                           SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0), use_tqdm=False)
        total = n = 0
        for w, r in zip(windows, res):
            for tok, lp in zip(w[1:], r.prompt_logprobs[1:]):
                total -= lp[tok].logprob
                n += 1
        return math.exp(total / n)

    for name in ("val", "heldout"):
        out[f"vllm_ppl_{name}"] = nll(args[name])

    def timed(prompt_len: int, new: int) -> float:
        if len(args["val"][0]) < prompt_len:  # the prompt is cut from the first perplexity window
            raise ValueError(f"latency prompt of {prompt_len} tokens is longer than an eval window "
                             f"({len(args['val'][0])}); raise eval.seq_len")
        ids = [{"prompt_token_ids": args["val"][0][:prompt_len]}]
        sp = SamplingParams(max_tokens=new, temperature=0, ignore_eos=True)
        for _ in range(args["warmup"]):
            llm.generate(ids, sp, use_tqdm=False)
        ts = []
        for _ in range(args["repeats"]):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate(ids, sp, use_tqdm=False)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return sum(ts) / len(ts)

    prefill = timed(args["prompt_len"], 1)
    full = timed(args["prompt_len"], args["decode_tokens"])
    out["vllm_prefill_ms"] = prefill * 1e3
    out["vllm_decode_ms_per_token"] = ((full - prefill) * 1e3 / (args["decode_tokens"] - 1)
                                       if args["decode_tokens"] > 1 else float("nan"))
    out["vllm_peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    out["vllm_kv_cache_reserved_gb"] = kv_bytes / 1e9
    return out


if __name__ == "__main__":
    print(json.dumps(main(json.load(open(sys.argv[1])))))
