"""Runs inside the Python that has vLLM installed (Linux; see eval.vllm_python): perplexity, speed and weight memory
of one GPTQ-format checkpoint. Usage: python vllm_worker.py <args.json>; prints one JSON line."""

from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # the model runs in this process, so memory is readable


def main(args: dict) -> dict:
    import torch
    from vllm import LLM, SamplingParams

    # A fixed KV-cache size instead of a share of the card: vLLM reserves the whole KV cache up front, so with
    # gpu_memory_utilization the peak memory measured the card (33 GB on an A100 at 0.4), not the model.
    # A fixed KV-cache size instead of a share of the card: vLLM reserves the whole KV cache up front, so with
    # gpu_memory_utilization the peak memory measured the card (33 GB on an A100 at 0.4), not the model.
    kv_bytes = args.get("kv_cache_memory_bytes", 1 << 30)
    llm = LLM(model=args["model_dir"], dtype="float16", max_model_len=args["seq_len"] + 8, seed=0,
              kv_cache_memory_bytes=kv_bytes)
    out: dict = {}
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
    out["vllm_decode_ms_per_token"] = (full - prefill) * 1e3 / (args["decode_tokens"] - 1)
    out["vllm_peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    out["vllm_kv_cache_reserved_gb"] = kv_bytes / 1e9
    return out


if __name__ == "__main__":
    print(json.dumps(main(json.load(open(sys.argv[1])))))
