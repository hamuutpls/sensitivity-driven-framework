"""Runs inside the Python that has TensorRT-LLM installed (Linux; see eval.trtllm_python): builds a TensorRT engine
from one checkpoint (FP16, or GPTQ format: int4 / int8 weight-only `W4A16_GPTQ` / `W8A16_GPTQ`) and measures engine
size, GPU memory, speed and perplexity. Usage: python trtllm_worker.py <args.json>; prints one JSON line.

A plan with several bit widths comes with `trtllm_mixed_precision.json` (per-layer algorithms under TensorRT-LLM's
module names) instead of hf_quant_config.json: it is first converted into a TensorRT-LLM checkpoint whose
config.json says MIXED_PRECISION and whose quant_cfg.json lists the layers, the format the engine build reads.

Same protocol as vllm_worker.py: perplexity of every window from the logits of its own tokens (positions 2..n),
prefill = one 128-token request writing one token, decode = (64-token request - prefill) / 63, warm-ups first.
TensorRT allocates through its own allocator, so memory is read from the device (used before/after), not torch."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path


def _used_gb() -> float:
    import torch

    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9


MIXED_FILE = "trtllm_mixed_precision.json"  # = trtllm_backend.MIXED_FILE (this file runs without sdf installed)


def mixed_checkpoint(model_dir: Path, out_dir: Path) -> Path:
    """Convert the GPTQ checkpoint in model_dir, whose layers have different bit widths (MIXED_FILE), into a
    TensorRT-LLM checkpoint in out_dir; returns out_dir."""
    from tensorrt_llm.models.automodel import AutoModelForCausalLM
    from tensorrt_llm.models.modeling_utils import LayerQuantConfig, QuantConfig
    from tensorrt_llm.quantization import QuantAlgo

    q = json.loads((model_dir / MIXED_FILE).read_text())["quantization"]
    layers = {name: QuantConfig(quant_algo=QuantAlgo(c["quant_algo"]), group_size=c["group_size"],
                                has_zero_point=c["has_zero_point"]) for name, c in q["quantized_layers"].items()}
    qc = LayerQuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION, quantized_layers=layers)
    model = AutoModelForCausalLM.get_trtllm_model_class(model_dir).from_hugging_face(
        str(model_dir), dtype="float16", quant_config=qc)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(out_dir), save_config=False)
    # config.json holds only the algorithm name (a plain QuantConfig cannot hold the layer list); TensorRT-LLM then
    # reads the per-layer settings from quant_cfg.json next to it
    model.config.quantization = QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION)
    model.config.to_json_file(str(out_dir / "config.json"))
    (out_dir / "quant_cfg.json").write_text(json.dumps({"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None,
                                                        "quantized_layers": q["quantized_layers"]}))
    return out_dir


def main(args: dict) -> dict:
    import torch
    from tensorrt_llm._tensorrt_engine import LLM  # the TensorRT engine backend (not the PyTorch one)
    from tensorrt_llm.llmapi import BuildConfig, KvCacheConfig, SamplingParams

    n = args["seq_len"] + 8
    base = _used_gb()
    bc = BuildConfig(max_seq_len=n, max_input_len=n, max_batch_size=8, max_num_tokens=8 * n,
                     gather_context_logits=True)
    t0 = time.perf_counter()
    hf_dir = model_dir = Path(args["model_dir"])
    if (model_dir / MIXED_FILE).exists():
        model_dir = mixed_checkpoint(model_dir, Path(args["engine_dir"]).with_name("trtllm-checkpoint"))
    # ~1 GB of KV cache for TinyLlama (22 layers x 2 x 4 heads x 64 x 2 bytes = 22.5 KB per token), as in vLLM.
    # The tokenizer comes from the exported HF folder: a converted mixed-bit checkpoint has none, and without one
    # TensorRT-LLM refuses to generate ("tokenizer is required to reset end_id").
    llm = LLM(model=str(model_dir), tokenizer=str(hf_dir), dtype="float16", build_config=bc,
              kv_cache_config=KvCacheConfig(max_tokens=args.get("kv_tokens", 46000)))
    out: dict = {"trtllm_build_s": time.perf_counter() - t0}
    eng = Path(args["engine_dir"])
    llm.save(str(eng))
    out["trtllm_engine_gb"] = sum(f.stat().st_size for f in eng.rglob("*.engine")) / 1e9
    out["trtllm_gpu_memory_loaded_gb"] = _used_gb() - base

    def ppl(windows: list[list[int]]) -> float:
        total = count = 0
        sp = SamplingParams(max_tokens=1, return_context_logits=True)
        for i in range(0, len(windows), 8):
            for w, r in zip(windows[i:i + 8], llm.generate(windows[i:i + 8], sp, use_tqdm=False)):
                logp = torch.log_softmax(r.context_logits.float(), dim=-1)  # (len, vocab)
                tok = torch.tensor(w[1:], device=logp.device)
                total -= logp[:-1].gather(1, tok[:, None]).sum().item()
                count += len(w) - 1
        return math.exp(total / count)

    for name in ("val", "heldout"):
        out[f"trtllm_ppl_{name}"] = ppl(args[name])

    def timed(new: int) -> float:
        ids = [args["val"][0][:args["prompt_len"]]]
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

    prefill = timed(1)
    full = timed(args["decode_tokens"])
    out["trtllm_prefill_ms"] = prefill * 1e3
    out["trtllm_decode_ms_per_token"] = ((full - prefill) * 1e3 / (args["decode_tokens"] - 1)
                                         if args["decode_tokens"] > 1 else float("nan"))
    out["trtllm_gpu_memory_after_run_gb"] = _used_gb() - base
    return out


if __name__ == "__main__":
    os.environ.setdefault("TLLM_LOG_LEVEL", "WARNING")
    print(json.dumps(main(json.load(open(sys.argv[1])))))
