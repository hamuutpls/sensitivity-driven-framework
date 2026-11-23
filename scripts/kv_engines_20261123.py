"""The engines' own KV-cache methods on the two final models (Mohammad, 2026-11-23: "do both"; advisor point C3).

SnapKV and InfiniGen run only in Hugging Face (simulated), so they are replaced here by what the engines run for
real: vLLM's 8-bit (FP8) KV cache and llama.cpp's 4-bit (q4_0) KV cache. Each model (FP16 baseline and the two final
models without their Stage 3 step) is measured with the engines' 16-bit cache and with their compressed cache, so
the only difference between a pair of rows is the KV cache. Writes kv_engines.csv and summary.md into
<OUTPUT_ROOT>/<run id>/review/ and prints that folder.

    python scripts/kv_engines_20261123.py [RUN_ID]

Settings come from main.py's build_config() (a Colab wrapper sets VLLM_PYTHON and LLAMACPP_DIR etc. on main).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

from review_20261122 import FINALS, _cfg, _f, _summary, _write_csv  # noqa: E402

RUN_ID = "20261123-kv-engines"
PRUNE = {"rtn_snapkv": 0.0, "omniquant_infinigen": 0.082}  # each final model's own pruning share
# (label, vLLM kv_cache_dtype, llama.cpp cache type): 16-bit first, so each compressed row has its partner
KV_SETTINGS = [("16-bit KV cache", "auto", "f16"), ("compressed KV cache", "fp8", "q4_0")]
COLS = ["vllm_size_gb", "vllm_ppl_val", "vllm_ppl_heldout", "vllm_decode_ms_per_token", "vllm_prefill_ms",
        "vllm_peak_memory_gb", "vllm_kv_cache_tokens", "llamacpp_size_gb", "llamacpp_ppl_val", "llamacpp_ppl_heldout",
        "llamacpp_decode_tokens_per_s", "llamacpp_prefill_tokens_per_s", "llamacpp_kv_cache_gb"]


def points() -> list[tuple[str, str, dict[str, Any]]]:
    """(model name, final it takes its Stage 0 settings from, search point): the FP16 baseline once, then each
    final model with its own Stage 1 and pruning and no Stage 3 step (the engine's KV cache replaces it)."""
    from sdf.search_space import NO_QUANT

    first = next(iter(FINALS))
    out = [("baseline_fp16", first, {**FINALS[first], "stage1_method": NO_QUANT, "prune_ratio_aggressive": 0.0,
                                     "stage3_method": "none"})]
    out += [(name, name, {**f, "prune_ratio_aggressive": PRUNE[name], "stage3_method": "none"})
            for name, f in FINALS.items()]
    return out


def run(run_id: str | None = None, inject: dict[str, Any] | None = None) -> Path:
    from sdf.run import start_run
    from sdf.search import evaluate, technique_space
    from sdf.search_space import NO_QUANT

    rows, out_dir, cfg = [], None, None
    for name, final, point in points():
        f = FINALS[final]
        for label, vllm_kv, lc_kv in KV_SETTINGS:
            cfg = _cfg("run3", run_id or RUN_ID, **{
                "search.joint_plan": False, "search.stage1_methods": [f["stage1_method"], NO_QUANT],
                "search.stage2_methods": [f["stage2_method"]], "search.stage3_methods": ["none"],
                "requirement.target_memory_gb": None, "requirement.target_ppl": None,
                "requirement.target_vllm_ppl": None, "requirement.target_latency_ms": None,
                "eval.downstream_tasks": [], "eval.trtllm_python": None,
                "eval.vllm_kv_cache_dtype": vllm_kv, "eval.llamacpp_kv_cache_type": lc_kv})
            ctx = start_run(cfg)
            out_dir = ctx.run_dir / "review"
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            r = evaluate(ctx, technique_space(cfg).make(point), **(inject or {}))
            rows.append({"model": name, "kv_cache": label, "vllm_kv_cache_dtype": vllm_kv,
                         "llamacpp_kv_cache_type": lc_kv, "stage1": point["stage1_method"],
                         "prune_share": point["prune_ratio_aggressive"],
                         **({"error": r["error"]} if "error" in r else {}), **(r.get("errors") or {}),
                         **r.get("metrics", {}), "dir": r.get("dir"), "wall_clock_s": time.time() - t0})
            _write_csv(out_dir / "kv_engines.csv", rows)
    _summary(out_dir, "The engines' own KV-cache methods on the final models", cfg, [
        "SnapKV and InfiniGen cannot run in vLLM or llama.cpp, so the KV step is replaced by each engine's own "
        "KV-cache storage: vLLM kv_cache_dtype fp8 (8 bits) and llama.cpp cache type q4_0 (4 bits, keys and "
        "values), against each engine's 16-bit cache. Every model is built without its Stage 3 step; the two rows "
        "of a model differ only in the KV cache. vLLM reserves a fixed 1 GB for the cache, so its peak memory "
        "does not change; vllm_kv_cache_tokens is how many tokens that space holds. llamacpp_kv_cache_gb is the "
        "cache llama-perplexity allocated (all windows it reads at once).", "",
        "| model | KV cache | " + " | ".join(COLS) + " | error |", "|---|---|" + "---|" * (len(COLS) + 1),
        *[f"| {r['model']} | {r['kv_cache']} | " + " | ".join(_f(r.get(c)) for c in COLS) + " | "
          + "; ".join(str(v) for k, v in r.items() if k.endswith("error")) + " |" for r in rows]])
    return out_dir


if __name__ == "__main__":
    print(run(*(sys.argv[1:2])))
