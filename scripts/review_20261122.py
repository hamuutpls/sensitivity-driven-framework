"""The three runs for the paper review of 22 Nov 2026 (spec: thesis/review-20261122/experiment-spec.md in the project
files; the functions are in src/sdf/review.py). Each run writes CSV files and a short summary.md into
<OUTPUT_ROOT>/<run id>/review/ and prints that folder.

    python scripts/review_20261122.py run1 [RUN_ID]   # per-layer damage and Spearman correlations
    python scripts/review_20261122.py run2 [RUN_ID]   # per-layer plan vs inverse / random / uniform 4 bits (needs run1)
    python scripts/review_20261122.py run3 [RUN_ID]   # stage ablation of the two final models (vLLM + accuracy)

Settings come from main.py's build_config() (so a Colab wrapper can set VLLM_PYTHON etc. on main), with the
overrides each run needs below. Run 2 reads run 1's per-layer CSV (the skip rise on the calibration text is the
Stage 0 score): pass RUN1_CSV in the environment or keep both runs under one OUTPUT_ROOT and run 1's default id.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

RUN_IDS = {"run1": "20261122-review-run1-layer-damage", "run2": "20261122-review-run2-equal-size",
           "run3": "20261122-review-run3-ablation"}
METHODS_RUN2 = ["rtn", "gptq", "omniquant", "squeezellm"]
TASKS = ["arc_easy", "piqa", "hellaswag", "winogrande"]
# the two final models within both limits (spec, Run 3), as search points
FINALS = {
    "rtn_snapkv": {"sensitive_threshold": 0.8, "gptq_groupsize": 64, "calib_dataset": "c4", "calib_samples": 128,
                   "stage1_method": "rtn", "stage2_method": "unstructured_prune", "stage3_method": "snapkv"},
    "omniquant_infinigen": {"sensitive_threshold": 0.857, "gptq_groupsize": -1, "calib_dataset": "c4",
                            "calib_samples": 32, "stage1_method": "omniquant", "stage2_method": "unstructured_prune",
                            "stage3_method": "infinigen"},
}
ABLATION_PRUNE = 0.082  # the OmniQuant model's pruning share; the RTN model's is 0, so its pruning rows use this


def _cfg(run: str, run_id: str | None, **over: Any):
    import main

    cfg = main.build_config()
    return cfg.with_overrides({"run.run_id": run_id or RUN_IDS[run], **over})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def _commit() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _common(cfg, inject: dict[str, Any] | None = None):
    from sdf.data import eval_windows
    from sdf.run import start_run
    from sdf.search_space import SEARCH_SPACE
    from sdf.stage0.run import calib_batches, setup

    ctx = start_run(cfg)
    inject = inject or {}  # tests: model_factory, tokenizer, text_loader instead of the Hugging Face downloads
    model = inject["model_factory"]() if "model_factory" in inject else None
    device, handle, text_loader = setup(ctx, model, inject.get("tokenizer"), inject.get("text_loader"))
    candidate = SEARCH_SPACE.make({p.name: cfg.hyperparams[p.name] for p in SEARCH_SPACE.params if p.name in cfg.hyperparams})
    calib = calib_batches(ctx, candidate, handle, text_loader, candidate["calib_samples"])
    val, held = eval_windows(text_loader(cfg.eval.dataset, "test"), handle.tokenizer, cfg.eval.seq_len,
                             cfg.eval.max_windows)
    out = ctx.run_dir / "review"
    out.mkdir(parents=True, exist_ok=True)
    return ctx, device, handle, candidate, calib, val, held, out


def run1(run_id: str | None = None, inject: dict[str, Any] | None = None) -> Path:
    from sdf.review import DAMAGE, layer_damage, spearman_table
    from sdf.stage0.sensitivity import int_zero

    cfg = _cfg("run1", run_id, **{"hyperparams.calib_dataset": "wikitext2", "hyperparams.calib_samples": 64})
    ctx, device, handle, candidate, calib, val, held, out = _common(cfg, inject)
    model = handle.model(cfg.stage0.profile_dtype)  # the dtype Stage 0 scores layers in (float32), as in the paper
    t0 = time.time()
    rows = layer_damage(model, {"calib": calib, "val": list(val.split(1))}, calib, group_size=128,
                        prune_ratio=0.25, int_zero=int_zero(cfg.stage0), device=device)
    _write_csv(out / "layer_damage.csv", rows)
    corr = []
    for text in ("calib", "val"):
        cols = {k: [r[f"{k}_{text}"] for r in rows] for k in DAMAGE}
        corr += [{"text": text, **c} for c in spearman_table(cols)]
    _write_csv(out / "spearman.csv", corr)
    _summary(out, "Run 1: does the layer-skipping score predict compression damage?", cfg, [
        f"{len(rows)} decoder layers; each changed alone. Calibration text: {candidate['calib_dataset']} train, "
        f"{len(calib)} batches of {cfg.calibration.seq_len} tokens (the text the score uses); model in "
        f"{cfg.stage0.profile_dtype} (as Stage 0 scores layers); validation: "
        f"{len(val)} windows of {cfg.eval.seq_len} tokens. Rounding: round-to-nearest, group 128, integer zero point. "
        "Pruning: 25% of each Linear's weights per output row, by magnitude and by Wanda (calibration inputs).",
        f"Wall clock {time.time() - t0:.0f} s.", "", "| text | a | b | Spearman | layers |", "|---|---|---|---|---|",
        *[f"| {c['text']} | {c['a']} | {c['b']} | {c['spearman']:.3f} | {c['layers']} |" for c in corr]])
    return out


def run2(run_id: str | None = None, run1_csv: str | None = None, inject: dict[str, Any] | None = None) -> Path:
    import torch

    from sdf.eval.metrics import perplexity
    from sdf.review import average_bits, bit_plans, bits_plan, equal_ppl_table, plan_row, ranked
    from sdf.stage0.planner import plan_compression, uniform_plan
    from sdf.stage0.run import weight_cost
    from sdf.stage0.sensitivity import SensitivityProfile, layer_shapes, normalize
    from sdf.stages.methods import METHODS, MethodCall

    cfg = _cfg("run2", run_id, **{"hyperparams.calib_dataset": "wikitext2", "hyperparams.calib_samples": 64,
                                  "hyperparams.gptq_groupsize": 128})
    ctx, device, handle, candidate, calib, val, held, out = _common(cfg, inject)
    path = Path(run1_csv or os.environ.get("RUN1_CSV") or
                Path(cfg.run.output_root) / RUN_IDS["run1"] / "review" / "layer_damage.csv")
    with path.open(encoding="utf-8") as f:
        raw = [float(r["skip_calib"]) for r in csv.DictReader(f)]
    scores = normalize(raw, cfg.stage0.normalization)
    model0 = handle.model(cfg.model.dtype)
    numel, rows_, other = layer_shapes(model0)
    profile = SensitivityProfile(raw, numel, rows_, other, "layer_removal")
    template = plan_compression(scores, 0.5, 0.0, 8, 4)
    n = len(raw)
    plans: dict[str, Any] = {}
    ks = [len(plan_compression(scores, thr, 0.0, 8, 4).protected_layers) for thr in (0.5, 0.8)]  # 11, 5 on TinyLlama
    for k in ks:
        for name, p in bit_plans(template, k).items():
            plans[f"A_k{k}_{name}"] = (p, "A", k, name.split("_")[0])  # family: sensitivity, inverse, random
    order = ranked(template)
    for k in (5, 4):
        plans[f"B_top{k}_8_rest_3"] = (bits_plan(template, order[:k], 8, 3, f"top{k}_8_rest_3"), "B", k,
                                       "top_k_8_rest_3")
    plans["B_uniform_4"] = (uniform_plan(scores, 4, 0.0), "B", 0, "uniform_4")
    # C: the threshold sweep (bits only, 8/4) measured again here under the same settings as the rows above; the
    # 2026-10-04 sweep used older code with pruning and guards, so its numbers are not comparable
    seen = set(ks)
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        p = plan_compression(scores, thr, 0.0, 8, 4)
        if (k := len(p.protected_layers)) not in seen:
            seen.add(k)
            plans[f"C_sweep_thr{thr}"] = (bits_plan(template, p.protected_layers, 8, 4, f"sweep_thr{thr}"), "C", k,
                                          "sensitivity")
    cpu = {k: v.detach().to("cpu", copy=True) for k, v in model0.state_dict().items()}

    def fresh():
        model0.load_state_dict(cpu)
        return model0

    def measure(model) -> dict[str, float]:
        return {"ppl_val": perplexity(model, val, device)[0], "ppl_heldout": perplexity(model, held, device)[0]}

    results = []
    for mname in METHODS_RUN2:
        m = METHODS[mname]
        cost = weight_cost(profile, 128, cfg.stage0, None, overhead=m.overhead)
        for pname, (plan, part, k, family) in plans.items():
            t0 = time.time()
            row = {"method": mname, "plan": pname, "part": part, "family": family, "layers_at_8": k,
                   "layers_8bit": " ".join(str(lp.layer) for lp in plan.layers if lp.bit_width == 8),
                   "avg_bits_plan": average_bits(plan, numel),
                   "avg_bits_per_weight": cost(plan).avg_bits_per_weight,
                   "predicted_weight_memory_gb": cost(plan).weight_memory_gb}
            try:
                row.update(plan_row(m, plan, fresh(), lambda mod, p: MethodCall(mod, p, candidate, cfg,
                                                                                lambda: calib), measure))
            except Exception as e:  # noqa: BLE001 - a row that fails is reported, the run goes on
                row["error"] = f"{type(e).__name__}: {e}"
            row["wall_clock_s"] = time.time() - t0
            results.append(row)
            _write_csv(out / "equal_size.csv", results)  # after every row, so a lost machine keeps what was done
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # C: the smallest row reaching the validation perplexity of uniform 4 bits and of the plan, per method
    equal = equal_ppl_table(results, ["B_uniform_4", *[f"A_k{k}_sensitivity" for k in ks]])
    _write_csv(out / "equal_ppl.csv", equal)
    _summary(out, "Run 2: the per-layer plan against other plans at equal size", cfg, [
        f"{n} layers; scores from {path} (skip rise on the calibration text, rank-normalised). Weights only, "
        "group 128 for round-to-nearest, GPTQ and OmniQuant (SqueezeLLM has no groups: one lookup table per output "
        "channel), Hugging Face perplexity (simulated rounding); size is the predicted weight memory (no llama.cpp "
        "file). A: k layers at 8 bits, the rest at 4: the plan (k most sensitive), the inverse (k least) and 5 random "
        "choices. B: the 5 or 4 most sensitive at 8 bits, the rest at 3, against 4 bits everywhere. C: the threshold "
        "sweep 0.1-0.9 (8/4 bits) measured again here; equal_ppl.csv gives, per method, target and family of plans, "
        "the smallest row reaching the target's validation perplexity (a target reaches itself).", "",
        "| method | plan | avg bits | GB | ppl val | ppl held-out |", "|---|---|---|---|---|---|",
        *[f"| {r['method']} | {r['plan']} | {r['avg_bits_per_weight']:.3f} | {r['predicted_weight_memory_gb']:.3f} | "
          f"{_f(r.get('ppl_val'))} | {_f(r.get('ppl_heldout'))} |" for r in results]])
    return out


def run3(run_id: str | None = None, finals: list[str] | None = None, inject: dict[str, Any] | None = None) -> Path:
    from sdf.review import VLLM_SAME, ablation_points, copy_backend
    from sdf.run import RunContext, start_run
    from sdf.search import evaluate, technique_space
    from sdf.search_space import NO_QUANT

    out_dir = None
    rows = []
    for name in finals or list(FINALS):
        f = FINALS[name]
        cfg = _cfg("run3", (run_id or RUN_IDS["run3"]), **{
            "search.joint_plan": False, "search.stage1_methods": [f["stage1_method"], NO_QUANT],
            "search.stage2_methods": [f["stage2_method"]], "search.stage3_methods": [f["stage3_method"], "none"],
            "requirement.target_memory_gb": None, "requirement.target_ppl": None,
            "requirement.target_vllm_ppl": None, "requirement.target_latency_ms": None,
            "eval.downstream_tasks": TASKS, "eval.llamacpp_dir": None, "eval.trtllm_python": None})
        ctx = start_run(cfg)
        # E3, E5 and E6 run vLLM on the same model as Baseline, E1 and E4 (no KV step in vLLM): measured once there
        no_vllm = RunContext(cfg.with_overrides({"eval.vllm_python": None}), ctx.run_id, ctx.run_dir, ctx.cache)
        out_dir = ctx.run_dir / "review"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = technique_space(cfg).make({k: v for k, v in f.items()})
        mine = []
        for row, point in ablation_points(base, ABLATION_PRUNE):
            t0 = time.time()
            r = evaluate(no_vllm if row in VLLM_SAME else ctx, point, **(inject or {}))
            m = r.get("metrics", {})
            mine.append({"model": name, "row": row, "stage1": point["stage1_method"],
                         "prune_share": point["prune_ratio_aggressive"], "stage3": point["stage3_method"],
                         **({"error": r["error"]} if "error" in r else {}),
                         **{k: (r.get("errors") or {}).get(k) for k in (r.get("errors") or {})},
                         **{k: v for k, v in m.items()}, "dir": r.get("dir"), "wall_clock_s": time.time() - t0})
            _write_csv(out_dir / "ablation.csv", rows + copy_backend(mine))
        rows += copy_backend(mine)
    cols = ["ppl_val", "ppl_heldout", *[f"acc_{t}" for t in TASKS], "downstream_acc_mean", "vllm_size_gb",
            "vllm_peak_memory_gb", "vllm_decode_ms_per_token", "vllm_ppl_val", "predicted_kv_memory_gb"]
    _summary(out_dir, "Run 3: ablation of the three stages", cfg, [
        "Each final model's settings with stages left out: no quantization (16-bit weights), pruning share 0, no KV "
        f"method. Pruning rows use unstructured pruning (Wanda) at {ABLATION_PRUNE} on the layers below each model's own "
        "threshold (so the RTN model, threshold 0.800, prunes a different set of layers than the OmniQuant model, "
        "0.857). vLLM runs the model before the KV step "
        "(it cannot run it), so vLLM was measured once, on Baseline, E1 and E4, and those numbers are repeated for "
        "E3, E5 and E6 (column vllm_from names the row they come from).", "",
        "| model | row | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols),
        *[f"| {r['model']} | {r['row']} | " + " | ".join(_f(r.get(c)) for c in cols) + " |" for r in rows]])
    return out_dir


def _f(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, (int, float)) and math.isfinite(v) else ("" if v is None else str(v))


def _summary(out: Path, title: str, cfg, lines: list[str]) -> None:
    head = [f"# {title}", "", f"Commit {_commit()}; model {cfg.model.name}; seed {cfg.run.seed}; written "
            f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.", ""]
    (out / "summary.md").write_text("\n".join(head + lines) + "\n", encoding="utf-8")
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what not in ("run1", "run2", "run3"):
        sys.exit(__doc__)
    print(globals()[what](*(sys.argv[2:3])))
