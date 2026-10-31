"""The settings search folded into the stages (Mohammad, 2026-10-24: "the search layer makes it look redundant and
unnecessary. we should integrate it into another stage"; option B, then 2026-10-25: "do 1a2a, no refinement for any
stage").

- Stage 1 searches its own settings (the WxAy technique, protection threshold, calibration text and passages, group
  size, SmoothQuant strength) on a model with Stage 1 only, following the joint plan; Stage 2 searches its own
  (pruning technique and share removed) on the uncompressed model pruned by the run's Stage 0 plan. Each runs the
  three searchers with the same budget and seed and keeps its best trade-off line: perplexity against predicted size
  (Stage 1 also against cfg.search.objectives' speed). No limit inside a stage: the limit is for the final model.
- Stage 3 tries every KV technique in cfg.search.stage3_methods once on its own (the technique is its only setting)
  and keeps the lowest validation perplexity; the KV cache does not change the model's size.
- The last step pairs every Stage 1 point on its line with every Stage 2 point on its line, orders the pairs by
  estimated perplexity (the Stage 1 point's plus the Stage 2 point's rise over the uncompressed model) and builds the
  chained model 0 -> 1 -> 2 -> 3 (sdf.search.evaluate, with Stage 3's technique) in that order until
  cfg.search.finalists are built and measured under the limit; a pair whose plan is over the limit is not built.
  The winner has the lowest cfg.search.objectives, in order.

Stage 0's joint refinement is off everywhere (stage0.joint_refine_rounds = 0): each try uses the one-shot joint plan.

Outputs in <run_dir>/stage_search/: stage_1/search/ and stage_2/search/ (each stage's search report, as MODE
"search" writes it), stage_3/ (the KV techniques), final/ (the chained models built), stage_search_report.md and
stage_search_results.json.
"""

from __future__ import annotations

import functools
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable

from sdf.config import FrameworkConfig, config_hash
from sdf.run import RunContext
from sdf.search import (SEARCHERS, Recorder, cached_point, evaluate, pareto, point_key, technique_space)
from sdf.search_space import SEARCH_SPACE, Param, SearchSpace
from sdf.utils.cache import atomic_write_text
from sdf.utils.logging import get_logger

log = get_logger(__name__)

VERSION = 1  # bump when a change here changes a stage try's numbers (the cache key)

# the search-space settings each stage searches; quarot_k_bits is left out: Our method rows do not read it
STAGE_PARAMS = {1: ("sensitive_threshold", "calib_dataset", "calib_samples", "gptq_groupsize", "smoothquant_alpha"),
                2: ("prune_ratio_aggressive",)}
# backends no stage objective reads, so a try does not run them (Stage 1 keeps vLLM, for its speed objective)
NO_BACKENDS = {"eval.vllm_python": None, "eval.trtllm_python": None, "eval.llamacpp_dir": None}
STAGE1_BACKENDS_OFF = {"eval.trtllm_python": None, "eval.llamacpp_dir": None}
NO_REFINE = {"stage0.pruning_levels": [], "stage0.joint_plan": True, "stage0.joint_refine_rounds": 0}


def stage_space(cfg: FrameworkConfig, stage: int) -> SearchSpace:
    """Stage 1's or Stage 2's settings plus its technique."""
    methods = getattr(cfg.search, f"stage{stage}_methods")
    return SearchSpace([p for p in SEARCH_SPACE.params if p.name in STAGE_PARAMS[stage]]
                       + [Param(f"stage{stage}_method", stage, methods[0], choices=tuple(methods))])


def objectives(cfg: FrameworkConfig, stage: int) -> list[str]:
    """Stage 1: cfg.search.objectives plus the memory metric; Stage 2: perplexity and the memory metric."""
    s = cfg.search
    return list(dict.fromkeys((s.objectives if stage == 1 else ["ppl_val"]) + [s.memory_metric]))


def full_candidate(cfg: FrameworkConfig, point: dict[str, Any]) -> dict[str, Any]:
    """The run's settings (cfg.hyperparams) with the point's search-space settings in place."""
    return SEARCH_SPACE.make({**cfg.hyperparams, **{p.name: point[p.name] for p in SEARCH_SPACE.params if p.name in point}})


def evaluate_stage(ctx: RunContext, point: dict[str, Any], max_windows: int | None = None, stage: int = 1,
                   model_factory: Callable | None = None, tokenizer=None,
                   text_loader: Callable | None = None) -> dict[str, Any]:
    """Build and measure one stage's Our method row of `point` (sdf.search.evaluate's arguments and result), cached;
    every measured point "meets" (a stage keeps a trade-off line, the limit is for the final model)."""
    point = stage_space(ctx.cfg, stage).validate(point)
    cfg = ctx.cfg.with_overrides({**NO_REFINE, **(NO_BACKENDS if stage != 1 else STAGE1_BACKENDS_OFF),
                                  **({"eval.max_windows": max_windows} if max_windows else {})})
    # the run's settings the point does not search (cfg.hyperparams) build its Stage 0 plan too
    key = {**point_key(ctx, cfg, point, [point[f"stage{stage}_method"]]), "stage": stage,
           "candidate": full_candidate(cfg, point), "stage_search": VERSION}
    tdir = ctx.run_dir / "points" / config_hash(key, length=10)
    return cached_point(ctx, "stage_search_point", key, tdir, lambda run_ctx: _build_stage(
        run_ctx, stage, point, model_factory, tokenizer, text_loader), cfg, limit=False)


def _build_stage(ctx: RunContext, stage: int, point: dict[str, Any], model_factory, tokenizer,
                 text_loader) -> dict[str, Any]:
    from sdf.stage0.run import run_stage0
    from sdf.stages.runner import run_stage

    candidate = full_candidate(ctx.cfg, point)
    kw = dict(tokenizer=tokenizer, text_loader=text_loader)
    stage0_dir = run_stage0(ctx, candidate, model=model_factory() if model_factory else None,
                            **kw).outputs["report"].parent
    method, plan_key = (point["stage1_method"], "joint_weights") if stage == 1 else (point["stage2_method"], "prune")
    out = run_stage(ctx, stage, [method], candidate, stage0_dir, only="framework", plan_keys=(plan_key,),
                    subdir=f"stage_{stage}", model_factory=model_factory, **kw)
    rows = json.loads(out["json"].read_text())["rows"]
    row = next(r for r in rows if r["variant"] == "framework")
    if row["status"] != "ok":  # the row's model was being built when it failed, so the try used up its cost
        return {"error": row.get("error") or "the row failed", "attempted": True}
    metrics = {k: v for k, v in row["metrics"].items() if isinstance(v, (int, float))}
    fp16 = next((r["metrics"].get("ppl_val") for r in rows if r["variant"] == "fp16"), None)
    if isinstance(fp16, (int, float)) and "ppl_val" in metrics:
        metrics["ppl_rise"] = metrics["ppl_val"] - fp16  # what Stage 2 adds to a Stage 1 point, in the pairing
    return {"metrics": metrics, "errors": {k: v for k, v in row["info"].items() if k.endswith("_error")}}


def pick_kv(ctx: RunContext, candidate: dict[str, Any], model_factory=None, tokenizer=None,
            text_loader=None) -> tuple[str | None, list[dict[str, Any]], dict[str, Path]]:
    """Every Stage 3 technique once (Our method rows, on the uncompressed model), and the one with the lowest
    validation perplexity among those within the KV memory limit, if one is set (None if none is left). Returns (technique, finished rows, report paths)."""
    from sdf.stage0.run import run_stage0
    from sdf.stages.runner import run_stage

    cfg = ctx.cfg.with_overrides({**NO_REFINE, **NO_BACKENDS})
    run_ctx = RunContext(cfg, ctx.run_id, ctx.run_dir / "stage_3", ctx.cache)
    kw = dict(tokenizer=tokenizer, text_loader=text_loader)
    stage0_dir = run_stage0(run_ctx, candidate, model=model_factory() if model_factory else None,
                            **kw).outputs["report"].parent
    out = run_stage(run_ctx, 3, cfg.search.stage3_methods, candidate, stage0_dir, only="framework", subdir="stage_3",
                    model_factory=model_factory, **kw)
    rows = [r for r in json.loads(out["json"].read_text())["rows"]
            if r["variant"] == "framework" and r["status"] == "ok" and _finite(r["metrics"].get("ppl_val"))
            and "kv_budget_gb" not in cfg.requirement.check(r["metrics"]).shortfall]
    best = min(rows, key=lambda r: r["metrics"]["ppl_val"], default=None)
    return (best["method"] if best else None), rows, out


def pair_up(ctx: RunContext, ones: list[dict[str, Any]], twos: list[dict[str, Any]], kv: str, finalists: int,
            **kw: Any) -> list[dict[str, Any]]:
    """Chained models of Stage 1 x Stage 2 trade-off points (trials), best estimated perplexity first, until
    `finalists` are measured under the limit, or finalists x search.max_tries_per_budget pairs were tried (a limit no
    pair meets would otherwise build them all). Returns every pair tried, in order, with its evaluate() result."""
    base = full_candidate(ctx.cfg, {})
    # a Stage 2 point without its rise (the uncompressed row was not measured) goes last
    estimate = lambda a, b: a["metrics"]["ppl_val"] + b["metrics"].get("ppl_rise", math.inf)  # noqa: E731
    tried, done = [], 0
    for a, b in sorted(itertools.product(ones, twos), key=lambda ab: estimate(*ab)):
        if done >= finalists or len(tried) >= finalists * ctx.cfg.search.max_tries_per_budget:
            break
        point = technique_space(ctx.cfg).validate({**base, **a["point"], **b["point"], "stage3_method": kv})
        r = evaluate(ctx, point, **kw)
        ok = "error" not in r and "not_built" not in r and all(_finite(r["metrics"].get(o))
                                                             for o in ctx.cfg.search.objectives)
        done += ok and r.get("met", False)
        tried.append({"stage1": [a["searcher"], a["n"]], "stage2": [b["searcher"], b["n"]], "point": point,
                      "estimate": estimate(a, b), "ok": ok, "met": ok and r.get("met", False),
                      **{k: r.get(k) for k in ("metrics", "error", "not_built", "dir", "cached")}})
        log.info("stage search pair %d: estimate %.4g -> %s", len(tried), tried[-1]["estimate"],
                 r.get("error") or r.get("not_built") or {o: r["metrics"].get(o) for o in ctx.cfg.search.objectives})
    return tried


def winner(cfg: FrameworkConfig, tried: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The measured pair under the limit with the lowest objectives, in order."""
    return min((t for t in tried if t["met"]), key=lambda t: [t["metrics"][o] for o in cfg.search.objectives],
               default=None)


def _searchers(names: list[str]) -> str:
    """The search methods in words, e.g. "2 search methods (MOBO, MFBO)"."""
    words = [{"nsga3": "NSGA-III"}.get(n, n.upper()) for n in names]
    return f"{len(words)} search method{'s' * (len(words) != 1)} ({', '.join(words)})"


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


_INTRO = {
    1: "Stage 1 searched its own settings: the quantization technique (weights and activations stored with fewer "
       "bits), the protection threshold, the calibration text and passages, the group size and the SmoothQuant "
       "strength. Each try built a model with Stage 1 only, following the joint weight and activation plan (no "
       "refinement), and measured it. There is no size limit here: the best trade-off line between perplexity, "
       "speed and size goes on to the pairing step, where Stage 2's pruning brings a pair under the limit.",
    2: "Stage 2 searched its own settings: the pruning technique and the share removed. Each try pruned the "
       "uncompressed model following the run's Stage 0 plan and measured it. There is no size limit here: the "
       "best trade-off line between perplexity and size goes on to the pairing step.",
}


def run_stage_search(ctx: RunContext, model_factory: Callable | None = None, tokenizer=None,
                     text_loader: Callable | None = None) -> dict[str, Path]:
    """Stage 1 and Stage 2 searches, the Stage 3 pick, the pairing step, then the report."""
    if ctx.cfg.search.low_threshold_max > 0:
        raise ValueError("search.low_threshold_max (3-bit layers) works in MODE \"search\" only")
    from sdf.data import eval_windows
    from sdf.reporting.search import write_search_report
    from sdf.stage0.run import setup

    s = ctx.cfg.search
    if unknown := [n for n in s.searchers if n not in SEARCHERS]:
        raise ValueError(f"unknown searcher(s) {unknown}; known: {sorted(SEARCHERS)}")
    if s.memory_metric != "predicted_weight_memory_gb":
        raise ValueError(f"search.memory_metric must be predicted_weight_memory_gb, not {s.memory_metric!r}: only "
                         "the predicted size is known before a model is built")
    if "ppl_heldout" in s.objectives or s.objectives[0] != "ppl_val":
        raise ValueError(f"search.objectives must start with ppl_val (held-out perplexity is never optimised), not "
                         f"{s.objectives}")
    if any(o.startswith("vllm_") for o in s.objectives) and not ctx.cfg.eval.vllm_python:
        raise ValueError(f"the objectives {s.objectives} need vLLM: set eval.vllm_python (Colab), or every try fails")
    req = ctx.cfg.requirement
    if (req.target_latency_ms is not None and s.latency_metric.startswith("vllm_")
            or req.target_ppl is not None and s.ppl_metric.startswith("vllm_")) and not ctx.cfg.eval.vllm_python:
        raise ValueError("a limit checked on a vLLM metric needs vLLM: set eval.vllm_python (Colab), or no pair can "
                         "meet it")
    if ctx.cfg.requirement.target_memory_gb is None:
        raise ValueError("set requirement.target_memory_gb: the pairing step keeps the pairs under it")
    root = ctx.run_dir / "stage_search"
    cfg = ctx.cfg.with_overrides({"stage0.joint_refine_rounds": 0})
    _, handle, loader = setup(ctx, None, tokenizer, text_loader)
    val, held = eval_windows(loader(cfg.eval.dataset, "test"), handle.tokenizer, cfg.eval.seq_len, cfg.eval.max_windows)
    kw = dict(model_factory=model_factory, tokenizer=tokenizer, text_loader=text_loader)
    outputs: dict[str, Path] = {}
    fronts: dict[int, list[dict[str, Any]]] = {}
    for stage in (1, 2):
        scfg = cfg.with_overrides({"search.objectives": objectives(cfg, stage), "requirement.target_memory_gb": None})
        sctx = RunContext(scfg, ctx.run_id, root / f"stage_{stage}", ctx.cache)
        space = stage_space(scfg, stage)
        trials: list[dict[str, Any]] = []
        for name in s.searchers:
            rec = Recorder(sctx, name, len(val) + len(held), evaluator=functools.partial(evaluate_stage, stage=stage),
                           **kw)
            SEARCHERS[name](rec, space, s.budget, cfg.run.seed, **({"eta": s.mfbo_eta} if name == "mfbo" else {}))
            trials += rec.trials
        outputs.update({f"stage_{stage}_{k}": p for k, p in write_search_report(sctx, trials, space,
                                                                                  _INTRO[stage]).items()})
        fronts[stage] = pareto(trials)
        if not fronts[stage]:
            raise RuntimeError(f"the Stage {stage} search measured no setting; see {outputs[f'stage_{stage}_search_report']}")
    kv, kv_rows, kv_out = pick_kv(RunContext(cfg, ctx.run_id, root, ctx.cache), full_candidate(cfg, {}), **kw)
    outputs.update({f"stage_3_{k}": p for k, p in kv_out.items()})
    if kv is None:
        raise RuntimeError(f"every Stage 3 technique failed; see {kv_out.get('report')}")
    tried = pair_up(RunContext(cfg, ctx.run_id, root / "final", ctx.cache), fronts[1], fronts[2], kv, s.finalists, **kw)
    best = winner(cfg, tried)
    results = {"stage_1_front": fronts[1], "stage_2_front": fronts[2], "stage_3_technique": kv,
               "stage_3_rows": [{"method": r["method"], "metrics": r["metrics"]} for r in kv_rows],
               "pairs": tried, "winner": best}
    outputs["stage_search_results"] = root / "stage_search_results.json"
    atomic_write_text(outputs["stage_search_results"], json.dumps(results, indent=2, default=str))
    outputs["stage_search_report"] = root / "stage_search_report.md"
    atomic_write_text(outputs["stage_search_report"], _report(cfg, fronts, kv, kv_rows, tried, best))
    return outputs


def _report(cfg: FrameworkConfig, fronts, kv: str, kv_rows, tried, best) -> str:
    from sdf.reporting.metrics import label
    from sdf.reporting.search import SETTING_NAMES, _fmt, _md, _req, setting_value

    s = cfg.search
    shown = list(dict.fromkeys(s.objectives + [s.memory_metric, "ppl_heldout"]))
    head = [label(m) + " (lower is better)" for m in shown]

    def front_table(stage: int) -> str:
        names = [p.name for p in stage_space(cfg, stage).params]
        return _md([["#"] + [SETTING_NAMES.get(n, n) for n in names] + head] + [[str(i + 1)] + [setting_value(n, t["point"][n]) for n in names]
                                             + [_fmt(t["metrics"].get(m)) for m in shown]
                                             for i, t in enumerate(sorted(fronts[stage], key=lambda t: t["values"][0]))])

    def outcome(t: dict[str, Any]) -> str:
        return ("measured, under the limit" if t["met"] else t["not_built"] or t["error"]
                or ("measured, over the limit" if t["ok"] else "measured, but an objective is missing"))

    lines = [
        "# Settings search inside the stages", "",
        "## What this shows", "",
        f"Each stage searched its own settings with {_searchers(s.searchers)} instead of one search over the whole "
        "chain: Stage 1 (quantization) and Stage 2 (pruning) each kept their best trade-off line, and Stage 3 tried "
        "every KV-cache technique once. The pairing step then put the best Stage 1 and Stage 2 settings together, "
        "built only the most promising pairs as full models (with Stage 3's technique) and kept the best one that "
        f"meets the limit ({_req(cfg)}). Stage 0's refinement was off: every plan is the one-shot joint plan.", "",
        "## What it means", "",
        (f"The final model: {setting_value('stage1_method', best['point']['stage1_method'])}, "
         f"{setting_value('stage2_method', best['point']['stage2_method'])} at "
         f"{setting_value('prune_ratio_aggressive', best['point']['prune_ratio_aggressive'])}, "
         f"{setting_value('stage3_method', kv)}; "
         + ", ".join(f"{label(m).lower()} {_fmt(best['metrics'].get(m))}" for m in shown) + "."
         if best else f"No pair met the limit among the {len(tried)} tried; see the pairs below."), "",
        "## Stage 1: best trade-off line", "", front_table(1),
        "## Stage 2: best trade-off line", "", front_table(2),
        "Perplexity rise over the uncompressed model is what the pairing step adds to a Stage 1 point to estimate "
        "the pair. The Stage 2 points were measured with the run's own Stage 0 settings, while a pair is built with "
        "its Stage 1 point's, so the estimate only sets the order; the built pairs' measured numbers decide.", "",
        "## Stage 3: KV-cache techniques", "",
        _md([["Technique", "Validation perplexity (lower is better)", "Held-out perplexity (lower is better)"]]
            + [[setting_value("stage3_method", r["method"]) + (" (chosen)" if r["method"] == kv else ""),
                _fmt(r["metrics"].get("ppl_val")), _fmt(r["metrics"].get("ppl_heldout"))] for r in kv_rows]),
        "## Pairs tried, best estimate first", "",
        _md([["#", "Stage 1 point (search method, try)", "Stage 2 point (search method, try)", "Estimated perplexity",
              "Outcome"] + head]
            + [[str(i + 1), f"{t['stage1'][0]} #{t['stage1'][1]}", f"{t['stage2'][0]} #{t['stage2'][1]}",
                _fmt(t["estimate"]), outcome(t)] + [_fmt((t["metrics"] or {}).get(m)) for m in shown]
               for i, t in enumerate(tried)]),
        f"Pairs are built in this order until {s.finalists} are measured under the limit; a pair whose plan is over "
        "the limit is not built. Held-out perplexity was measured but never used to choose.", "",
    ]
    return "\n".join(lines)
