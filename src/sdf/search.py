"""Search for the framework settings that meet the deployment requirement (Mohammad, 2026-10-19: "make the framework
dynamic and find the best settings that satisfy perplexity, memory, and latency"; for now memory below 1 GB, then the
lowest perplexity and latency).

A point is one set of settings: the SEARCH_SPACE hyperparameters plus a technique for each stage (technique_space).
evaluate() builds the chained model 0 -> 1 -> 2 -> 3 for it (the Stage 0 plan, the Stage 1 WxAy method, the Stage 2
pruning method at the plan's prune_ratio_aggressive, the Stage 3 KV method; Our method rows only) and measures it
(Stage 4). The searchers (MOBO: Optuna's Gaussian-process sampler; MFBO: successive halving over the number of
evaluation windows; NSGA-III: pymoo) minimise cfg.search.objectives under the requirement, each with the same
budget, and are compared. Held-out perplexity is measured, never optimised.

Outputs in <run_dir>/search/: points/<hash>/ (each evaluated point's Stage 0 and Stage 3 reports), trials.json,
search_report.md, search_results.xlsx (Trials, Pareto, Searcher comparison, Best configs) and the plots.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

from sdf.config import FrameworkConfig, config_hash
from sdf.requirements import RequirementCheck
from sdf.run import RunContext
from sdf.search_space import SEARCH_SPACE, Param, SearchSpace
from sdf.utils.cache import atomic_write_text
from sdf.utils.logging import get_logger

log = get_logger(__name__)

VERSION = 2  # 2: sizes count each method's stored extras (codebooks, outliers, scales). Bump when a change to the chain or the measurement changes a point's numbers (the cache key)
FAILED = 1e9  # objective and constraint value of a point that could not be built or measured


def technique_space(cfg: FrameworkConfig) -> SearchSpace:
    """SEARCH_SPACE plus the technique of each stage. Stage 1 always follows the joint weight x activation plan."""
    s = cfg.search
    return SearchSpace(list(SEARCH_SPACE.params) + [
        Param("stage1_method", 1, s.stage1_methods[0], choices=tuple(s.stage1_methods)),
        Param("stage2_method", 2, s.stage2_methods[0], choices=tuple(s.stage2_methods)),
        Param("stage3_method", 3, s.stage3_methods[0], choices=tuple(s.stage3_methods)),
    ])


def evaluate(ctx: RunContext, point: dict[str, Any], max_windows: int | None = None,
             model_factory: Callable | None = None, tokenizer=None,
             text_loader: Callable | None = None) -> dict[str, Any]:
    """Build and measure the chained model of `point` on `max_windows` evaluation windows (None: cfg.eval's).
    Cached by point and settings, so a point two searchers try is built once. Returns {"metrics", "met",
    "shortfall", "dir"}, or {"error", "dir"} when the setting cannot be built or measured."""
    from sdf.eval import vllm_backend
    from sdf.stage0.run import fp16_key, setup
    from sdf.stages.methods import METHODS

    point = technique_space(ctx.cfg).validate(point)
    cfg = ctx.cfg.with_overrides({"stage0.pruning_levels": [], "stage0.joint_plan": True,
                                  **({"eval.max_windows": max_windows} if max_windows else {})})
    names = [point["stage1_method"], point["stage2_method"], point["stage3_method"]]
    key = {"point": point, "model": asdict(cfg.model), "eval": asdict(cfg.eval), "calibration": asdict(cfg.calibration),
           "stage0": asdict(cfg.stage0), "seed": cfg.run.seed, "data": asdict(cfg.data),
           "hardware": fp16_key(ctx, setup(ctx)[0])["hardware"], "vllm": vllm_backend.VERSION, "search": VERSION,
           "versions": [_version(n, METHODS) for n in names]}
    tdir = ctx.run_dir / "search" / "points" / config_hash(key, length=10)

    def build() -> dict[str, Any]:
        try:
            return _build_and_measure(RunContext(cfg, ctx.run_id, tdir, ctx.cache), point, model_factory, tokenizer,
                                      text_loader)
        except Exception as e:  # noqa: BLE001 - a setting that fails is a result of the search, not a crash
            log.exception("search point failed: %s", point)
            return {"error": f"{type(e).__name__}: {e}"}

    path = ctx.cache.path("search_point", key)
    cached = path.exists()
    if cached:
        result = json.loads(path.read_text(encoding="utf-8"))["value"]
    else:
        result = build()
        if "error" not in result and not result.get("errors") and "not_built" not in result:  # a failure may be passing (a crash, out of
            atomic_write_text(path, json.dumps({"key": key, "value": result}, default=str))  # memory): retried
    if "error" not in result:  # checked after the cache, so a changed limit applies to points already measured
        check = check_requirement(ctx.cfg, result["metrics"])
        result = {**result, "met": check.met is True, "shortfall": check.shortfall}
    return {**result, "dir": str(tdir), "cached": cached}


def check_requirement(cfg: FrameworkConfig, metrics: dict[str, Any]) -> RequirementCheck:
    """The requirement checked with the memory limit on cfg.search.memory_metric, before and after the build alike
    (the requirement on its own prefers model_size_gb, which a simulated method fills with the 16-bit size)."""
    return cfg.requirement.check({**metrics, "model_size_gb": metrics.get(cfg.search.memory_metric)})


def _version(name: str, methods: dict) -> int:
    from sdf.stages.runner import stage1_row

    return stage1_row(name)[0].version if "_with_" in name else methods[name].version


def _build_and_measure(ctx: RunContext, point: dict[str, Any], model_factory, tokenizer, text_loader) -> dict:
    import torch

    from sdf.stage0.pruning_levels import layer_caps
    from sdf.stage0.run import run_stage0, setup
    from sdf.stages.methods import METHODS, series
    from sdf.stages.runner import (Stage0Plans, _capped, _free_memory, framework_plan, plan_metrics, run_stage,
                                   stage1_row)

    cfg = ctx.cfg
    given = model_factory is not None
    if model_factory is None:
        device = setup(ctx)[0]

        def model_factory():
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(
                cfg.model.name, torch_dtype=getattr(torch, cfg.model.dtype)).to(device)
    candidate = SEARCH_SPACE.make({p.name: point[p.name] for p in SEARCH_SPACE.params})
    kw = dict(tokenizer=tokenizer, text_loader=text_loader)
    stage0_dir = run_stage0(ctx, candidate, model=model_factory() if given else None, **kw).outputs["report"].parent
    plans = Stage0Plans.load(stage0_dir)
    first, plan_key = stage1_row(point["stage1_method"] + "_joint")
    prune = METHODS[point["stage2_method"]]
    m = series(first, prune, plan_key)
    def caps(r: float) -> tuple[float, ...]:  # structured pruning only; the model is freed before vLLM starts
        model = model_factory()
        try:
            return tuple(layer_caps(model, r))
        finally:
            del model
            _free_memory()

    plan, _ = _capped(m, framework_plan(m, plans, "prune"), caps, plans.profile.layer_numel)
    size = plan_metrics(plan, plans, cfg, candidate, m.storage, m.overhead)
    if check_requirement(cfg, size).met is False:  # the size follows from the plan: not worth building
        return {"metrics": size, "errors": {}, "not_built": "over the limit, so not built"}
    try:  # from here a model is built, so a failure still used up a try ("attempted")
        out = run_stage(ctx, 3, [point["stage3_method"]], candidate, stage0_dir, only="framework", subdir="stage_3",
                        before={"framework": (replace(m, name=prune.name), plan)}, model_factory=model_factory, **kw)
    except Exception as e:  # noqa: BLE001 - as in evaluate()
        log.exception("search point failed while building: %s", point)
        return {"error": f"{type(e).__name__}: {e}", "attempted": True}
    row = next(r for r in json.loads(out["json"].read_text())["rows"] if r["variant"] == "framework")
    if row["status"] != "ok":
        return {"error": row.get("error") or "the row failed", "attempted": True}
    metrics = {k: v for k, v in row["metrics"].items() if isinstance(v, (int, float))}
    errors = {k: v for k, v in row["info"].items() if k.endswith("_error")}  # e.g. a plan vLLM cannot load
    return {"metrics": metrics, "errors": errors}


# ----------------------------------------------------------------------------------------------------- searchers

class Recorder:
    """Every evaluation a searcher asks for, in order, with its cost: its share of the full evaluation windows for a
    model built for this searcher (even if measuring it then failed), else 0 (a point the plan puts over the limit, a
    crash before building, or a point this searcher already tried on as many windows that came back from the cache).
    Only built models use up the budget."""

    def __init__(self, ctx: RunContext, searcher: str, full_windows: int, **kw: Any):
        self.ctx, self.searcher, self.full, self.kw = ctx, searcher, full_windows, kw
        self.trials: list[dict[str, Any]] = []
        s = ctx.cfg.search
        self.objectives = s.objectives
        self.tries_per_budget = s.max_tries_per_budget  # stops a search whose points are almost all not built

    @property
    def spent(self) -> float:
        return sum(t["cost"] for t in self.trials)

    def more(self, budget: float) -> bool:
        """True while the budget (full tries of built models) is not used up and fewer than budget x max_tries_per_budget tries
        were made."""
        cap = self.tries_per_budget * budget
        if len(self.trials) >= cap and self.spent < budget:
            log.warning("search %s stopped after %d tries: only %.1f of %s full tries built (the rest were not built)",
                        self.searcher, len(self.trials), self.spent, budget)
        return self.spent < budget and len(self.trials) < cap

    def __call__(self, point: dict[str, Any], windows: int | None = None) -> dict[str, Any]:
        windows = windows or self.full
        t0 = time.perf_counter()
        r = evaluate(self.ctx, point, None if windows == self.full else windows, **self.kw)
        values = [r.get("metrics", {}).get(o) for o in self.objectives]
        built = "not_built" not in r and ("error" not in r or r.get("attempted", False))
        repeat = r["cached"] and any(t["point"] == point and t["windows"] == windows and t["built"]
                                     for t in self.trials)  # a failed build is not cached, so its repeat rebuilds
        ok = "error" not in r and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        t = {"searcher": self.searcher, "n": len(self.trials) + 1, "point": point, "windows": windows,
             "cost": windows / self.full if built and not repeat else 0.0, "built": built,
             "over_limit": "not_built" in r,
             "full": windows == self.full, "ok": ok, "met": ok and r.get("met", False),
             "values": values if ok else [FAILED] * len(values),
             "violation": FAILED if "error" in r else violation(r),
             "metrics": r.get("metrics", {}), "dir": r["dir"],
             "time_s": time.perf_counter() - t0, "cached": r["cached"],
             "error": r.get("error") or r.get("not_built") or (None if ok else "; ".join(f"{k}: {v}" for k, v in r.get("errors", {}).items())
                                         or f"no {[o for o, v in zip(self.objectives, values) if v is None]}")}
        self.trials.append(t)
        log.info("search %s #%d (%d windows): %s -> %s", self.searcher, t["n"], windows,
                 {k: point[k] for k in ("stage1_method", "stage2_method", "stage3_method")},
                 "failed: " + str(t["error"]) if not ok else dict(zip(self.objectives, values), met=t["met"]))
        return t


def violation(result: dict[str, Any]) -> float:
    """How far the point misses the requirement (0 when it meets it): the sum of the shortfalls."""
    return float(sum(result.get("shortfall", {}).values())) if not result.get("met") else 0.0


def _sample(space: SearchSpace, rng: random.Random) -> dict[str, Any]:
    return {p.name: rng.choice(p.choices) if p.choices else rng.uniform(p.low, p.high) for p in space.params}


def _decode(space: SearchSpace, x) -> dict[str, Any]:
    """A point from a vector in [0, 1]^n (NSGA-III's encoding): a choice by index, a range linearly."""
    out = {}
    for p, v in zip(space.params, x):
        v = min(max(float(v), 0.0), 1.0)
        out[p.name] = (p.choices[min(int(v * len(p.choices)), len(p.choices) - 1)] if p.choices
                       else p.low + v * (p.high - p.low))
    return out


def run_mobo(rec: Recorder, space: SearchSpace, budget: int, seed: int):
    """Multi-objective Bayesian optimisation: Optuna's Gaussian-process sampler, with the requirement as a
    constraint. GPSampler learns only from completed trials, so a point over the limit (not built, no values) is
    completed with the worst values built so far and its shortfall as the constraint: the sampler learns where the
    limit is. A point that crashed or could not be measured is completed the same way, with constraint 1, so the
    sampler does not propose it again (a failed trial would teach it nothing). Before the first built model there are
    no such values, so those points wait and are told then. Returns the study."""
    import optuna
    from optuna.distributions import CategoricalDistribution, FloatDistribution

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dists = {p.name: CategoricalDistribution(p.choices) if p.choices else FloatDistribution(p.low, p.high)
             for p in space.params}
    study = optuna.create_study(directions=["minimize"] * len(rec.objectives),
                                sampler=optuna.samplers.GPSampler(seed=seed))
    waiting, worst = [], None
    while rec.more(budget):
        trial = study.ask(dists)
        t = rec(dict(trial.params))
        if t["ok"]:
            trial.set_constraint("requirement", t["violation"])  # optuna >= 5
            study.tell(trial, t["values"])
            worst = [max(a, b) for a, b in zip(worst or t["values"], t["values"])]
        else:  # over the limit (its shortfall), or crashed or not measured (1: never a good place)
            trial.set_constraint("requirement", t["violation"] if 0 < t["violation"] < FAILED else 1.0)
            waiting.append(trial)
        if worst:
            for w in waiting:
                study.tell(w, worst)
            waiting = []
    return study


def run_nsga3(rec: Recorder, space: SearchSpace, budget: int, seed: int, pop_size: int = 10) -> None:
    """NSGA-III (pymoo), asked and told one generation at a time; settings are encoded in [0, 1] (_decode)."""
    import numpy as np
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.problem import Problem
    from pymoo.problems.static import StaticProblem
    from pymoo.util.ref_dirs import get_reference_directions

    k = len(rec.objectives)
    problem = Problem(n_var=len(space.params), n_obj=k, n_ieq_constr=1, xl=0.0, xu=1.0)
    ref_dirs = get_reference_directions("das-dennis", k, n_partitions=pop_size - 1 if k == 2 else 4)
    algo = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size)
    algo.setup(problem, termination=("n_eval", budget), seed=seed)
    while rec.more(budget):
        pop = algo.ask()
        pop = pop[: max(1, min(len(pop), int(math.ceil(budget - rec.spent))))]
        ts = [rec(_decode(space, x)) for x in pop.get("X")]
        Evaluator().eval(StaticProblem(problem, F=np.array([t["values"] for t in ts]),
                                       G=np.array([[t["violation"]] for t in ts])), pop)
        algo.tell(infills=pop)


def run_mfbo(rec: Recorder, space: SearchSpace, budget: int, seed: int, eta: int = 3, rungs: int = 3) -> None:
    """Multi-fidelity: successive halving. Each bracket measures eta^(rungs-1) random points on 1/eta^(rungs-1)
    of the evaluation windows, keeps the best 1/eta (requirement met first, then the non-dominated rank, then the
    first objective) on eta times the windows, and so on up to the full windows. Points not built or not measured
    are never promoted."""
    rng = random.Random(seed)
    windows = [max(2, rec.full // eta ** (rungs - 1 - r)) for r in range(rungs)]
    while rec.more(budget):
        points = [_sample(space, rng) for _ in range(eta ** (rungs - 1))]
        for w in windows:
            ts = []
            for p in points:
                if not rec.more(budget):
                    return
                ts.append(rec(p, w))
            ts = [t for t in ts if t["ok"]]  # a point not built or not measured is not worth more windows
            if not ts:
                break
            keep = max(1, len(ts) // eta)
            points = [t["point"] for t in sorted(ts, key=lambda t: _rank_key(t, ts))[:keep]]


def _rank_key(t: dict[str, Any], ts: list[dict[str, Any]]) -> tuple:
    feasible = [o for o in ts if o["met"]]
    dominated_by = sum(_dominates(o["values"], t["values"]) for o in feasible) if t["met"] else 0
    return (not t["met"], t["violation"], dominated_by, t["values"][0])


def _dominates(a: list[float], b: list[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The full-size trials that meet the requirement and no other such trial beats on every objective (one per
    point: a point two searchers found is listed once, from the first)."""
    ok, seen = [], set()
    for t in trials:
        k = json.dumps(t["point"], sort_keys=True)
        if t["full"] and t["met"] and k not in seen:
            seen.add(k)
            ok.append(t)
    return [t for t in ok if not any(_dominates(o["values"], t["values"]) for o in ok)]


def best_per_objective(trials: list[dict[str, Any]], objectives: list[str]) -> dict[str, dict[str, Any] | None]:
    """Per objective, the full-size trial that meets the requirement with its lowest value (None if no trial does);
    a tie goes to the one better on the other objectives, so the best is never beaten on every objective."""
    met = [t for t in trials if t["full"] and t["met"]]
    return {o: min(met, key=lambda t, i=i: (t["values"][i], t["values"]), default=None) for i, o in enumerate(objectives)}


def finalists(trials: list[dict[str, Any]], objectives: list[str]) -> list[dict[str, Any]]:
    """The settings that get the full comparison after the search (project rule: the final trade-off candidates and
    the best per objective): the Pareto trials, then any best-per-objective trial not among them, one per point."""
    out = pareto(trials)
    seen = {json.dumps(t["point"], sort_keys=True) for t in out}
    for t in best_per_objective(trials, objectives).values():
        if t and (k := json.dumps(t["point"], sort_keys=True)) not in seen:
            seen.add(k)
            out.append(t)
    return out


def hypervolume(trials: list[dict[str, Any]], ref: list[float]) -> float:
    """Area (volume) the met, full-size trials dominate up to `ref`; larger is better."""
    import numpy as np
    from pymoo.indicators.hv import HV

    pts = np.array([t["values"] for t in pareto(trials) if all(v < r for v, r in zip(t["values"], ref))])
    return float(HV(ref_point=np.array(ref))(pts)) if len(pts) else 0.0


SEARCHERS = {"mobo": run_mobo, "mfbo": run_mfbo, "nsga3": run_nsga3}


def run_search(ctx: RunContext, model_factory: Callable | None = None, tokenizer=None,
               text_loader: Callable | None = None) -> dict[str, Path]:
    """Every searcher in cfg.search.searchers with the same budget and seed, then the report."""
    from sdf.data import eval_windows
    from sdf.stage0.run import setup

    cfg = ctx.cfg
    s = cfg.search
    if unknown := [n for n in s.searchers if n not in SEARCHERS]:
        raise ValueError(f"unknown searcher(s) {unknown}; known: {sorted(SEARCHERS)}")
    if s.memory_metric != "predicted_weight_memory_gb":  # the limit is checked on the plan, before building
        raise ValueError(f"search.memory_metric must be predicted_weight_memory_gb, not {s.memory_metric!r}: only "
                         "the predicted size is known before a model is built")
    if "ppl_heldout" in s.objectives:
        raise ValueError("ppl_heldout is measured to check the choice, never optimised: use ppl_val")
    if any(o.startswith("vllm_") for o in s.objectives) and not cfg.eval.vllm_python:
        raise ValueError(f"the objectives {s.objectives} need vLLM: set eval.vllm_python (Colab), or every try fails")
    space = technique_space(cfg)
    _, handle, loader = setup(ctx, None, tokenizer, text_loader)
    val, held = eval_windows(loader(cfg.eval.dataset, "test"), handle.tokenizer, cfg.eval.seq_len, cfg.eval.max_windows)
    full = len(val) + len(held)
    kw = dict(model_factory=model_factory, tokenizer=tokenizer, text_loader=text_loader)
    trials: list[dict[str, Any]] = []
    for name in s.searchers:
        rec = Recorder(ctx, name, full, **kw)
        extra = {"eta": s.mfbo_eta} if name == "mfbo" else {}
        SEARCHERS[name](rec, space, s.budget, cfg.run.seed, **extra)
        trials += rec.trials
        atomic_write_text(ctx.run_dir / "search" / "trials.json", json.dumps(trials, indent=2, default=str))
    from sdf.reporting.search import write_search_report

    return write_search_report(ctx, trials, space)


def run_finalists(ctx: RunContext, trials: list[dict[str, Any]], model_factory: Callable | None = None,
                  tokenizer=None, text_loader: Callable | None = None) -> dict[str, Path]:
    """The full comparison of the search's final settings (finalists) on every backend (Mohammad, 2026-10-22:
    "Chained, Standard vs Ours"). Per setting, in <run_dir>/finalists/<n>_<hash>/: its Stage 0 plan, then the same
    chain 0 -> 1 -> 2 -> 3 the search built, each stage's report with FP16, Standard (the same technique with its
    default settings, no Stage 0) and Our method: Stage 1 the WxAy technique (Our method on the joint plan), Stage 2
    its pruning technique after each side's Stage 1 row, Stage 3 its KV technique on each side's Stage 2 model; then
    the master report. The backends cannot run the KV techniques, so on Stage 3 rows they measure the Stage 2
    model (as in every run). finalists/index.json lists each setting and its folder, rewritten after each one; a
    setting that fails is recorded there with its error and the next one runs. Rerun with the same run_id, a setting
    whose master report exists is not built again."""
    ev = ctx.cfg.eval
    if missing := [k for k in ("llamacpp_dir", "llamacpp_convert", "vllm_python", "trtllm_python")
                   if not getattr(ev, k)]:
        raise ValueError(f"the final settings run on all four backends: set eval.{', eval.'.join(missing)}")
    picked = finalists(trials, ctx.cfg.search.objectives)
    if not picked:
        raise ValueError("no full try of the search meets the requirement, so there is no final setting to compare")
    cfg = ctx.cfg.with_overrides({"stage0.pruning_levels": [], "stage0.joint_plan": True})  # as evaluate() builds
    kw = dict(model_factory=model_factory, tokenizer=tokenizer, text_loader=text_loader)
    index, outputs = [], {}
    outputs["finalists_index"] = ctx.run_dir / "finalists" / "index.json"
    for n, t in enumerate(picked, 1):
        point = technique_space(cfg).validate(t["point"])
        fdir = ctx.run_dir / "finalists" / f"{n}_{config_hash(point, length=10)}"
        entry = {"n": n, "dir": str(fdir), "point": point, "searcher": t["searcher"], "try": t["n"],
                 "search_metrics": t["metrics"]}
        log.info("final setting %d of %d: %s", n, len(picked), point)
        if (fdir / "master_report.md").exists():  # finished before this run was stopped: resume after it
            prev = next((e for e in _read_index(outputs["finalists_index"]) if e["dir"] == str(fdir)), {})
            index.append({**entry, **prev})
            continue
        try:
            outputs.update({f"finalist_{n}_{k}": p for k, p in _finalist(RunContext(cfg, ctx.run_id, fdir, ctx.cache),
                                                                           point, entry, **kw).items()})
        except Exception as e:  # noqa: BLE001 - one setting that fails must not lose the others
            log.exception("final setting %d failed", n)
            entry["error"] = f"{type(e).__name__}: {e}"
        index.append(entry)
        atomic_write_text(outputs["finalists_index"], json.dumps(index, indent=2, default=str))  # after each
    return outputs


def _read_index(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _finalist(ctx: RunContext, point: dict[str, Any], entry: dict[str, Any], model_factory: Callable | None,
              tokenizer, text_loader: Callable | None) -> dict[str, Path]:
    """One final setting's Stage 0 and chained Stages 1-3 (run_finalists); a side whose Stage 2 row failed gets no
    Stage 3 row, and `entry` says so."""
    from sdf.reporting.master import write_master
    from sdf.stage0.run import run_stage0
    from sdf.stages.runner import run_stage, stage1_row

    s1, s2, s3 = point["stage1_method"], point["stage2_method"], point["stage3_method"]
    candidate = SEARCH_SPACE.make({p.name: point[p.name] for p in SEARCH_SPACE.params})
    kw = dict(model_factory=model_factory, tokenizer=tokenizer, text_loader=text_loader)
    model = model_factory() if model_factory else None
    stage0_dir = run_stage0(ctx, candidate, model=model, tokenizer=tokenizer,
                            text_loader=text_loader).outputs["report"].parent
    del model
    ours1 = s1 + "_joint"
    run_stage(ctx, 1, [s1], candidate, stage0_dir, ours=stage1_row(ours1)[1], **kw)
    used: dict = {}
    run_stage(ctx, 2, [s2], candidate, stage0_dir, after={"original": s1, "framework": ours1}, used=used,
              ours="prune", **kw)
    before = {v: used[(s2, v)][:2] for v in ("original", "framework") if (s2, v) in used}  # absent: the row failed
    if missing := [v for v in ("original", "framework") if v not in before]:
        entry["stage3_not_run"] = f"the Stage 2 row failed, so no Stage 3 row for: {', '.join(missing)}"
    if before:
        run_stage(ctx, 3, [s3], candidate, stage0_dir, before=before, only=None if len(before) == 2 else next(iter(before)),
                  **kw)
    return write_master(ctx.run_dir)
