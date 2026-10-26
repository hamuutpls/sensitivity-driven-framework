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

import csv
import json
import math
import os
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
    point = technique_space(ctx.cfg).validate(point)
    cfg = ctx.cfg.with_overrides({"stage0.pruning_levels": [], "stage0.joint_plan": True,
                                  **({"eval.max_windows": max_windows} if max_windows else {})})
    key = point_key(ctx, cfg, point, [point["stage1_method"], point["stage2_method"], point["stage3_method"]])
    tdir = ctx.run_dir / "search" / "points" / config_hash(key, length=10)
    return cached_point(ctx, "search_point", key, tdir, lambda run_ctx: _build_and_measure(
        run_ctx, point, model_factory, tokenizer, text_loader), cfg)


def point_key(ctx: RunContext, cfg: FrameworkConfig, point: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """What decides a point's numbers: the point, the settings `cfg` builds it with and its techniques' versions."""
    from sdf.eval import vllm_backend
    from sdf.stage0.run import fp16_key, setup
    from sdf.stages.methods import METHODS

    return {"point": point, "model": asdict(cfg.model), "eval": asdict(cfg.eval), "calibration": asdict(cfg.calibration),
            "stage0": asdict(cfg.stage0), "seed": cfg.run.seed, "data": asdict(cfg.data),
            "hardware": fp16_key(ctx, setup(ctx)[0])["hardware"], "vllm": vllm_backend.VERSION, "search": VERSION,
            "versions": [_version(n, METHODS) for n in names]}


def cached_point(ctx: RunContext, kind: str, key: dict[str, Any], tdir: Path, build: Callable[[RunContext], dict],
                 cfg: FrameworkConfig, limit: bool = True) -> dict[str, Any]:
    """`build` (given a RunContext writing into `tdir`) cached under `kind` and `key`; a failure is returned as
    {"error"} and not cached, nor is a point not built. `limit`: check ctx.cfg's requirement on the metrics (after
    the cache, so a changed limit applies to points already measured); without it every measured point meets it
    (a per-stage search keeps a trade-off line, not a limit)."""
    point = key["point"]

    def run() -> dict[str, Any]:
        try:
            return build(RunContext(cfg, ctx.run_id, tdir, ctx.cache))
        except Exception as e:  # noqa: BLE001 - a setting that fails is a result of the search, not a crash
            log.exception("search point failed: %s", point)
            return {"error": f"{type(e).__name__}: {e}"}

    path = ctx.cache.path(kind, key)
    cached = path.exists()
    if cached:
        result = json.loads(path.read_text(encoding="utf-8"))["value"]
    else:
        result = run()
        if "error" not in result and not result.get("errors") and "not_built" not in result:  # a failure may be passing (a crash, out of
            atomic_write_text(path, json.dumps({"key": key, "value": result}, default=str))  # memory): retried
    if "error" not in result:
        check = check_requirement(ctx.cfg, result["metrics"]) if limit else None
        result = {**result, "met": check.met is True if check else True, "shortfall": check.shortfall if check else {}}
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
    Only built models use up the budget. `evaluator`: what measures a point, with evaluate()'s arguments and result
    (default evaluate(), the chained model); a per-stage search passes its own stage's."""

    def __init__(self, ctx: RunContext, searcher: str, full_windows: int, evaluator: Callable | None = None,
                 tried: TriedLog | None = None, **kw: Any):
        self.ctx, self.searcher, self.full, self.evaluator, self.kw = ctx, searcher, full_windows, evaluator, kw
        self.tried = tried  # combinations tried in earlier runs: their result is reused, never rebuilt
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
        earlier = self.tried.get(self.ctx, point, windows) if self.tried else None
        r = earlier or (self.evaluator or evaluate)(self.ctx, point, None if windows == self.full else windows, **self.kw)
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
        t["earlier_run"] = earlier["run_id"] if earlier else None
        if self.tried and not earlier:
            self.tried.add(self.ctx, self.searcher, t, r)
        self.trials.append(t)
        log.info("search %s #%d (%d windows): %s -> %s", self.searcher, t["n"], windows,
                 {k: v for k, v in point.items() if k.endswith("_method")},  # a per-stage search has one
                 "failed: " + str(t["error"]) if not ok else dict(zip(self.objectives, values), met=t["met"]))
        return t


class TriedLog:
    """Every combination the search has tried, in a CSV kept with the cache, so it outlives a run (Mohammad,
    2026-10-26: "adding tries to our runtime must not retry combinations we already tried", "maybe we should make a
    csv of all the combinations"). A combination is the point (numbers to 6 decimals) on a number of windows, under
    the same settings and technique versions as the point's cache key, so MFBO can still measure a point on more
    windows and a changed setting or technique makes it new. Proposed again in a later run, it is not built: its
    earlier result comes back, the requirement checked again, and it counts as that run's try (so 10 more tries =
    the budget raised by 10 with the same seed). A try that failed is logged but built again (a failure may pass, as
    in the cache), and a point not built (over a size limit) is not logged. Within one run the cache builds a point
    once."""

    METRICS = ("ppl_val", "ppl_heldout", "predicted_weight_memory_gb")

    def __init__(self, path: Path, run_id: str):
        self.path, self.run_id = Path(path), run_id
        self.rows: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:  # a row cut short by a crash while writing is skipped
                        ok = row["status"] == "ok" and "error" not in json.loads(row["result"])
                    except (TypeError, KeyError, ValueError):
                        log.warning("search_tried.csv: skipping an unreadable row")
                        continue
                    if ok and row["run_id"] != run_id:  # this run's own rows are its cache's business
                        self.rows[row["combination"]] = row

    @staticmethod
    def combination(ctx: RunContext, point: dict[str, Any], windows: int) -> str:
        settings = point_key(ctx, ctx.cfg, point, [point[k] for k in sorted(point) if k.endswith("_method")])
        settings.pop("point")
        return json.dumps({"settings": config_hash(settings, length=10), "windows": windows,
                           **{k: round(v, 6) if isinstance(v, float) else v for k, v in sorted(point.items())}},
                          sort_keys=True)

    def get(self, ctx: RunContext, point: dict[str, Any], windows: int) -> dict[str, Any] | None:
        row = self.rows.get(self.combination(ctx, point, windows))
        if row is None:
            return None
        r = json.loads(row["result"])
        check = check_requirement(ctx.cfg, r["metrics"])
        return {**r, "met": check.met is True, "shortfall": check.shortfall, "cached": True, "run_id": row["run_id"]}

    def add(self, ctx: RunContext, searcher: str, t: dict[str, Any], r: dict[str, Any]) -> None:
        if t["over_limit"]:
            return
        keep = {k: r[k] for k in ("metrics", "error", "errors", "attempted", "dir") if k in r}
        row = {"combination": self.combination(ctx, t["point"], t["windows"]), "run_id": self.run_id,
               "searcher": searcher, "n": t["n"], "windows": t["windows"], **t["point"],
               "status": "ok" if t["ok"] else "failed", "met": t["met"],
               **{m: t["metrics"].get(m) for m in self.METRICS}, "result": json.dumps(keep, default=str)}
        new = not self.path.exists()
        if new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fields = list(row)
        else:  # the file's own columns; "combination" and "result" hold everything a reuse needs
            with self.path.open(newline="", encoding="utf-8") as f:
                fields = next(csv.reader(f))
            with self.path.open("rb") as f:  # a line cut short by a crash: start on a new line
                f.seek(-1, os.SEEK_END)
                cut = f.read(1) not in (b"\n", b"\r")
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            elif cut:
                f.write("\r\n")
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())


def tried_path(ctx: RunContext) -> Path:
    return Path(ctx.cache.root) / "search_tried.csv"


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


def hypervolume(trials: list[dict[str, Any]], ref: list[float]) -> float:
    """Area (volume) the met, full-size trials dominate up to `ref`; larger is better."""
    import numpy as np
    from pymoo.indicators.hv import HV

    pts = np.array([t["values"] for t in pareto(trials) if all(v < r for v, r in zip(t["values"], ref))])
    return float(HV(ref_point=np.array(ref))(pts)) if len(pts) else 0.0


SEARCHERS = {"mobo": run_mobo, "mfbo": run_mfbo, "nsga3": run_nsga3}


def run_search(ctx: RunContext, model_factory: Callable | None = None, tokenizer=None,
               text_loader: Callable | None = None) -> dict[str, Path]:
    """Every searcher in cfg.search.searchers with the same budget and seed, then the report. Tries run no backend
    (Mohammad, 2026-10-25: "we do not test backends in stages 1 and 2. we test them in stage 4"): the backends set in
    cfg.eval measure only the final model (final_model)."""
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
    if backend := [o for o in s.objectives if o.startswith(BACKEND_PREFIXES)]:
        raise ValueError(f"the objectives {backend} come from a backend; tries run none (backends test only the "
                         "final model, in Stage 4): use HF metrics such as ppl_val and predicted_weight_memory_gb")
    if missing := [m for m in s.extra_stage1 if m not in s.stage1_methods]:
        raise ValueError(f"search.extra_stage1 {missing} must be in search.stage1_methods")
    if s.accuracy_tasks:
        try:
            import lm_eval  # noqa: F401 - checked now, not after hours of tries
        except ImportError as e:
            raise ValueError("search.accuracy_tasks needs lm-eval: pip install -e .[downstream]") from e
    space = technique_space(cfg)
    _, handle, loader = setup(ctx, None, tokenizer, text_loader)
    val, held = eval_windows(loader(cfg.eval.dataset, "test"), handle.tokenizer, cfg.eval.seq_len, cfg.eval.max_windows)
    full = len(val) + len(held)
    kw = dict(model_factory=model_factory, tokenizer=tokenizer, text_loader=text_loader)
    tries = RunContext(cfg.with_overrides({**NO_BACKENDS, "eval.downstream_tasks": []}), ctx.run_id, ctx.run_dir,
                       ctx.cache)  # accuracy tasks also only on the final model
    trials: list[dict[str, Any]] = []
    tried = TriedLog(tried_path(ctx), ctx.run_id)
    for name in s.searchers:
        rec = Recorder(tries, name, full, tried=tried, **kw)
        extra = {"eta": s.mfbo_eta} if name == "mfbo" else {}
        SEARCHERS[name](rec, space, s.budget, cfg.run.seed, **extra)
        trials += rec.trials
        atomic_write_text(ctx.run_dir / "search" / "trials.json", json.dumps(trials, indent=2, default=str))
    extras, final = [], None
    try:  # neither step may cost the report of hours of tries, nor the other step
        extras = extra_tries(tries, trials, s.extra_stage1, full, tried=tried, **kw)
    except Exception:  # noqa: BLE001
        log.exception("the extra Stage 1 tries failed; the report is written without them")
    try:
        final = final_model(ctx, trials, s.accuracy_tasks, **kw)
    except Exception:  # noqa: BLE001
        log.exception("the final model's Stage 4 test failed; the report is written without it")
    from sdf.reporting.search import write_search_report

    return write_search_report(ctx, trials, space, extras=extras, accuracy=final)


BACKEND_PREFIXES = ("vllm_", "llamacpp_", "trtllm_")
NO_BACKENDS = {"eval.vllm_python": None, "eval.trtllm_python": None, "eval.llamacpp_dir": None}


def final_model(ctx: RunContext, trials: list[dict[str, Any]], tasks: list[str], **kw: Any) -> dict[str, Any] | None:
    """Stage 4 for the final model under the limit that is best on the first objective (Mohammad, 2026-10-26: "lets
    see how much smaller can we get the model, while keeping 11 ppl or lower", so main.py puts the size first): built
    again with the backends in ctx.cfg.eval and tested for accuracy on `tasks` (eval.downstream_tasks, lm-eval;
    Mohammad, 2026-10-25: "then that accuracy will be compared with tinyllama accuracy"): "fp16" holds the uncompressed model's accuracy, measured in the same run
    under the same conditions. None when there is nothing to test (no task, no backend) or no try met the limit."""
    best = min((t for t in trials if t["full"] and t["met"]), key=lambda t: t["values"], default=None)
    if best is None or not tasks and all(getattr(ctx.cfg.eval, k.split(".")[1]) is None for k in NO_BACKENDS):
        return None
    from sdf.stage0.run import fp16_key, setup

    cfg = ctx.cfg.with_overrides({"eval.downstream_tasks": list(tasks)})
    actx = RunContext(cfg, ctx.run_id, ctx.run_dir, ctx.cache)
    r = evaluate(actx, best["point"], **kw)
    # the uncompressed model's accuracy, as the run_stage call measured and cached it (a cached point builds nothing)
    entry = ctx.cache.path("fp16_downstream", {**fp16_key(actx, setup(actx)[0]), "tasks": list(tasks),
                                               "limit": cfg.eval.downstream_limit})
    fp16 = json.loads(entry.read_text(encoding="utf-8"))["value"] if entry.exists() else {}
    return {"searcher": best["searcher"], "n": best["n"], "point": best["point"], "fp16": fp16,
            **{k: r.get(k) for k in ("metrics", "error", "errors", "dir")}}


def extra_tries(ctx: RunContext, trials: list[dict[str, Any]], methods: list[str], full_windows: int,
                **kw: Any) -> list[dict[str, Any]]:
    """One chained try for each of `methods` (Stage 1 techniques) whose weight technique no searcher measured on all
    the windows (Mohammad, 2026-10-25: "i want to see spqr quip quip# results included"): the search's best try
    under the limit (lowest first objective) with only its Stage 1 technique swapped, or the run's own settings if
    no try met the limit. Kept apart from the searchers' trials, so they change no comparison or trade-off line."""
    cfg = ctx.cfg
    space = technique_space(cfg)
    done = {t["point"]["stage1_method"].split("_with_")[0] for t in trials if t["full"] and t["ok"]}
    best = min((t for t in trials if t["full"] and t["met"]), key=lambda t: t["values"], default=None)
    base = best["point"] if best else space.make({k: v for k, v in cfg.hyperparams.items()
                                                  if k in {p.name for p in SEARCH_SPACE.params}})
    rec = Recorder(ctx, "extra", full_windows, **kw)
    for m in methods:
        if m.split("_with_")[0] not in done:
            rec({**base, "stage1_method": m})
    return rec.trials
