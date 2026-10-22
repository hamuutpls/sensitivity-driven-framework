"""The settings search's outputs in <run_dir>/search/: search_report.md, search_results.xlsx (Trials, Pareto,
Searcher comparison, Best configs), pareto_<objective>_vs_<metric>.png and hypervolume.png."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Font

from sdf.eval import vllm_backend  # noqa: F401 - registers the vLLM metric names
from sdf.config import FrameworkConfig
from sdf.reporting.metrics import METRICS, is_number, label
from sdf.run import RunContext
from sdf.search import FAILED, hypervolume, pareto, technique_space
from sdf.search_space import SearchSpace
from sdf.utils.cache import atomic_write_text
from sdf.utils.logging import get_logger

log = get_logger(__name__)

_SHOWN = ("ppl_val", "ppl_heldout", "predicted_weight_memory_gb", "vllm_decode_ms_per_token", "vllm_ppl_val",
          "vllm_size_gb", "decode_ms_per_token_mean", "avg_bits_per_weight", "avg_activation_bits",
          "decoder_weights_removed", "predicted_kv_memory_gb")


def compare_searchers(trials: list[dict[str, Any]], objectives: list[str]) -> list[dict[str, Any]]:
    """Per searcher: tries, models built (and how many on all the windows), cost, share meeting the requirement, its best first objective and the cost at
    which it first found it, hypervolume (one reference point for all: 10% past the worst met value), and its share
    of the combined trade-off line."""
    met = [t["values"] for t in trials if t["full"] and t["met"]]
    ref = [max(v[i] for v in met) * 1.1 for i in range(len(objectives))] if met else None
    front = {json.dumps(t["point"], sort_keys=True) for t in pareto(trials)}
    out = []
    for name in dict.fromkeys(t["searcher"] for t in trials):
        ts = [t for t in trials if t["searcher"] == name]
        full_met = [t for t in ts if t["full"] and t["met"]]
        best = min(full_met, key=lambda t: t["values"][0], default=None)
        mine = {json.dumps(t["point"], sort_keys=True) for t in full_met}
        out.append({
            "Search method": SEARCHER_NAMES.get(name, name), "Tries": len(ts),
            "Models built": sum(1 for t in ts if t["cost"] > 0),
            "Of those, on all evaluation windows": sum(1 for t in ts if t["cost"] > 0 and t["full"]),
            "Cost, in full tries": round(sum(t["cost"] for t in ts), 3),
            "Time, minutes": round(sum(t.get("time_s", 0.0) for t in ts) / 60, 1),
            "Share of tries under the limit (higher is better)": len([t for t in ts if t["met"]]) / len(ts),
            f"Best {_lower(label(objectives[0]))} (lower is better)": best["values"][0] if best else None,
            "Cost until it found that best (lower is better)":
                round(sum(t["cost"] for t in ts[: ts.index(best) + 1]), 3) if best else None,
            "Hypervolume (higher is better)": hypervolume(ts, ref) if ref else 0.0,
            "Share of the best trade-off line it found (higher is better)":
                len(front & mine) / len(front) if front else 0.0,
        })
    return out


SEARCHER_NAMES = {"mobo": "MOBO (multi-objective Bayesian optimisation)",
                  "mfbo": "MFBO (multi-fidelity: rough measurements first, full ones for the best)",
                  "nsga3": "NSGA-III (an evolutionary method)"}
SHORT_NAMES = {"mobo": "MOBO", "mfbo": "MFBO", "nsga3": "NSGA-III"}
SETTING_NAMES = {"sensitive_threshold": "Protection threshold", "prune_ratio_aggressive": "Pruning share",
                 "calib_dataset": "Calibration text", "calib_samples": "Calibration passages",
                 "gptq_groupsize": "Group size", "smoothquant_alpha": "SmoothQuant strength",
                 "quarot_k_bits": "QuaRot key bits", "stage1_method": "Stage 1 quantization",
                 "stage2_method": "Stage 2 pruning",
                 "stage3_method": "Stage 3 KV-cache (memory of earlier words) technique"}


def setting_value(name: str, value: Any) -> str:
    """A setting in words: techniques by their label, numbers rounded."""
    from sdf.stages.methods import METHODS
    from sdf.stages.runner import stage1_row

    if name == "stage1_method":
        return stage1_row(value)[0].label
    if name in ("stage2_method", "stage3_method"):
        return METHODS[value].label
    return _fmt(value)


def hypervolume_curve(trials: list[dict[str, Any]], searcher: str, ref: list[float]) -> list[tuple[float, float]]:
    """(cost spent, hypervolume so far) after each of `searcher`'s evaluations."""
    ts = [t for t in trials if t["searcher"] == searcher]
    return [(sum(t["cost"] for t in ts[: i + 1]), hypervolume(ts[: i + 1], ref)) for i in range(len(ts))]


def best_so_far(trials: list[dict[str, Any]], searcher: str) -> list[tuple[float, float]]:
    """(cost spent, lowest first objective so far among full tries under the limit), from its first such try."""
    out, low, spent = [], None, 0.0
    for t in (t for t in trials if t["searcher"] == searcher):
        spent += t["cost"]
        if t["full"] and t["met"] and (low is None or t["values"][0] < low):
            low = t["values"][0]
        if low is not None:
            out.append((spent, low))
    return out


def write_search_report(ctx: RunContext, trials: list[dict[str, Any]], space: SearchSpace,
                        tests: str | None = None, out_dir: Path | None = None) -> dict[str, Path]:
    """`tests`: the unit-test result of the code that made the numbers, e.g. "417 passed, 3 skipped at 5f440dc".
    `out_dir`: where a rebuild writes; trials.json is then left alone (only the search itself writes it)."""
    cfg = ctx.cfg
    s = cfg.search
    d = out_dir or ctx.run_dir / "search"
    d.mkdir(parents=True, exist_ok=True)
    comparison = compare_searchers(trials, s.objectives)
    by_name = {c["Search method"]: c for c in comparison}
    front = sorted(pareto(trials), key=lambda t: t["values"][0])
    best = {o: min((t for t in trials if t["full"] and t["met"]), key=lambda t, i=i: t["values"][i], default=None)
            for i, o in enumerate(s.objectives)}
    names = [p.name for p in space.params]
    outputs = {}
    if out_dir is None:
        outputs["search_trials"] = d / "trials.json"
        atomic_write_text(outputs["search_trials"], json.dumps(trials, indent=2, default=str))

    # --- workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Trials"
    settings = [SETTING_NAMES.get(n, n) for n in names]
    results = [label(m) for m in _SHOWN]
    values = lambda t: [setting_value(n, t["point"][n]) for n in names] + [t["metrics"].get(m) for m in _SHOWN]  # noqa: E731
    _table(ws, ["Search method", "Try", "Evaluation windows", "Cost, in full tries", "Time, s", "Under the limit",
                "Why it failed"] + settings + results,
           [[t["searcher"], t["n"], t["windows"], t["cost"], t.get("time_s"), t["met"], t["error"]] + values(t)
            for t in trials])
    _table(wb.create_sheet("Pareto"), ["Search method", "Try"] + settings + results,
           [[t["searcher"], t["n"]] + values(t) for t in front])
    cols = list(comparison[0]) if comparison else ["Search method"]
    _table(wb.create_sheet("Searcher comparison"), cols, [[c[k] for k in cols] for c in comparison])
    _table(wb.create_sheet("Best configs"), ["Best for"] + settings + results,
           [[label(o)] + values(t) for o, t in best.items() if t])
    outputs["search_xlsx"] = d / "search_results.xlsx"
    wb.save(outputs["search_xlsx"])

    charts, axes = _plots(d, trials, front, s.objectives, s.memory_metric, cfg.requirement.target_memory_gb)
    outputs.update(charts)

    # --- report: summary, then one section per question, each with a What this shows / What it means box; the
    # big tables go to the back
    shown = [m for m in (s.objectives[0], s.memory_metric, *s.objectives[1:], "ppl_heldout") if m]
    tech = ["stage1_method", "stage2_method", "prune_ratio_aggressive", "stage3_method"]
    rest = [n for n in names if n not in tech]
    head, cells = _metric_columns(shown)
    first = _lower(label(s.objectives[0]))
    built = [t for t in trials if t["cost"] > 0]
    met = [t for t in trials if t["full"] and t["met"]]
    top = front[0] if front else None
    ppl_x = [x for x in (s.memory_metric, *s.objectives[1:]) if f"search_pareto_{x}" in outputs]
    lines = [
        "# Settings search", "",
        f"Run {ctx.run_id}. Unit tests of the code that made these numbers: "
        + (tests or "not recorded for this report") + ".", "",
        "## Summary", "",
        *_box(
            "The framework tried many settings and built each one as a chained model: the Stage 0 plan (which layers "
            "are sensitive), then Stage 1 (weights and activations stored with fewer bits, following the joint weight "
            "and activation plan), then Stage 2 (pruning: removing numbers), then Stage 3 (compressing the KV cache, "
            "the model's memory of earlier words). A setting is the technique of each stage plus the numbers that tune "
            f"them. Only settings that meet the limit count ({_req(cfg)}). Among those the search looks for the lowest "
            + " and the lowest ".join(_lower(label(o)) for o in s.objectives) + ". " + f"{len(s.searchers)} search method(s) each had the "
            f"same budget of {s.budget} full tries: "
            + "; ".join(SEARCHER_NAMES[n] for n in s.searchers if n in SEARCHER_NAMES) + ".",
            f"There were {len(trials)} tries and {len(built)} models were built; {len(met)} tries measured on all the "
            "windows met the limit (a setting two methods both found counts twice). "
            f"{len(front)} distinct settings are on the best trade-off line (none can get better on one goal without "
            "getting worse on another)."
            + (f" The one with the lowest {first} ({_fmt(top['metrics'].get(s.objectives[0]))}) uses "
               f"{setting_value('stage1_method', top['point']['stage1_method'])}"
               + (f" ({_wxay(top)})" if _wxay(top) else "") + ", "
               f"{setting_value('stage2_method', top['point']['stage2_method'])} with a pruning share of "
               f"{setting_value('prune_ratio_aggressive', top['point']['prune_ratio_aggressive'])}, and "
               f"{setting_value('stage3_method', top['point']['stage3_method'])}." if top else
               " No setting met the limit, so there is no best setting to recommend.")),
        "## 1. Best trade-off settings", "",
        *_box(f"Every setting on the best trade-off line, lowest {first} first, with the model size, the speed "
              "and the held-out perplexity. Held-out perplexity was measured but never used to choose.",
              "Each row is a fair choice depending on whether accuracy or speed matters more. If held-out perplexity "
              "stays close to validation perplexity, the choice holds on text the search never saw. Speed comes from "
              "vLLM, which runs the model up to Stage 2 (it cannot run the Stage 3 techniques); unstructured pruning "
              "keeps the removed numbers as zeros, so it does not make vLLM faster."),
        _md([["#", "Stage 1 quantization", "Stage 2 pruning", "Pruning share", "Stage 3 technique"] + head]
            + [[str(i + 1)] + [setting_value(n, t["point"][n]) for n in tech] + cells(t) for i, t in enumerate(front)]),
        *[line for x in ppl_x for line in (
            f"![{label(s.objectives[0])} against {_lower(label(x))}](pareto_ppl_vs_{x}.png)", "",
            f"*Every model measured on all the evaluation windows, by search method: {first} (lower is better) against "
            f"{_lower(label(x))} (lower is better), so the best settings sit towards the bottom left. Filled circles met "
            "the limit; crosses (if any) were built but missed another part of the limit; black rings are the best trade-off settings of section 1"
            + (", joined by a line." if x in s.objectives else " (best on " + " and ".join(
                _lower(label(o)) for o in s.objectives) + " together, so not all at the bottom left here).")
            + (" The dashed line is the size limit." if x == s.memory_metric and cfg.requirement.target_memory_gb
               else "") + axes[f"search_pareto_{x}"] + "*", "")],
        "## 2. Search methods compared", "",
        *_box(f"How well each search method used the same budget: the best {first} it found, how much of its "
              "budget it spent before finding it, and how much of the space of good results its settings cover "
              "(hypervolume).",
              "The method with the higher hypervolume and the larger share of the best trade-off line found better "
              "settings overall; the one that reached its best for less cost is the faster one. MFBO's rough tries use "
              "fewer evaluation windows and count by that share, so it builds more models for the same budget."),
        _md([["Search method", "Models built", f"Best {first} (lower is better)",
              "Cost until it found that best, in full tries (lower is better)", "Hypervolume (higher is better)",
              "Share of the best trade-off line (higher is better)"]]
            + [[SHORT_NAMES.get(n, n), _fmt(c["Models built"]), _fmt(c[f"Best {first} (lower is better)"]),
                _fmt(c["Cost until it found that best (lower is better)"]), _fmt(c["Hypervolume (higher is better)"]),
                _pct(c["Share of the best trade-off line it found (higher is better)"])]
               for n, c in ((n, by_name[SEARCHER_NAMES.get(n, n)]) for n in dict.fromkeys(t["searcher"] for t in trials))]),
        *([f"![Best {first} found so far](best_so_far.png)", "",
           f"*The lowest {first} each method had found among models under the limit, against the budget spent "
           "(lower is better; a curve that drops earlier found good settings sooner)." + axes["search_best_so_far"]
           + "*", ""] if "search_best_so_far" in outputs else []),
        *(["![Hypervolume over the budget](hypervolume.png)", "",
           "*How much of the space of good results each method's settings cover, against the budget spent (higher is "
           "better). All methods share one reference point, 10% past the worst value under the limit."
           + axes["search_hypervolume"] + "*", ""]
          if "search_hypervolume" in outputs else []),
        "## 3. Best setting for each goal", "",
        *_box("The single setting under the limit with the lowest value of each goal.",
              "These are the ends of the best trade-off line: the most accurate model, and the best one on each "
              "other goal."),
        _md([["Best for", "Stage 1 quantization", "Stage 2 pruning", "Pruning share", "Stage 3 technique"] + head]
            + [[label(o)] + [setting_value(n, t["point"][n]) for n in tech] + cells(t)
               for o, t in best.items() if t]),
        "## Conditions", "",
        f"- Model: {cfg.model.name}; seed {cfg.run.seed}; evaluation windows: {cfg.eval.max_windows or 'all'} of "
        f"{cfg.eval.seq_len} tokens.",
        "- Techniques allowed: Stage 1 " + ", ".join(setting_value("stage1_method", m) for m in s.stage1_methods)
        + "; Stage 2 " + ", ".join(setting_value("stage2_method", m) for m in s.stage2_methods)
        + "; Stage 3 " + ", ".join(setting_value("stage3_method", m) for m in s.stage3_methods) + ".",
        "- A setting whose plan is already over the limit is not built and does not count against the budget, nor "
        "does one that failed before building or a repeat of the method's own try that came from the cache.",
        f"- Tries over the limit, so not built: {sum(1 for t in trials if t.get('over_limit'))} of {len(trials)}.",
        f"- Tries that failed (could not be built or measured): "
        f"{sum(1 for t in trials if not t['ok'] and not t.get('over_limit'))} of "
        f"{len(trials)}; the reasons are in the Trials sheet of search_results.xlsx.", "",
        "## Appendix A: the other settings of each best trade-off option", "",
        _md([["#"] + [SETTING_NAMES.get(n, n) for n in rest]]
            + [[str(i + 1)] + [setting_value(n, t["point"][n]) for n in rest] for i, t in enumerate(front)]),
        "## Appendix B: search methods, every column", "",
        _md([list(comparison[0]) if comparison else ["Search method"]]
            + [[_fmt(v) for v in c.values()] for c in comparison]),
        "A full try measures on every evaluation window (a slice of the test text); MFBO's rough tries use fewer "
        "windows and count by that share, though each still builds the whole model, so its time column is the fairer "
        "cost. Every try, with its settings, results and any error, is in search_results.xlsx.", "",
    ]
    outputs["search_report"] = d / "search_report.md"
    atomic_write_text(outputs["search_report"], "\n".join(lines))
    return outputs


def rebuild_search_report(run_dir: str | Path, tests: str | None = None) -> dict[str, Path]:
    """The search outputs again from a run's config.json and search/trials.json, with no model built. A finished run
    (the search wrote search_report.md at its end) is rebuilt in place; an unfinished one into search/partial/, so a
    rebuild never touches what the running search writes. trials.json is only read."""
    run_dir = Path(run_dir).resolve()
    d = run_dir / "search"
    out_dir = d if (d / "search_report.md").exists() else d / "partial"
    cfg = FrameworkConfig.from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    trials = json.loads((run_dir / "search" / "trials.json").read_text(encoding="utf-8"))
    ctx = RunContext(cfg=cfg, run_id=run_dir.name, run_dir=run_dir, cache=None)  # nothing is built, so no cache
    return write_search_report(ctx, trials, technique_space(cfg), tests, out_dir)


def _box(shows: str, means: str) -> list[str]:
    return [f"> **What this shows.** {shows}", ">", f"> **What it means.** {means}", ""]


def _metric_columns(metrics: list[str]):
    """Headers and a row maker for `metrics`, each with its better direction; a time per token is followed by the
    same speed in tokens per second."""
    head, makers = ["Average bits, weights and activations (WxAy)"], [_wxay]
    for m in metrics:
        spec = METRICS.get(m)
        head.append(label(m) + (f", {spec.unit}" if spec and spec.unit else "")
                    + (f", {spec.better} is better" if spec and spec.better else ""))
        makers.append(lambda t, m=m: _fmt(t["metrics"].get(m)))
        if m.endswith("ms_per_token") or m.endswith("ms_per_token_mean"):
            head.append("Same speed, tokens per second, higher is better")
            makers.append(lambda t, m=m: _fmt(_tokens_per_s(t["metrics"].get(m))))
    return head, lambda t: [f(t) for f in makers]


def _wxay(t: dict[str, Any]) -> str:
    w, a = t["metrics"].get("avg_bits_per_weight"), t["metrics"].get("avg_activation_bits")
    return f"W{w:.3g}A{a:.3g}" if is_number(w) and is_number(a) else ""


def _lower(text: str) -> str:
    """A label inside a sentence: first letter small (vLLM stays vLLM)."""
    return text[:1].lower() + text[1:]


def _tokens_per_s(ms: Any) -> float | None:
    return 1000.0 / ms if is_number(ms) and 0 < ms < FAILED else None


def _pct(v: Any) -> str:
    return f"{v:.0%}" if is_number(v) else ""


def _req(cfg) -> str:
    r = cfg.requirement
    unit = lambda m: f" {METRICS[m].unit}" if m in METRICS and METRICS[m].unit else ""  # noqa: E731
    parts = [f"{_lower(label(m))} at most {v}{unit(m)}" for m, v in ((cfg.search.memory_metric, r.target_memory_gb),
                                                     ("ppl_val", r.target_ppl),
                                                     ("decode_ms_per_token_mean", r.target_latency_ms),
                                                     ("predicted_kv_memory_gb", r.kv_budget_gb)) if v is not None]
    return "; ".join(parts) or "no limits set"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "" if v >= FAILED else f"{v:.4g}"
    return "" if v is None else str(v)


def _md(rows: list[list[str]]) -> str:
    head, *body = rows
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in body]) + "\n"


def _table(ws, header: list[str], rows: list[list[Any]]) -> None:
    ws.append(header)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        ws.append([v if isinstance(v, (int, float, str, bool)) or v is None else str(v) for v in r])


def _plots(d: Path, trials, front, objectives, memory, limit) -> tuple[dict[str, Path], dict[str, str]]:
    """The charts, and for each a caption note on any axis that does not start at zero."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib is not installed: no search plots (pip install .[figures])")
        return {}, {}
    out, notes = {}, {}
    y = objectives[0]
    ok = [t for t in trials if t["full"] and t["ok"]]
    # Okabe-Ito colours (colour-blind safe), one per search method; black is kept for the best trade-off rings
    palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442")
    searchers = list(dict.fromkeys(t["searcher"] for t in trials))
    if len(searchers) > len(palette):
        raise ValueError(f"{len(searchers)} search methods but only {len(palette)} colour-blind-safe colours")
    colors = dict(zip(searchers, palette))
    name = lambda n: SHORT_NAMES.get(n, n)  # noqa: E731

    def save(fig, ax, key, file):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        x0, y0 = ax.get_xlim()[0] > 0, ax.get_ylim()[0] > 0
        notes[key] = (" Neither axis starts at zero." if x0 and y0 else " The x axis does not start at zero." if x0
                      else " The y axis does not start at zero." if y0 else "")
        out[key] = d / file
        fig.savefig(out[key], dpi=150)
        plt.close(fig)

    for x in [memory] + objectives[1:]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for n, c in colors.items():
            for met, marker, word in ((True, "o", "under the limit"), (False, "x", "over the limit")):
                pts = [t for t in ok if t["searcher"] == n and t["met"] == met and is_number(t["metrics"].get(x))]
                if pts:
                    ax.scatter([t["metrics"][x] for t in pts], [t["metrics"][y] for t in pts], color=c, marker=marker,
                               alpha=0.6, label=f"{name(n)}, {word}")
        pts = sorted((t for t in front if is_number(t["metrics"].get(x))), key=lambda t: t["metrics"][x])
        if pts and x in objectives:  # the line joins the front only on the axes it was found on
            ax.plot([t["metrics"][x] for t in pts], [t["metrics"][y] for t in pts], color="black", lw=1, zorder=1)
        if pts:
            ax.scatter([t["metrics"][x] for t in pts], [t["metrics"][y] for t in pts], facecolors="none",
                       edgecolors="black", s=120, label="best trade-off")
        if x == memory and limit:
            ax.axvline(limit, color="grey", ls="--", lw=1, label=f"size limit ({limit} GB)")
        ax.set_xlabel(label(x) + (f", {METRICS[x].unit}" if x in METRICS and METRICS[x].unit else "")
                      + " (lower is better)")
        ax.set_ylabel(label(y) + " (lower is better)")
        save(fig, ax, f"search_pareto_{x}", f"pareto_ppl_vs_{x}.png")
    met = [t["values"] for t in ok if t["met"]]
    if met:
        ref = [max(v[i] for v in met) * 1.1 for i in range(len(objectives))]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for n, c in colors.items():
            curve = hypervolume_curve(trials, n, ref)
            ax.step([a for a, _ in curve], [b for _, b in curve], where="post", color=c, label=name(n))
        ax.set_xlabel("budget spent (full tries)")
        ax.set_ylabel("hypervolume (higher is better)")
        save(fig, ax, "search_hypervolume", "hypervolume.png")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for n, c in colors.items():
            curve = best_so_far(trials, n)
            if curve:
                ax.step([a for a, _ in curve], [b for _, b in curve], where="post", color=c, label=name(n))
        ax.set_xlabel("budget spent (full tries)")
        ax.set_ylabel(f"lowest {_lower(label(y))} so far (lower is better)")
        save(fig, ax, "search_best_so_far", "best_so_far.png")
    return out, notes
