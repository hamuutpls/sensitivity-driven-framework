"""The settings search's outputs in <run_dir>/search/: search_report.md, search_results.xlsx (Trials, Pareto,
Searcher comparison, Best configs), pareto_<objective>_vs_<metric>.png and hypervolume.png."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Font

from sdf.reporting.metrics import label
from sdf.run import RunContext
from sdf.search import FAILED, hypervolume, pareto
from sdf.search_space import SearchSpace
from sdf.utils.cache import atomic_write_text
from sdf.utils.logging import get_logger

log = get_logger(__name__)

_SHOWN = ("ppl_val", "ppl_heldout", "predicted_weight_memory_gb", "decode_ms_per_token_mean", "prefill_ms_mean",
          "avg_bits_per_weight", "avg_activation_bits", "decoder_weights_removed", "predicted_kv_memory_gb")


def compare_searchers(trials: list[dict[str, Any]], objectives: list[str]) -> list[dict[str, Any]]:
    """Per searcher: evaluations, cost, share meeting the requirement, its best first objective and the cost at
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
            "searcher": name, "evaluations": len(ts), "cost": round(sum(t["cost"] for t in ts), 3),
            "met_share": len([t for t in ts if t["met"]]) / len(ts),
            f"best_{objectives[0]}": best["values"][0] if best else None,
            "cost_to_best": round(sum(t["cost"] for t in ts[: ts.index(best) + 1]), 3) if best else None,
            "hypervolume": hypervolume(ts, ref) if ref else 0.0,
            "front_share": len(front & mine) / len(front) if front else 0.0,
        })
    return out


def hypervolume_curve(trials: list[dict[str, Any]], searcher: str, ref: list[float]) -> list[tuple[float, float]]:
    """(cost spent, hypervolume so far) after each of `searcher`'s evaluations."""
    ts = [t for t in trials if t["searcher"] == searcher]
    return [(sum(t["cost"] for t in ts[: i + 1]), hypervolume(ts[: i + 1], ref)) for i in range(len(ts))]


def write_search_report(ctx: RunContext, trials: list[dict[str, Any]], space: SearchSpace) -> dict[str, Path]:
    cfg = ctx.cfg
    s = cfg.search
    d = ctx.run_dir / "search"
    d.mkdir(parents=True, exist_ok=True)
    comparison = compare_searchers(trials, s.objectives)
    front = sorted(pareto(trials), key=lambda t: t["values"][0])
    best = {o: min((t for t in trials if t["full"] and t["met"]), key=lambda t, i=i: t["values"][i], default=None)
            for i, o in enumerate(s.objectives)}
    names = [p.name for p in space.params]
    outputs = {"search_trials": d / "trials.json"}
    atomic_write_text(outputs["search_trials"], json.dumps(trials, indent=2, default=str))

    # --- workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Trials"
    _table(ws, ["searcher", "n", "windows", "cost", "met", "error"] + names + list(_SHOWN),
           [[t["searcher"], t["n"], t["windows"], t["cost"], t["met"], t["error"]] + [t["point"][n] for n in names]
            + [t["metrics"].get(m) for m in _SHOWN] for t in trials])
    _table(wb.create_sheet("Pareto"), ["searcher", "n"] + names + list(_SHOWN),
           [[t["searcher"], t["n"]] + [t["point"][n] for n in names] + [t["metrics"].get(m) for m in _SHOWN]
            for t in front])
    cols = list(comparison[0]) if comparison else ["searcher"]
    _table(wb.create_sheet("Searcher comparison"), cols, [[c[k] for k in cols] for c in comparison])
    _table(wb.create_sheet("Best configs"), ["best for"] + names + list(_SHOWN),
           [[label(o)] + [t["point"][n] for n in names] + [t["metrics"].get(m) for m in _SHOWN]
            for o, t in best.items() if t])
    outputs["search_xlsx"] = d / "search_results.xlsx"
    wb.save(outputs["search_xlsx"])

    outputs.update(_plots(d, trials, front, s.objectives, s.memory_metric, comparison))

    # --- report
    lines = [
        "# Settings search", "",
        "## What this shows", "",
        "The framework tried different settings (how sensitive a layer must be to be protected, how much to prune, "
        "the calibration text, the techniques of Stages 1, 2 and 3) and built each one as a chained model: the "
        "Stage 0 plan, then quantization of weights and activations, then pruning, then KV-cache compression. Only "
        f"settings that meet the requirement count ({_req(cfg)}). Among those it looks for the lowest "
        + " and the lowest ".join(label(o) for o in s.objectives) + ". Three search methods each had the same budget "
        f"({s.budget} full evaluations): MOBO (Bayesian optimisation), MFBO (cheap rough measurements first, then "
        "full ones for the best) and NSGA-III (an evolutionary method).", "",
        "## What it means", "",
        f"{len(front)} setting(s) are on the best trade-off line: none of them can be improved on one goal without "
        "losing on another. They are listed below, lowest perplexity first. Held-out perplexity was measured but never "
        "used to choose.", "",
        "## Best trade-off settings", "",
        _md([["#"] + names + [label(m) for m in _SHOWN[:4]]]
            + [[str(i + 1)] + [_fmt(t["point"][n]) for n in names] + [_fmt(t["metrics"].get(m)) for m in _SHOWN[:4]]
               for i, t in enumerate(front)]),
        "## Search methods compared", "",
        _md([list(comparison[0]) if comparison else ["searcher"]] + [[_fmt(v) for v in c.values()] for c in comparison]),
        "Cost counts full evaluations (MFBO's rough ones count by their share of the windows). Hypervolume: the area of "
        "the goals the met settings cover (larger is better). Front share: the share of the best trade-off line this "
        "method found.", "",
        "## Conditions", "",
        f"- Model: {cfg.model.name}; seed {cfg.run.seed}; evaluation windows {cfg.eval.max_windows or 'all'} of "
        f"{cfg.eval.seq_len} tokens; techniques Stage 1 {s.stage1_methods}, Stage 2 {s.stage2_methods}, Stage 3 "
        f"{s.stage3_methods}.",
        f"- Failed settings: {sum(1 for t in trials if not t['ok'])} of {len(trials)} evaluations (see Trials).", "",
    ]
    outputs["search_report"] = d / "search_report.md"
    atomic_write_text(outputs["search_report"], "\n".join(lines))
    return outputs


def _req(cfg) -> str:
    r = cfg.requirement
    parts = [f"{label(m)} at most {v}" for m, v in (("predicted_weight_memory_gb", r.target_memory_gb),
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


def _plots(d: Path, trials, front, objectives, memory, comparison) -> dict[str, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib is not installed: no search plots (pip install .[figures])")
        return {}
    out = {}
    ok = [t for t in trials if t["full"] and t["ok"]]
    colors = dict(zip(dict.fromkeys(t["searcher"] for t in trials), ("#0072B2", "#E69F00", "#009E73", "#CC79A7")))
    for x in [memory] + objectives[1:]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, c in colors.items():
            pts = [t for t in ok if t["searcher"] == name and t["metrics"].get(x) is not None]
            ax.scatter([t["metrics"][x] for t in pts], [t["metrics"]["ppl_val"] for t in pts], c=c, label=name,
                       alpha=0.6, marker="o")
        pts = [t for t in front if t["metrics"].get(x) is not None]
        ax.scatter([t["metrics"][x] for t in pts], [t["metrics"]["ppl_val"] for t in pts], facecolors="none",
                   edgecolors="black", s=120, label="best trade-off")
        ax.set_xlabel(label(x) + " (lower is better)")
        ax.set_ylabel("validation perplexity (lower is better)")
        ax.legend()
        fig.tight_layout()
        out[f"search_pareto_{x}"] = d / f"pareto_ppl_vs_{x}.png"
        fig.savefig(out[f"search_pareto_{x}"], dpi=150)
        plt.close(fig)
    met = [t["values"] for t in ok if t["met"]]
    if met:
        from sdf.reporting.search import hypervolume_curve

        ref = [max(v[i] for v in met) * 1.1 for i in range(len(objectives))]
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, c in colors.items():
            curve = hypervolume_curve(trials, name, ref)
            ax.step([a for a, _ in curve], [b for _, b in curve], where="post", c=c, label=name)
        ax.set_xlabel("cost (full evaluations)")
        ax.set_ylabel("hypervolume (higher is better)")
        ax.legend()
        fig.tight_layout()
        out["search_hypervolume"] = d / "hypervolume.png"
        fig.savefig(out["search_hypervolume"], dpi=150)
        plt.close(fig)
    return out
