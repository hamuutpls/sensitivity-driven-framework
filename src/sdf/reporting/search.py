"""The settings search's outputs in <run_dir>/search/: search_report.md, search_results.xlsx (Trials, Pareto,
Searcher comparison, Best configs), pareto_<objective>_vs_<metric>.png and hypervolume.png."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Font

from sdf.eval import vllm_backend  # noqa: F401 - registers the vLLM metric names
from sdf.reporting.metrics import label
from sdf.run import RunContext
from sdf.search import FAILED, hypervolume, pareto
from sdf.search_space import SearchSpace
from sdf.utils.cache import atomic_write_text
from sdf.utils.logging import get_logger

log = get_logger(__name__)

_SHOWN = ("ppl_val", "ppl_heldout", "predicted_weight_memory_gb", "vllm_decode_ms_per_token", "vllm_ppl_val",
          "vllm_size_gb", "decode_ms_per_token_mean", "avg_bits_per_weight", "avg_activation_bits",
          "decoder_weights_removed", "predicted_kv_memory_gb")


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
            "Search method": SEARCHER_NAMES.get(name, name), "Tries": len(ts),
            "Cost, in full tries": round(sum(t["cost"] for t in ts), 3),
            "Time, minutes": round(sum(t.get("time_s", 0.0) for t in ts) / 60, 1),
            "Share of tries under the limit (higher is better)": len([t for t in ts if t["met"]]) / len(ts),
            f"Best {label(objectives[0]).lower()} (lower is better)": best["values"][0] if best else None,
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

    outputs.update(_plots(d, trials, front, s.objectives, s.memory_metric, comparison))

    # --- report
    shown = [m for m in (s.objectives[0], s.memory_metric, *s.objectives[1:], "ppl_heldout") if m]
    tech = ["stage1_method", "stage2_method", "prune_ratio_aggressive", "stage3_method"]
    rest = [n for n in names if n not in tech]
    lines = [
        "# Settings search", "",
        "## What this shows", "",
        "The framework tried many settings and built each one as a chained model: the Stage 0 plan (which layers are "
        "sensitive), then Stage 1 (weights and activations stored with fewer bits, following the joint weight and activation "
        "plan), then Stage 2 (pruning: removing "
        "numbers), then Stage 3 (compressing the KV cache, the model's memory of earlier words). A setting is the "
        "technique of each stage plus the numbers that tune them (protection threshold, pruning share, calibration "
        f"text and so on). Only settings that meet the limit count ({_req(cfg)}). Among those the search looks for "
        "the lowest " + " and the lowest ".join(label(o).lower() for o in s.objectives) + ". Speed comes from vLLM, "
        "which runs the model up to Stage 2 (it cannot run the Stage 3 techniques); unstructured pruning keeps the "
        "removed numbers as zeros, so it does not make vLLM faster. Three search methods each had the same budget of "
        f"{s.budget} built models: a setting whose plan is already over the limit is not built and does not count: " + "; ".join(SEARCHER_NAMES[n] for n in s.searchers
                                                                    if n in SEARCHER_NAMES) + ".", "",
        "## What it means", "",
        f"{len(front)} setting(s) are on the best trade-off line: none of them can get better on one goal without "
        "getting worse on another, so each is a fair choice depending on what matters more. They are listed below, "
        "lowest perplexity first (lower perplexity means more accurate; lower time means faster). Held-out perplexity "
        "was measured but never used to choose, so it shows how well the choice holds on unseen text.", "",
        "## Best trade-off settings", "",
        _md([["#", "Stage 1 quantization", "Stage 2 pruning", "Pruning share", "Stage 3 technique"]
             + [label(m) + " (lower is better)" for m in shown]]
            + [[str(i + 1)] + [setting_value(n, t["point"][n]) for n in tech] + [_fmt(t["metrics"].get(m)) for m in shown]
               for i, t in enumerate(front)]),
        "The other settings of each option:", "",
        _md([["#"] + [SETTING_NAMES.get(n, n) for n in rest]]
            + [[str(i + 1)] + [setting_value(n, t["point"][n]) for n in rest] for i, t in enumerate(front)]),
        "## Search methods compared", "",
        _md([list(comparison[0]) if comparison else ["Search method"]]
            + [[_fmt(v) for v in c.values()] for c in comparison]),
        "A full try measures on every evaluation window (a slice of the test text); MFBO's rough tries use fewer "
        "windows and count by that share, though each still builds the whole model, so its time column is the fairer "
        "cost. Hypervolume measures how much of the space of good results (low perplexity and low time) the method's "
        "settings under the limit cover; larger is better.", "",
        "## Conditions", "",
        f"- Model: {cfg.model.name}; seed {cfg.run.seed}; evaluation windows: {cfg.eval.max_windows or 'all'} of "
        f"{cfg.eval.seq_len} tokens.",
        "- Techniques allowed: Stage 1 " + ", ".join(setting_value("stage1_method", m) for m in s.stage1_methods)
        + "; Stage 2 " + ", ".join(setting_value("stage2_method", m) for m in s.stage2_methods)
        + "; Stage 3 " + ", ".join(setting_value("stage3_method", m) for m in s.stage3_methods) + ".",
        f"- Tries over the limit, so not built: {sum(1 for t in trials if t['cost'] == 0)} of {len(trials)}.",
        f"- Tries that failed (could not be built or measured): {sum(1 for t in trials if not t['ok'])} of "
        f"{len(trials)}; the reasons are in the Trials sheet of search_results.xlsx.", "",
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
