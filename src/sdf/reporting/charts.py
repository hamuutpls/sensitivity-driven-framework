"""PNG charts for the Stage 1-3 reports, written next to report.md and embedded in it with captions.

One colour per technique (Okabe-Ito, colour-blind safe), one marker per condition (standard method, our method...).
Perplexity is the Hugging Face measurement only: a backend's perplexity is its re-rounded file and stays in the tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sdf.reporting.metrics import METRICS, is_number
from sdf.utils.logging import get_logger

if TYPE_CHECKING:
    from sdf.reporting.reporter import ComparisonRow, StageReporter

log = get_logger(__name__)

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
_LEGEND_MAX = 12  # more techniques than this: the legend lists the conditions only


@dataclass
class Chart:
    path: Path
    shows: str  # "What this shows"
    means: str  # "What it means": which direction is better, and axes not at 0


def condition(row: "ComparisonRow") -> str:
    """The row's version within its technique: a pruning level's kind ("Our method"), else its plain name."""
    return row.info.get("kind_label") or row.plain_name


def write_charts(rep: "StageReporter") -> list[Chart]:
    """Perplexity chart (Stage 2: against the share removed), perplexity against memory, and against writing speed
    when some rows are not simulated. [] without matplotlib, outside Stages 1-3 or with nothing to compare."""
    if rep.stage not in (1, 2, 3):
        return []
    ok = [r for r in rep.rows if r.status == "ok" and is_number(r.metrics.get("ppl_val"))]
    rows = [r for r in ok if r.variant != "fp16"]
    if not rows:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib is not installed: no report charts (pip install .[figures])")
        return []
    fp16 = next((r for r in ok if r.variant == "fp16"), None)
    techniques = list(dict.fromkeys(rep.technique(r) for r in rows))
    # ponytail: colours repeat after eight techniques; the caption says so
    colour = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(techniques)}
    marker = {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(dict.fromkeys(condition(r) for r in rows))}
    style = dict(rep=rep, plt=plt, colour=colour, marker=marker, fp16=fp16)
    key = ("Each colour is a technique" + (" (colours repeat after eight)" if len(techniques) > len(PALETTE) else "")
           + ", each marker a version" + ("; the dashed line is the original, uncompressed model." if fp16 else ".")
           + (" With this many techniques the legend lists the versions only; the perplexity chart names every "
              "technique." if len(techniques) > _LEGEND_MAX else ""))
    out = []

    # older runs have only the share of the whole model removed
    removed = next((m for m in ("decoder_weights_removed", "sparsity")
                    if all(is_number(r.metrics.get(m)) for r in rows)), None) if rep.stage == 2 else None
    if removed:
        what = "decoder layers' weights" if removed == "decoder_weights_removed" else "model's numbers"
        path, notes = _scatter(rows, removed, f"Share of the {what} removed (%)",
                               rep.dir / "chart_perplexity_vs_share_removed.png", lines=True, **style)
        out.append(Chart(path, "Prediction error (perplexity, measured in Hugging Face) against the share of the "
                               f"{what} each version really removes. " + key,
                         "Lower is better: a line that stays low further to the right keeps the model accurate while "
                         "removing more. " + notes))
    else:
        path, notes = _dots(rows, rep.dir / "chart_perplexity.png", **style)
        out.append(Chart(path, "Prediction error (perplexity, measured in Hugging Face) of every version, one line per "
                               "technique. " + key, "Lower (further left) is better. " + notes))

    memory = "predicted_kv_memory_gb" if rep.stage == 3 else "predicted_weight_memory_gb"
    have = [r for r in rows if is_number(r.metrics.get(memory))]
    if have:
        plain = METRICS[memory].plain
        path, notes = _scatter(have, memory, f"{plain[:1].upper() + plain[1:]} (GB)",
                               rep.dir / "chart_perplexity_vs_memory.png", **style)
        out.append(Chart(path, f"Prediction error (perplexity) against the {plain}, for every version. " + key,
                         "Lower on both is better, so the best versions sit bottom-left. A version further right is "
                         "larger; that is shown, not counted against our method. " + notes))

    real = [r for r in rows if not r.info.get("simulated") and is_number(r.metrics.get("decode_tokens_per_s"))]
    if real:
        path, notes = _scatter(real, "decode_tokens_per_s", "Writing speed (word-pieces per second)",
                               rep.dir / "chart_perplexity_vs_speed.png", **style)
        out.append(Chart(path, "Prediction error (perplexity) against writing speed in Hugging Face, for every version "
                               "that is not simulated (a simulated version's speed is not a real compressed model's). "
                               "The time per word-piece is in the Results table. " + key,
                         "Lower perplexity and higher speed are better, so the best versions sit bottom-right. "
                         + notes))
    return out


def _scatter(rows, x: str, xlabel: str, path: Path, *, rep, plt, colour, marker, fp16, lines: bool = False):
    """Perplexity (y) against metric `x`: one series per (technique, condition), joined by a line when `lines`."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    scale = 100 if x in ("decoder_weights_removed", "sparsity") else 1
    series: dict[tuple[str, str], list] = {}
    for r in rows:
        series.setdefault((rep.technique(r), condition(r)), []).append(r)
    for (t, c), rs in series.items():
        pts = sorted((r.metrics[x] * scale, r.metrics["ppl_val"]) for r in rs)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour[t], marker=marker[c], ls="-" if lines else "",
                alpha=0.85, lw=1)
    notes = _finish(ax, plt, [r.metrics["ppl_val"] for r in rows], fp16, colour, marker, horizontal=True)
    ax.set_xlabel(xlabel)
    if ax.get_xlim()[0] > 0:
        notes += " The horizontal axis does not start at 0."
    return _save(fig, plt, path), notes.strip()


def _dots(rows, path: Path, *, rep, plt, colour, marker, fp16):
    """Perplexity (x) of every row, one line per technique (y)."""
    techniques = list(colour)
    fig, ax = plt.subplots(figsize=(8, 1.5 + 0.3 * len(techniques)))
    for r in rows:
        t = rep.technique(r)
        ax.plot(r.metrics["ppl_val"], techniques.index(t), color=colour[t], marker=marker[condition(r)], ls="",
                alpha=0.85)
    ax.set_yticks(range(len(techniques)), techniques, fontsize=7)
    ax.invert_yaxis()
    notes = _finish(ax, plt, [r.metrics["ppl_val"] for r in rows], fp16, colour, marker, horizontal=False)
    return _save(fig, plt, path), notes.strip()


def _finish(ax, plt, ppl: list[float], fp16, colour, marker, horizontal: bool) -> str:
    """Perplexity axis (log when it spans more than 10x), the original model's line, the legend; the axis note."""
    from matplotlib.lines import Line2D

    if fp16 is not None:
        ppl = ppl + [fp16.metrics["ppl_val"]]
        (ax.axhline if horizontal else ax.axvline)(fp16.metrics["ppl_val"], color="grey", ls="--", lw=1)
    log_scale = max(ppl) > 10 * min(ppl)
    (ax.set_yscale if horizontal else ax.set_xscale)("log" if log_scale else "linear")
    (ax.set_ylabel if horizontal else ax.set_xlabel)("Perplexity (lower is better)" + (", log scale" if log_scale else ""))
    handles = [Line2D([], [], color="grey", marker=m, ls="", label=c) for c, m in marker.items()]
    if len(colour) <= _LEGEND_MAX:
        handles = [Line2D([], [], color=c, lw=4, label=t) for t, c in colour.items()] + handles
    if fp16 is not None:
        handles.append(Line2D([], [], color="grey", ls="--", label="Original model (uncompressed)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    if log_scale:
        return "The perplexity axis is logarithmic (each step is ten times the last) and does not start at 0."
    low = (ax.get_ylim() if horizontal else ax.get_xlim())[0]
    return "The perplexity axis does not start at 0." if low > 0 else ""


def _save(fig, plt, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
