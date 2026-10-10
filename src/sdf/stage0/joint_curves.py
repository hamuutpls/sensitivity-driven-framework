"""Perplexity against WxAy, per decoder layer: the joint measurement (stage0/joint.py) as a table and figures.

Each point is the calibration perplexity with only that layer rounded: its weights to W bits and its inputs to A bits,
every other layer untouched (`baseline_bits`: not rounded). The pair the Stage 0 joint plan gives the layer is
marked (a pick off the measured grid is named in the panel title instead).
The table needs nothing beyond the standard library; the figures need matplotlib (the `figures` extra) and are
skipped without it.
"""

from __future__ import annotations

import csv
from pathlib import Path

from sdf.stage0.joint import JointProfile

CAPTION = ("Perplexity on the Stage 0 calibration text with only this layer rounded, at each weight x activation bit "
           "pair (WxAy), fewest total bits to the left. Lower is better. Colour: activation bits; marker: weight "
           "bits. The star is the final plan's choice under the bit budget shared by all layers, so it is not always "
           "the panel's lowest point. Points are raw measurements: more bits do not always score better, and some "
           "points fall below the dashed line. W16A16 is the unrounded model, not a separate measurement. ")
GRID_NOTE = "The y axes do not start at 0 and differ between panels."
SINGLE_NOTE = "The y axis does not start at 0."
A_COLOURS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e")  # activation bits, lowest first
W_MARKERS = ("o", "s", "^", "D", "v")  # weight bits, lowest first


def pairs_by_bits(profile: JointProfile) -> list[tuple[int, int]]:
    """Every (w, a) pair, fewest total bits first (ties: fewer weight bits first)."""
    return sorted(((w, a) for w in profile.w_options for a in profile.a_options), key=lambda p: (p[0] + p[1], p[0]))


def curve_rows(profile: JointProfile, picks: list[tuple[int, int]]) -> list[dict]:
    base, top = profile.meta["baseline_ppl"], profile.meta.get("baseline_bits", 16)
    rows = []
    for i in range(profile.num_layers):
        for w, a in pairs_by_bits(profile):
            rise = profile.rise[i][profile.w_options.index(w)][profile.a_options.index(a)]
            rows.append({"layer": i, "pair": f"W{w}A{a}", "weight_bits": w, "activation_bits": a,
                         "total_bits": w + a, "ppl": base + rise, "ppl_rise": rise,
                         "measured": not (w >= top and a >= top),  # unrounded: the baseline, copied
                         "chosen": tuple(picks[i]) == (w, a)})
    return rows


def write_csv(profile: JointProfile, picks: list[tuple[int, int]], path: str | Path) -> Path:
    rows = curve_rows(profile, picks)
    with open(path, "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    return Path(path)


def draw(profile: JointProfile, picks: list[tuple[int, int]], out_dir: str | Path, model: str = "",
         plan: str = "joint plan") -> list[Path]:
    """One grid figure (4 panels per row, a panel per layer) and one figure per layer, as PNG. Points only (pairs
    are not on one scale), coloured by activation bits, shaped by weight bits. `plan` names the plan the stars
    come from in the legend. Raises ImportError without matplotlib."""
    from matplotlib.figure import Figure  # no pyplot: leaves the caller's backend and open figures alone
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = profile.meta["baseline_ppl"]
    pairs = pairs_by_bits(profile)
    labels = [f"W{w}A{a}" for w, a in pairs]
    colour = {a: A_COLOURS[k % len(A_COLOURS)] for k, a in enumerate(profile.a_options)}
    marker = {w: W_MARKERS[k % len(W_MARKERS)] for k, w in enumerate(profile.w_options)}
    rows = curve_rows(profile, picks)
    title = f"Stage 0: perplexity against WxAy per layer{f' ({model})' if model else ''}"
    xlabel = "Weight bits x activation bits (WxAy), fewer total bits to the left"
    legend = ([Line2D([], [], ls="", marker="o", color=colour[a], label=f"activations {a}-bit")
               for a in profile.a_options]
              + [Line2D([], [], ls="", marker=marker[w], color="#555555", label=f"weights {w}-bit")
                 for w in profile.w_options]
              + [Line2D([], [], ls="", marker="*", color="black", ms=12, mfc="none", label=f"chosen by the {plan}"),
                 Line2D([], [], ls="--", color="#555555", label=f"nothing rounded ({base:.2f})")])

    def panel(ax, i, small):
        w0, a0 = picks[i]
        for x, r in enumerate(r for r in rows if r["layer"] == i):
            w, a = r["weight_bits"], r["activation_bits"]
            ax.plot([x], [r["ppl"]], ls="", marker=marker[w], color=colour[a], ms=4 if small else 8)
            if (w, a) == (w0, a0):
                ax.plot([x], [r["ppl"]], ls="", marker="*", ms=12 if small else 22, mfc="none", mec="black",
                        mew=1.0 if small else 1.5, zorder=3)
        ax.axhline(base, color="#555555", ls="--", lw=0.8)
        ax.set_xticks(range(len(pairs)), labels, rotation=90, fontsize=5 if small else 9)
        ax.tick_params(axis="y", labelsize=5 if small else 9)
        off = "" if (w0, a0) in pairs else " (not measured here)"
        ax.set_title(f"Layer {i}: chosen W{w0}A{a0}{off}", fontsize=7 if small else 12)
        ax.grid(alpha=0.3)
        ax.margins(x=0.06, y=0.15)  # room for the star at the edges

    files = []
    n = profile.num_layers
    cols = min(4, n)
    rows_n = -(-n // cols)
    fig = Figure(figsize=(8, 1.75 * rows_n + 1.6))
    axes = fig.subplots(rows_n, cols, squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < n:
            panel(ax, i, small=True)
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.legend(handles=legend, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.965), frameon=False)
    fig.supxlabel(xlabel, fontsize=8)
    fig.supylabel("Calibration perplexity (lower is better)", fontsize=8)
    fig.text(0.5, -0.02, CAPTION + GRID_NOTE, ha="center", va="top", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0.01, 0.02, 1, 1 - 0.75 / fig.get_figheight()))
    files.append(out_dir / "joint_curves_all_layers.png")
    fig.savefig(files[-1], dpi=200, bbox_inches="tight")

    for i in range(n):
        fig = Figure(figsize=(8, 5.2))
        ax = fig.subplots()
        panel(ax, i, small=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Calibration perplexity (lower is better)")
        ax.legend(handles=legend, fontsize=8, ncol=2)
        fig.suptitle(title, fontsize=11)
        fig.text(0.5, -0.01, CAPTION + SINGLE_NOTE, ha="center", va="top", fontsize=8, wrap=True)
        fig.tight_layout()
        files.append(out_dir / f"joint_curve_layer{i:02d}.png")
        fig.savefig(files[-1], dpi=150, bbox_inches="tight")
    return files
