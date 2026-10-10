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
           "pair (WxAy), ordered by total bits. Lower is better. The star is the final plan's choice under the "
           "bit budget shared by all layers, so it is not always the panel's lowest point. Points are raw measurements: "
           "small dips below the dashed line are noise. ")
GRID_NOTE = "The y axes do not start at 0 and differ between panels."
SINGLE_NOTE = "The y axis does not start at 0."


def pairs_by_bits(profile: JointProfile) -> list[tuple[int, int]]:
    """Every (w, a) pair, fewest total bits first (ties: fewer weight bits first)."""
    return sorted(((w, a) for w in profile.w_options for a in profile.a_options), key=lambda p: (p[0] + p[1], p[0]))


def curve_rows(profile: JointProfile, picks: list[tuple[int, int]]) -> list[dict]:
    base = profile.meta["baseline_ppl"]
    rows = []
    for i in range(profile.num_layers):
        for w, a in pairs_by_bits(profile):
            rise = profile.rise[i][profile.w_options.index(w)][profile.a_options.index(a)]
            rows.append({"layer": i, "pair": f"W{w}A{a}", "weight_bits": w, "activation_bits": a,
                         "total_bits": w + a, "ppl": base + rise, "ppl_rise": rise,
                         "chosen": tuple(picks[i]) == (w, a)})
    return rows


def write_csv(profile: JointProfile, picks: list[tuple[int, int]], path: str | Path) -> Path:
    rows = curve_rows(profile, picks)
    with open(path, "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    return Path(path)


def draw(profile: JointProfile, picks: list[tuple[int, int]], out_dir: str | Path, model: str = "") -> list[Path]:
    """One grid figure (a panel per layer) and one figure per layer, as PNG. Raises ImportError without
    matplotlib."""
    from matplotlib.figure import Figure  # no pyplot: leaves the caller's backend and open figures alone

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = profile.meta["baseline_ppl"]
    pairs = pairs_by_bits(profile)
    labels = [f"W{w}A{a}" for w, a in pairs]
    rows = curve_rows(profile, picks)
    title = f"Stage 0: perplexity against WxAy per layer{f' ({model})' if model else ''}"

    def panel(ax, i, small):
        ys = [r["ppl"] for r in rows if r["layer"] == i]
        w, a = picks[i]
        k = pairs.index((w, a)) if (w, a) in pairs else None
        ax.plot(range(len(pairs)), ys, "o-", color="#1f77b4", ms=3 if small else 6, lw=1.2,
                label="only this layer rounded")
        ax.plot([] if k is None else [k], [] if k is None else [ys[k]], "*", color="#d62728",
                ms=11 if small else 18, zorder=3, label="pair chosen by the Stage 0 plan")
        ax.axhline(base, color="#555555", ls="--", lw=1, label=f"model with nothing rounded ({base:.2f})")
        ax.set_xticks(range(len(pairs)), labels, rotation=90 if small else 45, fontsize=6 if small else 9)
        ax.tick_params(axis="y", labelsize=6 if small else 9)
        ax.set_title(f"Layer {i}: chosen W{w}A{a}{' (not measured here)' if k is None else ''}",
                     fontsize=8 if small else 12)
        ax.grid(alpha=0.3)

    files = []
    n = profile.num_layers
    cols = min(6, n)
    rows_n = -(-n // cols)
    fig = Figure(figsize=(3.2 * cols, 2.9 * rows_n))
    axes = fig.subplots(rows_n, cols, squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < n:
            panel(ax, i, small=True)
            if i % cols == 0:
                ax.set_ylabel("Calibration perplexity (lower is better)", fontsize=7)
        else:
            ax.axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(title, fontsize=14, y=1.0)
    fig.text(0.5, -0.005, CAPTION + GRID_NOTE, ha="center", fontsize=10, wrap=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.955))
    files.append(out_dir / "joint_curves_all_layers.png")
    fig.savefig(files[-1], dpi=150, bbox_inches="tight")

    for i in range(n):
        fig = Figure(figsize=(8, 5))
        ax = fig.subplots()
        panel(ax, i, small=False)
        ax.set_xlabel("Weight bits x activation bits (WxAy), fewer total bits to the left")
        ax.set_ylabel("Calibration perplexity (lower is better)")
        ax.legend(fontsize=9)
        fig.suptitle(title, fontsize=11)
        fig.text(0.5, 0.005, CAPTION + SINGLE_NOTE, ha="center", fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        files.append(out_dir / f"joint_curve_layer{i:02d}.png")
        fig.savefig(files[-1], dpi=150, bbox_inches="tight")
    return files
