"""Draws the two Chapter 4 figures (framework overview, search loop) as PDF.

    pip install matplotlib && python thesis/Figures/make_ch4_figures.py

Colours are from the Okabe-Ito colour-blind-safe palette; every box is also told apart by its text.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).parent
BLUE, ORANGE, GREEN, YELLOW, GREY = "#cfe3f3", "#f8dcb4", "#c9e8dc", "#f7efb0", "#ececec"
EDGE = "#333333"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "pdf.fonttype": 42})


def box(ax, x, y, w, h, text, color, bold=None, size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=EDGE, lw=0.9))
    if bold:
        ax.text(x + w / 2, y + h - 0.16, bold, ha="center", va="top", weight="bold", size=size)
        ax.text(x + w / 2, y + h - 0.56, text, ha="center", va="top", size=size - 0.5, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", size=size, linespacing=1.25)


def arrow(ax, a, b, text=None, rad=0.0, off=(0, 0.08), style="-|>", ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=EDGE, lw=1.0, linestyle=ls,
                                                    connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", va="bottom", size=8.5)


def overview(model_label="TinyLlama", out=None):
    fig, ax = plt.subplots(figsize=(8.0, 5.15))
    ax.set_xlim(0, 12.8), ax.set_ylim(0, 8.2), ax.axis("off")
    w, h, y = 2.36, 2.3, 2.6
    xs = [0.1 + i * 2.56 for i in range(5)]
    stages = [
        ("Stage 0", "Profile and plan\n\nmeasures every\nlayer, writes the\nper-layer plans", BLUE),
        ("Stage 1", "Quantization\n\nweights and\nactivations at\nthe planned bits", GREEN),
        ("Stage 2", "Pruning\n\nweights removed\nat the planned\nshare per layer", GREEN),
        ("Stage 3", "KV cache\n\nkeys and values\nrounded, tokens\ndropped per layer", GREEN),
        ("Stage 4", "Evaluation\n\nquality, size,\nmemory, latency\nin the engines", ORANGE),
    ]
    for x, (name, text, color) in zip(xs, stages):
        box(ax, x, y, w, h, text, color, bold=name)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + w, y + h / 2), (b, y + h / 2))
    # inputs
    box(ax, xs[0], 0.35, w, 1.3, model_label, GREY)
    arrow(ax, (xs[0] + w / 2, 1.65), (xs[0] + w / 2, y))
    # plans handed down
    ax.plot([xs[0] + w / 2 + 0.6, xs[0] + w / 2 + 0.6], [y, 2.05], color=EDGE, lw=1.0, ls="--")
    ax.plot([xs[0] + w / 2 + 0.6, xs[3] + w / 2], [2.05, 2.05], color=EDGE, lw=1.0, ls="--")
    for x in xs[1:4]:
        arrow(ax, (x + w / 2, 2.05), (x + w / 2, y), ls="--")
    ax.text((xs[1] + xs[3] + w) / 2, 1.95, "per-layer plans", ha="center", va="top", size=8.5)
    # search on top
    box(ax, xs[0], 6.3, xs[4] + w - xs[0], 1.5,
        "proposes the technique for Stages 1, 2 and 3 and the plan settings for the next try;\n"
        "limits on perplexity and decode latency; minimises perplexity, latency and size", YELLOW,
        bold="Search (Gaussian-process Bayesian optimisation) under the deployment requirement", size=9)
    arrow(ax, (xs[0] + w / 2, 6.3), (xs[0] + w / 2, y + h), "settings", off=(0.62, -0.55))
    # one technique for each of Stages 1, 2 and 3: a line down from the search that branches to the three stages
    ty = 5.5
    ax.plot([xs[2] + w / 2, xs[2] + w / 2], [6.3, ty], color=EDGE, lw=1.0)
    ax.plot([xs[1] + w / 2, xs[3] + w / 2], [ty, ty], color=EDGE, lw=1.0)
    for x in xs[1:4]:
        arrow(ax, (x + w / 2, ty), (x + w / 2, y + h))
    ax.text(xs[2] + w / 2 - 0.12, ty + 0.07, "one technique and settings for each stage", ha="right", va="bottom", size=8.5)
    arrow(ax, (xs[4] + w / 2, y + h), (xs[4] + w / 2, 6.3), "measurements", off=(-1.1, -0.55))
    fig.savefig(out or HERE / "ch4-framework-overview.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def search_loop():
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.set_xlim(0, 12.8), ax.set_ylim(0, 11.2), ax.axis("off")
    w, h, x = 7.1, 0.95, 0.75
    steps = [
        ("1. The optimiser proposes a point: one technique per stage\nand the plan settings", YELLOW),
        ("2. Stage 0 turns the settings into per-layer plans\n(measurements are cached and reused)", BLUE),
        ("3. Predicted size checked against the size limit;\nthe combination is looked up in the log of earlier tries", BLUE),
        ("4. One model is built in sequence: Stage 1, then Stage 2,\nthen Stage 3, on the same model", GREEN),
        ("5. The model is measured: perplexity in Hugging Face,\nperplexity and decode latency in vLLM", ORANGE),
        ("6. The requirement is checked; objective values and\nany shortfall are returned to the optimiser", YELLOW),
    ]
    ys = [10.0 - i * 1.45 for i in range(len(steps))]
    for y, (text, color) in zip(ys, steps):
        box(ax, x, y, w, h, text, color, size=8.8)
    for a, b in zip(ys, ys[1:]):
        arrow(ax, (x + w / 2, a), (x + w / 2, b + h))
    # loop back
    lx = x - 0.4
    ax.plot([x, lx, lx], [ys[5] + h / 2, ys[5] + h / 2, ys[0] + h / 2], color=EDGE, lw=1.0)
    arrow(ax, (lx, ys[0] + h / 2), (x, ys[0] + h / 2))
    ax.text(lx - 0.08, (ys[0] + ys[5] + h) / 2, "repeat until the budget of tries is used", rotation=90, ha="right",
            va="center", size=8.5)
    # side exits from step 3
    sx, sw = 8.25, 4.45
    box(ax, sx, ys[2] + 0.55, sw, 0.8, "over the size limit: not built,\nshortfall returned to the optimiser", GREY, size=8)
    box(ax, sx, ys[2] - 0.45, sw, 0.8, "already built in an earlier run:\nits logged measurements are reused", GREY, size=8)
    arrow(ax, (x + w, ys[2] + 0.7), (sx, ys[2] + 0.95))
    arrow(ax, (x + w, ys[2] + 0.25), (sx, ys[2] - 0.05))
    # final model
    fy = 0.25
    box(ax, 0.3, fy, 12.3, 1.45,
        "the try that meets every limit and is best on the first objective (ties: the next objective) is built again,\n"
        "its files are kept, and it is measured in the inference engines\n"
        "and on the four accuracy tasks, next to the FP16 model", ORANGE, bold="After the last try: final model (Stage 4)",
        size=8.6)
    arrow(ax, (x + w / 2, ys[5]), (x + w / 2, fy + 1.45))
    fig.savefig(HERE / "ch4-search-loop.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    overview()
    search_loop()
    # Wiley paper variant of the overview (input box names the model generically)
    overview("Desired model\n(TinyLlama in\nour case)", HERE.parent / "paper-wiley" / "Figures" / "ch4-framework-overview.pdf")
