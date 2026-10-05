"""Draw the annotated thesis figures in docs/figures/ (SVG + PNG).

    pip install matplotlib && python docs/figures/make_figures.py

The numbers are copied from the final Stage 0 run 20261001-184158_781ac2 (TinyLlama-1.1B-Chat, RTX 5070 Ti,
main 7af8ab0, main.py defaults): its handoff.md, plus the layer-removal perplexity rises from the scores
comparison run (f57e864). The run folders live on Mohammad's PC, not in the repo.
# ponytail: numbers are a snapshot; read results/<run_id>/stage_0/*.json instead once figures are needed per run.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

OUT = Path(__file__).parent
RUN = "Stage 0 run 20261003-144910_3df5df (same on Colab, 20261004-120724_3c2a04), TinyLlama-1.1B-Chat (22 layers, 1.10 B parameters, 2.20 GB at 16 bits)"

# Colour-blind-safe (Okabe-Ito) roles used in every figure.
PROTECT, COMPRESS, GUARD, DONE, PLANNED, NOTE = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#999999", "#444444"

LAYERS = list(range(22))
SENS = [1.00, .81, .95, .52, .24, .33, .38, .90, .29, .19, .00, .05, .14, .10, .48, .67, .71, .43, .62, .57, .76, .86]
GUARDED = {0, 1, 2, 7, 21}
REMOVAL_RISE = {0: 1176, 2: 368, 7: 36, 21: 24, 1: 12}  # perplexity added when the layer is skipped
ACT_BITS = [8, 8, 8, 4, 4, 4, 8, 4, 8, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8]
ACT_DMG4 = [.0687, .7304, .1571, .0444, .0470, .0350, .1301, .0662, .0691, .0398, .0400,
            .0649, .0329, .0652, .0553, .0683, .0761, .0831, .0879, .0902, .1134, .1950]
KV_KEPT = [100, 75, 75, 30, 10, 10, 20, 50, 10, 10, 10, 10, 20, 20, 20, 10, 20, 10, 20, 10, 10, 20]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def box(ax, x, y, w, h, text, color, dashed=False, fs=9.5, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", lw=1.6,
                                ec=color, fc=color + "18", ls="--" if dashed else "-"))
    lines = text.split("\n")
    if bold_first:
        ax.text(x + w / 2, y + h - 2.2, lines[0], ha="center", va="top", fontsize=fs + 1, weight="bold")
        lines = lines[1:]
    ax.text(x + w / 2, y + h / 2 - (1.6 if bold_first else 0), "\n".join(lines), ha="center", va="center",
            fontsize=fs, linespacing=1.35)


def arrow(ax, a, b, color=NOTE, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color,
                                 ls="--" if dashed else "-", shrinkA=2, shrinkB=2))


def canvas(w, h, title, subtitle):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100), ax.set_ylim(0, 100), ax.axis("off")
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, weight="bold")
    fig.text(0.02, 0.925, subtitle, fontsize=9.5, color=NOTE)
    return fig, ax


def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig1_pipeline():
    fig, ax = canvas(14, 7.6, "Figure 1. The framework at a glance",
                     "Measure once which parts of the model are fragile, then compress each part only as hard as it can take.")
    box(ax, 1, 31, 14, 26, "Original model\nTinyLlama 1.1B\n22 layers\n2.20 GB at 16 bits", NOTE)
    box(ax, 21, 31, 17, 26, "Stage 0  (done)\nSensitivity plan\nTests each layer and\nwrites one plan per\nlater stage", DONE)
    arrow(ax, (15.5, 44), (20.5, 44))
    stages = [("Stage 1  (planned)\nWeights: the stored numbers\nGPTQ, AWQ, pruning, low-rank", 56),
              ("Stage 2  (planned)\nActivations: numbers passed\nbetween layers\nSmoothQuant, QuaRot, RPTQ, SpinQuant", 31),
              ("Stage 3  (planned)\nKV cache: short-term memory\nwhile writing\nQuaRot KV, KVQuant, H2O, SnapKV", 6)]
    for text, y in stages:
        box(ax, 46, y, 26, 20, text, PLANNED, dashed=True, fs=8.8)
        arrow(ax, (38.5, 44), (45.5, y + 10))
        arrow(ax, (72.5, y + 10), (79.5, 44), dashed=True, color=PLANNED)
    box(ax, 80, 31, 19, 26, "Stage 4  (planned)\nEvaluation on real\nsoftware: HF, llama.cpp,\nvLLM, TensorRT-LLM", PLANNED, dashed=True)
    box(ax, 21, 83, 78, 12, "Search  (planned)\nTries many settings (threshold, share removed, group size, ...) with\nMOBO, MFBO and NSGA-III; keeps the best trade-offs of accuracy, memory, speed",
        PLANNED, dashed=True, fs=8.8)
    arrow(ax, (29.5, 82.5), (29.5, 57.5), dashed=True, color=PLANNED)
    fig.text(0.02, 0.0, "Solid green = built and tested.  Grey dashed = planned.\n"
            "Stages 1-3 each start from the original model plus the Stage 0 plan, so each kind of compression is measured on its own.\n"
            "Every stage compares three versions under identical conditions: the original model, the standard method, and the method guided by Stage 0.",
            fontsize=9, color=NOTE, linespacing=1.4, va="top")
    ax.annotate("Fragile layers get more bits\nand are never thinned out", (29.5, 31), (22, 14), fontsize=9,
                color=DONE, ha="center", arrowprops=dict(arrowstyle="-", color=DONE, lw=1))
    save(fig, "fig1_pipeline")


def fig2_stage0_method():
    fig, ax = canvas(14, 8.4, "Figure 2. How Stage 0 builds its plans",
                     "Each row is one test. Stage 0 changes one layer at a time on ordinary text (64 WikiText-2 passages) "
                     "and writes down how much worse the model predicts the next word.")
    rows = [
        (70, "Weights  (for Stage 1)", [
            "1. Skip one layer\nRun the text with that\nlayer taken out",
            "2. Measure the damage\nRise in perplexity\n(how surprised the model\nis by real text)",
            "3. Rank the layers\nScore 0 (least fragile)\nto 1 (most fragile)",
            "4. Decide per layer\nScore >= 0.5: 8 bits, keep all\nBelow: 4 bits, remove 30%\nTop 5 by damage:\nnever thinned out"],
         "compression_plan.json"),
        (40, "Activations  (for Stage 2)", [
            "1. Round the inputs\nRound one layer's inputs\nto 4 bits, then to 8 bits",
            "2. Measure the damage\nRise in perplexity\nat each bit width",
            "3. Share out the bits\nAverage 6 bits; 8 bits go to\nthe layers hurt most at 4",
            ""], "activation_plan.json"),
        (10, "KV cache  (for Stage 3)", [
            "1. Round the notes\nOne layer's keys, then\nvalues, to 2, 4 and 8 bits",
            "2. Measure the damage\nRise in perplexity, and\nwhere each layer looks\nback in the text",
            "3. Share out the bits\nAverage 4 bits (standard size)",
            "4. Decide what to forget\nKeep the fewest earlier words\nthat get 95% of the attention"],
         "kv_cache_plan.json"),
    ]
    for y, label, steps, out in rows:
        ax.text(0.5, y + 22.5, label, fontsize=11, weight="bold", color=DONE)
        xs = [0.5, 20.5, 40.5, 60.5]
        for i, (x, s) in enumerate(zip(xs, steps)):
            if not s:
                continue
            box(ax, x, y, 17.5, 19, s, PROTECT if i < 2 else DONE, fs=8.6)
            nxt = next((xs[j] for j in range(i + 1, 4) if steps[j]), None)
            arrow(ax, (x + 17.8, y + 9.5), ((nxt or 81) - 0.4, y + 9.5))
        ax.text(81.5, y + 9.5, out, fontsize=9, family="monospace", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=NOTE))
    ax.text(81.5, 95, "File each later stage loads", fontsize=9, color=NOTE)
    ax.text(0.5, 1, "Blue = measured on the model (slow, about 99 s for weights; cached and reused by every search trial).  "
            "Green = planning from the measurements (milliseconds).", fontsize=9, color=NOTE)
    save(fig, "fig2_stage0_method")


def fig3_weight_plan():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [2.1, 1.25]})
    fig.suptitle("Figure 3. Which layers are fragile, and what Stage 1 is told to do", x=0.02, ha="left",
                 fontsize=14, weight="bold")
    fig.text(0.02, 0.905, RUN, fontsize=9, color=NOTE)
    colors = [PROTECT if s >= 0.5 else COMPRESS for s in SENS]
    bars = ax.bar(LAYERS, SENS, color=colors, edgecolor=[GUARD if i in GUARDED else "none" for i in LAYERS],
                  linewidth=[2.2 if i in GUARDED else 0 for i in LAYERS])
    ax.axhline(0.5, color=NOTE, ls="--", lw=1)
    ax.text(10, 0.515, "threshold 0.5", ha="center", fontsize=8.5, color=NOTE)
    for i in GUARDED:
        ax.text(i, SENS[i] + 0.02, f"+{REMOVAL_RISE[i]:,}", ha="center", fontsize=8, color=GUARD, weight="bold")
    ax.set(xticks=LAYERS, xlabel="Layer (0 = where the text comes in)", ylabel="Sensitivity score (rank, 0 to 1;\nhigher = more fragile)",
           ylim=(0, 1.18))
    ax.annotate("Without layer 0 the model stops working:\nperplexity goes from 14 to about 1,190",
                (0, 1.06), (3.2, 1.08), fontsize=8.8, color=GUARD, va="center",
                arrowprops=dict(arrowstyle="->", color=GUARD))
    ax.annotate("Middle layers barely matter:\nskipping one adds 1.4 to 5.7",
                (11, 0.07), (11.2, 0.68), fontsize=8.8, ha="center",
                arrowprops=dict(arrowstyle="->", color=NOTE))
    fig.legend(handles=[Patch(color=PROTECT, label="Main plan, protected: 8 bits, nothing removed (11 layers)"),
                       Patch(color=COMPRESS, label="Main plan, compressed: 4 bits, 30% removed (11 layers)"),
                       Patch(fc="white", ec=GUARD, lw=2, label="Never pruned in any plan, even the smaller budget plan\n(number = perplexity added if skipped)")],
              loc="lower left", fontsize=8.5, frameon=False, bbox_to_anchor=(0.03, 0.0), ncol=3)

    names = ["Original\n(16-bit)", "Standard\n4-bit", "Stage 0\nmain plan", "Stage 0\nbudget plan"]
    mb = [2200, 777.0, 942.0, 768.0]
    c = [NOTE, PLANNED, DONE, DONE]
    ax2.bar(range(4), mb, color=c)
    for i, v in enumerate(mb):
        ax2.text(i, v + 30, f"{v:,.0f} MB", ha="center", fontsize=8.8)
    ax2.set(xticks=range(4), ylabel="Predicted size of the model's numbers\n(MB, smaller is better)", ylim=(0, 2550))
    ax2.set_xticklabels(names, fontsize=8.3)
    ax2.set_title("Predicted size of each version", fontsize=10.5, loc="left")
    ax2.text(2, 1350, "The budget plan fits within the\nstandard method's memory,\nso Stage 1 compares accuracy\nat no extra memory",
             ha="center", fontsize=8.5, color=NOTE)
    fig.text(0.02, -0.03, "Bar colours show the main plan: all 11 blue layers keep 8 bits and lose nothing. The orange border matters in the smaller "
             "budget plan, which protects only the 5 bordered layers; there the other blue layers drop to 4 bits with 30% removed.\n"
             "Higher score = more fragile layer; for size, smaller is better. The score is the layer's rank by damage, not the damage itself: the five 'never pruned' layers "
             "are where removing one layer hurts by far the most. Sizes are predictions; Stage 1 measures the real ones.",
             fontsize=8.8, color=NOTE)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    save(fig, "fig3_weight_plan")


def fig4_activation_plan():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    fig.suptitle("Figure 4. Activation plan for Stage 2: where rounding the passed-on numbers hurts",
                 x=0.02, ha="left", fontsize=14, weight="bold")
    fig.text(0.02, 0.905, RUN, fontsize=9, color=NOTE)
    ax.bar(LAYERS, ACT_DMG4, color=[PROTECT if b == 8 else COMPRESS for b in ACT_BITS])
    ax.set(xticks=LAYERS, xlabel="Layer", ylabel="Perplexity added when only this layer's\ninputs are rounded to 4 bits (lower is better)",
           ylim=(0, 0.85))
    ax.annotate("Layer 1 is by far the most fragile:\n+0.73 at 4 bits, +0.002 at 8 bits",
                (1, 0.73), (4, 0.7), fontsize=9, va="center", arrowprops=dict(arrowstyle="->", color=NOTE))
    ax.annotate("At 8 bits every layer's damage is\nwithin noise (at most 0.003)", (14, 0.06), (11, 0.24), fontsize=9,
                ha="center", color=NOTE)
    ax.legend(handles=[Patch(color=PROTECT, label="Gets 8 bits (11 layers)"),
                       Patch(color=COMPRESS, label="Gets 4 bits (11 layers)")], frameon=False, loc="upper right")
    ax.text(21.5, 0.5, "Average 6 bits per layer\n\nPredicted perplexity rise\n  this plan:  0.56\n"
            "  copied from weight plan:  0.67\n  standard 8-bit everywhere:  0.009",
            ha="right", fontsize=9, family="monospace", bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=NOTE))
    fig.text(0.02, -0.02, "Lower perplexity rise is better. Activations are the numbers one layer hands to the next while the model runs. Rises are added "
             "up per layer, which assumes they add; Stage 2 measures the real effect.", fontsize=8.8, color=NOTE)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig4_activation_plan")


def fig5_kv_plan():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [2.3, 1]})
    fig.suptitle("Figure 5. KV cache plan for Stage 3: how much of the past each layer remembers",
                 x=0.02, ha="left", fontsize=14, weight="bold")
    fig.text(0.02, 0.905, RUN + ". Measured on 64 passages.", fontsize=9, color=NOTE)
    ax.bar(LAYERS, KV_KEPT, color=[PROTECT if k >= 50 else COMPRESS for k in KV_KEPT])
    for i, k in enumerate(KV_KEPT):
        ax.text(i, k + 1.5, f"{k}", ha="center", fontsize=8)
    ax.set(xticks=LAYERS, xlabel="Layer", ylabel="Share of earlier words kept (%)", ylim=(0, 118))
    ax.annotate("Early layers read the whole text", (1, 80), (4, 95), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=NOTE))
    ax.annotate("Most layers look at a few words only:\n10-20% of the words get 95% of their attention",
                (10, 13), (12.5, 55), fontsize=9, ha="center", arrowprops=dict(arrowstyle="->", color=NOTE))
    ax.text(0, -26, "Bits: keys 4 everywhere; values 4 everywhere except layer 0 (2 bits, no measurable cost).",
            fontsize=8.8, color=NOTE, transform=ax.transData)

    names, mb = ["Standard\n4-bit", "Plan,\nbits only", "Plan, bits +\nforgetting"], [13.0, 12.8, 3.2]
    ax2.bar(range(3), mb, color=[PLANNED, DONE, DONE])
    for i, v in enumerate(mb):
        ax2.text(i, v + 0.3, f"{v} MB", ha="center", fontsize=9)
    ax2.set(xticks=range(3), ylabel="Predicted cache for one 2,048-token text\n(MB, smaller is better)", ylim=(0, 16))
    ax2.set_xticklabels(names, fontsize=8.6)
    ax2.set_title("Predicted memory", fontsize=10.5, loc="left")
    ax2.annotate("4x smaller, mostly\nfrom forgetting;\nkept words still get\n96.7% of attention",
                 (2, 4.4), (2, 8.5), fontsize=8.6, ha="center", arrowprops=dict(arrowstyle="->", color=NOTE))
    fig.text(0.02, -0.04, "Smaller memory is better. The KV cache is the model's notes on every earlier word, kept while it writes. Choosing bits "
             "per layer predicts no accuracy gain at a 4-bit average; Stage 3 must confirm the forgetting with a real "
             "method (H2O, SnapKV).", fontsize=8.8, color=NOTE)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig5_kv_plan")


def fig6_handoff():
    fig, ax = canvas(14, 8, "Figure 6. What Stage 0 hands to each later stage",
                     "Every plan is a JSON file next to the Stage 0 report; later stages load it instead of measuring again.")
    box(ax, 36, 40, 28, 22, "Stage 0 results\nTinyLlama, 22 layers\nPerplexity 10.18 (tuning half)\n10.62 (held-out half, never tuned on)", DONE)
    targets = [
        (2, 72, "Stage 1: weights\ncompression_plan.json\n11 layers protected at 8 bits;\n0, 1, 2, 7, 21 never pruned\n942 MB plan vs 777 MB standard\n(+ 768 MB budget plan)", (36, 56)),
        (66, 72, "Stage 2: activations\nactivation_plan.json\n8 or 4 bits per layer, average 6\nPredicted rise 0.56 vs 0.67 copied\nLayer 1 most fragile", (64, 56)),
        (2, 6, "Stage 3: KV cache\nkv_cache_plan.json\nKeys/values 4 bits (layer 0 values 2)\nWords kept 10-100% per layer\n13.0 MB -> 3.2 MB predicted", (36, 46)),
        (66, 6, "Stage 4 and search\nresults.json, sensitivity_profile.json\nReference: 47 ms to read a prompt,\n46 ms per word, 2.45 GB peak memory\nCached profile: trials re-plan in ms", (64, 46)),
    ]
    for x, y, text, anchor in targets:
        box(ax, x, y, 32, 22, text, PLANNED, dashed=True, fs=8.8)
        arrow(ax, anchor, (x + 16, y + (0 if y > 50 else 22)))
    ax.text(50, 30, "Each stage starts from the original model,\nnot from the previous stage's output",
            ha="center", fontsize=9, color=NOTE)
    save(fig, "fig6_handoff")


if __name__ == "__main__":
    for f in (fig1_pipeline, fig2_stage0_method, fig3_weight_plan, fig4_activation_plan, fig5_kv_plan, fig6_handoff):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("fig*")))
