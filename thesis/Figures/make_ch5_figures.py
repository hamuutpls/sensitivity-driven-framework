"""Draws the Chapter 5 figures as PDF from the measured results in thesis/data/*.csv.

    pip install matplotlib && python thesis/Figures/make_ch5_figures.py

Colours are from the Okabe-Ito colour-blind-safe palette; Standard and Our method also differ in hatch or marker.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
STD, OURS, SAME, FP = "#E69F00", "#0072B2", "#009E73", "#555555"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})


def rows(name):
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(HERE / name, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def quantization():
    r = rows("stage014-quantization.csv")
    fp = float(r[0]["HF ppl validation"])
    names, std, ours = [], [], []
    for x in r:
        if x["Method"].endswith("weights only"):
            m = x["Method"].replace(" weights only", "")
            if x["Version"].startswith("Standard"):
                names.append(m), std.append(float(x["HF ppl validation"]))
            else:
                ours.append(float(x["HF ppl validation"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    xs = range(len(names))
    ax.bar([i - 0.2 for i in xs], std, 0.4, color=STD, label="Standard (4 bits in every layer, 0.81 GB)")
    ax.bar([i + 0.2 for i in xs], ours, 0.4, color=OURS, hatch="//", edgecolor="white",
           label="Our method (Stage 0 plan, 1.05 GB)")
    ax.axhline(fp, color=FP, ls="--", lw=1, label="FP16 (2.20 GB)")
    ax.set_xticks(list(xs)), ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylim(9.9, 12.3), ax.set_ylabel("Validation perplexity (lower is better)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "ch5-quantization.pdf")


def pruning():
    r = rows("stage024-pruning.csv")
    fp = float(r[0]["HF ppl validation"])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for ax, (key, title) in zip(axes, (("Unstructured pruning (Wanda)", "Unstructured (Wanda)"),
                                       ("Structured pruning", "Structured"), ("Low-rank", "Low-rank"))):
        series = {"std": [], "ours": [], "same": []}
        for x in r[1:]:
            if not x["Method"].startswith(key):
                continue
            share = int(x["Method"].split(", ")[1].split("%")[0])
            k = "same" if "same size" in x["Method"] else "std" if x["Version"].startswith("Standard") else "ours"
            series[k].append((share, float(x["HF ppl validation"])))
        for k, c, mk, lab in (("std", STD, "o", "Standard"), ("ours", OURS, "s", "Our method"),
                              ("same", SAME, "^", "Our method, same total removed")):
            pts = sorted(series[k])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker=mk, ms=4, lw=1.3, label=lab)
        ax.axhline(fp, color=FP, ls="--", lw=1, label="FP16")
        ax.set_yscale("log"), ax.set_title(title, size=9), ax.set_xlabel("Pruning level (%)")
        ax.set_xticks(range(30, 100, 10))
    axes[0].set_ylabel("Validation perplexity\n(log scale, lower is better)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    save(fig, "ch5-pruning.pdf")


def kv():
    r = rows("stage034-kv.csv")
    fp = float(r[0]["HF ppl validation"])
    names, std, ours = [], [], []
    for x in r[1:]:
        if x["Version"].startswith("Standard"):
            names.append(x["Method"]), std.append(float(x["HF ppl validation"]))
        else:
            ours.append(float(x["HF ppl validation"]))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    xs = range(len(names))
    ax.bar([i - 0.2 for i in xs], std, 0.4, color=STD, label="Standard")
    ax.bar([i + 0.2 for i in xs], ours, 0.4, color=OURS, hatch="//", edgecolor="white", label="Our method (Stage 0 plan)")
    ax.axhline(fp, color=FP, ls="--", lw=1, label="FP16")
    ax.set_xticks(list(xs)), ax.set_xticklabels(names)
    ax.set_ylim(9.9, 12.5), ax.set_ylabel("Validation perplexity (lower is better)")
    ax.legend(frameon=False, loc="upper right", ncol=3)
    save(fig, "ch5-kv.pdf")


def sensitivity():
    r = rows("stage0-per-layer.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for x in r:
        guarded, prot = x["guarded"] == "True", x["protected"] == "True"
        c = "#D55E00" if guarded else OURS if prot else "#999999"
        h = "xx" if guarded else "//" if prot else ""
        ax.bar(int(x["layer"]), float(x["raw_score"]), 0.75, color=c, hatch=h, edgecolor="white")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#D55E00", hatch="xx", ec="white", label="protected (8 bits) and never pruned"),
                       Patch(fc=OURS, hatch="//", ec="white", label="protected (8 bits)"),
                       Patch(fc="#999999", label="compressed (4 bits, pruned)")], frameon=False, fontsize=8)
    ax.set_yscale("log"), ax.set_xticks(range(22)), ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Rise in calibration perplexity\nwhen the layer is bypassed (log scale)")
    save(fig, "ch5-sensitivity.pdf")


def tries():
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0), sharey=True)
    for ax, (name, title) in zip(axes, (("search-lowest-perplexity-smallest-size-40tries.csv",
                                         "Lowest-perplexity and smallest-size search"),
                                        ("search-fastest-smallest-size-40tries.csv",
                                         "Fastest and smallest-size search"))):
        r = rows(name)
        wo = [x for x in r if "_with_" not in x["stage1"]]
        wa = [x for x in r if "_with_" in x["stage1"]]
        ax.scatter([float(x["vllm_decode_ms_per_token"]) for x in wo], [float(x["vllm_ppl_val"]) for x in wo],
                   color=OURS, marker="o", s=22, label="weights only (%d tries)" % len(wo))
        ax.scatter([float(x["vllm_decode_ms_per_token"]) for x in wa], [float(x["vllm_ppl_val"]) for x in wa],
                   color=STD, marker="^", s=26, label="weights and activations (%d tries)" % len(wa))
        ax.axvline(2.5, color=FP, ls="--", lw=1), ax.axhline(11.5, color=FP, ls="--", lw=1)
        ax.scatter([2.54], [10.18], color="black", marker="*", s=70, label="FP16")
        ax.set_xscale("log"), ax.set_xticks([2, 2.5, 5, 10, 20, 40]), ax.set_xticklabels(["2", "2.5", "5", "10", "20", "40"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("vLLM decode latency (ms per token)"), ax.set_title(title, size=9)
        ax.set_ylim(9.5, 26.5)
    axes[0].set_ylabel("vLLM validation perplexity\n(lower is better)")
    axes[0].text(5, 11.7, "limits: 11.5 and 2.5 ms", color=FP, size=8, va="bottom")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper center")
    save(fig, "ch5-search-tries.pdf")


if __name__ == "__main__":
    quantization()
    pruning()
    kv()
    sensitivity()
    tries()
