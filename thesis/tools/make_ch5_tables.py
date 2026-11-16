"""Fills the tables of thesis/tools/Chapter5.template.tex from thesis/data/*.csv and writes thesis/Chapters/Chapter5v2.tex,
so that no number of those tables is typed by hand.

    python thesis/tools/make_ch5_tables.py
"""
import csv, collections, sys
from pathlib import Path
T = Path(__file__).resolve().parent.parent
D = str(T / "data") + "/"
def rows(n): return list(csv.DictReader(open(D + n, encoding="utf-8")))
f2 = lambda v: f"{float(v):.2f}"
def big(v):
    v = float(v)
    return f"{v:.2f}" if v < 100 else f"{v:,.0f}".replace(",", "{,}")
out = {}
q = rows("stage014-quantization.csv")
d = collections.OrderedDict()
for x in q[1:]:
    d.setdefault(x["Method"], {})["std" if x["Version"].startswith("Standard") else "ours"] = x
# weight-only table
b = []
for m, v in d.items():
    if m.endswith("weights only"):
        s, o = v["std"], v["ours"]
        b.append(" & ".join([m.replace(" weights only", "").replace("#", "\\#"),
            f2(s["HF ppl validation"]), f2(o["HF ppl validation"]), f2(s["HF ppl held-out"]), f2(o["HF ppl held-out"]),
            f2(s["llama.cpp ppl validation"]), f2(o["llama.cpp ppl validation"]),
            f"{float(s['llama.cpp decode (tokens/s)']):.0f}", f"{float(o['llama.cpp decode (tokens/s)']):.0f}"]) + " \\\\")
out["QUANT_WO"] = "\n".join(b)
# WxAy table
b = []
for m, v in d.items():
    if m.endswith("+ RTN activations") or m.endswith("+ SmoothQuant activations"):
        j = d[m + " (joint weight + activation plan)"]["ours"]; w8 = d[m + " (W8A8, 8 bits on all layers)"]["std"]
        name = m.replace(" weights + RTN activations", "").replace("RTN weights + SmoothQuant activations", "RTN with SmoothQuant").replace("#", "\\#")
        b.append(" & ".join([name, f2(v["std"]["HF ppl validation"]), f2(v["ours"]["HF ppl validation"]),
                             f2(j["HF ppl validation"]), f2(w8["HF ppl validation"])]) + " \\\\")
out["QUANT_WA"] = "\n".join(b)
# pruning
p = rows("stage024-pruning.csv")
pd = collections.defaultdict(dict)
for x in p[1:]:
    tech, rest = x["Method"].split(", ", 1); lvl = int(rest.split("%")[0])
    k = "same" if "same size" in x["Method"] else "std" if x["Version"].startswith("Standard") else "ours"
    pd[(tech, lvl)][k] = x
b = []
for tech, short in (("Unstructured pruning (Wanda)", "Unstructured (Wanda)"), ("Structured pruning", "Structured"), ("Low-rank", "Low-rank")):
    for lvl in (30, 40, 50, 60, 70):
        v = pd[(tech, lvl)]
        b.append(" & ".join([short if lvl == 30 else "", str(lvl), big(v["std"]["HF ppl validation"]), big(v["ours"]["HF ppl validation"]), big(v["same"]["HF ppl validation"])]) + " \\\\")
    b.append("\\hline")
out["PRUNE"] = "\n".join(b[:-1])
# kv
k = rows("stage034-kv-predicted.csv")
kd = collections.OrderedDict()
for x in k[1:]: kd.setdefault(x["method"], {})[x["variant"]] = x
names = {"rtn_kv": "Round-to-nearest KV", "quarot_kv": "QuaRot KV", "kvquant": "KVQuant", "h2o": "H2O", "snapkv": "SnapKV", "infinigen": "InfiniGen"}
b = []
for m, v in kd.items():
    s, o = v["original"], v["framework"]
    b.append(" & ".join([names[m], f2(s["hf_ppl_val"]), f2(s["hf_ppl_heldout"]), f"{float(s['predicted_kv_memory_gb'])*1000:.1f}",
                         f2(o["hf_ppl_val"]), f2(o["hf_ppl_heldout"]), f"{float(o['predicted_kv_memory_gb'])*1000:.1f}"]) + " \\\\")
out["KV"] = "\n".join(b)
# stage 0 per-layer
s0 = rows("stage0-per-layer.csv")
b = []
for x in s0:
    b.append(" & ".join([x["layer"], f"{float(x['raw_score']):.2f}", f"{float(x['norm_score']):.2f}", x["weight_bits"],
                         "yes" if x["guarded"] == "True" else "no", x["act_bits"], x["kv_key_bits"], x["kv_value_bits"],
                         f"{float(x['kv_keep_ratio'])*100:.0f}"]) + " \\\\")
out["STAGE0"] = "\n".join(b)
tpl = (T / "tools" / "Chapter5.template.tex").read_text(encoding="utf-8")
for k_, v_ in out.items():
    if k_ != "STAGE0":
        assert tpl.count("@@" + k_ + "@@") == 1, k_
        tpl = tpl.replace("@@" + k_ + "@@", v_)
assert "@@" not in tpl
(T / "Chapters" / "Chapter5v2.tex").write_text(tpl, encoding="utf-8", newline="\n")
