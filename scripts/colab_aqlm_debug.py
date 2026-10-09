"""Why is AQLM's perplexity 85k on Colab? Run in a Colab cell (GPU) after cloning the repo and installing it:

    %run /content/sdf/scripts/colab_aqlm_debug.py

Prints lines starting with DBG. Quantizes every decoder Linear once with AQLM (4 books, then 8), then measures
perplexity with only some module types quantized, and with a variant of the scale (one per 128 inputs).
"""
import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from sdf.stages import weights_b as wb  # noqa: E402

NAME = globals().get("MODEL") or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float16).to(dev).eval()
ids = tok("\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]), return_tensors="pt").input_ids[0]
WIN = [ids[i:i + 1024] for i in range(0, 20 * 1024, 1024)]
mods = {f"{i}.{n}": m for i, l in enumerate(model.model.layers) for n, m in l.named_modules() if isinstance(m, torch.nn.Linear)}
orig = {k: m.weight.data.clone() for k, m in mods.items()}


@torch.no_grad()
def ppl() -> float:
    nll = [model(input_ids=w[None].to(dev), labels=w[None].to(dev)).loss.float().item() for w in WIN]
    return math.exp(sum(nll) / len(nll))


def restore(keys=None):
    for k, m in mods.items():
        m.weight.data.copy_(orig[k])


def quantized(fn):
    out = {}
    for k, m in mods.items():
        w = m.weight.data.clone()
        fn(w)
        out[k] = w
    return out


def put(q, keep):
    restore()
    for k, w in q.items():
        if keep(k):
            mods[k].weight.data.copy_(w)


def aq_variant(books, norm):
    def fn(w):
        if norm == "row":
            return wb.aqlm_quantize_(w, books)
        x = w.float()  # one scale per 128 inputs
        g = x.reshape(x.shape[0], -1, 128)
        s = g.norm(dim=2, keepdim=True) + 1e-9
        t = (g / s).reshape(x.shape)
        wb.aqlm_quantize_(t, books)
        w.copy_((t.reshape(g.shape) * s).reshape(x.shape).to(w.dtype))
    return fn


print("DBG fp16 ppl", round(ppl(), 3), flush=True)
for books in (4, 8):
    q = quantized(aq_variant(books, "row"))
    rel = {k: ((q[k].float() - orig[k].float()).norm() / orig[k].float().norm()).item() for k in q}
    print("DBG books", books, "mean rel err", round(sum(rel.values()) / len(rel), 4),
          "worst", sorted(rel.items(), key=lambda t: -t[1])[:4], flush=True)
    print("DBG books", books, "ALL ppl", round((put(q, lambda k: True) or ppl()), 3), flush=True)
    if books == 4:
        for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            put(q, lambda k, t=t: k.endswith(t))
            print("DBG books 4 only", t, "ppl", round(ppl(), 3), flush=True)
        for i in (0, 1, 2, 21):
            put(q, lambda k, i=i: k.startswith(f"{i}."))
            print("DBG books 4 only layer", i, "ppl", round(ppl(), 3), flush=True)
for norm in ("group128",):
    q = quantized(aq_variant(4, norm))
    put(q, lambda k: True)
    print("DBG books 4 scale", norm, "ALL ppl", round(ppl(), 3), flush=True)
restore()
print("DBG done", flush=True)
