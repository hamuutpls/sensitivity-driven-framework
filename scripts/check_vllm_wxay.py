"""GPU check that vLLM really rounds a WxAy model's activations (the compressed-tensors export, `ct_export`).

TinyLlama is saved at one (weight bits, activation bits) pair on every layer, then at the Stage 0 joint plan's
pairs, and measured in vLLM. If vLLM rounded nothing, every A8/A4 row would equal its A16 row; if it does, the
perplexity rises from A16 to A8 (a little) to A4 (clearly). The worker also refuses a model whose activation
rounding is not on a Humming kernel. Usage (Linux, the framework's Python):

    python scripts/check_vllm_wxay.py /path/to/vllm-venv/bin/python [max_windows]
"""

import sys
from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdf.config import EvalConfig, _default_sources
from sdf.data import eval_windows, load_texts
from sdf.eval import vllm_backend
from sdf.stage0.activation import ActivationLayerPlan, ActivationPlan
from sdf.stage0.planner import uniform_plan

NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Stage 0 joint plan of run 20261010-182342_e5c3ce, layers 0-21 (weight bits, activation bits)
JOINT = [(4, 4), (8, 8), (8, 8), (4, 4), (4, 4), (4, 4), (8, 8), (4, 4), (4, 8), (4, 4), (4, 4), (4, 4), (8, 4),
         (8, 4), (8, 4), (8, 8), (8, 8), (8, 8), (4, 8), (8, 8), (4, 8), (8, 8)]


def plans(pairs):
    w = uniform_plan([0.0] * len(pairs), 4, 0.0)
    w = replace(w, layers=tuple(replace(lp, bit_width=b) for lp, (b, _) in zip(w.layers, pairs)))
    a = ActivationPlan(tuple(ActivationLayerPlan(i, b, b == 16) for i, (_, b) in enumerate(pairs)), "measured")
    return w, a


def main(vllm_python: str, max_windows: int) -> None:
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float16).cpu()  # leave the GPU to vLLM
    cfg = EvalConfig(vllm_python=vllm_python, max_windows=max_windows)
    val, held = eval_windows(load_texts(_default_sources(), "wikitext2", "test"), tok, cfg.seq_len, max_windows)
    n = model.config.num_hidden_layers
    rows = [(f"W{w}A{a} every layer", [(w, a)] * n) for w in (8, 4) for a in (16, 8, 4)]
    rows.append(("Stage 0 joint plan", JOINT))
    print("WXAY-BEGIN")
    for label, pairs in rows:
        w, a = plans(pairs)
        try:
            out = vllm_backend.measure(model, tok, w, val, held, cfg, 16, 128, acts=a, act_group_size=128)
            print(f"{label}: ppl_val {out['vllm_ppl_val']:.3f}, held-out {out['vllm_ppl_heldout']:.3f}, "
                  f"{out['vllm_size_gb']:.3f} GB, decode {out['vllm_decode_ms_per_token']:.2f} ms/token")
        except vllm_backend.VLLMError as e:
            print(f"{label}: FAILED {e}")
    print("WXAY-END")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
