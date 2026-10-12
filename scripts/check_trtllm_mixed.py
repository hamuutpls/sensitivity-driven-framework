"""GPU check that TensorRT-LLM builds and runs a mixed-bit plan (some layers 4-bit, some 8-bit, some FP16).

TinyLlama is saved in GPTQ format at three plans and measured in both TensorRT-LLM and vLLM: every layer 4-bit,
Our method's Stage 0 plan (8-bit on layers 0-3, 7, 15, 16, 18-21, 4-bit elsewhere), and that plan with layer 5 kept
at FP16. Both backends read the same rounded weights, so their perplexities must agree (they agree to about 1e-4
on FP16 and on uniform 4-bit). The script prints PASS or FAIL per plan. Usage (Linux, the framework's Python):

    python scripts/check_trtllm_mixed.py /path/to/trt-venv/bin/python /path/to/vllm-venv/bin/python [max_windows]
"""

import sys
from dataclasses import replace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdf.config import EvalConfig, _default_sources
from sdf.data import eval_windows, load_texts
from sdf.eval import trtllm_backend, vllm_backend
from sdf.stage0.planner import uniform_plan

NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EIGHT = {0, 1, 2, 3, 7, 15, 16, 18, 19, 20, 21}  # Our method's 8-bit layers (Stage 0 weight plan)


def plan(bits_of):
    p = uniform_plan([0.0] * 22, 4, 0.0)
    return replace(p, layers=tuple(replace(lp, bit_width=bits_of(lp.layer)) for lp in p.layers))


def main(trtllm_python: str, vllm_python: str, max_windows: int) -> None:
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float16).cpu()  # leave the GPU to the backends
    cfg = EvalConfig(trtllm_python=trtllm_python, vllm_python=vllm_python, max_windows=max_windows)
    val, held = eval_windows(load_texts(_default_sources(), "wikitext2", "test"), tok, cfg.seq_len, max_windows)
    rows = [("every layer 4-bit", plan(lambda i: 4)),
            ("Our method (Stage 0 plan)", plan(lambda i: 8 if i in EIGHT else 4)),
            ("Our method with layer 5 at FP16", plan(lambda i: 16 if i == 5 else 8 if i in EIGHT else 4))]
    print("TRTMIX-BEGIN")
    for label, p in rows:
        res = {}
        for name, mod in (("trtllm", trtllm_backend), ("vllm", vllm_backend)):
            try:
                out = mod.measure(model, tok, p, val, held, cfg, 16, 128)
                res[name] = out[f"{name}_ppl_val"]
                size = out.get("trtllm_engine_gb", out.get("vllm_size_gb"))
                print(f"{label} [{name}]: ppl_val {res[name]:.4f}, held-out {out[f'{name}_ppl_heldout']:.4f}, "
                      f"{size:.3f} GB, decode {out[f'{name}_decode_ms_per_token']:.2f} ms/token")
            except (trtllm_backend.TRTLLMError, vllm_backend.VLLMError) as e:
                print(f"{label} [{name}]: FAILED {e}")
        ok = len(res) == 2 and abs(res["trtllm"] - res["vllm"]) <= 0.01 * res["vllm"]
        print(f"{label}: {'PASS' if ok else 'FAIL'} ({res}: TensorRT-LLM must match vLLM within 1%)")
    print("TRTMIX-END")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40)
