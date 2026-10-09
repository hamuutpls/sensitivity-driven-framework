import copy

import pytest
import torch

from sdf.stage0.planner import uniform_plan
from sdf.stages.bnb import WEIGHT_METHODS_BNB, _int8_torch, _nf4_torch, bnb_


def run(model, bits):
    m = copy.deepcopy(model)
    bnb_(m, uniform_plan([1.0] * 4, bits, 0.0), [], 8, 16)
    ids = torch.arange(20).reshape(1, 20) % 64
    with torch.no_grad():
        return (m(input_ids=ids).logits - model(input_ids=ids).logits).pow(2).mean().item(), m


def test_nf4_and_int8_error_ordering(tiny_llama):
    e8, m8 = run(tiny_llama, 8)
    e4, m4 = run(tiny_llama, 4)
    print(f"\nbnb logit MSE: int8 {e8:.2e}  nf4 {e4:.2e}")
    assert 0 < e8 < e4 < 0.5
    w = m4.model.layers[0].mlp.up_proj.weight
    assert len(w.reshape(-1, 64)[0].unique()) <= 16


def test_16_bit_layers_are_left_alone(tiny_llama):
    e, _ = run(tiny_llama, 16)
    assert e == 0


def test_other_bit_widths_raise(tiny_llama):
    with pytest.raises(ValueError, match="only 4-bit and 8-bit"):
        run(tiny_llama, 3)


def test_pruning_plan_is_refused(tiny_llama):
    with pytest.raises(ValueError):
        bnb_(copy.deepcopy(tiny_llama), uniform_plan([1.0] * 4, 4, 0.5), [], 8, 16)


def test_torch_fallback_matches_bitsandbytes_on_cpu_when_it_can_run():
    F = pytest.importorskip("bitsandbytes.functional")
    w = torch.randn(32, 128)
    try:
        packed, st = F.quantize_4bit(w, blocksize=64, quant_type="nf4")
    except RuntimeError:
        pytest.skip("bitsandbytes 4-bit needs a GPU here")
    assert torch.allclose(F.dequantize_4bit(packed, st), _nf4_torch(w), atol=1e-5)
    q, s, _ = F.int8_vectorwise_quant(w.half())
    assert (F.int8_vectorwise_dequant(q, s) - _int8_torch(w.half().float())).abs().mean() < 1e-3  # ties may differ


def test_registry():
    assert WEIGHT_METHODS_BNB["bitsandbytes"][0] is bnb_
