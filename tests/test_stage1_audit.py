"""Audit checks every Stage 1 weight method must pass: bits come from the plan, nothing is removed or zeroed."""

import copy

import pytest
import torch

from sdf.stage0.planner import LayerPlan, CompressionPlan
from sdf.stage0.sensitivity import find_decoder_layers
from sdf.stages.methods import METHODS

WEIGHT_METHODS = [m.name for m in METHODS.values() if m.stage == 1 and m.apply is not None
                  and m.plans == ("quant",)]


def _plan(bits):
    return CompressionPlan(tuple(LayerPlan(layer=i, bit_width=b, pruning_ratio=0.0, protected=b > 4, sensitivity=0.5)
                                 for i, b in enumerate(bits)), "test", None, 0.0)


def _rel_err(a, b):
    return [((x.weight - y.weight).norm() / y.weight.norm()).item()
            for x, y in zip(_linears(a), _linears(b))]


def _linears(model):
    return [m for layer in find_decoder_layers(model) for m in layer.modules() if isinstance(m, torch.nn.Linear)]


@pytest.mark.parametrize("name", WEIGHT_METHODS)
def test_method_follows_plan_bits_and_removes_nothing(name, tiny_llama, small_cfg):
    from sdf.stages.methods import MethodCall

    m = METHODS[name]
    torch.manual_seed(0)
    batches = [torch.randint(0, 64, (2, 16)) for _ in range(4)]
    model = copy.deepcopy(tiny_llama)
    plan = _plan([4, 8, 4, 16])  # layer 3 is at the baseline: left alone
    cand = {"gptq_groupsize": 16, "calib_samples": 16}
    call = MethodCall(model, plan, cand, small_cfg, lambda: batches)
    with m.apply(call):  # (RTN restores the weights when the context closes)
        errs = _rel_err(model, tiny_llama)
    per_layer = [errs[i * 7:(i + 1) * 7] for i in range(4)]  # 7 Linears per layer
    assert all(e == 0 for e in per_layer[3]), "a layer at the baseline bits must stay untouched"
    for errs in per_layer[:3]:
        # AQLM is exact on this toy model (fewer 8-vectors per layer than the 2**8 codewords of one book)
        assert all((e > 0 or name == "aqlm") and e < 0.9 for e in errs), "changed, but not zeroed or pruned away"
    if not m.fixed_bits and name != "aqlm":  # the 8-bit layer is closer to FP than the 4-bit ones: bits come from the plan
        mean = lambda es: sum(es) / len(es)  # noqa: E731
        assert mean(per_layer[1]) < mean(per_layer[0]) and mean(per_layer[1]) < mean(per_layer[2])
