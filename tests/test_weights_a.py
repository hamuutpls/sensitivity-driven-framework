import copy

import pytest
import torch

from dataclasses import replace

from sdf.stage0.planner import uniform_plan
from sdf.stage0.sensitivity import find_decoder_layers, round_to_nearest
from sdf.stages.weights import gptq_
from sdf.stages.weights_a import WEIGHT_METHODS_A

BITS = [4, 4, 8, 4]
METHODS = {k: v[0] for k, v in WEIGHT_METHODS_A.items()}


@pytest.fixture
def batches():
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, 64, (2, 16), generator=g) for _ in range(4)]


def mixed_plan():
    p = uniform_plan([0.0] * 4, 4, 0.0)
    return replace(p, layers=tuple(replace(lp, bit_width=b) for lp, b in zip(p.layers, BITS)))


def linears(model):
    return [[m.weight.data.clone() for m in layer.modules() if isinstance(m, torch.nn.Linear)]
            for layer in find_decoder_layers(model)]


def logit_mse(model, ref, batches):
    with torch.no_grad():
        return sum((model(input_ids=b).logits - ref(input_ids=b).logits).pow(2).mean().item() for b in batches)


def rtn_(model, plan, batches, gs, baseline_bits):
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        for m in layer.modules():
            if isinstance(m, torch.nn.Linear) and lp.bit_width < baseline_bits:
                m.weight.data.copy_(round_to_nearest(m.weight.data, lp.bit_width, gs, True))


@pytest.mark.parametrize("name", METHODS)
def test_rounds_only_layers_below_baseline_and_grid_size(tiny_llama, batches, name):
    before, model = linears(tiny_llama), copy.deepcopy(tiny_llama)
    METHODS[name](model, mixed_plan(), batches, 16, 8)  # baseline 8: the 8-bit layer is left alone
    after = linears(model)
    assert all(torch.equal(a, b) for a, b in zip(after[2], before[2]))
    for i in (0, 1, 3):
        for a, b in zip(after[i], before[i]):
            assert not torch.equal(a, b)
            if name == "squeezellm":  # codebook of 16 per row plus the few (0.45%) outliers kept as they were
                assert all(r.unique().numel() <= 16 + 4 for r in a)
                continue
            assert (a == b).float().mean() < 0.05  # outliers stay exactly as they were
            for ra, rb in zip(a.reshape(-1, 16), b.reshape(-1, 16)):  # one grid per group of 16
                assert ra[ra != rb].unique().numel() <= 16


@pytest.mark.parametrize("name", METHODS)
def test_output_error_not_much_worse_than_rtn(tiny_llama, batches, name):
    plan, errs = uniform_plan([0.0] * 4, 4, 0.0), {}
    for label, fn in (("rtn", rtn_), ("gptq", gptq_), (name, METHODS[name])):
        model = copy.deepcopy(tiny_llama)
        fn(model, plan, batches, 16, 16)
        errs[label] = logit_mse(model, tiny_llama, batches)
    print(errs)
    assert errs[name] <= 2 * errs["rtn"]


@pytest.mark.parametrize("name", METHODS)
def test_rejects_pruning_plan(tiny_llama, batches, name):
    with pytest.raises(ValueError):
        METHODS[name](tiny_llama, uniform_plan([0.0] * 4, 4, 0.3), batches, 16, 16)
