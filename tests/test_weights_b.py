import copy
from dataclasses import replace

import pytest
import torch

from sdf.stage0.planner import uniform_plan
from sdf.stages import weights_b as wb
from sdf.stages.weights import gptq_

METHODS = {k: v[0] for k, v in wb.WEIGHT_METHODS_B.items()}
BATCHES = [torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(i)) for i in range(4)]


def mixed_plan(bits=(4, 4, 8, 4)):
    plan = uniform_plan([0.0] * 4, 4, 0.0)
    return replace(plan, layers=tuple(replace(lp, bit_width=b) for lp, b in zip(plan.layers, bits)))


def run(fn, model, plan, baseline_bits=16):
    model = copy.deepcopy(model)
    fn(model, plan, BATCHES, 8, baseline_bits)
    return model


def linears(model):
    return {n: p for n, p in model.named_parameters() if "layers" in n and p.ndim == 2}


def logits_mse(model, ref):
    with torch.no_grad():
        return sum((model(input_ids=b).logits - ref(input_ids=b).logits).pow(2).mean().item() for b in BATCHES)


@pytest.mark.parametrize("name", METHODS)
def test_only_layers_below_baseline_change_and_weights_stay_finite(tiny_llama, name):
    out = run(METHODS[name], tiny_llama, mixed_plan(), baseline_bits=8)
    ref = tiny_llama.state_dict()
    for n, p in out.named_parameters():
        layer8 = ".layers.2." in n
        changed = not torch.equal(p, ref[n])
        assert torch.isfinite(p).all()
        assert changed == (n in linears(out) and not layer8), n


@pytest.mark.parametrize("name", METHODS)
def test_bits_only_plan_required(tiny_llama, name):
    with pytest.raises(ValueError, match="removes nothing"):
        METHODS[name](copy.deepcopy(tiny_llama), uniform_plan([0.0] * 4, 4, 0.5), BATCHES, 8, 16)


def test_registry_shape():
    assert wb.BILLM_FIXED_BITS and set(wb.WEIGHT_METHODS_B) == {"aqlm", "quip", "quipsharp", "pbllm", "billm"}
    assert [v[4] for v in wb.WEIGHT_METHODS_B.values()] == [True, True, True, True, False]


def test_aqlm_rows_use_one_codebook_per_book_and_more_books_fit_better():
    torch.manual_seed(0)
    W = torch.randn(64, 64)
    scale = W.norm(dim=1, keepdim=True)
    errs = []
    for books in (1, 2):
        Q = W.clone()
        wb.aqlm_quantize_(Q, books)
        errs.append((Q - W).pow(2).sum().item())
        if books == 1:  # 512 vectors of 8 weights take at most 2**8 distinct values
            assert len(torch.unique((Q / scale).reshape(-1, 8).round(decimals=4), dim=0)) <= 256
    assert errs[1] < errs[0]


def test_e8p_codebook_and_residual_stages():
    C = wb._e8p("cpu")
    assert len(torch.unique(C, dim=0)) == 2 ** 16
    assert torch.isin(torch.cat([C - .25, C + .25]).sum(1) % 2, torch.tensor([0.0, 1.0])).all()  # integer-sum shifts of E8
    assert len(wb._e8_one_bit("cpu")) == 256
    V = torch.randn(512, 8)
    errs = [(wb._E8Residual(V, b)(V) - V).pow(2).mean().item() for b in (2, 3, 4)]
    assert errs[0] > errs[1] > errs[2]
    q = wb._E8Residual(V, 2)
    ((_, _, s),) = q.stages
    x = q(V) / s  # every output block is s times one codeword
    assert (x - C[wb._nearest(x, C, C.pow(2).sum(1) / 2)]).abs().max() < 1e-4


def test_pbllm_binarizes_the_planned_share_with_two_values_per_row():
    torch.manual_seed(0)
    W, X = torch.randn(32, 64), torch.randn(256, 64)
    for bits in (2, 4, 6):
        Q = W.clone()
        mask = wb.pbllm_quantize_(Q, 2 * X.T @ X / 256, bits)
        assert mask.float().mean().item() == pytest.approx(1 - (bits - 1) / 7, abs=1 / W.numel())
        assert all(len(torch.unique(row[m].round(decimals=5))) <= 2 for row, m in zip(Q, mask))


def test_billm_ignores_plan_bits_and_uses_few_values_per_row(tiny_llama):
    four = run(wb.billm_, tiny_llama, uniform_plan([0.0] * 4, 4, 0.0))
    eight = run(wb.billm_, tiny_llama, uniform_plan([0.0] * 4, 8, 0.0))
    for (n, a), b in zip(four.named_parameters(), eight.parameters()):
        assert torch.equal(a, b), n
    for n, p in linears(four).items():  # <= 2 groups x 2 values + 2 residual orders x 2 values, per 128-column block
        assert all(len(torch.unique(row)) <= 8 for row in p), n


def test_quality_at_4_bits_on_tiny_model(tiny_llama):
    plan = uniform_plan([0.0] * 4, 4, 0.0)
    mse = {k: logits_mse(run(METHODS[k], tiny_llama, plan), tiny_llama) for k in ("quip", "pbllm", "billm")}
    mse["gptq"] = logits_mse(run(gptq_, tiny_llama, plan), tiny_llama)
    assert all(v > 0 for v in mse.values())
    assert mse["quip"] < 3 * mse["gptq"]
    assert mse["billm"] > mse["quip"]  # about 1 bit is far worse than 4
