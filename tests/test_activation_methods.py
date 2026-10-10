import copy

import pytest
import torch

from sdf.stage0.activation import uniform_activation_plan
from sdf.stages.activation_methods import ACTIVATION_METHODS, _cayley_step, _orthogonal, spinquant
from sdf.stages.methods import _rtn_activations, MethodCall
from types import SimpleNamespace

GS = 8
NAMES = list(ACTIVATION_METHODS)


@pytest.fixture
def outlier_model(tiny_llama):
    with torch.no_grad():
        tiny_llama.model.embed_tokens.weight[:, 3] *= 30  # one outlier channel in the residual stream
    return tiny_llama


@pytest.fixture
def batches():
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, 64, (2, 24), generator=g) for _ in range(4)]


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def run(model, name, bits, batches):
    m = copy.deepcopy(model)
    ctx = ACTIVATION_METHODS[name][0](m, uniform_activation_plan(4, bits), batches, GS)
    ref = logits(model, batches[0])
    off = logits(m, batches[0])
    with ctx:
        on = logits(m, batches[0])
    return (off - ref).abs().max().item(), (on - ref).pow(2).mean().item()


def rtn_mse(model, bits, batches):
    call = MethodCall(copy.deepcopy(model), uniform_activation_plan(4, bits), {},
                      SimpleNamespace(stage0=SimpleNamespace(act_group_size=GS, baseline_bits=16)), lambda: batches)
    ref = logits(model, batches[0])
    with _rtn_activations(call):
        return (logits(call.model, batches[0]) - ref).pow(2).mean().item()


@pytest.mark.parametrize("name", NAMES)
def test_transform_is_equivalent_and_quantization_hurts_more_at_4_bits(outlier_model, batches, name):
    drift, mse8 = run(outlier_model, name, 8, batches)
    _, mse4 = run(outlier_model, name, 4, batches)
    assert drift < 1e-4
    assert mse4 >= mse8 and mse8 < 0.5
    print(f"\n{name}: 8-bit {mse8:.2e}  4-bit {mse4:.2e}  (rtn_act {rtn_mse(outlier_model, 8, batches):.2e} / "
          f"{rtn_mse(outlier_model, 4, batches):.2e})")


@pytest.mark.parametrize("name", ["smoothquant", "quarot", "spinquant"])
def test_method_beats_plain_rounding_at_4_bits_with_outliers(outlier_model, batches, name):
    assert run(outlier_model, name, 4, batches)[1] < rtn_mse(outlier_model, 4, batches)


def test_layers_at_16_bits_are_not_quantized(outlier_model, batches):
    for name in NAMES:
        _, mse = run(outlier_model, name, 16, batches)
        assert mse < 1e-8


def test_orthogonal_matrices_and_cayley_step_stay_orthogonal():
    gen = torch.Generator().manual_seed(0)
    for n in (16, 24):
        q = _orthogonal(n, gen)
        assert torch.allclose(q @ q.T, torch.eye(n), atol=1e-5)
    r = _cayley_step(q, torch.randn(24, 24), 0.1)
    assert torch.allclose(r @ r.T, torch.eye(24), atol=1e-4) and not torch.allclose(r, q)


def test_spinquant_learning_lowers_the_proxy_loss(outlier_model, batches):
    losses = []
    for steps in (0, 30):
        m = copy.deepcopy(outlier_model)
        ctx = spinquant(m, uniform_activation_plan(4, 4), batches, GS, steps=steps)
        with ctx:
            losses.append((logits(m, batches[0]) - logits(outlier_model, batches[0])).pow(2).mean().item())
    assert losses[1] <= losses[0]


def test_orthogonal_is_built_on_the_requested_device_both_ways():
    # the power-of-two (Hadamard) path once stayed on the CPU, so SpinQuant's x @ r failed on a GPU
    import torch
    from sdf.stages.activation_methods import _orthogonal
    for n in (8, 6):
        assert _orthogonal(n, torch.Generator().manual_seed(0), "meta").device.type == "meta"
