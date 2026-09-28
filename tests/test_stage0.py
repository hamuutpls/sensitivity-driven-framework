import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sdf.config import CalibrationConfig, CandidateConfig
from sdf.calibration import make_batches
from sdf.stage0 import (
    COMPRESSED_BITS,
    PROTECTED_BITS,
    CompressionPlan,
    SensitivityProfile,
    normalize,
    plan_compression,
    profile_sensitivity,
    run_stage0,
)


def tiny_llama(num_layers=4):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=num_layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return LlamaForCausalLM(cfg)


def test_normalize():
    assert normalize([2.0, 4.0, 3.0]) == [0.0, 1.0, 0.5]
    assert normalize([5.0, 5.0]) == [0.0, 0.0]


def test_plan_protects_layers_at_or_above_threshold():
    profile = SensitivityProfile(scores=[0.0, 0.3, 0.5, 1.0], raw_scores=[1, 2, 3, 4])
    plan = plan_compression(profile, sensitivity_threshold=0.5, pruning_ratio=0.25)

    assert plan.protected_layers == [2, 3]
    assert plan.compressed_layers == [0, 1]
    for lp in plan.layers:
        if lp.protected:
            assert (lp.bit_width, lp.pruning_ratio) == (PROTECTED_BITS, 0.0)
        else:
            assert (lp.bit_width, lp.pruning_ratio) == (COMPRESSED_BITS, 0.25)
    assert CompressionPlan.from_dict(plan.to_dict()) == plan


def test_run_stage0_report():
    profile = SensitivityProfile(scores=[0.0, 0.3, 0.5, 1.0], raw_scores=[1, 2, 3, 4], cost={"calibration_batches": 2})
    _, report = run_stage0(profile, CandidateConfig(sensitivity_threshold=0.4, pruning_ratio=0.3))

    assert report["protected"]["count"] == 2
    assert report["compressed"]["percent"] == 50.0
    assert report["bit_width_distribution"] == {"4": 2, "8": 2}
    assert report["pruning_ratio_distribution"] == {"0.0": 2, "0.3": 2}
    assert report["profiling_cost"] == {"calibration_batches": 2}


def test_plan_rejects_bad_pruning_ratio():
    profile = SensitivityProfile(scores=[0.0, 1.0], raw_scores=[0, 1])
    with pytest.raises(ValueError):
        plan_compression(profile, 0.5, 1.0)


def test_profile_sensitivity_on_tiny_llama(tmp_path):
    model = tiny_llama(num_layers=4)
    frozen = model.model.embed_tokens.weight
    frozen.requires_grad_(False)
    batches = [torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(i)) for i in range(3)]

    profile = profile_sensitivity(model, batches, device="cpu", meta={"model": "tiny"})

    assert profile.num_layers == 4
    assert min(profile.scores) == 0.0 and max(profile.scores) == 1.0
    assert all(r > 0 for r in profile.raw_scores)
    assert profile.cost["calibration_batches"] == 3
    assert profile.cost["calibration_tokens"] == 3 * 2 * 16
    # gradients cleared and requires_grad flags restored
    assert all(p.grad is None for p in model.parameters())
    assert frozen.requires_grad is False
    assert model.model.layers[0].self_attn.q_proj.weight.requires_grad is True

    profile.save(tmp_path / "profile.json")
    assert SensitivityProfile.load(tmp_path / "profile.json") == profile


class _CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 64 for c in text]}


def test_make_batches_shapes_and_determinism():
    cfg = CalibrationConfig(n_batches=3, batch_size=2, seq_len=8, seed=1)
    texts = ["hello world " * 20, "another document " * 20]
    a = make_batches(texts, _CharTokenizer(), cfg)
    b = make_batches(texts, _CharTokenizer(), cfg)
    assert len(a) == 3 and all(t.shape == (2, 8) for t in a)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
