import pytest
import torch

from sdf.stage0.kv_cache import (KVProfile, _coverage, keep_ratio_for, plan_kv, predict_kv, profile_kv,
                                 uniform_kv_plan)


def toy_kv():
    # layer 0 keys are fragile, everything else barely cares; layer 1 attends to few tokens
    return KVProfile(bits_options=[2, 4, 8], key_dims=[100, 100], value_dims=[100, 100],
                     key_rise=[[5.0, 1.0, 0.0], [0.1, 0.0, 0.0]], value_rise=[[0.2, 0.05, 0.0], [0.1, -0.01, 0.0]],
                     keep_ratios=[0.25, 0.5], coverage=[[0.5, 0.7], [0.96, 0.99]])


def test_kv_plan_spends_bits_on_fragile_keys():
    prof = toy_kv()
    plan = plan_kv(prof, avg_bits=4, coverage_target=0.95)
    assert plan.layers[0].key_bits == 8  # the fragile tensor gets the most bits
    bits = [b for lp in plan.layers for b in (lp.key_bits, lp.value_bits)]
    assert sum(bits) / len(bits) <= 4
    assert [lp.keep_ratio for lp in plan.layers] == [1.0, 0.25]
    assert plan_kv(prof, 4, None).layers[1].keep_ratio == 1.0
    assert keep_ratio_for([0.5, 0.7], [0.25, 0.5], 0.95) == 1.0


def test_predict_kv():
    prof = toy_kv()
    fp16 = predict_kv(uniform_kv_plan(2, 16), prof, 10, 1, 50, 32, 16)
    assert fp16.memory_gb == pytest.approx(10 * 400 * 16 / 8 / 1e9) and fp16.ppl_rise == 0
    uni = predict_kv(uniform_kv_plan(2, 4), prof, 10, 1, 50, 32, 16)
    assert uni.memory_gb == pytest.approx(10 * 400 * (4 + 32 / 50) / 8 / 1e9)
    assert uni.ppl_rise == pytest.approx(1.0 + 0.0 + 0.05 + 0.0)  # negative rise clipped to 0
    fw = predict_kv(plan_kv(prof, 4, 0.95), prof, 10, 1, 50, 32, 16)
    assert fw.ppl_rise < uni.ppl_rise and fw.memory_gb < uni.memory_gb
    assert fw.kept_share == pytest.approx((10 * 200 + 3 * 200) / (10 * 400))  # layer 1 keeps ceil(0.25 * 10) = 3 tokens
    assert fw.attention_kept == pytest.approx((1.0 + 0.96) / 2)


def test_coverage_uniform_attention():
    t = 8
    attn = torch.tril(torch.ones(t, t))
    attn = (attn / attn.sum(-1, keepdim=True)).view(1, 1, t, t)
    cov = _coverage(attn, [0.5, 1.0])
    assert cov[1] == pytest.approx(1.0) and 0.5 <= cov[0] < 0.7  # uniform attention: top half ~ half


def test_profile_kv_on_tiny_llama(tiny_llama):
    batches = [torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(i)) for i in range(2)]
    before = {k: v.clone() for k, v in tiny_llama.state_dict().items()}
    impl = tiny_llama.config._attn_implementation
    prof = profile_kv(tiny_llama, batches, [2, 8], 8, [0.25, 0.5], ("k_proj", "v_proj"), device="cpu")
    assert prof.key_dims == [16] * 4 and prof.value_dims == [16] * 4  # 2 KV heads x head dim 8
    assert len(prof.key_rise) == 4 and all(len(r) == 2 for r in prof.key_rise + prof.value_rise)
    assert any(r[0] != 0 for r in prof.key_rise)
    assert all(0 < c[0] <= c[1] <= 1 + 1e-6 for c in prof.coverage)
    assert tiny_llama.config._attn_implementation == impl
    assert all(torch.equal(v, before[k]) for k, v in tiny_llama.state_dict().items())
    assert KVProfile.from_dict(prof.to_dict()) == prof
    with pytest.raises(ValueError):
        profile_kv(tiny_llama, batches, [2], 8, [0.5], ("qkv", "v_proj"), device="cpu")
