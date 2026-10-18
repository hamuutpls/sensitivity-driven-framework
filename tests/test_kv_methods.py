"""Stage 3 KV-cache techniques (stages/kv_methods.py): QuaRot KV, KVQuant, H2O, SnapKV, InfiniGen, and the runner
rules for them (standard rows, the whole Stage 0 plan, backends run only the uncompressed row; Mohammad, 2026-10-18)."""

import copy
import json
import math

import pytest
import torch
from transformers.models.llama.modeling_llama import eager_attention_forward

from sdf.eval import llamacpp, trtllm_backend, vllm_backend
from sdf.run import start_run
from sdf.search_space import SEARCH_SPACE
from sdf.stage0.kv_cache import KVLayerPlan, KVPlan, kv_projections, uniform_kv_plan
from sdf.stage0.sensitivity import find_decoder_layers, round_to_nearest
from sdf.stages.kv_methods import (LayerKV, SDF_KV, budgets, calibrate_kvquant, group_mean, h2o_keep,
                                   infinigen_keep, kmeans_1d, kv_attention, kv_attention_forward, kvquant_hooks,
                                   kvquant_round, quarot_round, repeat_kv, snap_to_levels, snapkv_keep)
from sdf.stages.methods import METHODS, MethodCall
from sdf.stages.runner import Stage0Plans, original_plan, run_stage
from conftest import fake_texts
from test_stages import stage0_dir  # noqa: F401 - the Stage 0 folder fixture

KV_METHODS = ("rtn_kv", "quarot_kv", "kvquant", "h2o", "snapkv", "infinigen")


def _scores(b=1, hq=4, n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(b, hq, n, n, generator=g)
    return s.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), -math.inf)


def test_quarot_round_beats_plain_rounding_on_an_outlier_channel():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 2, 64, 16, generator=g)
    x[..., 3] *= 30  # one outlier channel, as keys have
    plain = round_to_nearest(x, 4, 16)
    assert (quarot_round(x, 4) - x).pow(2).mean() < (plain - x).pow(2).mean()
    assert (quarot_round(x, 8) - x).abs().max() < (quarot_round(x, 4) - x).abs().max()


def test_group_mean_and_repeat_kv_shapes():
    x = torch.arange(8.0).view(1, 4, 2)
    assert group_mean(x, 2).tolist() == [[[1.0, 2.0], [5.0, 6.0]]]
    assert repeat_kv(torch.zeros(1, 2, 3, 5), 2).shape == (1, 4, 3, 5)


def test_budgets_are_a_share_of_the_words_each_word_can_see():
    assert budgets(0.5, 6).tolist() == [1, 1, 2, 2, 3, 3]
    assert budgets(0.7, 10).tolist()[-1] == 7  # not 8 from float rounding
    assert budgets(0.2, 512).tolist()[-1] == 103


def _check_budgets(keep, r):
    n = keep.shape[-1]
    assert not (keep & ~torch.ones(n, n, dtype=torch.bool).tril()).any()  # never a future word
    assert keep[..., torch.arange(n), torch.arange(n)].all()  # every word sees itself
    assert torch.equal(keep.sum(-1), budgets(r, n).expand_as(keep.sum(-1)))


@pytest.mark.parametrize("r", [0.3, 0.5])
def test_h2o_keeps_its_share_with_the_recent_half(r):
    keep = h2o_keep(_scores(n=16), groups=2, keep_ratio=r)
    assert keep.shape == (1, 2, 16, 16)
    _check_budgets(keep, r)
    for i, b in enumerate(budgets(r, 16).tolist()):
        assert keep[..., i, i - max(b // 2, 1) + 1:i + 1].all()  # the recent half is always kept


def test_h2o_keeps_a_heavy_hitter():
    s = _scores(n=16)
    s[..., 2] = torch.where(torch.isinf(s[..., 2]), s[..., 2], torch.full_like(s[..., 2], 20.0))  # all attend to 2
    keep = h2o_keep(s, groups=2, keep_ratio=0.3)
    assert keep[..., 3:, 2].all()
    assert torch.equal(h2o_keep(s, 2, 1.0), torch.ones(16, 16, dtype=torch.bool).tril().expand(1, 2, 16, 16))


def test_snapkv_keeps_the_window_and_the_most_voted_word():
    s = _scores(n=20)
    s[..., 5] = torch.where(torch.isinf(s[..., 5]), s[..., 5], torch.full_like(s[..., 5], 20.0))
    keep = snapkv_keep(s, groups=2, keep_ratio=0.5, window=3, kernel=1)
    _check_budgets(keep, 0.5)
    for i in range(6, 20):
        assert keep[..., i, i - 2:i + 1].all()  # observation window
        assert keep[..., i, 5].all()  # the word every query looks at


def test_snapkv_window_votes_do_not_spill_onto_earlier_words():
    s = torch.full((1, 2, 30, 30), -5.0).masked_fill(~torch.ones(30, 30, dtype=torch.bool).tril(), -math.inf)
    s[..., 0] = torch.where(torch.isinf(s[..., 0]), s[..., 0], torch.zeros_like(s[..., 0]))  # word 0 mildly liked
    idx = torch.arange(30)
    s[..., idx, idx] = 10.0  # every query mostly looks at itself (inside the window)
    keep = snapkv_keep(s, groups=2, keep_ratio=0.2, window=4, kernel=7)  # word 25: 6 kept, 4 in the window
    assert keep[..., 25, 0].all()  # not a word next to the window, whose pooled vote would come from the window


def test_infinigen_fetches_the_words_with_the_highest_guess():
    g = torch.Generator().manual_seed(1)
    q, k = torch.randn(1, 4, 16, 8, generator=g).abs() + 1, torch.randn(1, 2, 16, 8, generator=g)
    k[:, :, 4] = 10.0  # word 4 matches every (positive) query, whichever channels are used
    keep = infinigen_keep(q, k, groups=2, keep_ratio=0.3)
    _check_budgets(keep, 0.3)
    rows = [i for i, b in enumerate(budgets(0.3, 16).tolist()) if b >= 2 and i > 4]
    assert keep[..., rows, 4].all()


def test_infinigen_is_causal():
    g = torch.Generator().manual_seed(3)
    q, k = torch.randn(1, 4, 24, 8, generator=g), torch.randn(1, 2, 24, 8, generator=g)
    q2, k2 = q.clone(), k.clone()
    q2[:, :, 20:] *= 50  # later words change, earlier rows must not
    k2[:, :, 20:] *= 50
    assert torch.equal(infinigen_keep(q, k, 2, 0.3)[..., :20, :], infinigen_keep(q2, k2, 2, 0.3)[..., :20, :])


def test_predict_kv_has_no_prediction_for_an_unmeasured_share(stage0_dir):
    from sdf.stage0.kv_cache import predict_kv
    prof = Stage0Plans.load(stage0_dir).kv_profile
    cost = predict_kv(uniform_kv_plan(prof.num_layers, 16, 0.37), prof, 64, 1, 8, 32, 16)
    assert math.isnan(cost.attention_kept) and cost.memory_gb > 0
    cost = predict_kv(uniform_kv_plan(prof.num_layers, 3, 1.0), prof, 64, 1, 8, 32, 16)  # 3 bits never measured
    assert math.isnan(cost.ppl_rise) and cost.avg_bits == 3


class _Attn(torch.nn.Module):
    num_key_value_groups = 2
    training = False

    def __init__(self, spec):
        super().__init__()
        self.sdf_kv = spec


def test_kv_attention_forward_matches_eager_and_masks_forgotten_words():
    g = torch.Generator().manual_seed(2)
    q, k, v = (torch.randn(1, h, 10, 8, generator=g) for h in (4, 2, 2))
    mask = torch.zeros(1, 1, 10, 10).masked_fill(~torch.ones(10, 10, dtype=torch.bool).tril(), -1e9)
    ref, _ = eager_attention_forward(_Attn(LayerKV()), q, k, v, mask, 0.3)
    out, _ = kv_attention_forward(_Attn(LayerKV()), q, k, v, mask, 0.3)
    assert torch.allclose(out, ref, atol=1e-6)
    out, probs = kv_attention_forward(_Attn(LayerKV(forget="h2o", keep_ratio=0.4)), q, k, v, mask, 0.3)
    assert torch.allclose(out[:, :1], ref[:, :1], atol=1e-6)  # the first word has nothing to forget
    assert torch.equal((probs > 0).sum(-1), budgets(0.4, 10).expand(1, 4, 10))
    out, _ = kv_attention_forward(_Attn(LayerKV(rotate_key_bits=4, rotate_value_bits=4)), q, k, v, mask, 0.3)
    assert not torch.allclose(out, ref) and (out - ref).abs().max() < 0.5


def test_kv_attention_switches_and_restores_the_model_attention(tiny_llama):
    ids = torch.randint(0, 64, (1, 16))
    ref = tiny_llama(input_ids=ids).logits
    old = tiny_llama.config._attn_implementation
    with kv_attention(tiny_llama, [LayerKV()] * 4):
        assert tiny_llama.config._attn_implementation == SDF_KV
        assert torch.allclose(tiny_llama(input_ids=ids).logits, ref, atol=1e-5)
    with kv_attention(tiny_llama, [LayerKV(forget="snapkv", keep_ratio=0.5)] * 4):
        out = tiny_llama(input_ids=ids).logits
    assert torch.allclose(out[:, :1], ref[:, :1], atol=1e-5) and not torch.allclose(out[:, 1:], ref[:, 1:])
    assert tiny_llama.config._attn_implementation == old
    assert not any(hasattr(layer.self_attn, "sdf_kv") for layer in find_decoder_layers(tiny_llama))
    with pytest.raises(ValueError):
        with kv_attention(tiny_llama, [LayerKV()] * 3):
            pass


def test_kmeans_and_snap_to_levels():
    x = torch.cat([torch.full((50,), -1.0), torch.full((50,), 0.5)]) + torch.linspace(-0.01, 0.01, 100)
    levels = kmeans_1d(x, 2)
    assert torch.allclose(levels, torch.tensor([-1.0, 0.5]), atol=0.02)
    assert snap_to_levels(torch.tensor([-0.9, 0.1, 0.3, 2.0]), levels).tolist() == pytest.approx([-1, 0.5, 0.5, 0.5],
                                                                                                abs=0.02)


def test_kvquant_round_keeps_outliers_and_the_first_word():
    x = torch.randn(1, 6, 4)
    x[0, 3, 1] = 100.0
    lo, hi = torch.full((4,), -2.0), torch.full((4,), 2.0)
    levels = torch.tensor([-1.0, 0.0, 1.0])
    out = kvquant_round(x, lo, hi, levels)
    assert out[0, 3, 1] == 100.0 and torch.equal(out[:, 0], x[:, 0])
    inside = (x[:, 1:].abs() <= 2)
    assert set(out[:, 1:][inside].tolist()) <= {-2.0, 0.0, 2.0}


def test_kvquant_calibrates_and_hooks_every_layer(tiny_llama):
    batches = [torch.randint(0, 64, (2, 16)) for _ in range(2)]
    names = ("k_proj", "v_proj")
    layers = calibrate_kvquant(tiny_llama, batches, [2, 4, 16, 4], [4, 4, 4, 16], names, 16)
    assert len(layers) == 4
    assert all((c.key_lo < c.key_hi).all() for c in layers)
    assert layers[0].key_levels[2].numel() == 4 and layers[1].value_levels[4].numel() == 16
    assert layers[2].key_levels == {} and layers[3].value_levels == {}
    ids = torch.randint(0, 64, (1, 16))
    ref = tiny_llama(input_ids=ids).logits
    k0, _ = kv_projections(find_decoder_layers(tiny_llama)[0], names)
    with kvquant_hooks(tiny_llama, layers, [2, 4, 16, 4], [4, 4, 4, 16], names, 16):
        assert not torch.allclose(tiny_llama(input_ids=ids).logits, ref)
        assert len(k0._forward_hooks) == 1
    assert not k0._forward_hooks and torch.allclose(tiny_llama(input_ids=ids).logits, ref)


def test_standard_rows_paper_defaults(stage0_dir, small_cfg):
    plans = Stage0Plans.load(stage0_dir)
    s0 = small_cfg.stage0
    assert uniform_kv_plan(2, 16, 0.2).layers[1] == KVLayerPlan(1, 16, 16, 0.2)
    for name in ("h2o", "snapkv", "infinigen"):
        assert {(lp.key_bits, lp.keep_ratio) for lp in original_plan(METHODS[name], plans, s0, 0.3).layers} == \
            {(16, s0.kv_standard_keep_ratio)}
    for name in ("rtn_kv", "quarot_kv", "kvquant"):
        assert {(lp.key_bits, lp.keep_ratio) for lp in original_plan(METHODS[name], plans, s0, 0.3).layers} == \
            {(s0.kv_uniform_bits, 1.0)}
    assert all(METHODS[m].plans == ("kv",) for m in KV_METHODS)  # every technique follows the whole plan


def test_quarot_standard_row_uses_quarot_k_bits(stage0_dir, tiny_llama, small_cfg, monkeypatch):
    plans, s0 = Stage0Plans.load(stage0_dir), small_cfg.stage0
    orig = original_plan(METHODS["quarot_kv"], plans, s0, 0.3, quarot_k_bits=2)
    assert {(lp.key_bits, lp.value_bits, lp.keep_ratio) for lp in orig.layers} == {(2, s0.kv_uniform_bits, 1.0)}
    assert original_plan(METHODS["kvquant"], plans, s0, 0.3, quarot_k_bits=2).layers[0].key_bits == s0.kv_uniform_bits
    seen = []
    import sdf.stages.methods as mm
    monkeypatch.setattr(mm, "kv_attention", lambda model, specs: seen.append(specs) or kv_attention(model, specs))
    cand = SEARCH_SPACE.make({"calib_samples": 16})
    with METHODS["quarot_kv"].apply(MethodCall(tiny_llama, orig, cand, small_cfg, lambda: [])):
        pass
    assert {(s.rotate_key_bits, s.rotate_value_bits, s.forget) for s in seen[0]} == {(2, 4, None)}
    framework = KVPlan(tuple(KVLayerPlan(i, 8, 2, 0.5 if i else 1.0) for i in range(4)), "sensitivity")
    with METHODS["quarot_kv"].apply(MethodCall(tiny_llama, framework, cand, small_cfg, lambda: [])):
        pass
    assert seen[1][0] == LayerKV(8, 2, None, 1.0) and seen[1][1] == LayerKV(8, 2, "h2o", 0.5)


def test_stage3_backends_run_only_the_uncompressed_row(stage0_dir, tiny_llama, tokenizer, small_cfg, monkeypatch):
    calls = []

    def fake(name, keys):
        def measure(model, tokenizer, plan, *rest, **kw):
            calls.append((name, plan))
            return {k: 1.0 for k in keys}
        return measure

    monkeypatch.setattr(llamacpp, "measure", fake("llamacpp", llamacpp.MAIN_METRICS))
    monkeypatch.setattr(vllm_backend, "measure", fake("vllm", vllm_backend.MAIN_METRICS))
    monkeypatch.setattr(trtllm_backend, "measure", fake("trtllm", trtllm_backend.MAIN_METRICS))
    cfg = small_cfg.with_overrides({"eval.llamacpp_dir": "lc", "eval.llamacpp_convert": "conv.py",
                                    "eval.vllm_python": "vp", "eval.trtllm_python": "tp"})
    out = run_stage(start_run(cfg), 3, ["h2o", "rtn_kv"], SEARCH_SPACE.make({"calib_samples": 16}), stage0_dir,
                    model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    rows = {f"{r['method']}/{r['variant']}": r for r in json.loads(out["json"].read_text())["rows"]}
    assert sorted(n for n, _ in calls) == ["llamacpp", "trtllm", "vllm"]  # the uncompressed row only
    assert rows["baseline/fp16"]["metrics"]["llamacpp_ppl_val"] == 1.0
    for k, r in rows.items():
        if k != "baseline/fp16":
            assert r["status"] == "ok", (k, r.get("error"))
            for name in ("llamacpp", "vllm", "trtllm"):
                assert r["info"][f"{name}_error"].startswith("not run: this backend cannot forget words")


def test_backend_reasons_follow_this_runs_backends_not_the_cache(stage0_dir, tiny_llama, tokenizer, small_cfg,
                                                                 monkeypatch):
    monkeypatch.setattr(llamacpp, "measure", lambda *a, **k: {m: 1.0 for m in llamacpp.MAIN_METRICS})
    kwargs = dict(model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    cand = SEARCH_SPACE.make({"calib_samples": 16})
    ctx = start_run(small_cfg)
    run_stage(ctx, 3, ["rtn_kv"], cand, stage0_dir, **kwargs)  # caches the standard row without backends
    cfg = small_cfg.with_overrides({"eval.llamacpp_dir": "lc", "eval.llamacpp_convert": "conv.py"})
    out = run_stage(start_run(cfg), 3, ["rtn_kv"], cand, stage0_dir, **kwargs)
    rows = {f"{r['method']}/{r['variant']}": r for r in json.loads(out["json"].read_text())["rows"]}
    assert rows["rtn_kv/original"]["info"]["cached"]
    assert rows["rtn_kv/original"]["info"]["llamacpp_error"].startswith("not run")
    assert "vllm_error" not in rows["rtn_kv/original"]["info"]
