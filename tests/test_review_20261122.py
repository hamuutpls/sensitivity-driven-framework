"""Experiments for the paper review of 22 Nov 2026 (sdf.review) and the Stage 1 option that leaves Stage 1 out."""

import copy
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from sdf import review
from sdf.run import start_run
from sdf.search import evaluate, technique_space
from sdf.search_space import NO_QUANT
from sdf.stage0.planner import plan_compression
from sdf.stage0.sensitivity import find_decoder_layers, layer_shapes
from sdf.stages.methods import METHODS
from conftest import fake_texts


def test_spearman_matches_known_values_and_averages_ties():
    assert review.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert review.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # textbook example: d = (0, -1, 1, 0, 0) -> 1 - 6*2 / (5*24) = 0.9
    assert review.spearman([1, 2, 3, 4, 5], [1, 3, 2, 4, 5]) == pytest.approx(0.9)
    # ties share their average rank: ranks a = (1, 2.5, 2.5, 4), b = (1, 2, 3, 4) -> Pearson of the ranks
    ra, rb = [1, 2.5, 2.5, 4], [1, 2, 3, 4]
    ma, mb = sum(ra) / 4, sum(rb) / 4
    want = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(
        sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    assert review.spearman([1, 5, 5, 9], [1, 2, 3, 4]) == pytest.approx(want)
    assert math.isnan(review.spearman([1, 1, 1], [1, 2, 3]))
    with pytest.raises(ValueError):
        review.spearman([1, 2], [1, 2, 3])
    t = review.spearman_table({"x": [1, 2, 3], "y": [3, 2, 1], "z": [1, 3, 2]})
    assert [(r["a"], r["b"]) for r in t] == [("x", "y"), ("x", "z"), ("y", "z")] and t[0]["layers"] == 3
    assert t[0]["spearman"] == pytest.approx(-1.0)


def _batches(n=2, seq=16):
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, 64, (2, seq), generator=g) for _ in range(n)]


@pytest.mark.parametrize("how", ["magnitude", "wanda"])
def test_prune_layer_prunes_a_quarter_of_each_row_and_restores(tiny_llama, how):
    layer = find_decoder_layers(tiny_llama)[1]
    linears = [m for m in layer.modules() if isinstance(m, nn.Linear)]
    before = [m.weight.data.clone() for m in linears]
    with review.prune_layer(tiny_llama, layer, 0.25, how, _batches()):
        for m in linears:
            zeros = (m.weight.data == 0).sum(dim=1)
            assert torch.all(zeros == round(0.25 * m.in_features))
        others = [m for m in find_decoder_layers(tiny_llama)[0].modules() if isinstance(m, nn.Linear)]
        assert all((m.weight.data == 0).sum() == 0 for m in others)  # only this layer
    assert all(torch.equal(m.weight.data, w) for m, w in zip(linears, before))
    if how == "magnitude":  # the smallest |w| go, whatever their sign
        with review.prune_layer(tiny_llama, layer, 0.25):
            w, orig = linears[0].weight.data, before[0]
            kept_min = orig.abs().masked_fill(w == 0, float("inf")).min(dim=1).values
            dropped_max = orig.abs().masked_fill(w != 0, -1).max(dim=1).values
            assert torch.all(dropped_max <= kept_min)


def test_prune_layer_rejects_bad_settings(tiny_llama):
    layer = find_decoder_layers(tiny_llama)[0]
    with pytest.raises(ValueError, match="unknown pruning"):
        with review.prune_layer(tiny_llama, layer, 0.25, "random"):
            pass
    with pytest.raises(ValueError, match="calibration"):
        with review.prune_layer(tiny_llama, layer, 0.25, "wanda"):
            pass


def test_layer_damage_gives_one_row_per_layer_and_leaves_the_model_as_it_was(tiny_llama):
    state = copy.deepcopy(tiny_llama.state_dict())
    tiny_llama.train()
    texts = {"calib": _batches(2), "val": _batches(1)}
    rows = review.layer_damage(tiny_llama, texts, _batches(2), group_size=8, device="cpu")
    assert [r["layer"] for r in rows] == [0, 1, 2, 3]
    want = {f"{k}_{s}" for k in review.DAMAGE for s in ("calib", "val")} | {"layer", "base_calib", "base_val"}
    assert all(set(r) == want for r in rows)
    assert all(abs(r["w8_calib"]) < abs(r["skip_calib"]) for r in rows)  # 8 bits barely changes a layer
    assert any(r["skip_calib"] != 0 for r in rows)
    assert tiny_llama.training  # mode given back
    assert all(torch.equal(v, state[k]) for k, v in tiny_llama.state_dict().items())  # weights given back
    only = review.layer_damage(tiny_llama, {"val": _batches(1)}, [], kinds=("skip",), device="cpu")
    assert set(only[0]) == {"layer", "base_val", "skip_val"}
    with pytest.raises(ValueError, match="unknown damage"):
        review.layer_damage(tiny_llama, {"val": _batches(1)}, [], kinds=("w2",), device="cpu")


def _template(n=8):
    scores = [0.3, 0.9, 0.1, 0.7, 0.5, 0.2, 0.8, 0.4][:n]
    return plan_compression(scores, 0.5, 0.0, 8, 4)


def test_bit_plans_pick_the_most_least_and_random_layers_at_the_same_size():
    t = _template()
    order = review.ranked(t)
    assert order[:3] == sorted(range(8), key=lambda i: -t.layers[i].sensitivity)[:3]
    plans = review.bit_plans(t, 3, seeds=[0, 1])
    assert set(plans) == {"sensitivity", "inverse", "random_0", "random_1"}
    high = {k: {lp.layer for lp in p.layers if lp.bit_width == 8} for k, p in plans.items()}
    assert high["sensitivity"] == set(order[:3]) and high["inverse"] == set(order[-3:])
    assert all(len(h) == 3 for h in high.values())
    again = review.bit_plans(t, 3, seeds=[0])["random_0"]
    assert high["random_0"] == {lp.layer for lp in again.layers if lp.bit_width == 8}  # same seed, same layers
    numel = [100] * 8
    sizes = {review.average_bits(p, numel) for p in plans.values()}
    assert sizes == {(3 * 8 + 5 * 4) / 8}  # all the same size
    assert all(lp.pruning_ratio == 0 and not lp.guarded for p in plans.values() for lp in p.layers)
    assert all(lp.sensitivity == tl.sensitivity for lp, tl in zip(plans["random_1"].layers, t.layers))
    b = review.bits_plan(t, order[:2], 8, 3, "B")
    assert review.average_bits(b, numel) == pytest.approx((2 * 8 + 6 * 3) / 8) and b.kind == "B"
    with pytest.raises(ValueError):
        review.bit_plans(t, 9)
    with pytest.raises(ValueError, match="not in the plan"):
        review.bits_plan(t, [8], 8, 4, "x")


def test_equal_ppl_size_takes_the_smallest_row_that_reaches_the_target():
    rows = [{"ppl_val": 12.0, "predicted_weight_memory_gb": 0.5}, {"ppl_val": 11.0, "predicted_weight_memory_gb": 0.7},
            {"ppl_val": 10.9, "predicted_weight_memory_gb": 0.9}, {"ppl_val": None, "predicted_weight_memory_gb": 0.1}]
    assert review.equal_ppl_size(rows, 11.0)["predicted_weight_memory_gb"] == 0.7
    assert review.equal_ppl_size(rows, 10.0) is None


def test_ablation_points_leave_out_exactly_the_stages_each_row_drops():
    point = {"stage1_method": "rtn", "stage3_method": "snapkv", "prune_ratio_aggressive": 0.0, "x": 1}
    rows = dict(review.ablation_points(point, 0.082))
    assert list(rows) == ["Baseline", "E1", "E2", "E3", "E4", "E5", "E6"]
    got = {k: (v["stage1_method"], v["prune_ratio_aggressive"], v["stage3_method"]) for k, v in rows.items()}
    assert got == {"Baseline": (NO_QUANT, 0.0, "none"), "E1": ("rtn", 0.0, "none"), "E2": (NO_QUANT, 0.082, "none"),
                   "E3": (NO_QUANT, 0.0, "snapkv"), "E4": ("rtn", 0.082, "none"), "E5": ("rtn", 0.0, "snapkv"),
                   "E6": ("rtn", 0.082, "snapkv")}
    assert all(v["x"] == 1 for v in rows.values()) and point["stage1_method"] == "rtn"  # input untouched


def test_equal_ppl_table_gives_the_smallest_row_of_each_family_per_target():
    rows = [{"method": "rtn", "plan": "u4", "family": "uniform", "ppl_val": 11.0, "predicted_weight_memory_gb": 0.6},
            {"method": "rtn", "plan": "s1", "family": "sens", "ppl_val": 11.5, "predicted_weight_memory_gb": 0.5},
            {"method": "rtn", "plan": "s2", "family": "sens", "ppl_val": 10.9, "predicted_weight_memory_gb": 0.65},
            {"method": "rtn", "plan": "s3", "family": "sens", "ppl_val": 10.8, "predicted_weight_memory_gb": 0.7},
            {"method": "rtn", "plan": "i1", "family": "inv", "ppl_val": 12.0, "predicted_weight_memory_gb": 0.65},
            {"method": "gptq", "plan": "u4", "family": "uniform", "ppl_val": None, "predicted_weight_memory_gb": 0.6}]
    t = {(r["method"], r["target"], r["family"]): r for r in review.equal_ppl_table(rows, ["u4"])}
    assert t[("rtn", "u4", "sens")]["smallest_plan"] == "s2" and t[("rtn", "u4", "sens")]["smallest_gb"] == 0.65
    assert t[("rtn", "u4", "uniform")]["smallest_plan"] == "u4"  # the target reaches itself
    assert t[("rtn", "u4", "inv")]["smallest_plan"] is None
    assert not any(k[0] == "gptq" for k in t)  # a failed row is no target and no candidate


def test_copy_backend_repeats_the_vllm_numbers_of_the_row_with_the_same_vllm_model():
    rows = [{"row": r, "ppl_val": i, **({"vllm_ppl_val": 10 + i} if r in ("Baseline", "E1", "E4") else {})}
            for i, r in enumerate(["Baseline", "E1", "E2", "E3", "E4", "E5", "E6"])]
    out = {r["row"]: r for r in review.copy_backend(rows)}
    assert out["E3"]["vllm_ppl_val"] == 10 and out["E5"]["vllm_ppl_val"] == 11 and out["E6"]["vllm_ppl_val"] == 14
    assert out["E3"]["ppl_val"] == 3 and out["E3"]["vllm_from"] == "Baseline"  # its own Hugging Face numbers kept
    assert "vllm_from" not in out["E1"] and "vllm_ppl_val" not in out["E2"]


def test_plan_row_applies_a_stage1_method_with_the_plan_and_measures_it(tiny_llama, small_cfg):
    from sdf.stages.methods import MethodCall

    t = _template(4)
    plan = review.bits_plan(t, review.ranked(t)[:1], 8, 4, "x")
    before = copy.deepcopy(tiny_llama.state_dict())
    seen = {}

    def measure(model):
        w = find_decoder_layers(model)[0].self_attn.q_proj.weight
        seen["changed"] = not torch.equal(w, before["model.layers.0.self_attn.q_proj.weight"])
        return {"x": 1.0}

    call = lambda m, p: MethodCall(m, p, {"gptq_groupsize": 16, "calib_samples": 16}, small_cfg, lambda: _batches())  # noqa: E731
    assert review.plan_row(METHODS["rtn"], plan, tiny_llama, call, measure) == {"x": 1.0}
    assert seen["changed"]  # measured while the rounding is applied
    with pytest.raises(ValueError, match="Stage 1"):
        review.plan_row(METHODS["unstructured_prune"], plan, tiny_llama, call, measure)


# ------------------------------------------------------------------------------------------ leaving Stage 1 out


@pytest.fixture
def cfg(small_cfg):
    return small_cfg.with_overrides({"requirement.target_memory_gb": 1.0,
                                     "search.objectives": ["ppl_val", "decode_ms_per_token_mean"],
                                     "search.joint_plan": False,
                                     "search.stage1_methods": ["rtn_with_rtn_act", NO_QUANT],
                                     "search.stage2_methods": ["unstructured_prune"],
                                     "search.stage3_methods": ["rtn_kv", "none"]})


def _kw(tiny_llama, tokenizer):
    return dict(model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)


def test_no_quant_is_a_stage1_method_that_needs_the_threshold_plan(cfg):
    assert METHODS[NO_QUANT].stage == 1 and not METHODS[NO_QUANT].simulated
    space = technique_space(cfg)
    assert NO_QUANT in next(p for p in space.params if p.name == "stage1_method").choices
    with pytest.raises(ValueError, match=NO_QUANT):
        technique_space(cfg.with_overrides({"search.joint_plan": True}))


def test_leaving_stage1_out_keeps_16_bit_weights_and_still_prunes(cfg, tiny_llama, tokenizer):
    ctx = start_run(cfg)
    space = technique_space(cfg)
    base = {"calib_samples": 16, "stage3_method": "none", "sensitive_threshold": 0.5}
    fp = evaluate(ctx, space.make({**base, "stage1_method": NO_QUANT, "prune_ratio_aggressive": 0.0}),
                  **_kw(tiny_llama, tokenizer))
    assert "error" not in fp, fp
    numel, _, fixed = layer_shapes(tiny_llama)
    fp16_gb = (sum(numel) + fixed) * 16 / 8 / 1e9
    assert fp["metrics"]["predicted_weight_memory_gb"] == pytest.approx(fp16_gb, rel=0.02)
    q = evaluate(ctx, space.make({**base, "stage1_method": "rtn_with_rtn_act", "prune_ratio_aggressive": 0.0}),
                 **_kw(tiny_llama, tokenizer))
    assert q["metrics"]["predicted_weight_memory_gb"] < fp["metrics"]["predicted_weight_memory_gb"]
    pr = evaluate(ctx, space.make({**base, "stage1_method": NO_QUANT, "prune_ratio_aggressive": 0.4}),
                  **_kw(tiny_llama, tokenizer))
    assert "error" not in pr, pr
    assert pr["metrics"]["decoder_weights_removed"] > 0
    assert pr["metrics"]["predicted_weight_memory_gb"] < fp["metrics"]["predicted_weight_memory_gb"]
    assert fp["metrics"]["avg_bits_per_weight"] == 16.0
    # no rounding: with nothing pruned and no KV method, the model scores exactly the unchanged model's perplexity
    rows = json.loads((Path(fp["dir"]) / "stage_3" / "results.json").read_text())["rows"]
    ppl = {x["variant"]: x["metrics"]["ppl_val"] for x in rows}
    assert ppl["framework"] == ppl["fp16"] == fp["metrics"]["ppl_val"]
    assert q["metrics"]["ppl_val"] != ppl["fp16"]  # the quantized row does change the model


def test_the_three_run_drivers_write_their_tables(small_cfg, tiny_llama, tokenizer, monkeypatch):
    """scripts/review_20261122.py end to end on the toy model: the wiring between the runs and their CSV columns."""
    import csv
    import importlib.util

    spec = importlib.util.spec_from_file_location("review_20261122", Path(__file__).parents[1] / "scripts" /
                                                  "review_20261122.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    monkeypatch.setattr(R, "_cfg", lambda run, run_id, **over: small_cfg.with_overrides(
        {"run.run_id": run_id or R.RUN_IDS[run], **over}))
    monkeypatch.setattr(R, "METHODS_RUN2", ["rtn"])
    monkeypatch.setattr(R, "TASKS", [])  # no lm-eval in the tests
    monkeypatch.setattr(R, "FINALS", {"toy": {"sensitive_threshold": 0.5, "gptq_groupsize": 32, "calib_dataset": "c4",
                                              "calib_samples": 16, "stage1_method": "rtn",
                                              "stage2_method": "unstructured_prune", "stage3_method": "rtn_kv"}})
    inject = dict(model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    read = lambda p: list(csv.DictReader(p.open(encoding="utf-8")))  # noqa: E731

    out1 = R.run1(inject=inject)
    assert len(read(out1 / "layer_damage.csv")) == 4
    assert {r["text"] for r in read(out1 / "spearman.csv")} == {"calib", "val"}
    out2 = R.run2(inject=inject, run1_csv=str(out1 / "layer_damage.csv"))
    rows = read(out2 / "equal_size.csv")
    assert not [r for r in rows if r.get("error")], rows
    assert {r["part"] for r in rows} >= {"A", "B"} and "B_uniform_4" in {r["plan"] for r in rows}
    assert {r["family"] for r in rows} >= {"sensitivity", "inverse", "random", "uniform_4", "top_k_8_rest_3"}
    assert read(out2 / "equal_ppl.csv") and (out2 / "summary.md").exists()
    out3 = R.run3(inject=inject)
    ab = read(out3 / "ablation.csv")
    assert [r["row"] for r in ab] == ["Baseline", "E1", "E2", "E3", "E4", "E5", "E6"]
    assert not [r for r in ab if r.get("error")], ab
    assert ab[0]["avg_bits_per_weight"] == "16.0" and float(ab[1]["avg_bits_per_weight"]) < 16
    assert float(ab[2]["decoder_weights_removed"]) > 0 and float(ab[0]["decoder_weights_removed"]) == 0
    assert (out3 / "summary.md").exists()
