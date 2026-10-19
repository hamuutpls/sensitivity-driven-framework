"""Chained runs (0 -> 1 -> 2 -> 3 -> 4): Stage 2 after a different Stage 1 method per side, Stage 3 on a Stage 2 row."""

import copy
import json

import pytest

from sdf.run import start_run
from sdf.search_space import SEARCH_SPACE
from sdf.stage0.run import run_stage0
from sdf.stages.runner import Stage0Plans, run_chain, run_stage, trade_off_rows
from conftest import fake_texts


@pytest.fixture
def stage0_dir(tiny_llama, tokenizer, small_cfg):
    ctx = start_run(small_cfg)
    run_stage0(ctx, SEARCH_SPACE.make({"calib_samples": 16}), model=copy.deepcopy(tiny_llama), tokenizer=tokenizer,
               text_loader=fake_texts)
    return ctx.run_dir / "stage_0"


def _rows(out):
    return {f"{r['method']}/{r['variant']}": r for r in json.loads(out["json"].read_text())["rows"]}


def test_stage2_after_a_different_stage1_method_per_side(stage0_dir, tiny_llama, tokenizer, small_cfg):
    kw = dict(model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    used = {}
    out = run_stage(start_run(small_cfg), 2, ["unstructured_prune"], SEARCH_SPACE.make({"calib_samples": 16}),
                    stage0_dir, after={"original": "rtn", "framework": "gptq"}, used=used, **kw)
    rows = _rows(out)
    # the rows keep the pruning method's name, so Our method is compared with Standard
    assert set(rows) == {"baseline/fp16", "unstructured_prune/original", "unstructured_prune/framework",
                         "unstructured_prune_same_size/framework"}
    assert all(r["status"] == "ok" for r in rows.values()), {k: r.get("error") for k, r in rows.items()}
    assert "after Round-to-nearest" in rows["unstructured_prune/original"]["info"]["description"]
    assert "bits [4]" in rows["unstructured_prune/original"]["info"]["description"]
    for k in ("unstructured_prune/framework", "unstructured_prune_same_size/framework"):
        assert "after GPTQ" in rows[k]["info"]["description"]
        assert "vs_original_abs" in rows[k]["deltas"]["ppl_val"]
    assert "(Round-to-nearest)" in json.loads(out["json"].read_text())["conditions"]["starting point"]
    m, plan, level = used[("unstructured_prune_same_size", "framework")]
    assert m.name == "unstructured_prune" and "GPTQ" in m.label and level is None
    assert min(lp.bit_width for lp in plan.layers) < 16 and max(lp.pruning_ratio for lp in plan.layers) > 0
    # a different Stage 1 method for Standard is a different cache entry
    out2 = run_stage(start_run(small_cfg), 2, ["unstructured_prune"], SEARCH_SPACE.make({"calib_samples": 16}),
                     stage0_dir, after={"original": "awq", "framework": "gptq"}, **kw)
    assert not _rows(out2)["unstructured_prune/original"]["info"]["cached"]


def test_stage3_on_a_stage2_row(stage0_dir, tiny_llama, tokenizer, small_cfg):
    kw = dict(model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    cand = SEARCH_SPACE.make({"calib_samples": 16})
    ctx = start_run(small_cfg)
    used = {}
    run_stage(ctx, 2, ["unstructured_prune"], cand, stage0_dir, after={"original": "rtn", "framework": "rtn"},
              used=used, **kw)
    alone = _rows(run_stage(ctx, 3, ["rtn_kv"], cand, stage0_dir, **kw))
    before = {"original": used[("unstructured_prune", "original")][:2],
              "framework": used[("unstructured_prune_same_size", "framework")][:2]}
    out = run_stage(ctx, 3, ["rtn_kv"], cand, stage0_dir, before=before, subdir="stage_3_chain", **kw)
    assert out["report"].parent.name == "stage_3_chain"
    rows = _rows(out)
    assert all(r["status"] == "ok" for r in rows.values()), {k: r.get("error") for k, r in rows.items()}
    for v in ("original", "framework"):
        r = rows[f"rtn_kv/{v}"]
        assert r["info"]["built_on"] == "unstructured_prune"
        assert r["metrics"]["decoder_weights_removed"] > 0
        assert r["metrics"]["predicted_weight_memory_gb"] < rows["baseline/fp16"]["metrics"]["predicted_weight_memory_gb"]
        assert r["metrics"]["predicted_kv_memory_gb"] == alone[f"rtn_kv/{v}"]["metrics"]["predicted_kv_memory_gb"]
        assert r["metrics"]["ppl_val"] != alone[f"rtn_kv/{v}"]["metrics"]["ppl_val"]  # the Stage 2 row was built
    # the Stage 2 row is part of the cache key: not the unchained entry, but reused on a second chained run
    assert not rows["rtn_kv/original"]["info"]["cached"]
    again = _rows(run_stage(ctx, 3, ["rtn_kv"], cand, stage0_dir, before=before, subdir="stage_3_chain", **kw))
    assert again["rtn_kv/original"]["info"]["cached"]
    assert again["rtn_kv/original"]["metrics"]["ppl_val"] == rows["rtn_kv/original"]["metrics"]["ppl_val"]


def test_after_and_before_belong_to_their_stage(stage0_dir, small_cfg):
    with pytest.raises(ValueError, match="after= is for Stage 2"):
        run_stage(start_run(small_cfg), 3, ["rtn_kv"], {}, stage0_dir, after={"original": "rtn", "framework": "rtn"})


def test_trade_off_rows_keep_every_row_no_other_beats_on_memory_and_perplexity():
    def row(name, mem, ppl, status="ok", variant="original"):
        return {"method": name, "variant": variant, "status": status,
                "metrics": {"predicted_weight_memory_gb": mem, "ppl_val": ppl, "ppl_heldout": 0.0}}
    rows = [row("small", 0.5, 20.0), row("mid", 0.7, 12.0), row("big", 1.0, 10.5),
            row("worse", 0.8, 13.0),  # mid is smaller and better
            row("tie", 0.7, 12.0),  # the same point as mid: both stay
            row("broken", 0.1, float("nan")), row("failed", 0.1, 1.0, "failed")]
    assert [r["method"] for r in trade_off_rows(rows)] == ["small", "mid", "big", "tie"]
    assert trade_off_rows([]) == []


def test_run_chain_follows_each_side_along_its_trade_off_line(tiny_llama, tokenizer, small_cfg):
    cfg = small_cfg.with_overrides({"stage0.pruning_levels": [0.3, 0.5],
                                    "stages.stage1_methods": ["rtn", "rtn_with_rtn_act", "gptq_with_rtn_act"],
                                    "stages.stage2_methods": ["unstructured_prune", "low_rank"],
                                    "stages.stage3_methods": ["rtn_kv"]})
    ctx = start_run(cfg)
    cand = SEARCH_SPACE.make({"calib_samples": 16})
    kw = dict(model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    run_stage0(ctx, cand, model=copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    out = run_chain(ctx, cand, ctx.run_dir / "stage_0", **kw)

    s1 = json.loads(out["stage_1_json"].read_text())["rows"]
    stage2 = {k.removesuffix("_json"): json.loads(p.read_text()) for k, p in out.items()
              if k.startswith("stage_2_") and k.endswith("_json")}
    stage3 = {k.removesuffix("_json"): json.loads(p.read_text()) for k, p in out.items()
              if k.startswith("stage_3_") and k.endswith("_json")}
    for variant, side in (("original", "standard"), ("framework", "ours")):
        # Stage 1: exactly the WxAy rows on the line go on (rtn, weights only, never does)
        wxay = [r for r in s1 if r["variant"] == variant and "_with_" in r["method"]]
        expected = {f"stage_2_{side}_after_{r['method']}" for r in trade_off_rows(wxay)}
        assert {k for k in stage2 if k.startswith(f"stage_2_{side}_")} == expected and expected
        for k in expected:
            rows = stage2[k]["rows"]
            assert {r["variant"] for r in rows} == {"fp16", variant}  # only this side's rows
            assert next(r for r in rows if r["method"] == "low_rank_030")["status"] == "failed"
        # Stage 2: the rows on the line of all of this side's Stage 2 rows go on, one Stage 3 report each
        all2 = [(k, r) for k in expected for r in stage2[k]["rows"] if r["variant"] == variant]
        line = trade_off_rows([r for _, r in all2])
        assert {k for k in stage3 if k.startswith(f"stage_3_{side}_")} == {
            f"stage_3_{side}_after_{k.removeprefix(f'stage_2_{side}_after_')}_{r['method']}"
            for k, r in all2 if r in line}
        for k, rep in stage3.items():
            if k.startswith(f"stage_3_{side}_"):
                r3 = next(r for r in rep["rows"] if r["method"] == "rtn_kv")
                assert r3["variant"] == variant and r3["status"] == "ok", r3.get("error")
                assert k.endswith(r3["info"]["built_on"]) and r3["metrics"]["avg_activation_bits"] < 16


def test_run_chain_needs_a_wxay_row_and_stage3_needs_before_only_there(stage0_dir, tiny_llama, tokenizer, small_cfg):
    kw = dict(model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    cfg = small_cfg.with_overrides({"stages.stage1_methods": ["rtn", "rtn_act"]})
    with pytest.raises(ValueError, match="has none"):  # before Stage 1 runs
        run_chain(start_run(cfg), SEARCH_SPACE.make({"calib_samples": 16}), stage0_dir, **kw)
    cfg = small_cfg.with_overrides({"stages.stage1_methods": ["qtip_with_rtn_act"]})  # listed, but its rows fail
    with pytest.raises(RuntimeError, match="no finished WxAy"):
        run_chain(start_run(cfg), SEARCH_SPACE.make({"calib_samples": 16}), stage0_dir, **kw)
    with pytest.raises(ValueError, match="before= for Stage 3"):
        run_stage(start_run(small_cfg), 2, ["unstructured_prune"], {}, stage0_dir, before={"original": None})


@pytest.mark.parametrize("ours", ["gptq_with_rtn_act", "gptq_with_rtn_act_joint"])
def test_stage2_after_a_wxay_row_keeps_its_activation_bits(stage0_dir, tiny_llama, tokenizer, small_cfg, ours):
    kw = dict(model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    used = {}
    out = run_stage(start_run(small_cfg), 2, ["unstructured_prune", "low_rank"], SEARCH_SPACE.make({"calib_samples": 16}),
                    stage0_dir, after={"original": "rtn_with_rtn_act_w8a8", "framework": ours}, used=used, **kw)
    rows = _rows(out)
    for k in ("unstructured_prune/original", "unstructured_prune/framework", "unstructured_prune_same_size/framework"):
        assert rows[k]["status"] == "ok", rows[k].get("error")
        assert rows[k]["metrics"]["avg_activation_bits"] < 16 and rows[k]["metrics"]["decoder_weights_removed"] > 0
    assert rows["low_rank/original"]["status"] == "failed"  # still cannot follow Stage 1
    std, std_plan, _ = used[("unstructured_prune", "original")]
    assert {lp.bit_width for lp in std_plan.weights.layers} == {8} and std_plan.acts.avg_bits == 8  # W8A8 standard
    _, plan, _ = used[("unstructured_prune_same_size", "framework")]
    s0 = Stage0Plans.load(stage0_dir)
    acts = s0.plans["joint_acts" if ours.endswith("_joint") else "activations"]
    weights = s0.plans["joint_weights" if ours.endswith("_joint") else "quant"]
    assert plan.acts == acts and [lp.bit_width for lp in plan.weights.layers] == [lp.bit_width for lp in weights.layers]
    assert "after GPTQ + round-to-nearest activations" in rows["unstructured_prune/framework"]["info"]["description"]

    # Stage 3 on such a row: the activations stay rounded under the KV method
    before = {v: used[(n, v)][:2] for n, v in (("unstructured_prune", "original"),
                                                ("unstructured_prune_same_size", "framework"))}
    s3 = _rows(run_stage(start_run(small_cfg), 3, ["rtn_kv"], SEARCH_SPACE.make({"calib_samples": 16}), stage0_dir,
                         before=before, subdir="stage_3_chain", **kw))
    for v in ("original", "framework"):
        assert s3[f"rtn_kv/{v}"]["status"] == "ok" and s3[f"rtn_kv/{v}"]["metrics"]["avg_activation_bits"] < 16
