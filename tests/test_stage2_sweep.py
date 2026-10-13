"""Stage 2 share sweep: Stage 0 saves one pruning plan per level, Stage 2 runs every method at each level (capped
for structured pruning) and sends every row to the backends (Mohammad, 2026-10-13)."""

import copy
import json
from dataclasses import replace

import pytest

from sdf.eval import llamacpp, trtllm_backend, vllm_backend
from sdf.run import start_run
from sdf.search_space import SEARCH_SPACE
from sdf.stage0.planner import CompressionPlan, cap_pruning, level_tag, uniform_plan
from sdf.stage0.run import run_stage0
from sdf.stages import methods as methods_mod
from sdf.stages.methods import METHODS, methods_for
from sdf.stages.runner import Stage0Plans, run_stage, sweep
from conftest import fake_texts

LEVELS = [0.3, 0.9]


@pytest.fixture
def stage0_dir(tiny_llama, tokenizer, small_cfg):
    cfg = small_cfg.with_overrides({"stage0.pruning_levels": LEVELS, "stage0.joint_plan": False})
    ctx = start_run(cfg)
    stale = ctx.run_dir / "stage_0" / "prune_plan_050.json"  # from an earlier run with other levels
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("{}")
    run_stage0(ctx, SEARCH_SPACE.make({"calib_samples": 16}), model=copy.deepcopy(tiny_llama), tokenizer=tokenizer,
               text_loader=fake_texts)
    return ctx.run_dir / "stage_0"


def test_level_tag_names_a_share_in_percent():
    assert [level_tag(r) for r in (0.3, 0.9, 0.05, 1.0, 0.7000001)] == ["030", "090", "005", "100", "070"]


def test_cap_pruning_lowers_only_the_layers_above_the_cap():
    plan = replace(uniform_plan([0.1, 0.2, 0.3], 16, 0.9),
                   layers=tuple(replace(lp, pruning_ratio=r) for lp, r in
                                zip(uniform_plan([0.1, 0.2, 0.3], 16, 0.9).layers, (0.9, 0.5, 0.75))))
    capped = cap_pruning(plan, [0.75, 0.75, 0.6])
    assert [lp.pruning_ratio for lp in capped.layers] == [0.75, 0.5, 0.6]
    with pytest.raises(ValueError):
        cap_pruning(plan, [0.75])
    assert [lp.bit_width for lp in capped.layers] == [16, 16, 16]


def test_structured_pruning_max_ratio_is_75_percent_and_series_methods_keep_it():
    assert METHODS["structured_prune"].max_ratio == 0.75
    assert METHODS["unstructured_prune"].max_ratio is None and METHODS["low_rank"].max_ratio is None
    (after,) = methods_for(2, ["structured_prune_after_gptq"])
    assert after.max_ratio == 0.75


def test_stage0_saves_one_plan_per_level_and_removes_old_ones(stage0_dir):
    names = sorted(f.name for f in stage0_dir.glob("prune_plan_*.json"))
    assert names == ["prune_plan_030.json", "prune_plan_090.json", "prune_plan_same_size.json",
                     "prune_plan_same_size_030.json", "prune_plan_same_size_090.json"]
    base = CompressionPlan.load(stage0_dir / "prune_plan.json")  # the 30% plan, as before
    ours = CompressionPlan.load(stage0_dir / "prune_plan_090.json")
    assert ours.prune_ratio_aggressive == 0.9
    for b, o in zip(base.layers, ours.layers):  # the same layers pruned, at the level's share; nothing rounded
        assert o.bit_width == 16 and (o.pruning_ratio == 0.9) == (b.pruning_ratio > 0)
    assert CompressionPlan.load(stage0_dir / "prune_plan_030.json").to_dict() == base.to_dict()
    same = CompressionPlan.load(stage0_dir / "prune_plan_same_size_090.json")
    assert same.prune_ratio_aggressive == 0.9 and max(lp.pruning_ratio for lp in same.layers) == 1.0
    assert all(lp.pruning_ratio == 0 for lp in same.layers if lp.guarded)  # the guard still holds


def test_stage0_plans_load_the_levels_and_swap_them_in(stage0_dir):
    plans = Stage0Plans.load(stage0_dir)
    assert sorted(plans.levels) == LEVELS
    at = plans.at_level(0.9)
    assert at.plans["prune"].prune_ratio_aggressive == 0.9
    assert at.plans["prune_same_size"].prune_ratio_aggressive == 0.9
    assert at.files["prune"] == "prune_plan_090.json"
    assert at.files["prune_same_size"] == "prune_plan_same_size_090.json"
    assert at.plans["quant"] is plans.plans["quant"]  # other plans unchanged
    assert plans.plans["prune"].prune_ratio_aggressive == pytest.approx(0.3)  # the original is left as it was


def test_sweep_repeats_only_pruning_methods_once_per_level(stage0_dir):
    plans = Stage0Plans.load(stage0_dir)
    (prune,) = methods_for(2, ["low_rank"])
    rows = sweep(prune, plans)
    assert [(m.name, level) for m, _, level in rows] == [("low_rank_030", 0.3), ("low_rank_090", 0.9)]
    assert rows[1][1].plans["prune"].prune_ratio_aggressive == 0.9
    (rtn,) = methods_for(1, ["rtn"])
    assert sweep(rtn, plans) == [(rtn, plans, None)]
    no_levels = replace(plans, levels={})
    assert sweep(prune, no_levels) == [(prune, no_levels, None)]  # an older Stage 0 folder: one 30% group


def test_stage2_runs_every_level_on_every_backend(stage0_dir, tiny_llama, tokenizer, small_cfg, monkeypatch):
    # the toy model's feed-forward share is 67%, below the real 75% cap, so the test caps structured at 60%
    monkeypatch.setitem(methods_mod.METHODS, "structured_prune",
                        replace(METHODS["structured_prune"], max_ratio=0.6))
    calls = []

    def fake(name, keys):
        def measure(model, tokenizer, plan, val, held, cfg, baseline_bits, *rest, **kw):
            calls.append((name, plan))
            return {k: 1.0 for k in keys}
        return measure

    monkeypatch.setattr(llamacpp, "measure", fake("llamacpp", llamacpp.MAIN_METRICS))
    monkeypatch.setattr(vllm_backend, "measure", fake("vllm", vllm_backend.MAIN_METRICS))
    monkeypatch.setattr(trtllm_backend, "measure", fake("trtllm", trtllm_backend.MAIN_METRICS))
    cfg = small_cfg.with_overrides({"eval.llamacpp_dir": "lc", "eval.llamacpp_convert": "conv.py",
                                    "eval.vllm_python": "vp", "eval.trtllm_python": "tp"})
    out = run_stage(start_run(cfg), 2, ["unstructured_prune", "structured_prune", "low_rank"],
                    SEARCH_SPACE.make({"calib_samples": 16}), stage0_dir,
                    model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    rows = {f"{r['method']}/{r['variant']}": r for r in json.loads(out["json"].read_text())["rows"]}
    expected = {"baseline/fp16"} | {f"{m}_{t}{s}/{v}" for m in ("unstructured_prune", "structured_prune", "low_rank")
                                    for t in ("030", "090") for s, v in (("", "original"), ("", "framework"),
                                                                         ("_same_size", "framework"))}
    assert set(rows) == expected
    for k, r in rows.items():
        assert r["status"] == "ok", (k, r.get("error"))
        assert r["metrics"]["vllm_ppl_val"] == 1.0 and r["metrics"]["llamacpp_ppl_val"] == 1.0
        assert r["metrics"]["trtllm_ppl_val"] == 1.0
    # every pruned row went to all three backends with its pruning plan (16 bits: saved as plain FP16)
    pruned = [p for _, p in calls if p is not None]
    assert len(pruned) == 3 * (len(rows) - 1)
    assert all(isinstance(p, CompressionPlan) and {lp.bit_width for lp in p.layers} == {16} for p in pruned)

    # the standard row prunes every layer at the level; structured is capped and says so
    std = rows["unstructured_prune_090/original"]
    assert "level 90%, 90.0% of the layer weights removed" in std["info"]["label"]
    capped = rows["structured_prune_090/original"]
    assert "lowest cap 60.0%; the plan asked for up to 90%" in capped["info"]["note"]
    assert "level 90%, 60.0% of the layer weights removed" in capped["info"]["label"]
    assert capped["metrics"]["sparsity"] < std["metrics"]["sparsity"]
    assert "capped" not in rows["structured_prune_030/original"]["info"].get("note", "")
    # same size at 90%: the unguarded layers cannot hold it all, so it removes less, and the label says how much
    same = rows["unstructured_prune_090_same_size/framework"]
    assert same["info"]["plan_file"] == "prune_plan_same_size_090.json"
    assert "level 90%, 75.0% of the layer weights removed" in same["info"]["label"]  # 3 of 4 equal layers emptied
    assert same["info"]["label"].startswith("Unstructured pruning (Wanda): Our method, same amount removed")
    labels = [r["info"]["label"] for k, r in rows.items() if k != "baseline/fp16"]
    assert len(set(labels)) == len(labels)  # every row can be told apart in the plain report
    assert rows["low_rank_030/framework"]["info"]["compare_to"] == "low_rank_030"


def test_series_rows_sweep_too_with_the_stage1_bits_and_the_cap(stage0_dir, tiny_llama, tokenizer, small_cfg,
                                                                  monkeypatch):
    monkeypatch.setitem(methods_mod.METHODS, "structured_prune",
                        replace(METHODS["structured_prune"], max_ratio=0.6))
    calls = []

    def measure(model, tokenizer, plan, *rest, **kw):
        calls.append(plan)
        return {k: 1.0 for k in llamacpp.MAIN_METRICS}

    monkeypatch.setattr(llamacpp, "measure", measure)
    cfg = small_cfg.with_overrides({"eval.llamacpp_dir": "lc", "eval.llamacpp_convert": "conv.py"})
    out = run_stage(start_run(cfg), 2, ["structured_prune_after_rtn"], SEARCH_SPACE.make({"calib_samples": 16}),
                    stage0_dir, model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer,
                    text_loader=fake_texts)
    rows = {f"{r['method']}/{r['variant']}": r for r in json.loads(out["json"].read_text())["rows"]}
    assert {k for k in rows if k != "baseline/fp16"} == {
        f"structured_prune_after_rtn_{t}{s}/{v}" for t in ("030", "090")
        for s, v in (("", "original"), ("", "framework"), ("_same_size", "framework"))}
    for k, r in rows.items():
        assert r["status"] == "ok", (k, r.get("error"))
    ours = rows["structured_prune_after_rtn_090/framework"]
    assert "lowest cap 60.0%" in ours["info"]["note"] and "level 90%" in ours["info"]["label"]
    plans = [p for p in calls if p is not None]
    assert len(plans) == 6 and all(min(lp.bit_width for lp in p.layers) < 16 for p in plans)  # the Stage 1 bits
    assert all(max(lp.pruning_ratio for lp in p.layers) <= 0.6 for p in plans if p.layers[0].pruning_ratio)


def test_without_level_plans_stage2_runs_each_method_once(stage0_dir, tiny_llama, tokenizer, small_cfg, caplog):
    for f in stage0_dir.glob("prune_plan_*0.json"):  # an older Stage 0 folder: no level plans
        f.unlink()
    out = run_stage(start_run(small_cfg), 2, ["unstructured_prune"], SEARCH_SPACE.make({"calib_samples": 16}),
                    stage0_dir, model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer,
                    text_loader=fake_texts)
    rows = {f"{r['method']}/{r['variant']}": r for r in json.loads(out["json"].read_text())["rows"]}
    assert set(rows) == {"baseline/fp16", "unstructured_prune/original", "unstructured_prune/framework",
                         "unstructured_prune_same_size/framework"}
    assert all(r["status"] == "ok" for r in rows.values())
    assert "level" not in rows["unstructured_prune/original"]["info"].get("label", "")
    assert "no pruning level plans" in caplog.text


def test_a_level_missing_its_same_size_plan_fails_only_that_row(stage0_dir, tiny_llama, tokenizer, small_cfg):
    (stage0_dir / "prune_plan_same_size_090.json").unlink()
    out = run_stage(start_run(small_cfg), 2, ["unstructured_prune"], SEARCH_SPACE.make({"calib_samples": 16}),
                    stage0_dir, model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer,
                    text_loader=fake_texts)
    rows = {f"{r['method']}/{r['variant']}": r for r in json.loads(out["json"].read_text())["rows"]}
    bad = rows["unstructured_prune_090_same_size/framework"]
    assert bad["status"] == "failed" and "prune_plan_same_size_090.json" in bad["error"]
    assert all(r["status"] == "ok" for k, r in rows.items() if k != "unstructured_prune_090_same_size/framework")


def test_structured_cap_drops_to_the_feed_forward_share_when_that_is_lower(stage0_dir, tiny_llama, tokenizer,
                                                                           small_cfg):
    """The toy model's feed-forward share (67%) is below 75%, as on models without grouped-query attention: the
    rows run at that share instead of failing (Mohammad, 2026-10-13)."""
    out = run_stage(start_run(small_cfg), 2, ["structured_prune"], SEARCH_SPACE.make({"calib_samples": 16}),
                    stage0_dir, model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer,
                    text_loader=fake_texts)
    rows = {f"{r['method']}/{r['variant']}": r for r in json.loads(out["json"].read_text())["rows"]}
    for k, r in rows.items():
        assert r["status"] == "ok", (k, r.get("error"))
    capped = rows["structured_prune_090/original"]
    assert "capped at 75% of each layer, or the layer's feed-forward share when lower (lowest cap 66.7%" in capped["info"]["note"]
    assert "level 90%, 66.7% of the layer weights removed" in capped["info"]["label"]


def test_a_llamacpp_failure_keeps_the_row_and_records_why(stage0_dir, tiny_llama, tokenizer, small_cfg, monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("llama-quantize exited with 1")

    monkeypatch.setattr(llamacpp, "measure", broken)
    cfg = small_cfg.with_overrides({"eval.llamacpp_dir": "lc", "eval.llamacpp_convert": "conv.py"})
    out = run_stage(start_run(cfg), 2, ["unstructured_prune"], SEARCH_SPACE.make({"calib_samples": 16}),
                    stage0_dir, model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer,
                    text_loader=fake_texts, measure_fp16=False)
    rows = {f"{r['method']}/{r['variant']}": r for r in json.loads(out["json"].read_text())["rows"]}
    row = rows["unstructured_prune_030/framework"]
    assert row["status"] == "ok" and row["metrics"]["ppl_val"] > 0
    assert row["info"]["llamacpp_error"] == "llama-quantize exited with 1"


def test_same_size_plan_moves_what_a_capped_layer_cannot_take_to_the_other_layers():
    from sdf.stage0.planner import same_size_pruning_plan
    scores, numel = [0.9, 0.1, 0.2, 0.5, 0.0], [100, 100, 100, 100, 100]
    free = same_size_pruning_plan(scores, 16, 0.5, numel, frozenset({0}))
    assert max(lp.pruning_ratio for lp in free.layers) > 0.7  # without caps the most robust layer goes above 70%
    capped = same_size_pruning_plan(scores, 16, 0.5, numel, frozenset({0}), [0.7] * 5)
    ratios = [lp.pruning_ratio for lp in capped.layers]
    assert ratios[0] == 0 and max(ratios) <= 0.7 + 1e-9
    assert sum(ratios) / 5 == pytest.approx(0.5, abs=1e-6)  # the same total as the standard row
    full = same_size_pruning_plan(scores, 16, 0.7, numel, frozenset({0}), [0.7] * 5)  # more than 4 x 70% can hold
    assert [lp.pruning_ratio for lp in full.layers] == pytest.approx([0, 0.7, 0.7, 0.7, 0.7])
    assert same_size_pruning_plan(scores, 16, 0.5, numel, frozenset({0}), None).to_dict() == free.to_dict()
    with pytest.raises(ValueError):
        same_size_pruning_plan(scores, 16, 0.5, numel, frozenset({0}), [0.7])


def test_capping_a_same_size_plan_keeps_its_total_and_the_stage1_bits():
    from sdf.stage0.planner import combine_plans, same_size_pruning_plan
    from sdf.stages.runner import _capped
    scores, numel = [0.9, 0.1, 0.2, 0.5, 0.0], [100, 100, 100, 100, 100]
    same = same_size_pruning_plan(scores, 16, 0.5, numel, frozenset({0}))
    bits = replace(same, layers=tuple(replace(lp, bit_width=b, pruning_ratio=0.0)
                                      for lp, b in zip(same.layers, (8, 4, 4, 8, 4))))
    after_stage1 = combine_plans(bits, same)  # a series row: Stage 1 bits, same-size pruning
    m = METHODS["structured_prune"]
    plan, note = _capped(m, after_stage1, lambda r: (0.7,) * 5, numel)
    ratios = [lp.pruning_ratio for lp in plan.layers]
    assert max(ratios) <= 0.7 + 1e-9 and sum(ratios) / 5 == pytest.approx(0.5, abs=1e-6)
    assert [lp.bit_width for lp in plan.layers] == [8, 4, 4, 8, 4]
    assert "moved to the other layers" in note
    std, std_note = _capped(m, uniform_plan(scores, 16, 0.9), lambda r: (0.7,) * 5, numel)
    assert {lp.pruning_ratio for lp in std.layers} == {0.7} and "moved" not in std_note
    assert _capped(m, uniform_plan(scores, 16, 0.3), lambda r: (0.7,) * 5, numel)[1] == ""


def test_capped_same_size_matches_the_capped_standard_total_when_caps_differ():
    from sdf.stage0.planner import same_size_pruning_plan
    from sdf.stages.runner import _capped
    scores, numel, caps = [0.1, 0.2, 0.3, 0.4], [100, 100, 100, 100], (0.5, 0.75, 0.75, 0.75)
    m = METHODS["structured_prune"]
    std, _ = _capped(m, uniform_plan(scores, 16, 0.7), lambda r: caps, numel)
    same, note = _capped(m, same_size_pruning_plan(scores, 16, 0.7, numel), lambda r: caps, numel)
    removed = lambda p: sum(lp.pruning_ratio for lp in p.layers) / 4  # noqa: E731
    assert removed(std) == pytest.approx(0.65)
    assert removed(same) == pytest.approx(removed(std), abs=1e-6)
    assert all(lp.pruning_ratio <= c + 1e-9 for lp, c in zip(same.layers, caps))
    assert "lowest cap 50.0%" in note
