"""Chained runs (0 -> 1 -> 2 -> 3 -> 4): Stage 2 after a different Stage 1 method per side, Stage 3 on a Stage 2 row."""

import copy
import json

import pytest

from sdf.run import start_run
from sdf.search_space import SEARCH_SPACE
from sdf.stage0.run import run_stage0
from sdf.stages.methods import METHODS
from sdf.stages.runner import best_rows, run_chain, run_stage
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


def test_best_rows_pick_the_lowest_validation_perplexity_per_level(tmp_path):
    rows = [{"method": n, "variant": v, "status": st, "metrics": {"ppl_val": p, "ppl_heldout": 1.0}}
            for n, v, st, p in [("a_030", "original", "ok", 12.0), ("b_030", "original", "ok", 11.0),
                                ("c_030", "original", "failed", 1.0), ("d_030", "original", "ok", float("nan")),
                                ("a_090", "original", "ok", 30.0), ("b_090", "framework", "ok", 5.0),
                                ("e_090", "original", "ok", 2.0)]]
    f = tmp_path / "results.json"
    f.write_text(json.dumps({"rows": rows}))
    level = lambda n: int(n[-3:]) / 100  # noqa: E731
    assert best_rows(f, "original", lambda n: not n.startswith("e"), level) == {0.3: "b_030", 0.9: "a_090"}
    assert best_rows(f, "framework", lambda n: True, level) == {0.9: "b_090"}
    assert best_rows(f, "original", lambda n: False) == {}


def test_run_chain_feeds_each_stage_the_best_row_of_the_one_before(tiny_llama, tokenizer, small_cfg):
    cfg = small_cfg.with_overrides({"stage0.pruning_levels": [0.3, 0.5], "stage0.joint_plan": False,
                                    "stages.stage1_methods": ["rtn", "gptq", "rtn_act"],
                                    "stages.stage2_methods": ["unstructured_prune", "structured_prune", "low_rank"],
                                    "stages.stage3_methods": ["rtn_kv", "h2o"]})
    ctx = start_run(cfg)
    cand = SEARCH_SPACE.make({"calib_samples": 16})
    kw = dict(model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    run_stage0(ctx, cand, model=copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    out = run_chain(ctx, cand, ctx.run_dir / "stage_0", **kw)

    s1 = _rows({"json": out["stage_1_json"]})
    s2 = json.loads(out["stage_2_json"].read_text())
    for v, label in (("original", "Standard"), ("framework", "Our method")):
        # the best weight row by validation perplexity (rtn_act, activations only, cannot be picked)
        best = min(("rtn", "gptq"), key=lambda n: s1[f"{n}/{v}"]["metrics"]["ppl_val"])
        assert f"({METHODS[best].label})" in s2["conditions"]["starting point"].split(";")[0 if v == "original" else 1]
    # low-rank cannot follow Stage 1: its rows fail and say so, they are never picked
    s2rows = {f"{r['method']}/{r['variant']}": r for r in s2["rows"]}
    assert s2rows["low_rank_030/original"]["status"] == "failed"
    for tag in ("030", "050"):
        s3 = json.loads(out[f"stage_3_after_{tag}_json"].read_text())
        rows = {f"{r['method']}/{r['variant']}": r for r in s3["rows"]}
        # the lowest validation perplexity at this share, on each side (Our method: a same-amount row)
        for v, suffix in (("original", ""), ("framework", "_same_size")):
            names = [f"{p}_{tag}{suffix}" for p in ("unstructured_prune", "structured_prune")]
            best = min(names, key=lambda n: s2rows[f"{n}/{v}"]["metrics"]["ppl_val"])
            assert rows[f"rtn_kv/{v}"]["info"]["built_on"] == best
        # Our method goes on with the same amount removed as Standard
        assert rows["h2o/framework"]["metrics"]["decoder_weights_removed"] == pytest.approx(
            rows["h2o/original"]["metrics"]["decoder_weights_removed"], abs=0.02)
        assert rows["h2o/framework"]["status"] == "ok"
    assert set(out) >= {f"stage_3_after_{t}_report" for t in ("030", "050")}


def test_run_chain_needs_a_weight_row_and_stage3_needs_before_only_there(stage0_dir, tiny_llama, tokenizer, small_cfg):
    cfg = small_cfg.with_overrides({"stages.stage1_methods": ["rtn_act"]})
    with pytest.raises(RuntimeError, match="no finished weight row"):
        run_chain(start_run(cfg), SEARCH_SPACE.make({"calib_samples": 16}), stage0_dir,
                  model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    with pytest.raises(ValueError, match="before= for Stage 3"):
        run_stage(start_run(small_cfg), 2, ["unstructured_prune"], {}, stage0_dir, before={"original": None})
