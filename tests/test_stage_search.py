"""The search folded into the stages (sdf.stage_search): each stage's space and try, the pairing step, the whole run."""

import json
from pathlib import Path

import pytest

from sdf import stage_search
from sdf.eval import vllm_backend
from sdf.run import start_run
from sdf.stage_search import (evaluate_stage, objectives, pair_up, run_stage_search, stage_space, winner)
from test_search import HF, _kw, cfg  # noqa: F401 - the cfg fixture


def test_each_stage_searches_its_own_settings_and_technique(cfg):  # noqa: F811
    one, two = ([p.name for p in stage_space(cfg, s).params] for s in (1, 2))
    assert one == ["sensitive_threshold", "calib_dataset", "calib_samples", "gptq_groupsize", "smoothquant_alpha",
                   "stage1_method"]
    assert two == ["prune_ratio_aggressive", "stage2_method"]
    assert objectives(cfg, 1) == HF + ["predicted_weight_memory_gb"]
    assert objectives(cfg, 2) == ["ppl_val", "predicted_weight_memory_gb"]


def _rows(r, stage):
    return json.loads((Path(r["dir"]) / f"stage_{stage}" / "results.json").read_text())["rows"]


def test_a_stage1_try_builds_only_the_joint_row_without_refinement_and_has_no_limit(cfg, tiny_llama,  # noqa: F811
                                                                                     tokenizer):
    ctx = start_run(cfg.with_overrides({"requirement.target_memory_gb": 1e-9}))  # nothing could meet it
    point = stage_space(cfg, 1).make({"calib_samples": 16})
    r = evaluate_stage(ctx, point, stage=1, **_kw(tiny_llama, tokenizer))
    assert "error" not in r, r
    assert r["met"] and r["metrics"]["predicted_weight_memory_gb"] > 1e-9  # a stage keeps a line, no limit
    rows = _rows(r, 1)
    assert [(x["variant"], x["method"]) for x in rows] == [("fp16", "baseline"), ("framework", "rtn_with_rtn_act_joint")]
    stage0 = json.loads((Path(r["dir"]) / "stage_0" / "results.json").read_text())["rows"]
    joint = next(x for x in stage0 if x["method"] == "joint_weights_activations")
    assert "refinement" not in joint["info"]  # the one-shot joint plan
    again = evaluate_stage(ctx, point, stage=1, **_kw(tiny_llama, tokenizer))
    assert again["cached"] and {**again, "cached": False} == r
    # a run setting the point does not search (here the pruning share in Stage 1) is part of the key
    other = start_run(cfg.with_overrides({"requirement.target_memory_gb": 1e-9, "hyperparams.prune_ratio_aggressive": 0.5}))
    assert evaluate_stage(other, point, stage=1, **_kw(tiny_llama, tokenizer))["dir"] != r["dir"]


def test_a_stage2_try_prunes_the_uncompressed_model_with_no_backend(cfg, tiny_llama, tokenizer,  # noqa: F811
                                                                    monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("Stage 2 tries run no backend")

    monkeypatch.setattr(vllm_backend, "measure", refuse)
    ctx = start_run(cfg.with_overrides({"eval.vllm_python": "py", "hyperparams.calib_samples": 16}))
    r = evaluate_stage(ctx, stage_space(cfg, 2).make({"prune_ratio_aggressive": 0.4}), stage=2,
                       **_kw(tiny_llama, tokenizer))
    assert "error" not in r, r
    m = r["metrics"]
    fp16 = next(x for x in _rows(r, 2) if x["variant"] == "fp16")["metrics"]["ppl_val"]
    assert m["decoder_weights_removed"] > 0 and m["ppl_rise"] == pytest.approx(m["ppl_val"] - fp16)
    assert "avg_activation_bits" not in m
    assert [(x["variant"], x["method"]) for x in _rows(r, 2)] == [("fp16", "baseline"),
                                                                  ("framework", "unstructured_prune")]


def _t(searcher, n, point, **metrics):
    return {"searcher": searcher, "n": n, "point": point, "metrics": metrics}


def test_pairs_are_built_best_estimate_first_until_enough_meet_the_limit(cfg, monkeypatch):  # noqa: F811
    ones = [_t("mobo", 1, {"stage1_method": "rtn_with_rtn_act", "sensitive_threshold": 0.2}, ppl_val=10.0),
            _t("mfbo", 4, {"stage1_method": "rtn_with_rtn_act", "sensitive_threshold": 0.6}, ppl_val=11.0)]
    twos = [_t("nsga3", 2, {"stage2_method": "unstructured_prune", "prune_ratio_aggressive": 0.0}, ppl_rise=0.0),
            _t("mobo", 3, {"stage2_method": "structured_prune", "prune_ratio_aggressive": 0.5}, ppl_rise=1.5)]
    seen = []

    def fake(ctx, point, **kw):  # the smallest threshold with no pruning is over the limit
        seen.append(point)
        if point["sensitive_threshold"] == 0.2 and point["prune_ratio_aggressive"] == 0.0:
            return {"metrics": {}, "not_built": "over the limit, so not built", "met": False, "dir": "d",
                    "cached": False}
        ppl = 10 + point["sensitive_threshold"] + point["prune_ratio_aggressive"]
        return {"metrics": {"ppl_val": ppl, "decode_ms_per_token_mean": 1.0}, "met": True, "dir": "d",
                "cached": False}

    monkeypatch.setattr(stage_search, "evaluate", fake)
    tried = pair_up(start_run(cfg), ones, twos, "h2o", finalists=2)
    # estimates: 10.0 (over the limit), 11.0, 11.5, 12.5: built in that order until 2 are measured under it
    assert [t["estimate"] for t in tried] == [10.0, 11.0, 11.5]
    assert [t["met"] for t in tried] == [False, True, True] and tried[0]["not_built"]
    assert all(p["stage3_method"] == "h2o" and p["quarot_k_bits"] == 4 for p in seen)  # the run's other settings
    assert tried[1]["stage1"] == ["mfbo", 4] and tried[1]["stage2"] == ["nsga3", 2]
    assert winner(cfg, tried)["metrics"]["ppl_val"] == pytest.approx(10.6)  # 0.6 + 0.0 beats 0.2 + 0.5
    assert winner(cfg, tried[:1]) is None
    # a limit no pair meets: it stops after finalists x max_tries_per_budget pairs, not after every pair
    monkeypatch.setattr(stage_search, "evaluate", lambda ctx, point, **kw: {**fake(ctx, point), "met": False})
    c = cfg.with_overrides({"search.max_tries_per_budget": 1})
    assert len(pair_up(start_run(c), ones, twos, "h2o", finalists=3)) == 3


def test_run_stage_search_writes_each_stage_and_the_final_choice(cfg, tiny_llama, tokenizer):  # noqa: F811
    c = cfg.with_overrides({"search.budget": 1, "search.searchers": ["mobo", "nsga3"], "search.finalists": 1,
                            "hyperparams.calib_samples": 16})
    out = run_stage_search(start_run(c), **_kw(tiny_llama, tokenizer))
    res = json.loads(out["stage_search_results"].read_text())
    assert res["stage_1_front"] and res["stage_2_front"] and res["stage_3_technique"] in ("rtn_kv", "h2o")
    assert {r["method"] for r in res["stage_3_rows"]} == {"rtn_kv", "h2o"}
    best = res["winner"]
    assert best and best["met"] and best["point"]["stage3_method"] == res["stage_3_technique"]
    assert sum(t["met"] for t in res["pairs"]) == 1  # finalists = 1
    trials = json.loads(out["stage_1_search_trials"].read_text())
    assert {t["searcher"] for t in trials} == {"mobo", "nsga3"} and "stage2_method" not in trials[0]["point"]
    text = out["stage_search_report"].read_text()
    assert "The final model:" in text and "Pairs tried" in text and "at most 1.0" in text
    assert "with 2 search methods (MOBO, NSGA-III)" in text  # the run's searchers, not always three
    assert "Stage 1 searched its own settings" in out["stage_1_search_report"].read_text()


def test_a_stage_search_needs_perplexity_first_and_vllm_for_a_vllm_objective(cfg):  # noqa: F811
    with pytest.raises(ValueError, match="ppl_val"):
        run_stage_search(start_run(cfg.with_overrides({"search.objectives": ["decode_ms_per_token_mean"]})))
    with pytest.raises(ValueError, match="eval.vllm_python"):
        run_stage_search(start_run(cfg.with_overrides({"search.objectives": ["ppl_val", "vllm_decode_ms_per_token"]})))
    with pytest.raises(ValueError, match="target_memory_gb"):
        run_stage_search(start_run(cfg.with_overrides({"requirement.target_memory_gb": None})))


def test_the_kv_pick_keeps_to_the_kv_memory_limit(cfg, tiny_llama, tokenizer):  # noqa: F811
    from sdf.stage_search import full_candidate, pick_kv

    c = cfg.with_overrides({"hyperparams.calib_samples": 16})
    kv, rows, _ = pick_kv(start_run(c), full_candidate(c, {}), **_kw(tiny_llama, tokenizer))
    assert kv in ("rtn_kv", "h2o") and len(rows) == 2
    tight = c.with_overrides({"requirement.kv_budget_gb": 1e-12})  # no technique fits
    assert pick_kv(start_run(tight), full_candidate(tight, {}), **_kw(tiny_llama, tokenizer))[0] is None


def test_extra_tries_measure_only_the_techniques_no_searcher_measured(cfg, monkeypatch):  # noqa: F811
    from sdf.stage_search import _report, extra_tries

    seen = []

    def fake(ctx, point, max_windows=None, stage=1, **kw):
        seen.append((point, max_windows, stage))
        return {"metrics": {"ppl_val": 11.0, "vllm_decode_ms_per_token": 30.0, "decode_ms_per_token_mean": 30.0,
                            "predicted_weight_memory_gb": 0.9},
                "met": True, "dir": "d", "cached": False}

    monkeypatch.setattr(stage_search, "evaluate_stage", fake)
    c = cfg.with_overrides({"hyperparams.calib_samples": 64, "search.stage1_methods": ["rtn_with_rtn_act",
                            "gptq_with_rtn_act", "awq_with_rtn_act", "awq_with_smoothquant"]})
    measured = {"full": True, "ok": True, "point": {"stage1_method": "gptq_with_smoothquant"}}
    rough = {"full": False, "ok": True, "point": {"stage1_method": "awq_with_rtn_act"}}
    out = extra_tries(start_run(c), [measured, rough], ["gptq_with_rtn_act", "awq_with_rtn_act"], 6)
    # gptq was measured on all the windows (with another activation option); awq only on fewer, so it gets its try
    assert [p["stage1_method"] for p, _, _ in seen] == ["awq_with_rtn_act"]
    assert seen[0][1:] == (None, 1) and seen[0][0]["calib_samples"] == 64  # all windows, the run's own settings
    assert out[0]["searcher"] == "extra" and out[0]["ok"]
    assert extra_tries(start_run(c), [measured], [], 6) == []
    text = _report(c, {1: [], 2: []}, "rtn_kv", [], [], None, out)
    assert "## Stage 1: extra techniques" in text and "11" in text.split("extra techniques")[1]
    assert "extra techniques" not in _report(c, {1: [], 2: []}, "rtn_kv", [], [], None)
