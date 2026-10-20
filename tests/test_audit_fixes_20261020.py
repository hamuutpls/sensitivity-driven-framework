"""Fixes from the 2026-10-19/20 ponytail audits (numbers are the audit's finding numbers)."""

import copy
import json
from dataclasses import replace

import pytest

from sdf.eval import llamacpp, vllm_backend
from sdf.run import start_run
from sdf.search_space import SEARCH_SPACE
from sdf.stage0.planner import level_tag, uniform_plan
from sdf.stage0.run import run_stage0
from sdf.stages import methods as methods_mod
from sdf.stages.methods import METHODS
from sdf.stages.runner import Stage0Plans, _no_passing_error, _plan_key, _ROTATED, run_stage
from sdf.utils.cache import ArtifactCache
from conftest import fake_texts


@pytest.fixture
def stage0_dir(tiny_llama, tokenizer, small_cfg):
    cfg = small_cfg.with_overrides({"stage0.pruning_levels": [0.3, 0.8, 0.9], "stage0.joint_plan": False})
    ctx = start_run(cfg)
    run_stage0(ctx, SEARCH_SPACE.make({"calib_samples": 16}), model=copy.deepcopy(tiny_llama), tokenizer=tokenizer,
               text_loader=fake_texts)
    return ctx.run_dir / "stage_0"


def _rows(out):
    return {f"{r['method']}/{r['variant']}": r for r in json.loads(out["json"].read_text())["rows"]}


def _run(cfg, stage, methods, stage0_dir, tiny_llama, tokenizer, **kw):
    return _rows(run_stage(start_run(cfg), stage, methods, SEARCH_SPACE.make({"calib_samples": 16}), stage0_dir,
                           model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer,
                           text_loader=fake_texts, **kw))


def test_cache_keeps_a_value_only_when_store_accepts_it(tmp_path):
    cache, calls = ArtifactCache(tmp_path), []
    compute = lambda: calls.append(1) or {"v": len(calls)}  # noqa: E731
    assert cache.get_or_compute("k", 1, compute, store=lambda v: False) == ({"v": 1}, False)
    assert cache.get_or_compute("k", 1, compute) == ({"v": 2}, False)  # not saved before, computed again
    assert cache.get_or_compute("k", 1, compute) == ({"v": 2}, True)


def test_only_backend_errors_that_always_repeat_are_saved():  # 1
    assert _no_passing_error({"info": {}}) and _no_passing_error({})
    assert _no_passing_error({"info": {"vllm_error": _ROTATED, "llamacpp_activations": "16-bit"}})
    assert not _no_passing_error({"info": {"vllm_error": "only 0.10 of 40.00 GB of GPU memory free"}})


def test_a_one_off_backend_failure_on_a_standard_row_is_measured_again(stage0_dir, tiny_llama, tokenizer,
                                                                      small_cfg, monkeypatch):  # 1
    state = {"fail": True}

    def measure(*a, **kw):
        if state["fail"]:
            raise vllm_backend.VLLMError("only 0.10 of 40.00 GB of GPU memory free")
        return {k: 1.0 for k in vllm_backend.MAIN_METRICS}

    monkeypatch.setattr(vllm_backend, "measure", measure)
    cfg = small_cfg.with_overrides({"eval.vllm_python": "vp"})
    first = _run(cfg, 1, ["rtn"], stage0_dir, tiny_llama, tokenizer, measure_fp16=False)
    assert "vllm_error" in first["rtn/original"]["info"]
    state["fail"] = False
    second = _run(cfg, 1, ["rtn"], stage0_dir, tiny_llama, tokenizer, measure_fp16=False)
    assert second["rtn/original"]["info"]["cached"] is False
    assert second["rtn/original"]["metrics"]["vllm_ppl_val"] == 1.0
    third = _run(cfg, 1, ["rtn"], stage0_dir, tiny_llama, tokenizer, measure_fp16=False)
    assert third["rtn/original"]["info"]["cached"] is True  # a clean result is saved as before


def test_a_backend_failure_on_the_fp16_row_keeps_its_hf_numbers(stage0_dir, tiny_llama, tokenizer, small_cfg,
                                                               monkeypatch):  # 2
    def broken(*a, **kw):
        raise RuntimeError("llama-perplexity exited with 1")

    monkeypatch.setattr(llamacpp, "measure", broken)
    cfg = small_cfg.with_overrides({"eval.llamacpp_dir": "lc", "eval.llamacpp_convert": "conv.py"})
    rows = _run(cfg, 1, ["rtn"], stage0_dir, tiny_llama, tokenizer)
    fp16 = rows["baseline/fp16"]
    assert fp16["status"] == "ok" and fp16["metrics"]["ppl_val"] > 0
    assert fp16["info"]["llamacpp_error"] == "llama-perplexity exited with 1"


def test_a_level_is_the_share_its_plans_were_built_for(stage0_dir):  # 4
    plans = Stage0Plans.load(stage0_dir)
    for ratio, ps in plans.levels.items():
        assert all(p.prune_ratio_aggressive == ratio for p in ps.values())
    # a level that is not a whole percent: the file is named 033, the level stays 0.333
    p = plans.levels[0.3]["prune"]
    replace(p, prune_ratio_aggressive=0.333).save(stage0_dir / "prune_plan_033.json")
    replace(plans.levels[0.3]["prune_same_size"], prune_ratio_aggressive=0.333).save(
        stage0_dir / "prune_plan_same_size_033.json")
    assert 0.333 in Stage0Plans.load(stage0_dir).levels and level_tag(0.333) == "033"


def test_the_standard_key_ignores_scores_and_flags():  # 5
    a = uniform_plan([0.1, 0.9, 0.5], 4, 0.3)
    b = uniform_plan([1.0, 0.0, 0.2], 4, 0.3)
    assert a.to_dict() != b.to_dict() and _plan_key(a) == _plan_key(b)
    assert _plan_key(a) != _plan_key(uniform_plan([0.1, 0.9, 0.5], 8, 0.3))
    assert _plan_key(a) != _plan_key(uniform_plan([0.1, 0.9, 0.5], 4, 0.5))


def test_stages_score_exposure_against_stage0s_reference(stage0_dir):  # 18
    plans = Stage0Plans.load(stage0_dir)
    assert plans.reference is None  # the test run scores layer removal: its own ranks are the reference
    (stage0_dir / "exposure_reference.json").write_text(json.dumps([1.0] + [0.0] * (plans.num_layers - 1)))
    assert Stage0Plans.load(stage0_dir).reference[0] == 1.0


def test_billm_standard_row_has_no_predicted_size(stage0_dir, tiny_llama, tokenizer, small_cfg, monkeypatch):  # 9
    monkeypatch.setitem(methods_mod.METHODS, "rtn", replace(METHODS["rtn"], fixed_bits="it binarizes every layer"))
    rows = _run(small_cfg, 1, ["rtn"], stage0_dir, tiny_llama, tokenizer, measure_fp16=False)
    std = rows["rtn/original"]
    assert std["status"] == "ok"
    assert "predicted_weight_memory_gb" not in std["metrics"] and "avg_bits_per_weight" not in std["metrics"]


def test_capped_levels_with_the_same_plans_run_once(stage0_dir, tiny_llama, tokenizer, small_cfg, monkeypatch):  # 14
    # the toy model's caps are 66.7%, so 80% and 90% give the same capped plans
    out = run_stage(start_run(small_cfg), 2, ["structured_prune", "unstructured_prune"],
                    SEARCH_SPACE.make({"calib_samples": 16}), stage0_dir, model_factory=lambda: copy.deepcopy(tiny_llama),
                    tokenizer=tokenizer, text_loader=fake_texts, measure_fp16=False)
    rows = _rows(out)
    assert "structured_prune_080/original" in rows and "structured_prune_090/original" not in rows
    assert "structured_prune_090_same_size/framework" not in rows
    assert "unstructured_prune_090/original" in rows  # not capped: every level runs
    assert "at 90% was not run: after the cap its plans are the same as at 80%" in out["report"].read_text()


def test_pruning_rows_key_the_export_group_size_only_when_they_round(stage0_dir, tiny_llama, tokenizer, small_cfg,
                                                                     monkeypatch):  # 15
    monkeypatch.setattr(vllm_backend, "measure", lambda *a, **kw: {k: 1.0 for k in vllm_backend.MAIN_METRICS})
    keys = []
    real = ArtifactCache.get_or_compute
    monkeypatch.setattr(ArtifactCache, "get_or_compute",
                        lambda self, kind, key, *a, **kw: (keys.append((kind, key)), real(self, kind, key, *a, **kw))[1])
    cfg = small_cfg.with_overrides({"eval.vllm_python": "vp"})
    _run(cfg, 2, ["unstructured_prune"], stage0_dir, tiny_llama, tokenizer, measure_fp16=False)
    std = [k for kind, k in keys if kind == "stage2_original"]
    assert std and all("export_group_size" not in k for k in std)  # 16 bits: the export rounds nothing


def test_plan_metrics_use_stage0s_exposure_reference(stage0_dir, small_cfg):  # 18
    from sdf.stages.runner import plan_metrics
    plans = Stage0Plans.load(stage0_dir)
    cand = SEARCH_SPACE.make({"calib_samples": 16})
    quant = plans.plans["quant"]
    own = plan_metrics(quant, plans, small_cfg, cand)["sensitivity_exposure"]
    ref = [0.0 if lp.protected else 1.0 for lp in quant.layers]  # the reference rates every protected layer robust
    other = plan_metrics(quant, replace(plans, reference=ref), small_cfg, cand)["sensitivity_exposure"]
    assert other != own


def test_without_levels_the_standard_row_removes_stage0s_share(stage0_dir, tiny_llama, tokenizer, small_cfg):  # 4
    for f in stage0_dir.glob("prune_plan_*0.json"):
        f.unlink()
    built_for = Stage0Plans.load(stage0_dir).plans["prune_same_size"].prune_ratio_aggressive
    out = run_stage(start_run(small_cfg), 2, ["unstructured_prune"],
                    SEARCH_SPACE.make({"calib_samples": 16, "prune_ratio_aggressive": 0.5}), stage0_dir,
                    model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts,
                    measure_fp16=False)
    std = _rows(out)["unstructured_prune/original"]
    assert built_for != 0.5 and std["metrics"]["decoder_weights_removed"] == pytest.approx(built_for)
