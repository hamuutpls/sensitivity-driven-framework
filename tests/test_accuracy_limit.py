"""Accuracy as a search limit (paper review, 2026-11-22: Mohammad "add both"): requirement.min_accuracy."""

import sys

import pytest

from sdf import search
from sdf.requirements import DeploymentRequirement
from sdf.run import start_run


def test_min_accuracy_is_a_lower_bound_with_its_shortfall():
    req = DeploymentRequirement(min_accuracy=0.6, target_ppl=11.5)
    ok = req.check({"downstream_acc_mean": 0.62, "ppl_val": 11.0})
    assert ok.met and ok.checked["min_accuracy"] == "downstream_acc_mean"
    low = req.check({"downstream_acc_mean": 0.55, "ppl_val": 11.0})
    assert low.met is False and low.shortfall == {"min_accuracy": pytest.approx(0.05)}
    assert req.check({"ppl_val": 11.0}).met is None  # accuracy not measured: not met yet, not failed
    both = req.check({"downstream_acc_mean": 0.5, "ppl_val": 12.0})
    assert set(both.shortfall) == {"min_accuracy", "target_ppl"}
    assert DeploymentRequirement(min_accuracy=0.6).check({"downstream_acc_mean": float("nan")}).met is False


@pytest.fixture
def cfg(small_cfg):
    return small_cfg.with_overrides({"search.objectives": ["ppl_val", "predicted_weight_memory_gb"],
                                     "search.searchers": ["mobo"], "search.budget": 1,
                                     "search.stage1_methods": ["rtn_with_rtn_act"],
                                     "search.stage2_methods": ["unstructured_prune"],
                                     "search.stage3_methods": ["rtn_kv"]})


def test_the_accuracy_limit_needs_the_tasks(cfg):
    with pytest.raises(ValueError, match="min_accuracy"):
        search.run_search(start_run(cfg.with_overrides({"requirement.min_accuracy": 0.6})))


@pytest.mark.parametrize("limit, want", [(None, []), (0.6, ["arc_easy", "piqa"])])
def test_every_try_runs_the_accuracy_tasks_only_when_accuracy_is_a_limit(cfg, tokenizer, monkeypatch, limit, want):
    monkeypatch.setitem(sys.modules, "lm_eval", object())  # checked before the tries; no test is run here
    seen = []

    class Stop(Exception):
        pass

    def recorder(ctx, *a, **k):
        seen.append(ctx.cfg.eval.downstream_tasks)
        raise Stop

    monkeypatch.setattr(search, "Recorder", recorder)
    from conftest import fake_texts

    run = start_run(cfg.with_overrides({"requirement.min_accuracy": limit,
                                        "search.accuracy_tasks": ["arc_easy", "piqa"]}))
    with pytest.raises(Stop):
        search.run_search(run, tokenizer=tokenizer, text_loader=fake_texts)
    assert seen == [want]
