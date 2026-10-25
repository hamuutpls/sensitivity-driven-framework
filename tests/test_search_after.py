"""What the chained search does after its searchers: the extra Stage 1 techniques and the final model's accuracy."""

import json
import sys

import pytest

from sdf import search
from sdf.eval import downstream
from sdf.reporting.search import write_search_report
from sdf.run import start_run
from sdf.search import extra_tries, final_accuracy, technique_space
from test_search import _kw, _point, cfg  # noqa: F401 - the cfg fixture


def _trial(point, n, full=True, ok=True, met=True, **metrics):
    return {"searcher": "mobo", "n": n, "point": point, "full": full, "ok": ok, "met": met, "windows": 6, "cost": 1.0,
            "values": [metrics.get("ppl_val", 0.0), 1.0], "metrics": metrics, "error": None, "dir": "d"}


def test_extra_tries_swap_only_the_stage1_technique_of_the_best_try(cfg, monkeypatch):  # noqa: F811
    c = cfg.with_overrides({"search.stage1_methods": ["rtn_with_rtn_act", "gptq_with_rtn_act", "awq_with_rtn_act",
                                                      "awq_with_smoothquant"]})
    seen = []

    def fake(ctx, point, max_windows=None, **kw):
        seen.append((point, max_windows))
        return {"metrics": {"ppl_val": 11.0, "decode_ms_per_token_mean": 1.0, "predicted_weight_memory_gb": 0.5,
                            "decoder_weight_bits": 4.5, "avg_activation_bits": 8},
                "met": True, "shortfall": {}, "dir": "d", "cached": False}

    monkeypatch.setattr(search, "evaluate", fake)
    best = _trial(_point(c, sensitive_threshold=0.2, stage1_method="gptq_with_rtn_act"), 1, ppl_val=10.0)
    worse = _trial(_point(c, sensitive_threshold=0.7, stage1_method="rtn_with_rtn_act"), 2, ppl_val=12.0)
    rough = _trial(_point(c, stage1_method="awq_with_smoothquant"), 3, full=False, ppl_val=9.0)
    out = extra_tries(start_run(c), [best, worse, rough], ["rtn_with_rtn_act", "awq_with_rtn_act"], 6)
    # rtn was measured on all the windows; awq only on fewer, so it gets its try: the best try with awq in Stage 1
    assert [p for p, _ in seen] == [{**best["point"], "stage1_method": "awq_with_rtn_act"}] and seen[0][1] is None
    assert out[0]["searcher"] == "extra" and out[0]["met"]
    # no try met the limit: the run's own settings
    seen.clear()
    extra_tries(start_run(c), [{**worse, "met": False}], ["awq_with_rtn_act"], 6)
    assert seen[0][0] == technique_space(c).make({"stage1_method": "awq_with_rtn_act"})
    report = write_search_report(start_run(c), [best, worse], technique_space(c), extras=out)["search_report"]
    assert "## Extra Stage 1 techniques" in report.read_text() and "| W4.5A8 |" in report.read_text()


def test_the_lowest_perplexity_final_model_is_tested_for_accuracy_next_to_the_uncompressed_one(  # noqa: F811
        cfg, tiny_llama, tokenizer, monkeypatch):
    calls = []

    def fake_accuracy(model, tok, tasks, limit, batch_size, seed):
        calls.append(tasks)
        return {"acc_arc_easy": 0.5 if len(calls) == 1 else 0.4, "downstream_acc_mean": 0.5 if len(calls) == 1 else 0.4}

    monkeypatch.setattr(downstream, "downstream_accuracy", fake_accuracy)
    c = cfg.with_overrides({"requirement.target_memory_gb": 1e9})
    ctx = start_run(c)
    better = _trial(_point(c, calib_samples=16, sensitive_threshold=0.2), 1, ppl_val=10.0)
    worse = _trial(_point(c, calib_samples=16, sensitive_threshold=0.8), 2, ppl_val=12.0)
    over = _trial(_point(c, calib_samples=16), 3, met=False, ppl_val=9.0)
    assert final_accuracy(ctx, [better], [], **_kw(tiny_llama, tokenizer)) is None  # no task: no test
    assert final_accuracy(ctx, [over], ["arc_easy"], **_kw(tiny_llama, tokenizer)) is None
    acc = final_accuracy(ctx, [worse, over, better], ["arc_easy"], **_kw(tiny_llama, tokenizer))
    assert acc["n"] == 1 and calls == [["arc_easy"], ["arc_easy"]]
    assert acc["metrics"]["acc_arc_easy"] == 0.4 and acc["fp16"] == {"acc_arc_easy": 0.5, "downstream_acc_mean": 0.5}
    text = write_search_report(ctx, [worse, better], technique_space(c), accuracy=acc)["search_report"].read_text()
    assert "## Accuracy of the final model" in text and text.count("| 0.4 | 0.5 | -0.1 | -20 |") == 2  # + the mean
    # a rerun in a new run folder finds the point cached, builds nothing, and still has the uncompressed accuracy
    again = final_accuracy(start_run(c.with_overrides({"run.run_id": "again"})), [better], ["arc_easy"],
                           **_kw(tiny_llama, tokenizer))
    assert len(calls) == 2 and again["fp16"] == acc["fp16"] and again["metrics"] == acc["metrics"]
    assert json.dumps(acc, default=str)


def test_a_failing_after_step_drops_only_itself(cfg, monkeypatch):  # noqa: F811
    from test_search import _fake_evaluate

    monkeypatch.setattr(search, "evaluate", _fake_evaluate(over=lambda p: False))
    monkeypatch.setattr(search, "extra_tries", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(search, "final_accuracy", lambda *a, **k: {"searcher": "mobo", "n": 1, "fp16": {},
                                                                   "metrics": {"acc_arc_easy": 0.4}})
    c = cfg.with_overrides({"search.budget": 1, "search.searchers": ["mobo"]})
    text = search.run_search(start_run(c))["search_report"].read_text()
    assert "## Accuracy of the final model" in text and "Extra Stage 1" not in text


def test_the_search_checks_its_after_steps_before_any_try(cfg, monkeypatch):  # noqa: F811
    with pytest.raises(ValueError, match="extra_stage1"):
        search.run_search(start_run(cfg.with_overrides({"search.extra_stage1": ["awq_with_rtn_act"]})))
    monkeypatch.setitem(sys.modules, "lm_eval", None)  # not installed
    with pytest.raises(ValueError, match="lm-eval"):
        search.run_search(start_run(cfg.with_overrides({"search.accuracy_tasks": ["arc_easy"]})))
