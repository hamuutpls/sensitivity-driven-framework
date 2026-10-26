"""What the chained search does after its searchers: the extra Stage 1 techniques and the final model's accuracy."""

import json
import sys

import pytest

from sdf import search
from sdf.eval import downstream
from sdf.reporting.search import write_search_report
from sdf.run import start_run
from sdf.search import extra_tries, final_model, technique_space
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
    assert final_model(ctx, [better], [], **_kw(tiny_llama, tokenizer)) is None  # no task: no test
    assert final_model(ctx, [over], ["arc_easy"], **_kw(tiny_llama, tokenizer)) is None
    acc = final_model(ctx, [worse, over, better], ["arc_easy"], **_kw(tiny_llama, tokenizer))
    assert acc["n"] == 1 and calls == [["arc_easy"], ["arc_easy"]]
    assert acc["metrics"]["acc_arc_easy"] == 0.4 and acc["fp16"] == {"acc_arc_easy": 0.5, "downstream_acc_mean": 0.5}
    text = write_search_report(ctx, [worse, better], technique_space(c), accuracy=acc)["search_report"].read_text()
    assert "## Accuracy of the final model" in text and text.count("| 0.4 | 0.5 | -0.1 | -20 |") == 2  # + the mean
    # a rerun in a new run folder finds the point cached, builds nothing, and still has the uncompressed accuracy
    again = final_model(start_run(c.with_overrides({"run.run_id": "again"})), [better], ["arc_easy"],
                           **_kw(tiny_llama, tokenizer))
    assert len(calls) == 2 and again["fp16"] == acc["fp16"] and again["metrics"] == acc["metrics"]
    assert json.dumps(acc, default=str)


def test_the_final_model_is_the_best_on_the_first_objective(cfg, monkeypatch):  # noqa: F811
    # size first (Mohammad, 2026-10-26: "how much smaller can we get the model, while keeping 11 ppl or lower"):
    # the smallest final model under the limit goes to Stage 4, not the one with the lowest perplexity
    built = []
    monkeypatch.setattr(search, "evaluate", lambda ctx, point, **kw: built.append(point) or {"metrics": {}, "dir": "d"})
    c = cfg.with_overrides({"eval.vllm_python": "py"})
    small = {**_trial(_point(c, sensitive_threshold=0.9), 1, ppl_val=10.9), "values": [0.7, 10.9]}
    accurate = {**_trial(_point(c, sensitive_threshold=0.3), 2, ppl_val=10.2), "values": [0.95, 10.2]}
    smaller_over = {**_trial(_point(c, sensitive_threshold=0.8), 3, met=False, ppl_val=11.5), "values": [0.6, 11.5]}
    out = final_model(start_run(c), [accurate, smaller_over, small], [])
    assert out["n"] == 1 and built == [small["point"]]


def test_a_failing_after_step_drops_only_itself(cfg, monkeypatch):  # noqa: F811
    from test_search import _fake_evaluate

    monkeypatch.setattr(search, "evaluate", _fake_evaluate(over=lambda p: False))
    monkeypatch.setattr(search, "extra_tries", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(search, "final_model", lambda *a, **k: {"searcher": "mobo", "n": 1, "fp16": {},
                                                                   "earlier_run": "b30",
                                                                   "metrics": {"acc_arc_easy": 0.4}})
    c = cfg.with_overrides({"search.budget": 1, "search.searchers": ["mobo"]})
    text = search.run_search(start_run(c))["search_report"].read_text()
    assert "## Accuracy of the final model" in text and "Extra Stage 1" not in text
    assert "MOBO try #1 of the earlier run b30" in text  # a model an earlier run built says which run


def test_tries_run_no_backend_and_the_final_model_runs_them_in_stage4(cfg, monkeypatch):  # noqa: F811
    seen = []

    def fake(ctx, point, max_windows=None, **kw):
        seen.append((ctx.cfg.eval.vllm_python, list(ctx.cfg.eval.downstream_tasks), point["stage1_method"]))
        if point["stage1_method"].startswith("awq") and ctx.cfg.eval.vllm_python is None:
            return {"error": "boom", "cached": False, "dir": "d"}  # never built in a try, so the extra try runs it
        vllm = {"vllm_decode_ms_per_token": 3.0} if ctx.cfg.eval.vllm_python else {}
        return {"metrics": {"ppl_val": 10.0 + point["sensitive_threshold"], "predicted_weight_memory_gb": 0.5, **vllm},
                "met": True, "shortfall": {}, "dir": "d", "cached": False}

    monkeypatch.setattr(search, "evaluate", fake)
    c = cfg.with_overrides({"search.budget": 2, "search.searchers": ["mobo"], "eval.vllm_python": "py",
                            "eval.downstream_tasks": ["arc_easy"],
                            "search.objectives": ["ppl_val", "predicted_weight_memory_gb"],
                            "search.stage1_methods": ["rtn_with_rtn_act", "awq_with_rtn_act"],
                            "search.extra_stage1": ["awq_with_rtn_act"], "search.accuracy_tasks": []})
    out = search.run_search(start_run(c))
    tries, last = seen[:-1], seen[-1]
    assert any(m.startswith("awq") for _, _, m in tries)  # the extra try ran
    assert all(v is None and t == [] for v, t, _ in tries)  # tries and extra tries: no backend, no accuracy test
    assert last[0] == "py"  # the final model: vLLM
    text = out["search_report"].read_text()
    assert "## The final model on the backends (Stage 4)" in text and "| 3 |" in text
    assert "as tokens per second | 333" in text  # 3 ms per token next to its 333 tokens/s
    # nothing to test in Stage 4 (no backend, no task): no final step
    assert final_model(start_run(cfg), [_trial(_point(cfg), 1, ppl_val=10.0)], []) is None


def test_the_search_checks_its_after_steps_before_any_try(cfg, monkeypatch):  # noqa: F811
    with pytest.raises(ValueError, match="extra_stage1"):
        search.run_search(start_run(cfg.with_overrides({"search.extra_stage1": ["awq_with_rtn_act"]})))
    monkeypatch.setitem(sys.modules, "lm_eval", None)  # not installed
    with pytest.raises(ValueError, match="lm-eval"):
        search.run_search(start_run(cfg.with_overrides({"search.accuracy_tasks": ["arc_easy"]})))
