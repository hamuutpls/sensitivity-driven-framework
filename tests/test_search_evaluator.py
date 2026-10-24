"""Recorder with its own evaluator: what a per-stage search passes instead of the chained evaluate()."""

from sdf.run import start_run
from sdf.search import Recorder, run_mobo, technique_space
from test_search import _fake_evaluate, _point, cfg  # noqa: F401 - the cfg fixture


def test_recorder_uses_the_evaluator_it_is_given(cfg):  # noqa: F811
    calls = []
    fake = _fake_evaluate(over=lambda p: False)

    def evaluator(ctx, point, max_windows=None, **kw):
        calls.append((max_windows, kw))
        return fake(ctx, point, max_windows)

    rec = Recorder(start_run(cfg), "x", 6, evaluator=evaluator, extra=1)
    t = rec(_point(cfg, sensitive_threshold=0.2))
    assert t["ok"] and t["cost"] == 1 and calls == [(None, {"extra": 1})]  # all windows: None, as evaluate() gets
    rec(_point(cfg, sensitive_threshold=0.2), 3)
    assert calls[-1][0] == 3
    run_mobo(rec, technique_space(cfg), budget=4, seed=0)  # a searcher drives it like the default
    assert len(calls) == len(rec.trials) and rec.spent >= 4
