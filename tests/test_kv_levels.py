"""KV-cache levels in the search (Mohammad, 2026-11-24: "start with small kv cache compression options, if the
accuracy doesnt reach 0.6 then we do no kv step"; 2026-11-25 he agreed to the order light, medium, strong, none)."""

import json

import pytest

from sdf import search
from sdf.config import FrameworkConfig
from sdf.run import start_run
from sdf.search import TriedLog, kv_level_configs
from conftest import fake_texts
from test_search import _point, cfg  # noqa: F401 - the cfg fixture

LEVELS = [
    {"name": "light", "stage0.kv_avg_bits": 8, "stage0.kv_attention_coverage": 0.99},
    {"name": "medium", "stage0.kv_avg_bits": 4, "stage0.kv_attention_coverage": 0.98},
    {"name": "strong", "stage0.kv_avg_bits": None, "stage0.kv_attention_coverage": 0.95},
    {"name": "no KV step", "search.stage3_methods": ["none"]},
]


def _fake(seen, met_when):
    """evaluate() without building: every try is measured; it meets the limits when `met_when(cfg)`."""
    def fake(ctx, point, max_windows=None, **kw):
        s0 = ctx.cfg.stage0
        seen.append((s0.kv_avg_bits, s0.kv_attention_coverage, point["stage3_method"]))
        met = met_when(ctx.cfg)
        return {"metrics": {"ppl_val": 10 + point["sensitive_threshold"], "decode_ms_per_token_mean": 1.0,
                            "predicted_weight_memory_gb": 0.5 + point["sensitive_threshold"]},
                "errors": {}, "met": met, "shortfall": {} if met else {"min_accuracy": 0.1}, "dir": "d",
                "cached": False}
    return fake


def _search(cfg, budget=2, **over):  # noqa: F811
    return cfg.with_overrides({"search.searchers": ["mobo"], "search.budget": budget, "search.kv_levels": LEVELS,
                               "search.extra_stage1": [], **over})


def _run(tokenizer, ctx):
    return search.run_search(ctx, tokenizer=tokenizer, text_loader=fake_texts)


def test_a_level_needs_a_name_and_may_change_only_the_kv_settings():
    with pytest.raises(ValueError, match="needs a name"):
        FrameworkConfig().with_overrides({"search.kv_levels": [{"stage0.kv_avg_bits": 8}]})
    with pytest.raises(ValueError, match="cannot be set"):
        FrameworkConfig().with_overrides({"search.kv_levels": [{"name": "a", "stage0.uniform_bits": 8}]})
    with pytest.raises(ValueError, match="must differ"):
        FrameworkConfig().with_overrides({"search.kv_levels": [{"name": "a"}, {"name": "a"}]})
    assert FrameworkConfig().search.kv_levels == []  # off unless asked: earlier runs' settings stay the same


def test_each_level_is_the_run_settings_with_its_overrides(cfg):  # noqa: F811
    assert kv_level_configs(cfg) == [("", cfg)]
    levels = kv_level_configs(_search(cfg))
    assert [n for n, _ in levels] == ["light", "medium", "strong", "no KV step"]
    light, none = levels[0][1], levels[3][1]
    assert (light.stage0.kv_avg_bits, light.stage0.kv_attention_coverage) == (8, 0.99)
    assert light.search.stage3_methods == cfg.search.stage3_methods
    assert none.search.stage3_methods == ["none"] and none.stage0 == cfg.stage0


def test_the_next_level_runs_only_when_no_try_of_the_one_before_met_the_limits(cfg, tokenizer, monkeypatch):  # noqa: F811
    seen = []
    # here only medium's tries meet the limits
    monkeypatch.setattr(search, "evaluate", _fake(seen, lambda c: c.stage0.kv_attention_coverage == 0.98))
    out = _run(tokenizer, start_run(_search(cfg)))
    levels = [(b, c) for b, c, _ in seen]
    assert set(levels) == {(8, 0.99), (4, 0.98)}  # light, then medium; strong and "none" never searched
    assert levels.index((4, 0.98)) > max(i for i, lv in enumerate(levels) if lv == (8, 0.99))  # in that order
    trials = json.loads(out["search_trials"].read_text())
    assert [t["kv_level"] for t in trials] == ["light"] * 2 + ["medium"] * 2
    assert [t["n"] for t in trials] == [1, 2, 3, 4]  # numbered on across the levels
    text = out["search_report"].read_text()
    assert "## KV-cache levels" in text
    assert "| light | average KV bits 8; attention kept 0.99 | 2 | 0 |" in text
    assert "| medium | average KV bits 4; attention kept 0.98 | 2 | 2 |" in text
    assert "| no KV step | Stage 3 techniques none | not searched | - |" in text


def test_no_kv_step_is_the_last_choice(cfg, tokenizer, monkeypatch):  # noqa: F811
    seen = []
    monkeypatch.setattr(search, "evaluate", _fake(seen, lambda c: c.search.stage3_methods == ["none"]))
    trials = json.loads(_run(tokenizer, start_run(_search(cfg)))["search_trials"].read_text())
    assert [t["kv_level"] for t in trials] == ["light"] * 2 + ["medium"] * 2 + ["strong"] * 2 + ["no KV step"] * 2
    assert {m for *_, m in seen[-2:]} == {"none"} and "none" not in {m for *_, m in seen[:-2]}
    assert all(t["point"]["stage3_method"] == "none" for t in trials[-2:])


def test_a_level_met_by_an_earlier_run_ends_the_search_there(cfg, tokenizer, monkeypatch):  # noqa: F811
    seen = []
    monkeypatch.setattr(search, "evaluate", _fake(seen, lambda c: c.stage0.kv_avg_bits == 8))
    c = _search(cfg)
    _run(tokenizer, start_run(c.with_overrides({"run.run_id": "a"})))
    seen.clear()  # a later run with more tries replays light's tries, which met the limits, and stops there
    trials = json.loads(_run(tokenizer, start_run(c.with_overrides({"run.run_id": "b", "search.budget": 3})))
                        ["search_trials"].read_text())
    assert {t["kv_level"] for t in trials} == {"light"} and len(seen) == 1  # only the one new try is built


def test_the_same_point_is_a_new_combination_at_each_level(cfg):  # noqa: F811
    ctxs = [start_run(lcfg) for _, lcfg in kv_level_configs(_search(cfg))[:3]]
    point = _point(ctxs[0].cfg)
    assert len({TriedLog.combination(c, point, 6) for c in ctxs}) == 3  # never reused from another level's try


def test_a_search_for_the_smallest_model_alone_writes_its_report(cfg, tokenizer, monkeypatch):  # noqa: F811
    # Mohammad, 2026-11-25, chose "a": the smallest model with accuracy at least 0.60 and decode under 2.5 ms, so
    # size is the only objective
    monkeypatch.setattr(search, "evaluate", _fake([], lambda c: c.stage0.kv_avg_bits == 4))
    out = _run(tokenizer, start_run(_search(cfg, budget=3, **{"search.objectives": ["predicted_weight_memory_gb"]})))
    trials = json.loads(out["search_trials"].read_text())
    assert {t["kv_level"] for t in trials} == {"light", "medium"} and all(len(t["values"]) == 1 for t in trials)
    assert "## KV-cache levels" in out["search_report"].read_text()


def test_a_budget_of_the_most_bits_puts_every_tensor_at_them():
    # the review of PR #101: allocate_bits left a tensor whose measured rise is already 0 at 2 bits
    from sdf.stage0.kv_cache import KVProfile, plan_kv_bits

    prof = KVProfile.__new__(KVProfile)
    object.__setattr__(prof, "bits_options", [2, 4, 8])
    object.__setattr__(prof, "key_dims", [64, 64])
    object.__setattr__(prof, "value_dims", [64, 64])
    object.__setattr__(prof, "key_rise", [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    object.__setattr__(prof, "value_rise", [[0.3, -0.01, 0.0], [0.0, 0.0, 0.0]])
    assert set(plan_kv_bits(prof, 8).values()) == {8}
    assert set(plan_kv_bits(prof, 4).values()) != {8}  # below it, the measured plan as before


def test_the_extra_tries_run_at_the_level_the_search_ends_on(cfg, tokenizer, monkeypatch):  # noqa: F811
    c = _search(cfg, budget=1, **{"search.stage1_methods": ["rtn_with_rtn_act", "awq_with_rtn_act"],
                                  "search.extra_stage1": ["awq_with_rtn_act"]})
    seen = []
    monkeypatch.setattr(search, "evaluate", _fake(seen, lambda c: c.stage0.kv_avg_bits == 4))
    asked = []
    real = search.extra_tries
    monkeypatch.setattr(search, "extra_tries", lambda ctx, trials, *a, **k: asked.append(
        (ctx.cfg.stage0.kv_avg_bits, [t["kv_level"] for t in trials])) or real(ctx, trials, *a, **k))
    _run(tokenizer, start_run(c))
    assert asked == [(4, ["medium"])]


def test_the_final_model_is_built_at_the_level_the_search_ends_on(cfg, tokenizer, monkeypatch):  # noqa: F811
    monkeypatch.setattr(search, "evaluate", _fake([], lambda c: c.stage0.kv_avg_bits == 4))
    asked = []
    monkeypatch.setattr(search, "final_model", lambda ctx, trials, *a, **k: asked.append(
        (ctx.cfg.stage0.kv_avg_bits, ctx.cfg.stage0.kv_attention_coverage)))
    _run(tokenizer, start_run(_search(cfg)))
    assert asked == [(4, 0.98)]  # medium's settings, not light's or the run's own


def test_the_levels_table_counts_earlier_runs_tries_that_met_the_limits(cfg):  # noqa: F811
    from sdf.reporting.search import _kv_levels_section

    c = _search(cfg)
    mine = [{"kv_level": "light", "full": True, "met": False}]
    logged = [{"kv_level": "light", "full": True, "met": True, "from_log": True}]
    text = "\n".join(_kv_levels_section(c, mine, logged))
    assert "| light | average KV bits 8; attention kept 0.99 | 1 | 0 (and 1 from earlier runs) |" in text
