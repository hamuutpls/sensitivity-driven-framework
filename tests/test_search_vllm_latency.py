"""The search's vLLM latency limit and weight-only Stage 1 techniques (Mohammad, 2026-10-29: hard limits val
perplexity <= 11.5 and vLLM decode < 2.5 ms per token, "we need to beat the uncompressed version")."""

import copy

import pytest

from sdf import search
from sdf.reporting import search as search_report
from sdf.run import start_run
from sdf.search import check_requirement, evaluate, run_search, technique_space, tries_need_vllm
from conftest import fake_texts

VLLM = "vllm_decode_ms_per_token"


@pytest.fixture
def cfg(small_cfg):
    return small_cfg.with_overrides({"requirement.target_ppl": 11.5, "requirement.target_latency_ms": 2.5,
                                     "search.latency_metric": VLLM,
                                     "search.objectives": ["predicted_weight_memory_gb", "ppl_val"],
                                     "search.stage1_methods": ["rtn", "gptq"], "search.joint_plan": False,
                                     "search.stage2_methods": ["unstructured_prune"],
                                     "search.stage3_methods": ["rtn_kv"]})


def _kw(tiny_llama, tokenizer):
    return dict(model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)


def test_the_latency_limit_is_checked_on_the_search_latency_metric(cfg):
    base = {"ppl_val": 10.0, "predicted_weight_memory_gb": 0.5, "decode_ms_per_token_mean": 30.0}  # HF: slow
    met = check_requirement(cfg, {**base, VLLM: 2.0})
    assert met.met is True and met.checked["target_latency_ms"] == "decode_ms_per_token_mean"
    over = check_requirement(cfg, {**base, VLLM: 2.6})
    assert over.met is False and over.shortfall["target_latency_ms"] == pytest.approx(0.1)
    missing = check_requirement(cfg, base)  # vLLM failed: the HF latency must not stand in for it
    assert missing.met is None and missing.unmeasured == ["target_latency_ms"]
    # the default metric is the HF one, as before
    hf = cfg.with_overrides({"search.latency_metric": "decode_ms_per_token_mean"})
    assert check_requirement(hf, {**base, VLLM: 2.0}).met is False


def test_a_try_without_a_vllm_latency_is_built_but_not_met(cfg, monkeypatch):
    built = []
    monkeypatch.setattr(search, "_build_and_measure", lambda *a: built.append(1) or {
        "metrics": {"ppl_val": 10.0, "predicted_weight_memory_gb": 0.5}, "errors": {"vllm_error": "no kernel"}})
    r = evaluate(start_run(cfg), technique_space(cfg).make({}))
    assert built and r["met"] is False  # met None (unmeasured) counts as not met
    monkeypatch.setattr(search, "_build_and_measure", lambda *a: {
        "metrics": {"ppl_val": 10.0, "predicted_weight_memory_gb": 0.5, VLLM: 2.4}, "errors": {}})
    assert evaluate(start_run(cfg), technique_space(cfg).make({"calib_samples": 16}))["met"] is True


def test_the_size_check_before_building_skips_only_a_model_over_a_limit(cfg):
    # before the build the latency is unknown (met None): that alone must not skip building
    assert check_requirement(cfg, {"predicted_weight_memory_gb": 0.5}).met is None
    tight = cfg.with_overrides({"requirement.target_memory_gb": 0.1})
    assert check_requirement(tight, {"predicted_weight_memory_gb": 0.5}).met is False


@pytest.mark.parametrize("over, needed", [
    ({"search.objectives": ["ppl_val", VLLM], "requirement.target_latency_ms": None,
      "search.latency_metric": "decode_ms_per_token_mean"}, True),  # a vLLM objective
    ({}, True),  # the latency limit on vLLM
    ({"requirement.target_latency_ms": None}, False),  # a vLLM latency metric with no latency limit
    ({"search.latency_metric": "decode_ms_per_token_mean"}, False),  # the limit on the HF latency
])
def test_tries_run_vllm_only_when_the_search_needs_it(cfg, tiny_llama, tokenizer, monkeypatch, over, needed):
    c = cfg.with_overrides({**over, "eval.vllm_python": "py", "eval.llamacpp_dir": "ll", "eval.trtllm_python": "trt"})
    assert tries_need_vllm(c) is needed
    seen = []
    monkeypatch.setitem(search.SEARCHERS, "mobo", lambda rec, *a, **k: seen.append(rec.ctx.cfg.eval))
    monkeypatch.setattr(search_report, "write_search_report", lambda *a, **k: {})
    run_search(start_run(c.with_overrides({"search.searchers": ["mobo"]})), **_kw(tiny_llama, tokenizer))
    (ev,) = seen
    assert ev.vllm_python == ("py" if needed else None)
    assert ev.llamacpp_dir is None and ev.trtllm_python is None  # never in a try


def test_a_vllm_metric_needs_vllm_and_other_backends_stay_forbidden(cfg):
    for over in ({"search.objectives": ["ppl_val", VLLM]}, {}):  # an objective, or the latency metric
        with pytest.raises(ValueError, match="vllm_python"):
            run_search(start_run(cfg.with_overrides(over)))
    for o in ("llamacpp_decode_ms_per_token", "trtllm_decode_ms_per_token"):
        with pytest.raises(ValueError, match="Stage 4"):
            run_search(start_run(cfg.with_overrides({"eval.vllm_python": "py", "eval.llamacpp_dir": "ll",
                                                     "search.objectives": ["ppl_val", o]})))


@pytest.mark.parametrize("method", ["rtn", "gptq"])
def test_a_weight_only_stage1_technique_builds_the_chained_model(cfg, tiny_llama, tokenizer, method):
    ctx = start_run(cfg.with_overrides({"search.latency_metric": "decode_ms_per_token_mean",
                                        "requirement.target_latency_ms": None,
                                        "requirement.target_ppl": None,  # the toy model is far above 11.5
                                        "requirement.target_memory_gb": 1.0}))
    point = technique_space(ctx.cfg).make({"calib_samples": 16, "prune_ratio_aggressive": 0.4,
                                           "stage1_method": method})
    r = evaluate(ctx, point, **_kw(tiny_llama, tokenizer))
    assert "error" not in r, r
    m = r["metrics"]
    assert m["ppl_val"] > 1 and m["predicted_weight_memory_gb"] > 0 and m["decoder_weights_removed"] > 0
    assert m.get("avg_activation_bits", 16) == 16  # weights only: activations stay 16-bit
    assert r["met"]


def test_a_weight_only_try_gets_its_vllm_latency_and_the_limit_is_checked_on_it(cfg, tiny_llama, tokenizer,
                                                                                 monkeypatch):
    from sdf.eval import vllm_backend

    seen = []

    def measure(model, tok, plan, val, held, ev, base, group, acts=None, act_group_size=128):
        seen.append((plan, acts))
        return {VLLM: 2.0 if plan is not None else 2.53, "vllm_ppl_val": 9.0, "vllm_size_gb": 0.1}

    monkeypatch.setattr(vllm_backend, "measure", measure)
    c = cfg.with_overrides({"eval.vllm_python": "py", "requirement.target_ppl": None,
                            "search.objectives": ["predicted_weight_memory_gb", "ppl_val", VLLM]})
    ctx = start_run(c)
    rec = search.Recorder(ctx, "mobo", 6, **_kw(tiny_llama, tokenizer))
    t = rec(technique_space(c).make({"calib_samples": 16, "prune_ratio_aggressive": 0.2, "stage1_method": "gptq"}))
    assert t["ok"] and t["values"][2] == 2.0 and t["met"], t["error"]
    (plan, acts), = [x for x in seen if x[0] is not None]
    assert acts is None  # weights only: the GPTQ export, no activation rounding
    monkeypatch.setattr(vllm_backend, "measure", lambda *a, **k: {**measure(*a, **k), VLLM: 2.6})
    t = rec(technique_space(c).make({"calib_samples": 16, "prune_ratio_aggressive": 0.1, "stage1_method": "gptq"}))
    assert t["ok"] and not t["met"] and t["violation"] == pytest.approx(0.1)


def test_a_mobo_search_on_three_objectives_with_the_vllm_limit_writes_the_report(cfg, tiny_llama, tokenizer,
                                                                                  monkeypatch):
    from sdf.eval import vllm_backend

    monkeypatch.setattr(vllm_backend, "measure", lambda model, tok, plan, *a, **k: {
        VLLM: 2.53 if plan is None else 2.0, "vllm_ppl_val": 9.0, "vllm_size_gb": 0.1})
    c = cfg.with_overrides({"eval.vllm_python": "py", "requirement.target_ppl": None, "search.searchers": ["mobo"],
                            "search.budget": 2, "search.objectives": ["predicted_weight_memory_gb", "ppl_val", VLLM],
                            "search.stage1_methods": ["rtn", "rtn_with_rtn_act"]})
    out = run_search(start_run(c), **_kw(tiny_llama, tokenizer))
    text = out["search_report"].read_text(encoding="utf-8")
    assert out and all(p.exists() for p in out.values()), out
    assert "mobo" in text.lower() and out["search_xlsx"].exists() and "search_hypervolume" in out
