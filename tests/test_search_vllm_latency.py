"""The search's vLLM latency limit and weight-only Stage 1 techniques (Mohammad, 2026-10-29: hard limits val
perplexity <= 11.5 and vLLM decode < 2.5 ms per token, "we need to beat the uncompressed version")."""

import copy
from pathlib import Path

import pytest
import torch

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
    import json
    rows = json.loads((Path(t["dir"]) / "stage_3" / "results.json").read_text())["rows"]
    fw = next(x for x in rows if x["variant"] == "framework")
    assert fw["requirement"]["met"] is True  # the point's own report checks the vLLM latency too, not HF's
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
    assert "vLLM decode time per token at most 2.5" in text  # the limit is labelled with the vLLM metric


# ---------------------------------------------------------------- pre-merge review fixes (2026-10-29)

def test_stage_reports_check_the_latency_limit_on_the_search_latency_metric(tmp_path):
    from sdf.reporting.reporter import StageReporter
    from sdf.requirements import DeploymentRequirement

    def rep(**kw):
        return StageReporter(stage=1, run_dir=tmp_path, title="t", config={}, environment={}, conditions={},
                             requirement=DeploymentRequirement(target_latency_ms=2.5), main_metrics=["ppl_val"],
                             **kw)

    vllm = rep(latency_metric=VLLM)
    with vllm.method("gptq", "framework") as r:  # HF decode ~30 ms, vLLM 2.0: the vLLM value decides
        r.metrics.update(ppl_val=10.0, decode_ms_per_token_mean=30.0, **{VLLM: 2.0})
    with vllm.method("awq", "framework") as r:  # no vLLM run: unmeasured, never the HF value
        r.metrics.update(ppl_val=10.0, decode_ms_per_token_mean=30.0)
    assert vllm.rows[0].requirement["met"] is True
    assert vllm.rows[0].requirement["checked"]["target_latency_ms"] == "decode_ms_per_token_mean"
    assert vllm.rows[1].requirement["met"] is None and vllm.rows[1].requirement["unmeasured"] == ["target_latency_ms"]
    hf = rep()  # the default: the HF latency, as before
    with hf.method("gptq", "framework") as r:
        r.metrics.update(ppl_val=10.0, decode_ms_per_token_mean=30.0, **{VLLM: 2.0})
    assert hf.rows[0].requirement["met"] is False


def test_a_try_whose_vllm_latency_failed_violates_the_limit_for_mobo(cfg, monkeypatch):
    # the latency limit is on vLLM but vLLM is no objective: the try is "ok", yet the limit was not measured
    monkeypatch.setattr(search, "_build_and_measure", lambda *a: {
        "metrics": {"ppl_val": 10.0, "predicted_weight_memory_gb": 0.5}, "errors": {"vllm_error": "out of memory"}})
    ctx = start_run(cfg)
    t = search.Recorder(ctx, "mobo", 6)(technique_space(cfg).make({}))
    assert t["ok"] and not t["met"] and t["violation"] == 1.0  # never 0: MOBO would take it as met
    assert search.violation({"met": False, "shortfall": {"target_ppl": 0.5}, "unmeasured": ["target_latency_ms"]}) == 1.5
    assert search.violation({"met": True, "shortfall": {}, "unmeasured": []}) == 0.0
    assert search.violation({"met": False, "shortfall": {}, "unmeasured": []}) == 0.0  # no limits set: as before


def test_a_try_whose_backend_failed_is_never_reused_from_the_tried_log(cfg):
    ctx = start_run(cfg)
    point = technique_space(cfg).make({})
    m = {"ppl_val": 10.0, "ppl_heldout": 10.2, "predicted_weight_memory_gb": 0.5, VLLM: 2.0}

    def logged(r):
        path = ctx.run_dir / "tried.csv"
        if path.exists():
            path.unlink()
        log = search.TriedLog(path, "old")
        log.add(ctx, "mobo", {"n": 1, "point": point, "windows": 6, "over_limit": False, "ok": True, "met": False,
                              "metrics": r["metrics"]}, r)
        return search.TriedLog(path, "new")

    assert logged({"metrics": m, "errors": {}}).get(ctx, point, 6) is not None  # a good try is reused
    failed = logged({"metrics": {k: v for k, v in m.items() if k != VLLM}, "errors": {"vllm_error": "out of memory"}})
    assert failed.get(ctx, point, 6) is None  # retried, like a crash
    assert failed.known(ctx, 6, ["predicted_weight_memory_gb", "ppl_val"]) == []  # never a final-model candidate


def test_main_py_builds_a_valid_search_config(monkeypatch, tmp_path):
    import main
    from sdf.stage0 import run as stage0_run

    c = main.build_config()
    assert c.search.searchers == ["mobo"] and c.requirement.target_latency_ms == 2.5 and c.requirement.target_ppl == 11.5
    # the final model: lowest perplexity under both limits, then vLLM latency, then size (Mohammad, 2026-10-29: "b")
    assert c.search.objectives == ["ppl_val", VLLM, "predicted_weight_memory_gb"] and c.search.latency_metric == VLLM
    assert c.search.low_threshold_max == 0 and c.requirement.target_memory_gb is None and not c.search.joint_plan
    space = technique_space(c)
    weight_only = ["rtn", "gptq", "awq", "omniquant", "efficientqat"]
    for m in weight_only + c.search.stage1_methods:
        space.validate({**space.make({}), "stage1_method": m})
    assert set(weight_only) <= set(c.search.stage1_methods) and "gptq_with_rtn_act" in c.search.stage1_methods
    with pytest.raises(ValueError, match="vllm_python"):  # main.py's VLLM_PYTHON is None until it is set
        run_search(start_run(c := c.with_overrides({"run.output_root": str(tmp_path / "results"),
                                                     "run.cache_dir": str(tmp_path / "cache")})))

    class Passed(Exception):
        pass

    def setup(*a, **k):  # the first step after the checks: they all passed
        raise Passed

    monkeypatch.setattr(stage0_run, "setup", setup)
    monkeypatch.setitem(__import__("sys").modules, "lm_eval", object())  # SEARCH_ACCURACY_TASKS needs lm-eval
    with pytest.raises(Passed):
        run_search(start_run(c.with_overrides({"eval.vllm_python": "/content/vllm-venv/bin/python"})))


# ---------------------------------------------------------------- the vLLM perplexity limit, no KV method (2026-10-31)

VLLM_PPL = "vllm_ppl_val"


def test_the_perplexity_limit_is_checked_on_the_search_ppl_metric(cfg):
    c = cfg.with_overrides({"search.ppl_metric": VLLM_PPL, "requirement.target_latency_ms": None})
    base = {"ppl_val": 10.0, "predicted_weight_memory_gb": 0.5}  # HF under the limit
    assert check_requirement(c, {**base, VLLM_PPL: 11.0}).met is True
    over = check_requirement(c, {**base, VLLM_PPL: 12.0})
    assert over.met is False and over.shortfall["target_ppl"] == pytest.approx(0.5)
    missing = check_requirement(c, base)  # vLLM failed: the HF perplexity must not stand in for it
    assert missing.met is None and missing.unmeasured == ["target_ppl"]
    # the default is the HF perplexity, as before
    hf = c.with_overrides({"search.ppl_metric": "ppl_val"})
    assert check_requirement(hf, {**base, "ppl_val": 12.0, VLLM_PPL: 11.0}).met is False


def test_a_try_without_the_vllm_perplexity_violates_the_limit_for_mobo(cfg, monkeypatch):
    c = cfg.with_overrides({"search.ppl_metric": VLLM_PPL, "requirement.target_latency_ms": None})
    monkeypatch.setattr(search, "_build_and_measure", lambda *a: {
        "metrics": {"ppl_val": 10.0, "predicted_weight_memory_gb": 0.5}, "errors": {"vllm_error": "out of memory"}})
    t = search.Recorder(start_run(c), "mobo", 6)(technique_space(c).make({}))
    assert t["ok"] and not t["met"] and t["violation"] == 1.0


def test_a_vllm_perplexity_limit_makes_the_tries_run_vllm(cfg):
    off = {"requirement.target_latency_ms": None}
    assert not tries_need_vllm(cfg.with_overrides(off))
    assert tries_need_vllm(cfg.with_overrides({**off, "search.ppl_metric": VLLM_PPL}))
    # no perplexity limit: nothing to check on vLLM
    assert not tries_need_vllm(cfg.with_overrides({**off, "search.ppl_metric": VLLM_PPL,
                                                   "requirement.target_ppl": None}))
    with pytest.raises(ValueError, match="vllm_python"):
        run_search(start_run(cfg.with_overrides({**off, "search.ppl_metric": VLLM_PPL,  # only the ppl metric is vLLM
                                                 "search.latency_metric": "decode_ms_per_token_mean"})))
    with pytest.raises(ValueError, match="ppl_heldout"):
        run_search(start_run(cfg.with_overrides({"search.ppl_metric": "ppl_heldout"})))


def test_a_diverged_vllm_number_keeps_the_constraint_finite(cfg):
    c = cfg.with_overrides({"search.ppl_metric": VLLM_PPL, "requirement.target_latency_ms": None})
    check = check_requirement(c, {"ppl_val": 10.0, "predicted_weight_memory_gb": 0.5, VLLM_PPL: float("inf")})
    assert check.met is False
    assert search.violation({"met": False, "shortfall": check.shortfall, "unmeasured": check.unmeasured}) == 1.0


def test_stage_search_stops_when_a_limit_needs_vllm_it_does_not_have(cfg):
    from sdf.stage_search import run_stage_search

    c = cfg.with_overrides({"search.objectives": ["ppl_val", "predicted_weight_memory_gb"],
                            "search.latency_metric": "decode_ms_per_token_mean", "search.ppl_metric": VLLM_PPL})
    with pytest.raises(ValueError, match="vllm_python"):
        run_stage_search(start_run(c))


def test_main_py_and_the_stage_reporter_get_the_ppl_metric(cfg, tiny_llama, tokenizer, monkeypatch):
    import main
    from sdf.stage0.run import stage_reporter
    import inspect

    monkeypatch.setattr(main, "SEARCH_PPL_METRIC", VLLM_PPL)
    assert main.build_config().search.ppl_metric == VLLM_PPL
    assert "ppl_metric=cfg.search.ppl_metric" in inspect.getsource(stage_reporter)


def test_stage_reports_and_the_search_report_use_the_ppl_metric(cfg, tmp_path):
    from sdf.reporting.reporter import StageReporter
    from sdf.reporting.search import _req
    from sdf.requirements import DeploymentRequirement

    rep = StageReporter(stage=1, run_dir=tmp_path, title="t", config={}, environment={}, conditions={},
                        requirement=DeploymentRequirement(target_ppl=11.5), main_metrics=["ppl_val"],
                        ppl_metric=VLLM_PPL)
    with rep.method("gptq", "framework") as r:
        r.metrics.update(ppl_val=12.0, **{VLLM_PPL: 11.0})  # only the vLLM value counts
    with rep.method("awq", "framework") as r:
        r.metrics.update(ppl_val=11.0)
    assert rep.rows[0].requirement["met"] is True
    assert rep.rows[1].requirement["unmeasured"] == ["target_ppl"]
    text = _req(cfg.with_overrides({"search.ppl_metric": VLLM_PPL}))
    assert "vLLM perplexity" in text and "at most 11.5" in text


def test_no_kv_method_passes_the_stage2_model_on_unchanged(cfg, tiny_llama, tokenizer):
    # "none" as one choice: built and measured, cached under its own key, with the uncompressed cache's KV memory
    c = cfg.with_overrides({"search.latency_metric": "decode_ms_per_token_mean", "requirement.target_latency_ms": None,
                            "requirement.target_ppl": None, "search.stage1_methods": ["rtn_with_rtn_act"],
                            "search.stage3_methods": ["none", "rtn_kv"]})
    ctx = start_run(c)
    kw = _kw(tiny_llama, tokenizer)
    over = {"calib_samples": 16, "prune_ratio_aggressive": 0.4}
    none = evaluate(ctx, technique_space(c).make({**over, "stage3_method": "none"}), **kw)
    kv = evaluate(ctx, technique_space(c).make({**over, "stage3_method": "rtn_kv"}), **kw)
    assert "error" not in none and "error" not in kv, (none, kv)
    assert none["dir"] != kv["dir"]
    assert evaluate(ctx, technique_space(c).make({**over, "stage3_method": "none"}), **kw)["cached"]
    m, k = none["metrics"], kv["metrics"]
    assert "No KV method" in (Path(none["dir"]) / "stage_3" / "report.md").read_text(encoding="utf-8")
    assert m["kv_kept_share"] == 1.0 and m["avg_kv_bits"] == 16 and m["predicted_kv_memory_gb"] > k["predicted_kv_memory_gb"]
    assert m["decoder_weights_removed"] > 0 and m["predicted_weight_memory_gb"] == k["predicted_weight_memory_gb"]


def test_no_kv_method_alone_hooks_nothing_and_needs_no_kv_plan(cfg, tiny_llama, tokenizer, monkeypatch):
    from sdf.stages import methods
    from sdf.stages.methods import METHODS, MethodCall

    # the model comes out of "none" answering exactly as it went in, with nothing hooked
    x = torch.arange(8).unsqueeze(0)
    model = copy.deepcopy(tiny_llama).eval()
    with torch.no_grad():
        before = model(x).logits.clone()
        with METHODS["none"].apply(MethodCall(model, None, {}, cfg, lambda: [])) as extra:
            assert extra == {} and not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
            assert torch.equal(model(x).logits, before)

    def refuse(*a, **k):
        raise AssertionError("no KV technique may run")

    monkeypatch.setattr(methods, "kv_attention", refuse)  # what every real KV technique enters
    c = cfg.with_overrides({"search.latency_metric": "decode_ms_per_token_mean", "requirement.target_latency_ms": None,
                            "requirement.target_ppl": None, "search.stage1_methods": ["rtn_with_rtn_act"],
                            "search.stage3_methods": ["none"], "stage0.kv_cache": False})
    r = evaluate(start_run(c), technique_space(c).make({"calib_samples": 16, "prune_ratio_aggressive": 0.4}),
                 **_kw(tiny_llama, tokenizer))
    assert "error" not in r, r
    assert r["metrics"]["decoder_weights_removed"] > 0 and r["metrics"]["ppl_val"] > 1  # the Stage 2 model, measured


def test_no_kv_method_is_reused_from_the_tried_log(cfg, tmp_path):
    c = cfg.with_overrides({"search.stage3_methods": ["none"]})
    space = technique_space(c)
    assert [p.choices for p in space.params if p.name == "stage3_method"] == [("none",)]
    ctx = start_run(c)
    point = space.make({})
    m = {"ppl_val": 10.0, "ppl_heldout": 10.2, "predicted_weight_memory_gb": 0.5, VLLM: 2.0}
    path = tmp_path / "tried.csv"
    search.TriedLog(path, "old").add(ctx, "mobo", {"n": 1, "point": point, "windows": 6, "over_limit": False,
                                                   "ok": True, "met": True, "metrics": m}, {"metrics": m, "errors": {}})
    assert search.TriedLog(path, "new").get(ctx, point, 6) is not None


# ---------------------------------------------------------------- both perplexity limits (2026-10-31)

def _both(cfg):
    return cfg.with_overrides({"requirement.target_vllm_ppl": 11.5, "requirement.target_latency_ms": None})


def test_both_perplexity_limits_must_hold(cfg):
    c = _both(cfg)
    size = {"predicted_weight_memory_gb": 0.5}
    over = check_requirement(c, {**size, "ppl_val": 12.3, VLLM_PPL: 11.3})  # passed with search.ppl_metric alone
    assert over.met is False and set(over.shortfall) == {"target_ppl"}
    assert check_requirement(c, {**size, "ppl_val": 11.4, VLLM_PPL: 11.3}).met is True
    assert set(check_requirement(c, {**size, "ppl_val": 11.4, VLLM_PPL: 11.6}).shortfall) == {"target_vllm_ppl"}
    missing = check_requirement(c, {**size, "ppl_val": 11.4})  # no HF fallback
    assert missing.met is None and missing.unmeasured == ["target_vllm_ppl"]
    assert search.violation({"met": False, "shortfall": missing.shortfall, "unmeasured": missing.unmeasured}) == 1.0
    # unset, as before
    off = cfg.with_overrides({"requirement.target_latency_ms": None})
    assert check_requirement(off, {**size, "ppl_val": 11.4}).met is True


def test_a_try_without_the_vllm_perplexity_misses_both_limits_for_mobo(cfg, monkeypatch):
    c = _both(cfg)
    monkeypatch.setattr(search, "_build_and_measure", lambda *a: {
        "metrics": {"ppl_val": 11.4, "predicted_weight_memory_gb": 0.5}, "errors": {"vllm_error": "out of memory"}})
    t = search.Recorder(start_run(c), "mobo", 6)(technique_space(c).make({}))
    assert t["ok"] and not t["met"] and t["violation"] == 1.0


def test_the_vllm_perplexity_limit_needs_vllm_in_every_try(cfg, tiny_llama, tokenizer, monkeypatch):
    from sdf.reporting.search import _req
    from sdf.stage_search import run_stage_search

    hf = {"search.latency_metric": "decode_ms_per_token_mean"}
    c = _both(cfg).with_overrides(hf)
    assert tries_need_vllm(c) and not tries_need_vllm(cfg.with_overrides({**hf, "requirement.target_latency_ms": None}))
    with pytest.raises(ValueError, match="target_vllm_ppl"):
        run_search(start_run(c))
    with pytest.raises(ValueError, match="vllm_python"):
        run_stage_search(start_run(c.with_overrides({"search.objectives": ["ppl_val", "predicted_weight_memory_gb"]})))
    seen = []
    monkeypatch.setitem(search.SEARCHERS, "mobo", lambda rec, *a, **k: seen.append(rec.ctx.cfg.eval.vllm_python))
    monkeypatch.setattr(search_report, "write_search_report", lambda *a, **k: {})
    run_search(start_run(c.with_overrides({"eval.vllm_python": "py", "search.searchers": ["mobo"]})),
               **_kw(tiny_llama, tokenizer))
    assert seen == ["py"]
    text = _req(c)
    assert "at most 11.5" in text and "vLLM perplexity" in text and "alidation" in text


def test_main_py_passes_the_vllm_perplexity_limit(monkeypatch):
    import main

    assert main.build_config().requirement.target_vllm_ppl is None
    monkeypatch.setattr(main, "TARGET_VLLM_PPL", 11.5)
    assert main.build_config().requirement.target_vllm_ppl == 11.5
