import copy
import json

import pytest

from sdf.run import start_run
from sdf.search_space import SEARCH_SPACE
from sdf.stage0.run import run_stage0
from sdf.stages.methods import METHODS, methods_for
from sdf.stages.runner import Stage0Plans, run_stage, run_stages
from conftest import fake_texts


@pytest.fixture
def stage0_dir(tiny_llama, tokenizer, small_cfg):
    ctx = start_run(small_cfg)
    cand = SEARCH_SPACE.make({"calib_samples": 16})
    run_stage0(ctx, cand, model=copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    return ctx.run_dir / "stage_0"


def test_registry_lists_every_spec_method():
    spec = {1: {"gptq", "awq", "structured_prune", "unstructured_prune", "low_rank"},
            2: {"smoothquant", "quarot", "rptq", "spinquant"},
            3: {"quarot_kv", "kvquant", "h2o", "snapkv", "infinigen"}}
    for stage, names in spec.items():
        assert names <= {m.name for m in METHODS.values() if m.stage == stage}
    with pytest.raises(ValueError):
        methods_for(1, ["h2o"])  # a Stage 3 method


def test_stage0_plans_load(stage0_dir):
    plans = Stage0Plans.load(stage0_dir)
    assert {"weights", "weights_same_size", "activations", "kv", "kv_bits_only"} <= set(plans.plans)
    assert plans.num_layers == 4 and plans.kv_profile is not None
    with pytest.raises(FileNotFoundError):
        Stage0Plans.load(stage0_dir.parent)


@pytest.mark.parametrize("stage,method", [(1, "rtn"), (2, "rtn_act"), (3, "rtn_kv")])
def test_run_stage_end_to_end(stage0_dir, tiny_llama, tokenizer, small_cfg, stage, method):
    ctx = start_run(small_cfg)
    cand = SEARCH_SPACE.make({"calib_samples": 16})
    pristine = {k: v.clone() for k, v in tiny_llama.state_dict().items()}
    unimplemented = next(m.name for m in METHODS.values() if m.stage == stage and m.apply is None)
    kwargs = dict(model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    out = run_stage(ctx, stage, [method, unimplemented], cand, stage0_dir, **kwargs)

    res = json.loads(out["json"].read_text())
    rows = {f"{r['method']}/{r['variant']}": r for r in res["rows"]}
    assert rows["baseline/fp16"]["info"]["cached"]  # same cache entry Stage 0 measured
    assert rows[f"{method}/original"]["status"] == "ok"
    assert rows[f"{method}/framework"]["status"] == "ok"
    for r in rows.values():
        if r["status"] == "ok":
            assert r["metrics"]["ppl_val"] > 1 and r["metrics"]["ppl_heldout"] > 1
    # a method that is not written yet is a failed row saying so, not a crash or a silent gap
    assert rows[f"{unimplemented}/original"]["status"] == "failed"
    assert "not implemented" in rows[f"{unimplemented}/original"]["error"]
    # framework rows are compared with the original row of their method
    assert "vs_original_abs" in rows[f"{method}/framework"]["deltas"]["ppl_val"]
    assert out["report"].exists() and out["xlsx"].exists() and out["report"].parent.name == f"stage_{stage}"
    # every variant starts from a fresh model: the source model is untouched
    assert all((v == tiny_llama.state_dict()[k]).all() for k, v in pristine.items())

    if stage == 1:
        assert rows["rtn_same_size/framework"]["info"]["compare_to"] == "rtn"
        assert "model_size_gb" not in rows["rtn/framework"]["metrics"]  # simulated: FP16 in memory
        assert rows["rtn/original"]["metrics"]["avg_bits_per_weight"] < 16
    if stage == 3:
        assert rows["rtn_kv/framework"]["metrics"]["predicted_kv_memory_gb"] < \
            rows["baseline/fp16"]["metrics"]["predicted_kv_memory_gb"]

    # the original row is cached by config: a second run reuses it
    out2 = run_stage(ctx, stage, [method], cand, stage0_dir, **kwargs)
    rows2 = {f"{r['method']}/{r['variant']}": r for r in json.loads(out2["json"].read_text())["rows"]}
    assert rows2[f"{method}/original"]["info"]["cached"]


def test_run_stages_uses_config_methods(stage0_dir, tiny_llama, tokenizer, small_cfg):
    cfg = small_cfg.with_overrides({"stages.stage2_methods": [], "stages.stage3_methods": []})
    out = run_stages(start_run(cfg), SEARCH_SPACE.make({"calib_samples": 16}), stage0_dir,
                     model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    assert set(out) == {"stage_1_report", "stage_1_xlsx", "stage_1_json"}


def test_downstream_tasks_and_master_report(stage0_dir, tiny_llama, tokenizer, small_cfg, monkeypatch):
    from openpyxl import load_workbook

    from sdf.eval import downstream
    from sdf.reporting.master import write_master
    from sdf.reporting.metrics import METRICS

    calls = []

    def fake_accuracy(model, tok, tasks, limit, batch_size, seed):
        calls.append(tasks)
        return {**{downstream.task_metric(t): 0.5 for t in tasks}, "downstream_acc_mean": 0.5}

    monkeypatch.setattr(downstream, "downstream_accuracy", fake_accuracy)
    cfg = small_cfg.with_overrides({"eval.downstream_tasks": ["piqa"], "stages.stage2_methods": [],
                                    "stages.stage3_methods": []})
    ctx = start_run(cfg)
    run_stages(ctx, SEARCH_SPACE.make({"calib_samples": 16}), stage0_dir,
               model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    rows = json.loads((ctx.run_dir / "stage_1" / "results.json").read_text())["rows"]
    assert all(r["metrics"]["acc_piqa"] == 0.5 for r in rows if r["status"] == "ok")
    assert METRICS["acc_piqa"].better == "higher"

    out = write_master(ctx.run_dir)  # the fixture's Stage 0 ran in the same run folder
    text = out["master_report"].read_text()
    assert "Stage 1: Weight compression" in text and "(lower is better)" in text and "(higher is better)" in text
    ws = load_workbook(out["all_stages_xlsx"])["All stages"]
    s0_rows = json.loads((stage0_dir / "results.json").read_text())["rows"]
    assert ws.max_row == 1 + len(s0_rows) + len(rows)
    assert "Stage 0:" in text


def test_fp16_cache_miss_builds_the_model_with_the_factory(stage0_dir, tiny_llama, tokenizer, small_cfg, tmp_path):
    ctx = start_run(small_cfg.with_overrides({"run.cache_dir": str(tmp_path / "fresh_cache")}))
    out = run_stage(ctx, 1, ["rtn"], SEARCH_SPACE.make({"calib_samples": 16}), stage0_dir,
                    model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    fp16 = next(r for r in json.loads(out["json"].read_text())["rows"] if r["variant"] == "fp16")
    assert fp16["status"] == "ok" and not fp16["info"]["cached"]  # measured on the factory's model
