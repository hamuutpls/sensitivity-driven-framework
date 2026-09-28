import json

import pytest
import torch
from openpyxl import load_workbook

from sdf.data import calibration_batches, eval_windows
from sdf.eval import measure_model, model_size_gb
from sdf.run import start_run
from sdf.search_space import PER_CHANNEL, SEARCH_SPACE
from sdf.stage0 import (
    SensitivityProfile,
    baseline_cost,
    budget_matched_plan,
    normalize,
    outlier_layers,
    plan_compression,
    predict_cost,
    profile_sensitivity,
    run_stage0,
    uniform_plan,
)
from conftest import fake_texts


def toy_profile():
    return SensitivityProfile(scores=[0.0, 0.3, 0.5, 1.0], raw_scores=[1, 2, 3, 4],
                              layer_numel=[1000] * 4, layer_rows=[10] * 4, other_numel=500)


def test_normalize():
    assert normalize([2.0, 4.0, 3.0], "minmax") == [0.0, 1.0, 0.5]
    assert normalize([2.0, 9.0, 3.0]) == [0.0, 1.0, 0.5]  # rank ignores magnitudes
    assert normalize([1.0, 2.0, 2.0, 3.0]) == [0.0, 0.5, 0.5, 1.0]  # ties share their average rank
    assert normalize([5.0, 5.0]) == [0.0, 0.0] and normalize([5.0, 5.0], "minmax") == [0.0, 0.0]
    with pytest.raises(ValueError):
        normalize([1.0, 2.0], "zscore")


def test_outlier_layer_does_not_protect_everything():
    # TinyLlama on WikiText-2: layer 0 is a low outlier, the other 21 layers sit in 6596..8647.
    raw = [2705.0] + [6596.0 + i * (8647.0 - 6596.0) / 20 for i in range(21)]

    minmax = plan_compression(SensitivityProfile(normalize(raw, "minmax"), raw), 0.5, 0.3, 8, 4)
    assert len(minmax.protected_layers) == 21  # the degenerate plan seen on the real model

    plan = plan_compression(SensitivityProfile(normalize(raw), raw), 0.5, 0.3, 8, 4)
    assert len(plan.protected_layers) == 11  # threshold 0.5 protects the more sensitive half
    assert 0 in plan.compressed_layers and 21 in plan.protected_layers
    assert len(plan_compression(SensitivityProfile(normalize(raw), raw), 0.8, 0.3, 8, 4).protected_layers) == 5


def test_budget_matched_plan_fits_uniform_size():
    raw = [2705.0] + [6596.0 + i * (8647.0 - 6596.0) / 20 for i in range(21)]
    prof = SensitivityProfile(normalize(raw), raw, [44_000_000] * 22, [12_000] * 22, 260_000_000)
    cost = lambda p: predict_cost(p, prof, 128, 32, 16)  # noqa: E731
    budget = cost(uniform_plan(prof, 4, 0.0)).weight_memory_gb
    plan = budget_matched_plan(prof, budget, 0.3, 8, 4, cost)
    assert cost(plan).weight_memory_gb <= budget
    k = len(plan.protected_layers)
    assert 0 < k < 22
    assert set(plan.protected_layers) == set(sorted(range(22), key=lambda i: raw[i])[-k:])  # the most sensitive
    one_more = budget_matched_plan(prof, budget * 10, 0.3, 8, 4, cost)
    assert len(one_more.protected_layers) == 22  # a generous budget protects everything


def test_outlier_layers():
    raw = [2705.0] + [6596.0 + i * (8647.0 - 6596.0) / 20 for i in range(21)]
    assert outlier_layers(raw) == [0]
    assert outlier_layers([1.0, 1.1, 0.9, 1.05, 0.95]) == []


def test_plan_and_uniform():
    plan = plan_compression(toy_profile(), 0.5, 0.25, protected_bits=8, compressed_bits=4)
    assert plan.protected_layers == [2, 3] and plan.compressed_layers == [0, 1]
    assert [(lp.bit_width, lp.pruning_ratio) for lp in plan.layers] == [(4, 0.25), (4, 0.25), (8, 0.0), (8, 0.0)]
    uni = uniform_plan(toy_profile(), 4, 0.0)
    assert {lp.bit_width for lp in uni.layers} == {4} and not uni.protected_layers
    with pytest.raises(ValueError):
        plan_compression(toy_profile(), 0.5, 1.0, 8, 4)


def test_predict_cost():
    prof = toy_profile()
    fp16 = baseline_cost(prof, 16)
    assert fp16.weight_memory_gb == pytest.approx(4500 * 2 / 1e9)
    uni = predict_cost(uniform_plan(prof, 4, 0.0), prof, group_size=100, group_overhead_bits=32, baseline_bits=16)
    # per layer: 1000 * 4 bits + 10 groups * 32 bits = 4320 bits; + 500 * 16 bits outside the layers
    assert uni.weight_memory_gb == pytest.approx((4 * 4320 + 500 * 16) / 8 / 1e9)
    assert uni.sensitivity_exposure == pytest.approx(0.75)
    per_ch = predict_cost(uniform_plan(prof, 4, 0.0), prof, PER_CHANNEL, 32, 16)
    assert per_ch.per_layer_mb[0] == pytest.approx((4000 + 10 * 32) / 8 / 1e6)
    fw = predict_cost(plan_compression(prof, 0.5, 0.3, 8, 4), prof, 100, 32, 16)
    assert fw.sensitivity_exposure < uni.sensitivity_exposure  # sensitive layers are spared
    assert fw.sparsity == pytest.approx(2 * 300 / 4500)


def test_profile_sensitivity_on_tiny_llama(tiny_llama):
    frozen = tiny_llama.model.embed_tokens.weight
    frozen.requires_grad_(False)
    batches = [torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(i)) for i in range(3)]
    prof = profile_sensitivity(tiny_llama, batches, device="cpu")
    assert prof.num_layers == 4 and min(prof.scores) == 0.0 and max(prof.scores) == 1.0
    assert prof.cost["calibration_tokens"] == 96
    # q, o: 32x32; k, v: 32x16 (2 KV heads); gate, up, down: 32x64
    assert prof.layer_numel == [2 * 1024 + 2 * 512 + 3 * 2048] * 4
    assert all(p.grad is None for p in tiny_llama.parameters())
    assert frozen.requires_grad is False
    assert SensitivityProfile.from_dict(prof.to_dict()) == prof


def test_data_windows(tokenizer):
    a = calibration_batches(fake_texts("x", "train"), tokenizer, n_samples=6, seq_len=8, batch_size=4, seed=1)
    b = calibration_batches(fake_texts("x", "train"), tokenizer, n_samples=6, seq_len=8, batch_size=4, seed=1)
    assert [t.shape for t in a] == [(4, 8), (2, 8)] and all(torch.equal(x, y) for x, y in zip(a, b))
    val, held = eval_windows(fake_texts("x", "test"), tokenizer, seq_len=16, max_windows=5)
    assert val.shape == (2, 16) and held.shape == (3, 16)


def test_measure_model(tiny_llama, tokenizer, small_cfg):
    val, held = eval_windows(fake_texts("x", "test"), tokenizer, 16, 4)
    metrics, raw = measure_model(tiny_llama, val, held, small_cfg.eval, torch.device("cpu"))
    assert metrics["ppl_val"] > 1 and metrics["model_size_gb"] == pytest.approx(model_size_gb(tiny_llama))
    assert sum(r["measurement"] == "latency" for r in raw) == small_cfg.eval.latency_repeats


def test_run_stage0_end_to_end(tiny_llama, tokenizer, small_cfg):
    ctx = start_run(small_cfg)
    cand = SEARCH_SPACE.make({"calib_samples": 16, "sensitive_threshold": 0.5})
    res = run_stage0(ctx, cand, model=tiny_llama, tokenizer=tokenizer, text_loader=fake_texts)

    stage_dir = ctx.run_dir / "stage_0"
    for name in ("report.md", "stage_0_comparison.xlsx", "results.json", "compression_plan.json",
                 "sensitivity_profile.json"):
        assert (stage_dir / name).exists(), name
    data = json.loads((stage_dir / "results.json").read_text())
    rows = {f"{r['method']}/{r['variant']}": r for r in data["rows"]}
    assert set(rows) == {"baseline/fp16", "allocation/original", "allocation/framework",
                         "allocation_same_size/framework"}
    same = rows["allocation_same_size/framework"]
    assert same["metrics"]["predicted_weight_memory_gb"] <= rows["allocation/original"]["metrics"][
        "predicted_weight_memory_gb"] * (1 + 1e-9)
    assert "vs_original_abs" in same["deltas"]["predicted_weight_memory_gb"]  # compared to the uniform plan
    assert all(r["status"] == "ok" for r in rows.values())
    assert rows["baseline/fp16"]["metrics"]["ppl_val"] > 1
    fw = rows["allocation/framework"]
    assert fw["metrics"]["protected_layers"] == len(res.plan.protected_layers)
    assert "vs_original_abs" in fw["deltas"]["sensitivity_exposure"]
    assert len(data["per_layer"]) == 4
    wb = load_workbook(stage_dir / "stage_0_comparison.xlsx")
    assert wb["Per-layer"].max_row == 5

    # second trial with other Stage 0 hyperparameters: profile and FP16 baseline come from cache
    res2 = run_stage0(ctx, SEARCH_SPACE.make({"calib_samples": 16, "sensitive_threshold": 0.9}),
                      model=None, tokenizer=tokenizer, text_loader=fake_texts)
    assert res2.profile == res.profile
    data2 = json.loads((stage_dir / "results.json").read_text())
    assert data2["rows"][0]["info"]["cached"] is True
    assert data2["rows"][2]["info"]["profile_cached"] is True
    report = (stage_dir / "report.md").read_text(encoding="utf-8")
    assert "same-size version of the framework" in report and "Size floor" in report
