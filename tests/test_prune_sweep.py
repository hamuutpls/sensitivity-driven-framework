import json

import torch
from torch import nn

from sdf.run import start_run
from sdf.search_space import SEARCH_SPACE
from sdf.stage0.planner import same_size_pruning_plan, uniform_plan
from sdf.stage0.prune_sweep import apply_plan, magnitude_mask, run_prune_sweep
from conftest import fake_texts


def test_magnitude_mask_drops_smallest_per_row():
    w = torch.tensor([[0.1, -3.0, 0.5, 2.0], [4.0, 0.0, -1.0, 0.2]])
    assert magnitude_mask(w, 0.5).tolist() == [[False, True, False, True], [True, False, True, False]]
    assert magnitude_mask(w, 0.0).all() and not magnitude_mask(w, 1.0).any()


def test_apply_plan_prunes_rounds_and_restores(tiny_llama):
    before = {n: p.clone() for n, p in tiny_llama.named_parameters()}
    linear = tiny_llama.model.layers[0].mlp.down_proj
    with apply_plan(tiny_llama, uniform_plan([0.0] * 4, 4, 0.5), 8, quantize=True):
        w = linear.weight
        assert (w == 0).float().mean().item() >= 0.5  # half of each row removed, and kept at zero after rounding
        assert len(torch.unique(w[0, :8])) <= 2 ** 4 + 1
    with apply_plan(tiny_llama, uniform_plan([0.0] * 4, 4, 1.0), 8, quantize=False):
        assert all((m.weight == 0).all() for m in tiny_llama.model.layers[3].modules() if isinstance(m, nn.Linear))
    assert all(torch.equal(p, before[n]) for n, p in tiny_llama.named_parameters())


def test_run_prune_sweep(tiny_llama, tokenizer, small_cfg):
    cfg = small_cfg.with_overrides({"stage0.prune_sweep_ratios": [0.5, 0.1, 1.0]})
    ctx = start_run(cfg)
    cand = SEARCH_SPACE.make({"calib_samples": 16})
    out = run_prune_sweep(ctx, cand, model=tiny_llama, tokenizer=tokenizer, text_loader=fake_texts)
    data = json.loads(out["json"].read_text())
    keys = [r["method"] + "/" + r["variant"] for r in data["rows"]]
    assert keys[0] == "baseline/fp16" and keys[1:4] == ["prune_010/original", "prune_010/framework",
                                                         "same_010/framework"]
    assert len(keys) == 10 and all(r["status"] == "ok" for r in data["rows"])
    rows = {k: r["metrics"] for k, r in zip(keys, data["rows"])}
    # more pruning, less memory; the framework never prunes its guarded layer, so it removes less
    assert rows["prune_100/original"]["predicted_weight_memory_gb"] < rows["prune_010/original"]["predicted_weight_memory_gb"]
    assert rows["prune_100/framework"]["sparsity"] < rows["prune_100/original"]["sparsity"]
    # the fair test matches the standard method's size and share removed where the guard allows it
    for lvl in ("prune_010", "prune_050"):
        assert abs(rows[f"same_{lvl[6:]}/framework"]["predicted_weight_memory_gb"]
                   - rows[f"{lvl}/original"]["predicted_weight_memory_gb"]) < 1e-9
        assert abs(rows[f"same_{lvl[6:]}/framework"]["sparsity"] - rows[f"{lvl}/original"]["sparsity"]) < 1e-9
    report = out["report"].read_text(encoding="utf-8")
    assert "## Fair test: same size, same share removed" in report
    assert "## Pruning curve" in report and "## Original model" in report and "What each column means" in report
    assert out["report"].parent.name == "stage_0_prune_sweep"
    # second run is served from the cache
    again = run_prune_sweep(ctx, cand, model=tiny_llama, tokenizer=tokenizer, text_loader=fake_texts)
    assert all(r["info"]["cached"] for r in json.loads(again["json"].read_text())["rows"])


def test_same_size_pruning_plan_moves_pruning_to_robust_layers():
    scores, numel = [0.0, 0.25, 0.5, 0.75, 1.0], [100, 100, 100, 100, 200]
    plan = same_size_pruning_plan(scores, 4, 0.3, numel, frozenset({4}))
    ratios = [lp.pruning_ratio for lp in plan.layers]
    assert abs(sum(r * m for r, m in zip(ratios, numel)) - 0.3 * sum(numel)) < 1e-6  # same weights removed
    assert ratios == sorted(ratios, reverse=True) and ratios[4] == 0.0 and ratios[3] > 0  # robust lose most
    assert all(lp.bit_width == 4 for lp in plan.layers)
    full = same_size_pruning_plan(scores, 4, 1.0, numel, frozenset({4}))  # guard can't be pruned: capped
    assert [lp.pruning_ratio for lp in full.layers] == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_run_threshold_sweep(tiny_llama, tokenizer, small_cfg):
    from sdf.stage0.threshold_sweep import run_threshold_sweep

    cfg = small_cfg.with_overrides({"stage0.threshold_sweep": [0.2, 0.9], "stage0.guard_sweep": [0, 1]})
    out = run_threshold_sweep(start_run(cfg), SEARCH_SPACE.make({"calib_samples": 16}), model=tiny_llama,
                              tokenizer=tokenizer, text_loader=fake_texts)
    rows = {r["method"]: r for r in json.loads(out["json"].read_text())["rows"]}
    assert all(r["status"] == "ok" for r in rows.values()) and len(rows) == 2 + 2 * 4  # fp16, standard, 4 per guard
    # a lower threshold protects more layers and needs more memory
    lo, hi = rows["t020_k1"]["metrics"], rows["t090_k1"]["metrics"]
    assert lo["protected_layers"] > hi["protected_layers"]
    assert lo["predicted_weight_memory_gb"] > hi["predicted_weight_memory_gb"]
    assert "## Threshold sweep" in out["report"].read_text(encoding="utf-8")
