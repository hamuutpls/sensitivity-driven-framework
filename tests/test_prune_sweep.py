import json

import torch
from torch import nn

from sdf.run import start_run
from sdf.search_space import SEARCH_SPACE
from sdf.stage0.planner import uniform_plan
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
    assert keys[0] == "baseline/fp16" and keys[1:3] == ["prune_010/original", "prune_010/framework"]
    assert len(keys) == 7 and all(r["status"] == "ok" for r in data["rows"])
    rows = {k: r["metrics"] for k, r in zip(keys, data["rows"])}
    # more pruning, less memory; the framework never prunes its guarded layer, so it removes less
    assert rows["prune_100/original"]["predicted_weight_memory_gb"] < rows["prune_010/original"]["predicted_weight_memory_gb"]
    assert rows["prune_100/framework"]["sparsity"] < rows["prune_100/original"]["sparsity"]
    report = out["report"].read_text(encoding="utf-8")
    assert "## Pruning curve" in report and "## Original model" in report and "What each column means" in report
    assert out["report"].parent.name == "stage_0_prune_sweep"
    # second run is served from the cache
    again = run_prune_sweep(ctx, cand, model=tiny_llama, tokenizer=tokenizer, text_loader=fake_texts)
    assert all(r["info"]["cached"] for r in json.loads(again["json"].read_text())["rows"])
