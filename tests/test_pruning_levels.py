"""Stage 0 pruning levels test (stage0/pruning_levels.py)."""

import copy
import csv
import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sdf.run import start_run
from sdf.search_space import SEARCH_SPACE
from sdf.stage0.pruning_levels import (CSV_NAME, SUBDIR, capped_plan, feed_forward_share, layer_set, level_plan, restored,
                                       share_removed)
from sdf.stage0.run import run_stage0
from sdf.stage0.sensitivity import find_decoder_layers
from sdf.stages.pruning import mlp_linears, structured_prune_
from conftest import fake_texts


def test_layer_sets_split_the_layers_by_the_guard():
    assert layer_set("all_layers", 5, {0, 3}) == [0, 1, 2, 3, 4]
    assert layer_set("non_guarded_layers", 5, {0, 3}) == [1, 2, 4]
    assert layer_set("guarded_layers_only", 5, {3, 0}) == [0, 3]
    assert layer_set("guarded_layers_only", 5, set()) == []
    with pytest.raises(ValueError):
        layer_set("protected_layers", 5, {0})


def test_level_plan_prunes_only_the_chosen_layers_and_rounds_nothing():
    plan = level_plan(4, [1, 3], 0.7, 16)
    assert [lp.pruning_ratio for lp in plan.layers] == [0.0, 0.7, 0.0, 0.7]
    assert {lp.bit_width for lp in plan.layers} == {16}
    assert [lp.layer for lp in plan.layers] == [0, 1, 2, 3]


def test_feed_forward_share_of_tinyllama_is_78_6_percent():
    cfg = LlamaConfig(vocab_size=32000, hidden_size=2048, intermediate_size=5632, num_hidden_layers=1,
                      num_attention_heads=32, num_key_value_heads=4)
    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)
    assert feed_forward_share(model, find_decoder_layers(model)[0]) == pytest.approx(0.786, abs=5e-4)


def test_structured_levels_above_the_feed_forward_share_are_capped_and_remove_every_channel(tiny_llama):
    share = feed_forward_share(tiny_llama, find_decoder_layers(tiny_llama)[0])
    assert share == pytest.approx(6144 / 9216)  # toy model: 3 x 32 x 64 feed-forward of 9216 Linear weights
    plan = level_plan(4, [0, 1, 2, 3], 0.9, 16)
    with pytest.raises(ValueError):  # uncapped, structured pruning refuses
        structured_prune_(copy.deepcopy(tiny_llama), plan, [torch.randint(0, 64, (2, 8))], -1, 16)
    capped = capped_plan(tiny_llama, plan)
    assert [lp.pruning_ratio for lp in capped.layers] == pytest.approx([share] * 4)
    assert capped_plan(tiny_llama, level_plan(4, [0], 0.3, 16)).layers[0].pruning_ratio == 0.3  # below: unchanged
    m = copy.deepcopy(tiny_llama)
    structured_prune_(m, capped, [torch.randint(0, 64, (2, 8))], -1, 16)
    ups, down = mlp_linears(find_decoder_layers(m)[0], 32)
    assert down.weight.abs().sum() == 0 and all(u.weight.abs().sum() == 0 for u in ups)


def test_share_removed_weighs_layers_by_size(tiny_llama):
    assert share_removed(tiny_llama, level_plan(4, [0, 1, 2, 3], 0.5, 16)) == pytest.approx(0.5)
    assert share_removed(tiny_llama, level_plan(4, [2], 0.8, 16)) == pytest.approx(0.2)
    assert share_removed(tiny_llama, level_plan(4, [], 0.8, 16)) == 0.0


def test_restored_puts_back_every_decoder_weight(tiny_llama):
    before = copy.deepcopy(tiny_llama.state_dict())
    with restored(tiny_llama):
        for m in tiny_llama.model.layers.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.zero_()
    after = tiny_llama.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_every_stage0_run_measures_each_technique_layer_set_and_level(small_cfg, tiny_llama, tokenizer):
    cfg = small_cfg.with_overrides({"stage0.pruning_levels": [0.9, 0.3], "stage0.joint_plan": False})
    ctx = start_run(cfg)
    before = copy.deepcopy(tiny_llama.state_dict())
    res = run_stage0(ctx, SEARCH_SPACE.make({"calib_samples": 16}), model=tiny_llama, tokenizer=tokenizer,
                     text_loader=fake_texts)
    levels_dir = res.outputs["report"].parent.parent / SUBDIR
    with open(levels_dir / CSV_NAME) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 3 * 2
    assert {(r["technique"], r["layer_set"], r["level"]) for r in rows} == {
        (t, s, l) for t in ("wanda", "structured", "low_rank")
        for s in ("all_layers", "non_guarded_layers", "guarded_layers_only") for l in ("0.3", "0.9")}
    assert all(r["status"] == "ok" and math.isfinite(float(r["ppl_val"])) for r in rows)
    by = {(r["technique"], r["layer_set"], r["level"]): r for r in rows}
    assert by[("wanda", "guarded_layers_only", "0.3")]["layers_pruned"] == "1"  # guard_top_k = 1
    assert by[("wanda", "non_guarded_layers", "0.3")]["layers_pruned"] == "3"
    assert float(by[("structured", "all_layers", "0.9")]["share_removed_per_pruned_layer"]) == pytest.approx(6144 / 9216)
    assert float(by[("wanda", "all_layers", "0.9")]["share_of_decoder_weights_removed"]) == pytest.approx(0.9)
    # pruning changes the model (the toy model is untrained, so only a difference is checked, not a direction),
    # and every row starts again from the unpruned model, which is left as it was
    assert by[("wanda", "all_layers", "0.9")]["ppl_val"] != by[("wanda", "all_layers", "0.3")]["ppl_val"]
    after = tiny_llama.state_dict()
    assert all(torch.equal(before[k], after[k].to(before[k].dtype)) for k in before)
    assert f"../{SUBDIR}/report.md" in res.outputs["report"].read_text()  # pointer in Stage 0's report
    report = (levels_dir / "report.md").read_text()
    assert "Perplexity by technique, layer set and level" in report and "capped from 90%" in report
    assert "Share of the decoder layers' weights removed" in report and "Weight sparsity" not in report
    assert "no standard-versus-our-method comparison" in report and "see the failures below" not in report
    assert (levels_dir / "stage_0_comparison.xlsx").exists()
    stage0_rows = res.outputs["json"].read_text()
    assert "wanda_all_layers_030" not in stage0_rows  # the 63 rows stay out of Stage 0's own tables


def test_pruning_levels_can_be_turned_off(small_cfg, tiny_llama, tokenizer):
    cfg = small_cfg.with_overrides({"stage0.pruning_levels": [], "stage0.joint_plan": False})
    res = run_stage0(start_run(cfg), SEARCH_SPACE.make({"calib_samples": 16}), model=tiny_llama, tokenizer=tokenizer,
                     text_loader=fake_texts)
    assert not (res.outputs["report"].parent.parent / SUBDIR).exists()


def test_pruning_levels_must_be_shares(small_cfg):
    for bad in ([0.3, 1.5], [0.0], [0.3, 0.301]):
        with pytest.raises(ValueError, match="pruning_levels"):
            small_cfg.with_overrides({"stage0.pruning_levels": bad})


def test_without_a_guard_only_the_all_layers_set_runs(small_cfg, tiny_llama, tokenizer):
    cfg = small_cfg.with_overrides({"stage0.pruning_levels": [0.5], "stage0.joint_plan": False,
                                    "stage0.guard_top_k": 0})
    res = run_stage0(start_run(cfg), SEARCH_SPACE.make({"calib_samples": 16}), model=tiny_llama, tokenizer=tokenizer,
                     text_loader=fake_texts)
    with open(res.outputs["report"].parent.parent / SUBDIR / CSV_NAME) as f:
        rows = list(csv.DictReader(f))
    assert {r["layer_set"] for r in rows} == {"all_layers"} and len(rows) == 3
    assert all(r["status"] == "ok" for r in rows)


def test_feed_forward_share_refuses_a_layer_without_a_feed_forward_block(tiny_llama):
    with pytest.raises(ValueError, match="feed-forward"):
        feed_forward_share(tiny_llama, torch.nn.Sequential(torch.nn.Linear(32, 32)))
