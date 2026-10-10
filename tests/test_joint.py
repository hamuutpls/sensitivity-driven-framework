import copy
import itertools
import json
import random

import pytest

from sdf.run import start_run
from sdf.search_space import SEARCH_SPACE
from sdf.stage0.joint import JointProfile, interaction, joint_plans, plan_joint, profile_joint, refine_joint
from sdf.stage0.planner import uniform_plan
from sdf.stage0.run import run_stage0
from conftest import fake_texts


def _profile(layers=5, sizes=None, seed=0):
    rnd = random.Random(seed)
    rise = []
    for _ in range(layers):
        s, k = rnd.random() * 3, rnd.random()
        rise.append([[0.0 if (w, a) == (16, 16) else s * ((16 / w - 1) + k * (16 / a - 1) + (16 / w - 1) * (16 / a - 1))
                      for a in (4, 8, 16)] for w in (4, 8, 16)])
    return JointProfile([4, 8, 16], [4, 8, 16], rise, sizes or [10] * layers)


@pytest.mark.parametrize("sizes", [None, [10, 20, 10, 10, 20]])
def test_plan_joint_is_the_exact_optimum_under_both_budgets(sizes):
    p = _profile(sizes=sizes)
    picks = plan_joint(p, 6.0, 6.0)
    size = [s // 10 for s in p.layer_numel]
    assert sum(w * s for (w, _), s in zip(picks, size)) <= 6 * sum(size)
    assert sum(a for _, a in picks) <= 6 * len(picks)
    got = sum(p.rise_at(i, w, a) for i, (w, a) in enumerate(picks))
    best = min(sum(p.rise_at(i, w, a) for i, (w, a) in enumerate(c))
               for c in itertools.product(itertools.product((4, 8, 16), (4, 8, 16)), repeat=len(picks))
               if sum(w * s for (w, _), s in zip(c, size)) <= 6 * sum(size) and sum(a for _, a in c) <= 6 * len(c))
    assert got == pytest.approx(best)


def test_monotone_and_budget_errors():
    p = _profile()
    p.rise[0][1][1] = 99.0  # W8A8 measured worse than W4A4: noise, clipped
    assert p.rise_at(0, 8, 8) <= p.rise_at(0, 4, 4)
    with pytest.raises(ValueError):
        plan_joint(p, 3.0, 6.0)
    assert interaction(p, 4, 4)["ratio"] > 1  # the synthetic rises have a product term


def test_joint_plans_keep_nothing_pruned_and_allow_16_bits():
    p = _profile()
    picks = plan_joint(p, 12.0, 12.0)
    w, a = joint_plans(picks, uniform_plan([0.0] * 5, 4, 0.3))
    assert all(lp.pruning_ratio == 0 for lp in w.layers) and w.kind == "joint" and a.kind == "joint"
    assert [lp.bit_width for lp in w.layers] == [x for x, _ in picks] and 16 in [x for x, _ in picks]


def test_profile_joint_measures_every_pair(tiny_llama):
    import torch
    batches = [torch.randint(0, tiny_llama.config.vocab_size, (2, 16)) for _ in range(2)]
    before = {k: v.clone() for k, v in tiny_llama.state_dict().items()}
    p = profile_joint(tiny_llama, batches, [4, 16], [4, 16], 16, 16)
    assert p.num_layers == 4 and all(r[1][1] == 0.0 for r in p.rise)
    assert all(r[0][0] != 0.0 for r in p.rise)
    assert all((v == tiny_llama.state_dict()[k]).all() for k, v in before.items())  # weights restored


def test_refine_joint_keeps_the_budget_never_gets_worse_and_restores_weights(tiny_llama):
    import torch
    batches = [torch.randint(0, tiny_llama.config.vocab_size, (2, 16)) for _ in range(2)]
    before = {k: v.clone() for k, v in tiny_llama.state_dict().items()}
    p = profile_joint(tiny_llama, batches, [4, 8, 16], [4, 8, 16], 16, 16)
    start = plan_joint(p, 6.0, 6.0)
    worst = [(4, 4)] * p.num_layers
    picks, rec = refine_joint(tiny_llama, batches, p, start, 6.0, 6.0, 16, 16, rounds=2, compare={"worst": worst})
    assert sum(w for w, _ in picks) <= 6 * len(picks) and sum(a for _, a in picks) <= 6 * len(picks)
    assert rec["ppl"] <= rec["rounds"][0]["ppl"] and rec["rounds"][0]["picks"] == [list(x) for x in start]
    assert all(r["kept"] == (r["ppl"] < rec["rounds"][i]["ppl"]) for i, r in enumerate(rec["rounds"][1:]) if r["kept"])
    assert len(rec["context_rise"]) == p.num_layers and rec["compare_ppl"]["worst"] >= rec["ppl"] - 1e-6
    assert all((v == tiny_llama.state_dict()[k]).all() for k, v in before.items())  # weights restored


def test_stage0_writes_the_joint_plan(tiny_llama, tokenizer, small_cfg):
    ctx = start_run(small_cfg)
    cand = SEARCH_SPACE.make({"calib_samples": 16})
    run_stage0(ctx, cand, model=copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    d = ctx.run_dir / "stage_0"
    for f in ("joint_weight_plan.json", "joint_activation_plan.json", "joint_profile.json"):
        assert (d / f).exists()
    rows = {f"{r['method']}/{r['variant']}": r for r in json.loads((d / "results.json").read_text())["rows"]}
    row = rows["joint_weights_activations/framework"]
    assert row["status"] == "ok", row.get("error")
    quant = json.loads((d / "quant_plan.json").read_text())["layers"]
    assert row["metrics"]["decoder_weight_bits"] <= sum(lp["bit_width"] for lp in quant) / len(quant) + 1e-9
    assert row["metrics"]["avg_activation_bits"] <= small_cfg.stage0.act_avg_bits
    assert "separate_plans" in row["info"] and "combinations" in row["info"]
    # the per-layer joint plan never predicts more damage than the separate plans at the same budgets
    per_layer = row["info"]["per_layer_plan"]["predicted_joint_ppl_rise"]
    assert per_layer <= row["info"]["separate_plans"]["predicted_joint_ppl_rise"] + 1e-9
    # refinement keeps a plan only if the fully rounded model does better than the per-layer plan
    ref = row["info"]["refinement"]
    assert ref["calib_ppl"] <= ref["rounds"][0]["ppl"] and "separate" in ref["compare_calib_ppl"]
