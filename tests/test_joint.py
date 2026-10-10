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


def test_refine_joint_starts_from_the_best_plan_and_keeps_only_improvements(tiny_llama, monkeypatch):
    import torch
    import sdf.stage0.joint as joint
    batches = [torch.randint(0, tiny_llama.config.vocab_size, (2, 16)) for _ in range(2)]
    before = {k: v.clone() for k, v in tiny_llama.state_dict().items()}
    p = profile_joint(tiny_llama, batches, [4, 8, 16], [4, 8, 16], 16, 16)
    plans = {"w4a4": [(4, 4)] * p.num_layers, "w8a8": [(8, 8)] * p.num_layers}  # both within an 8/8 budget
    _, rec = refine_joint(tiny_llama, batches, p, plans, 8.0, 8.0, 16, 16, rounds=0)
    assert rec["starts"]["w4a4"] != rec["starts"]["w8a8"]  # a random tiny model: either may score better
    better, worse = sorted(plans.values(), key=lambda q: rec["starts"]["w4a4" if q == plans["w4a4"] else "w8a8"])
    # round 1 proposes the better plan (kept), round 2 the worse one (rejected)
    proposals = iter([better, worse])
    monkeypatch.setattr(joint, "plan_joint", lambda *a: next(proposals))
    picks, rec = refine_joint(tiny_llama, batches, p, {"a": worse, "b": worse}, 8.0, 8.0, 16, 16, rounds=3)
    assert rec["rounds"][0]["start"] == "a" and set(rec["starts"]) == {"a", "b"}
    assert [r["kept"] for r in rec["rounds"]] == [True, True, False]  # stops at the first rejected plan
    assert picks == better and rec["ppl"] == rec["rounds"][1]["ppl"] < rec["rounds"][0]["ppl"]
    assert all((v == tiny_llama.state_dict()[k]).all() for k, v in before.items())  # weights restored
    # the better start is chosen; real re-planning stays within the budget
    monkeypatch.undo()
    picks, rec = refine_joint(tiny_llama, batches, p, {"worse": worse, "better": better}, 8.0, 8.0, 16, 16, rounds=1)
    assert rec["rounds"][0]["start"] == "better" and rec["ppl"] <= min(rec["starts"].values())
    assert sum(w for w, _ in picks) <= 8 * len(picks) and sum(a for _, a in picks) <= 8 * len(picks)


def test_stage0_writes_the_joint_plan(tiny_llama, tokenizer, small_cfg):
    ctx = start_run(small_cfg)
    cand = SEARCH_SPACE.make({"calib_samples": 16})
    run_stage0(ctx, cand, model=copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    d = ctx.run_dir / "stage_0"
    for f in ("joint_weight_plan.json", "joint_activation_plan.json", "joint_profile.json", "joint_curves.csv"):
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
    # refinement starts from the better of the per-layer and separate plans and never ends worse on calibration
    ref = row["info"]["refinement"]
    assert ref["calib_ppl"] <= min(ref["start_calib_ppl"].values()) and set(ref["start_calib_ppl"]) == {"per_layer", "separate"}


def test_joint_curves_table_and_figures(tmp_path):
    import csv
    from sdf.stage0 import joint_curves
    p = _profile(layers=3)
    p.meta["baseline_ppl"] = 10.0
    picks = [(8, 8), (4, 8), (16, 4)]
    with open(joint_curves.write_csv(p, picks, tmp_path / "c.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 9 and [r["pair"] for r in rows[:9]] == [
        "W4A4", "W4A8", "W8A4", "W8A8", "W4A16", "W16A4", "W8A16", "W16A8", "W16A16"]
    assert [r["pair"] for r in rows if r["chosen"] == "True"] == ["W8A8", "W4A8", "W16A4"]
    assert float(rows[8]["ppl"]) == 10.0 and float(rows[0]["ppl"]) == 10.0 + p.rise[0][0][0]
    pytest.importorskip("matplotlib")
    files = joint_curves.draw(p, picks, tmp_path / "fig")
    assert [f.name for f in files] == ["joint_curves_all_layers.png"] + [f"joint_curve_layer{i:02d}.png" for i in range(3)]
    assert all(f.stat().st_size > 0 for f in files)
    joint_curves.draw(p, [(3, 6)] + picks[1:], tmp_path / "off")  # a pick off the measured grid still draws
