"""Fixes the final test waits for (ponytail audit 2026-10-20, findings 19, 23 and 24)."""

import pytest

from sdf.config import Stage0Config
from sdf.stage0.activation import uniform_activation_plan
from sdf.stage0.run import joint_act_budget


def test_the_joint_plan_gets_the_separate_activation_plans_real_average():  # 19
    s0 = Stage0Config()
    assert s0.act_plan != "from_weights" and s0.act_avg_bits == 6.0
    short = uniform_activation_plan(4, 5)  # a measured plan that spent less than its 6-bit budget
    assert joint_act_budget(s0, short) == 5.0
    assert joint_act_budget(s0, None) == 6.0
    assert joint_act_budget(Stage0Config(joint_avg_act_bits=4.0), short) == 4.0


def test_awq_rounds_o_proj_plainly_under_grouped_query_attention(tiny_llama):  # 24
    import torch
    from sdf.stage0.planner import uniform_plan
    from sdf.stage0.sensitivity import round_to_nearest
    from sdf.stages.weights import awq_

    attn = tiny_llama.model.layers[0].self_attn
    assert attn.v_proj.out_features != attn.o_proj.in_features  # 2 key/value heads for 4 query heads
    before = {n: getattr(attn, n).weight.data.clone() for n in ("q_proj", "o_proj")}
    torch.manual_seed(1)
    batches = [torch.randint(0, 64, (2, 16))]
    awq_(tiny_llama, uniform_plan([0.0] * 4, 3, 0.0), batches, 16, 16)
    assert torch.equal(attn.o_proj.weight.data, round_to_nearest(before["o_proj"], 3, 16, True))
    assert not torch.equal(attn.q_proj.weight.data, round_to_nearest(before["q_proj"], 3, 16, True))


def test_size_counts_what_each_method_stores():  # 23
    from sdf.stage0.planner import predict_cost, uniform_plan
    from sdf.stage0.sensitivity import SensitivityProfile
    from sdf.stages.methods import METHODS
    shapes = [(2048, 2048), (256, 2048), (256, 2048), (2048, 2048), (5632, 2048), (5632, 2048), (2048, 5632)]
    numel, rows = sum(o * i for o, i in shapes), sum(o for o, _ in shapes)  # one TinyLlama decoder layer
    prof = SensitivityProfile(raw_scores=[1.0], layer_numel=[numel], layer_rows=[rows], other_numel=0,
                              method="layer_removal")

    def bits_per_weight(name, b):
        return predict_cost(uniform_plan([0.0], b, 0.0), prof, 128, 32, 16,
                            overhead=METHODS[name].overhead).avg_bits_per_weight

    assert bits_per_weight("rtn", 4) == 4.25  # one fp16 scale and zero per 128 weights, as before
    assert bits_per_weight("squeezellm", 8) == pytest.approx(9.81, abs=0.01)
    assert bits_per_weight("squeezellm", 4) == pytest.approx(4.25, abs=0.01)
    assert bits_per_weight("spqr", 4) == pytest.approx(4 + 0.375 + 0.25 + 0.32)
    assert bits_per_weight("bitsandbytes", 4) == 4.5
    assert bits_per_weight("bitsandbytes", 8) == pytest.approx(8.0, abs=0.02)
    assert 2 < bits_per_weight("aqlm", 2) < 2.05 and 3 <= bits_per_weight("quipsharp", 3) < 3.001
    # pruning half the weights halves the per-weight parts, not the per-row codebook
    half = predict_cost(uniform_plan([0.0], 8, 0.5), prof, 128, 32, 16, sparse_storage="free",
                        overhead=METHODS["squeezellm"].overhead).avg_bits_per_weight
    assert half == pytest.approx(4 + rows * 256 * 16 / numel + 0.0045 * 32 / 2)
    # a fully removed layer is stored as nothing, codebooks included
    gone = [predict_cost(uniform_plan([0.0], 8, 1.0), prof, 128, 32, 16, sparse_storage="free",
                         overhead=METHODS[m].overhead).avg_bits_per_weight for m in ("squeezellm", "aqlm", "rtn")]
    assert gone == [0.0, 0.0, 0.0]


def test_every_row_of_a_method_counts_its_size_the_same_way(tiny_llama):  # 23
    from sdf.config import FrameworkConfig as Config
    from sdf.search_space import SEARCH_SPACE
    from sdf.stage0.planner import uniform_plan
    from sdf.stage0.sensitivity import SensitivityProfile
    from sdf.stages.methods import METHODS, joint, series
    from sdf.stages.runner import Stage0Plans, plan_metrics
    sq = METHODS["squeezellm"]
    assert series(sq, METHODS["unstructured_prune"]).overhead is sq.overhead
    assert joint(METHODS["spqr"], "rtn_act").overhead is METHODS["spqr"].overhead
    prof = SensitivityProfile(raw_scores=[1.0], layer_numel=[10_000], layer_rows=[100], method="layer_removal")
    plans = Stage0Plans(None, {}, prof, None)
    cand, plan = SEARCH_SPACE.make({}), uniform_plan([0.0], 4, 0.0)
    own = plan_metrics(plan, plans, Config(), cand, None, sq.overhead)["avg_bits_per_weight"]
    assert own == pytest.approx(4 + (100 * 16 * 16 + 0.0045 * 10_000 * 32) / 10_000)
    assert plan_metrics(plan, plans, Config(), cand)["avg_bits_per_weight"] != own
