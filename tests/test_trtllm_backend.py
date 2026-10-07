from dataclasses import replace

from torch import nn

from sdf.eval import trtllm_backend as t
from sdf.stage0.planner import uniform_plan


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 64, bias=False)


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block() for _ in range(3)])
        self.lm_head = nn.Linear(128, 64, bias=False)


def test_one_bit_width_is_one_gptq_algorithm():
    q = t.quant_file(M(), uniform_plan([0.0] * 3, 4, 0.0), 128)["quantization"]
    assert q == {"quant_algo": "W4A16_GPTQ", "group_size": 128, "has_zero_point": True}


def test_mixed_plan_lists_every_decoder_linear():
    plan = uniform_plan([0.0] * 3, 4, 0.0)
    plan = replace(plan, layers=tuple(replace(lp, bit_width=8) if lp.layer == 1 else lp for lp in plan.layers))
    q = t.quant_file(M(), plan, 128)["quantization"]
    assert q["quant_algo"] == "MIXED_PRECISION"
    assert {k: v["quant_algo"] for k, v in q["quantized_layers"].items()} == {
        "model.layers.0.q_proj": "W4A16_GPTQ", "model.layers.1.q_proj": "W8A16_GPTQ",
        "model.layers.2.q_proj": "W4A16_GPTQ"}  # lm_head stays FP16, as in the plan
