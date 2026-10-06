from dataclasses import replace

from sdf.eval.llamacpp import quantize_args
from sdf.stage0.planner import uniform_plan


def test_quantize_args_follow_the_plan():
    plan = uniform_plan([0.0] * 12, 4, 0.0)
    plan = replace(plan, layers=tuple(replace(lp, bit_width=8) if lp.layer in (0, 10) else lp for lp in plan.layers))
    opts, ftype = quantize_args(plan, 16)
    assert ftype == "Q4_K"  # most common bits = file type
    overrides = [opts[i + 1] for i, o in enumerate(opts) if o == "--tensor-type"]
    assert overrides == [r"blk\.0\.=q8_0", r"blk\.10\.=q8_0"]  # "blk\.1\." would not match blk.10
    assert opts[opts.index("--output-tensor-type") + 1] == "f16" and "--pure" in opts
