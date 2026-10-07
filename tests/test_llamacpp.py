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


def test_bench_json_survives_stderr_after_it(monkeypatch):
    from sdf.config import EvalConfig
    from sdf.eval import llamacpp

    rows = '[{"n_prompt": 8, "n_gen": 0, "avg_ts": 1.0, "stddev_ts": 0.1}, ' \
           '{"n_prompt": 0, "n_gen": 4, "avg_ts": 2.0, "stddev_ts": 0.2}]'
    monkeypatch.setattr(llamacpp, "_run", lambda cmd: rows + "\nload_backend: [CUDA0] loaded\n")
    monkeypatch.setattr(llamacpp, "_exe", lambda cfg, name: name)
    assert llamacpp._bench(EvalConfig(), "m.gguf")["llamacpp_decode_tokens_per_s"] == 2.0
