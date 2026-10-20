"""Fixes from the 2026-10-19 ponytail audit (numbers are the audit's findings). Every test here fails on main 0c3c70d."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sdf.config import EvalConfig, FrameworkConfig
from sdf.stage0.planner import uniform_plan


# 17 -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key, value", [("eval.latency_mode", "cuda-graph"), ("eval.latency_repeats", 0),
                                        ("eval.latency_decode_tokens", 0)])
def test_timing_settings_are_checked_when_the_config_is_built(key, value):
    with pytest.raises(ValueError, match=key):
        FrameworkConfig().with_overrides({key: value})
    FrameworkConfig().with_overrides({"eval.latency_mode": "eager", "eval.latency_repeats": 1,
                                      "eval.latency_decode_tokens": 1})


# 3 ------------------------------------------------------------------------------------------------------------
def test_a_hung_backend_program_times_out_with_the_backends_own_error(monkeypatch, tmp_path):
    from sdf.eval import llamacpp, worker

    monkeypatch.setattr(llamacpp, "TIMEOUT_S", 0.5)
    with pytest.raises(RuntimeError, match="timed out"):
        llamacpp._run([sys.executable, "-c", "import time; time.sleep(30)"])

    class BackendError(RuntimeError):
        pass

    def hang(cmd, timeout=None, **kw):
        assert timeout == worker.TIMEOUT_S
        raise subprocess.TimeoutExpired(cmd, timeout)
    monkeypatch.setattr(worker.subprocess, "run", hang)
    with pytest.raises(BackendError, match="vllm_worker timed out"):
        worker.run_worker(sys.executable, "vllm_worker.py", {}, tmp_path, BackendError)


# 11 -----------------------------------------------------------------------------------------------------------
def _fake_llamacpp(monkeypatch):
    """llamacpp with every program faked: records each command (program name, then its arguments) and writes the
    file the program would write. The F16 file is 4 bytes, the quantized one 1 byte."""
    from sdf.eval import llamacpp

    calls = []

    def run(cmd):
        cmd = [str(c) for c in cmd]
        name = "python" if cmd[0] == sys.executable else Path(cmd[0]).name
        calls.append([name, *cmd[1:]])
        if name == "python":
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"f16!")
        elif name == "llama-quantize":
            Path(cmd[-2]).write_bytes(b"q")
        elif name == "llama-bench":
            return ('[{"n_prompt": 8, "n_gen": 0, "avg_ts": 100.0, "stddev_ts": 1.0}, '
                    '{"n_prompt": 0, "n_gen": 4, "avg_ts": 20.0, "stddev_ts": 0.5}]')
        else:
            return "Final estimate: PPL = 7.5 +/- 0.1"
        return ""

    class Model:
        def save_pretrained(self, d):
            Path(d).mkdir()

    class Tok:
        def decode(self, ids):
            return " ".join(map(str, ids.tolist()))

    monkeypatch.setattr(llamacpp, "_run", run)
    monkeypatch.setattr(llamacpp, "_save_tokenizer", lambda tok, out: None)
    return llamacpp, Model(), Tok(), calls


def test_llamacpp_measure_benchmarks_and_scores_the_quantized_file(monkeypatch):
    llamacpp, model, tok, calls = _fake_llamacpp(monkeypatch)
    cfg = EvalConfig(llamacpp_dir="bin", llamacpp_convert="convert_hf_to_gguf.py")
    val, held = torch.zeros(3, 8, dtype=torch.long), torch.zeros(2, 8, dtype=torch.long)
    out = llamacpp.measure(model, tok, uniform_plan([0.0] * 4, 4, 0.0), val, held, cfg, 16)

    assert [c[0] for c in calls] == ["python", "llama-quantize", "llama-bench", "llama-perplexity",
                                     "llama-perplexity"]
    convert, quantize = calls[0], calls[1]
    f16 = convert[convert.index("--outfile") + 1]
    assert convert[1] == "convert_hf_to_gguf.py" and convert[convert.index("--outtype") + 1] == "f16"
    assert quantize[-3] == f16 and quantize[-2].endswith("q.gguf") and quantize[-1] == "Q4_K"
    assert all(c[c.index("-m") + 1] == quantize[-2] for c in calls[2:])  # the quantized file, not the F16 one
    assert [c[c.index("--chunks") + 1] for c in calls[3:]] == ["3", "2"]  # one llama.cpp chunk per window
    assert out == {"llamacpp_size_gb": 1e-9, "llamacpp_prefill_tokens_per_s": 100.0,
                   "llamacpp_decode_tokens_per_s": 20.0, "llamacpp_decode_tokens_per_s_std": 0.5,
                   "llamacpp_ppl_val": 7.5, "llamacpp_ppl_heldout": 7.5}

    for plan in (None, uniform_plan([0.0] * 4, 16, 0.0)):  # nothing below the baseline: the F16 file as is
        calls.clear()
        out = llamacpp.measure(model, tok, plan, val, held, cfg, 16)
        assert [c[0] for c in calls] == ["python", "llama-bench", "llama-perplexity", "llama-perplexity"]
        assert all(c[c.index("-m") + 1].endswith("f16.gguf") for c in calls[1:])
        assert out["llamacpp_size_gb"] == 4e-9
