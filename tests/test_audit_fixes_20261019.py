"""Fixes from the 2026-10-19 ponytail audit (numbers are the audit's findings). Every test here fails on main 0c3c70d."""

import subprocess
import sys

import pytest

from sdf.config import FrameworkConfig


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
