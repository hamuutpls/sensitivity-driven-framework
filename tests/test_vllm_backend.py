import subprocess
from types import SimpleNamespace

import pytest
import torch

from sdf.config import EvalConfig
from sdf.eval import vllm_backend


def test_failed_worker_raises_vllm_error_with_the_cause(monkeypatch, tmp_path):
    """A plan vLLM cannot load must surface vLLM's own reason (the runner records it and keeps the HF row)."""
    stderr = ("Traceback (most recent call last):\n  ...\npydantic_core._pydantic_core.ValidationError: 1 validation error\n"
              "  Value error, Unsupported quantization config: bits=3, sym=True [type=value_error]\n"
              "    For further information visit https://errors.pydantic.dev/2.13/v/value_error\n")
    seen = {}

    def fake_run(cmd, **kw):
        seen["env"] = kw["env"]
        return SimpleNamespace(returncode=1, stdout="", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)

    class Model:
        def save_pretrained(self, d, safe_serialization=True):
            d.mkdir(parents=True, exist_ok=True)

    class Tok:
        def save_pretrained(self, d):
            pass

    cfg = EvalConfig(vllm_python="/opt/vllm-venv/bin/python")
    w = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(vllm_backend.VLLMError, match="Unsupported quantization config: bits=3, sym=True"):
        vllm_backend.measure(Model(), Tok(), None, w, w, cfg, 16, 128)
    assert seen["env"]["PATH"].split(__import__("os").pathsep)[0].replace("\\", "/") == "/opt/vllm-venv/bin"
