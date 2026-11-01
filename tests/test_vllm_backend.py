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
    monkeypatch.setattr("sdf.eval.worker.run_killing_group", lambda cmd, timeout, **kw: subprocess.run(cmd, **kw))  # it uses Popen

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


def test_memory_share_leaves_room_for_what_is_already_on_the_card():
    from sdf.eval.vllm_worker import memory_share

    assert memory_share(80.0, 80.0) == 0.92  # an empty card: vLLM's default
    # the 40 GB A100 check that failed at 0.92: 36.27 of 39.49 GiB free
    share = memory_share(36.27, 39.49)
    assert share * 39.49 < 36.27 and abs(share - (36.27 / 39.49 - 0.02)) < 1e-12
    with pytest.raises(RuntimeError, match="only 0.10 of 40.00 GB"):  # a full card: a clear reason, not a bad share
        memory_share(0.1e9, 40e9)


class _Stop(Exception):
    pass


def _worker_share(monkeypatch, args):
    """Run vllm_worker.main against a stand-in vLLM up to the LLM call; returns the share it asked for."""
    import sys
    import types

    from sdf.eval import vllm_worker

    seen = {}

    def llm(**kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=llm, SamplingParams=None))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (36.27e9, 39.49e9))
    with pytest.raises(_Stop):
        vllm_worker.main({"model_dir": "m", "seq_len": 8, "prompt_len": 4, "decode_tokens": 2, "linear_backend": None,
                          **args})
    return seen["gpu_memory_utilization"]


def test_worker_takes_the_free_share_unless_one_is_set(monkeypatch):
    assert abs(_worker_share(monkeypatch, {}) - (36.27 / 39.49 - 0.02)) < 1e-12
    assert _worker_share(monkeypatch, {"gpu_memory_utilization": 0.5}) == 0.5


def test_backend_hands_the_memory_share_setting_to_the_worker(monkeypatch, tmp_path):
    import json

    seen = []

    def fake_run(cmd, **kw):
        seen.append(json.loads(open(cmd[-1]).read())["gpu_memory_utilization"])
        return SimpleNamespace(returncode=1, stdout="", stderr="stop")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("sdf.eval.worker.run_killing_group", lambda cmd, timeout, **kw: subprocess.run(cmd, **kw))  # it uses Popen

    class Model:
        def save_pretrained(self, d, safe_serialization=True):
            d.mkdir(parents=True, exist_ok=True)

    class Tok:
        def save_pretrained(self, d):
            pass

    w = torch.zeros(2, 4, dtype=torch.long)
    for share in (None, 0.6):
        with pytest.raises(vllm_backend.VLLMError):
            vllm_backend.measure(Model(), Tok(), None, w, w,
                                 EvalConfig(vllm_python="py", vllm_gpu_memory_utilization=share), 16, 128)
    assert seen == [None, 0.6]


def test_the_final_model_is_kept_and_other_exports_are_deleted(monkeypatch, tmp_path):
    # Mohammad, 2026-11-01: "save every model produced by each test", "i mean the model at the end of each run
    # after stage 4"
    from sdf.eval.keep import keep_models
    from sdf.stage0.planner import CompressionPlan, LayerPlan

    import tempfile

    monkeypatch.setattr(vllm_backend, "run_worker", lambda *a: {"vllm_decode_ms_per_token": 2.0})
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))  # where the exports go before they are kept

    class Model:
        def save_pretrained(self, d, safe_serialization=True):
            d.mkdir(parents=True, exist_ok=True)
            (d / "model.safetensors").write_bytes(b"w")

    class Tok:
        def save_pretrained(self, d):
            (d / "tokenizer.json").write_text("{}")

    w = torch.zeros(2, 4, dtype=torch.long)
    plan = CompressionPlan((LayerPlan(0, 16, 0.0, True, 1.0),), "sensitivity", 0.5, 0.0)
    cfg = EvalConfig(vllm_python="py")
    vllm_backend.measure(Model(), Tok(), plan, w, w, cfg, 16, 128)  # a search try: nothing kept
    with keep_models(tmp_path / "final") as folder:
        vllm_backend.measure(Model(), Tok(), None, w, w, cfg, 16, 128)  # the uncompressed row: not kept
        assert not (folder / "vllm").exists()
        vllm_backend.measure(Model(), Tok(), plan, w, w, cfg, 16, 128)
    kept = tmp_path / "final" / "vllm" / "model"
    assert (kept / "model.safetensors").exists() and (kept / "tokenizer.json").exists()
    assert not any((tmp_path / "tmp").iterdir())  # every other export deleted


def test_keep_moves_every_export_and_replaces_an_older_copy(tmp_path):
    from sdf.eval.keep import keep, keep_models

    src = tmp_path / "tmp"
    (src / "model").mkdir(parents=True)
    (src / "engine").mkdir()
    (src / "q.gguf").write_bytes(b"g")
    assert keep("llamacpp", src / "q.gguf") is None and (src / "q.gguf").exists()  # outside keep_models
    with keep_models(tmp_path / "final"):
        (tmp_path / "final" / "trtllm" / "old").mkdir(parents=True)
        dest = keep("trtllm", src / "model", src / "engine")
        keep("llamacpp", src / "q.gguf")
    assert sorted(p.name for p in dest.iterdir()) == ["engine", "model"]  # the older copy replaced
    assert (tmp_path / "final" / "llamacpp" / "q.gguf").exists() and not any(src.iterdir())
