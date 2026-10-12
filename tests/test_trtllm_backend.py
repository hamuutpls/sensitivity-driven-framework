import json
from dataclasses import replace
from pathlib import Path

import pytest

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


def _mixed(bits_of_layer_1=8):
    plan = uniform_plan([0.0] * 3, 4, 0.0)
    return replace(plan, layers=tuple(replace(lp, bit_width=bits_of_layer_1) if lp.layer == 1 else lp
                                      for lp in plan.layers))


def test_mixed_plan_lists_every_decoder_layer_under_tensorrt_llm_names():
    q = t.quant_file(M(), _mixed(), 128)["quantization"]
    assert q["quant_algo"] == "MIXED_PRECISION"
    assert {k: v["quant_algo"] for k, v in q["quantized_layers"].items()} == {
        "transformer.layers.0.attention.*": "W4A16_GPTQ", "transformer.layers.0.mlp.*": "W4A16_GPTQ",
        "transformer.layers.1.attention.*": "W8A16_GPTQ", "transformer.layers.1.mlp.*": "W8A16_GPTQ",
        "transformer.layers.2.attention.*": "W4A16_GPTQ", "transformer.layers.2.mlp.*": "W4A16_GPTQ"}
    assert all(v["group_size"] == 128 and v["has_zero_point"] for v in q["quantized_layers"].values())


def test_mixed_patterns_match_the_tensorrt_llm_linears_of_their_layer_only():
    from fnmatch import fnmatch

    pats = t.quant_file(M(), _mixed(), 128)["quantization"]["quantized_layers"]
    hit = lambda name: [p for p in pats if fnmatch(name, p)]  # how LayerQuantConfig.layer_quant_mode matches
    for name in ("transformer.layers.1.attention.qkv", "transformer.layers.1.attention.dense",
                 "transformer.layers.1.mlp.fc", "transformer.layers.1.mlp.gate", "transformer.layers.1.mlp.proj"):
        assert len(hit(name)) == 1 and pats[hit(name)[0]]["quant_algo"] == "W8A16_GPTQ", name
    for name in ("transformer.layers.1", "transformer.layers.1.input_layernorm", "transformer.layers.10.mlp.fc",
                 "transformer.vocab_embedding", "transformer.ln_f", "lm_head"):
        assert hit(name) == [], name  # norms, embeddings, the output head and other layers stay as they are


def test_a_16_bit_layer_in_a_mixed_plan_stays_fp16():
    pats = t.quant_file(M(), _mixed(16), 128)["quantization"]["quantized_layers"]
    assert not any(p.startswith("transformer.layers.1.") for p in pats) and len(pats) == 4


def test_a_plan_for_another_model_is_refused():
    plan = uniform_plan([0.0] * 4, 4, 0.0)
    plan = replace(plan, layers=tuple(replace(lp, bit_width=8) if lp.layer == 1 else lp for lp in plan.layers))
    with pytest.raises(ValueError, match="4 layers, the model 3"):
        t.quant_file(M(), plan, 128)


def _written_files(monkeypatch, plan):
    """The files measure() hands to the worker, captured when it launches the worker."""
    import subprocess

    import torch

    from sdf.config import EvalConfig

    seen = {}

    def fake_run(cmd, **kw):
        model_dir = Path(json.loads(Path(cmd[-1]).read_text())["model_dir"])
        seen.update({f.name: f.read_text() for f in model_dir.iterdir() if f.suffix == ".json"})
        return subprocess.CompletedProcess(cmd, 1, "", "stop here")

    monkeypatch.setattr(t.subprocess, "run", fake_run)
    monkeypatch.setattr(t.gptq_export, "export", lambda model, tok, plan, g, out: out.mkdir(parents=True))
    ids = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(t.TRTLLMError):
        t.measure(M(), None, plan, ids, ids, EvalConfig(trtllm_python="python"), 16, 128)
    return seen


def test_measure_writes_the_mixed_file_not_hf_quant_config_for_a_mixed_plan(monkeypatch):
    seen = _written_files(monkeypatch, _mixed())
    assert t.MIXED_FILE in seen and "hf_quant_config.json" not in seen
    assert json.loads(seen[t.MIXED_FILE])["quantization"]["quant_algo"] == "MIXED_PRECISION"


def test_measure_writes_hf_quant_config_for_one_bit_width(monkeypatch):
    seen = _written_files(monkeypatch, uniform_plan([0.0] * 3, 8, 0.0))
    assert t.MIXED_FILE not in seen
    assert json.loads(seen["hf_quant_config.json"])["quantization"]["quant_algo"] == "W8A16_GPTQ"


def test_worker_and_backend_agree_on_the_mixed_file_name():
    from sdf.eval import trtllm_worker

    assert trtllm_worker.MIXED_FILE == t.MIXED_FILE


def test_worker_turns_the_mixed_file_into_a_tensorrt_llm_checkpoint(tmp_path, monkeypatch):
    """mixed_checkpoint with a stand-in for tensorrt_llm: the per-layer config reaches from_hugging_face as a
    LayerQuantConfig, config.json names only MIXED_PRECISION, and quant_cfg.json carries the layer list."""
    import sys
    import types
    from dataclasses import asdict, dataclass

    from sdf.eval import trtllm_worker

    @dataclass
    class QuantConfig:
        quant_algo: str | None = None
        group_size: int = 128
        has_zero_point: bool = False

    class LayerQuantConfig(QuantConfig):
        def __init__(self, quant_algo, quantized_layers):
            super().__init__(quant_algo)
            self.quantized_layers = quantized_layers

    calls = {}

    class Model:
        def __init__(self, qc):
            self.config = types.SimpleNamespace(quantization=qc)
            self.config.to_json_file = lambda path: Path(path).write_text(
                json.dumps({"quantization": asdict(self.config.quantization)}))

        @classmethod
        def from_hugging_face(cls, model_dir, dtype, quant_config):
            calls.update(model_dir=model_dir, dtype=dtype, qc=quant_config)
            return cls(quant_config)

        def save_checkpoint(self, out, save_config):
            calls["save_config"] = save_config
            (Path(out) / "rank0.safetensors").write_text("weights")

    class QuantAlgo(str):  # tensorrt_llm's QuantAlgo is a StrEnum: QuantAlgo("W4A16_GPTQ") == "W4A16_GPTQ"
        MIXED_PRECISION = "MIXED_PRECISION"

    auto = types.SimpleNamespace(get_trtllm_model_class=lambda d: Model)
    fakes = {"tensorrt_llm": types.ModuleType("tensorrt_llm"),
             "tensorrt_llm.models": types.ModuleType("tensorrt_llm.models"),
             "tensorrt_llm.models.automodel": types.SimpleNamespace(AutoModelForCausalLM=auto),
             "tensorrt_llm.models.modeling_utils": types.SimpleNamespace(LayerQuantConfig=LayerQuantConfig,
                                                                         QuantConfig=QuantConfig),
             "tensorrt_llm.quantization": types.SimpleNamespace(QuantAlgo=QuantAlgo)}
    for k, v in fakes.items():
        monkeypatch.setitem(sys.modules, k, v)

    src = tmp_path / "hf"
    src.mkdir()
    q = t.quant_file(M(), _mixed(), 64)
    (src / t.MIXED_FILE).write_text(json.dumps(q))
    out = trtllm_worker.mixed_checkpoint(src, tmp_path / "ckpt")

    assert calls["model_dir"] == str(src) and calls["dtype"] == "float16" and calls["save_config"] is False
    qc = calls["qc"]
    assert qc.quant_algo == "MIXED_PRECISION"
    assert {k: (v.quant_algo, v.group_size, v.has_zero_point) for k, v in qc.quantized_layers.items()} == {
        k: (v["quant_algo"], 64, True) for k, v in q["quantization"]["quantized_layers"].items()}
    assert json.loads((out / "config.json").read_text())["quantization"]["quant_algo"] == "MIXED_PRECISION"
    assert "quantized_layers" not in json.loads((out / "config.json").read_text())["quantization"]
    assert json.loads((out / "quant_cfg.json").read_text()) == {
        "quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None,
        "quantized_layers": q["quantization"]["quantized_layers"]}
    assert (out / "rank0.safetensors").exists()
