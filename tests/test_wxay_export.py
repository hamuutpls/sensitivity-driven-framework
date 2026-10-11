"""WxAy (weights and activations per layer) Stage 1 models on the Stage 4 backends: the compressed-tensors export
vLLM reads, and which backend gets what from the runner."""

import copy
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sdf.config import EvalConfig
from sdf.eval import ct_export as c
from sdf.eval import llamacpp, trtllm_backend, vllm_backend
from sdf.eval.gptq_export import symmetric_grid
from sdf.run import start_run
from sdf.search_space import SEARCH_SPACE
from sdf.stage0.activation import ActivationLayerPlan, ActivationPlan
from sdf.stage0.planner import CompressionPlan, uniform_plan
from sdf.stage0.run import run_stage0
from sdf.stages.runner import run_stage
from conftest import fake_texts


def wplan(bits: list[int]) -> CompressionPlan:
    p = uniform_plan([0.0] * len(bits), 4, 0.0)
    return replace(p, layers=tuple(replace(lp, bit_width=b) for lp, b in zip(p.layers, bits)))


def aplan(bits: list[int]) -> ActivationPlan:
    return ActivationPlan(tuple(ActivationLayerPlan(i, b, b == 16) for i, b in enumerate(bits)), "measured")


def unpack_rows(words: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of c.pack_rows, written from compressed-tensors' layout (value k of a word at bits k*bits..)."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    per = 32 // bits
    vals = torch.stack([(w >> (bits * k)) & (2 ** bits - 1) for k in range(per)], dim=-1)
    return vals.reshape(w.shape[0], -1)


@pytest.mark.parametrize("bits", c.BITS)
def test_pack_rows_matches_compressed_tensors(bits):
    helpers = pytest.importorskip("compressed_tensors.compressors.pack_quantized.helpers")
    q = torch.randint(0, 2 ** bits, (6, 64), generator=torch.Generator().manual_seed(bits))
    ref = helpers.pack_to_int32((q - 2 ** (bits - 1)).to(torch.int8), bits)  # it packs signed values
    assert torch.equal(c.pack_rows(q, bits), ref)


@pytest.mark.parametrize("bits", c.BITS)
def test_pack_rows_round_trip(bits):
    q = torch.randint(0, 2 ** bits, (5, 64), generator=torch.Generator().manual_seed(bits))
    packed = c.pack_rows(q, bits)
    assert packed.dtype == torch.int32 and packed.shape == (5, 64 * bits // 32)
    assert torch.equal(unpack_rows(packed, bits), q)


def test_three_bits_has_no_compressed_tensors_packing():
    with pytest.raises(ValueError, match="no compressed-tensors packing"):
        c.quantize_linear(torch.randn(64, 128), 3, 128)


@pytest.mark.parametrize("bits", c.BITS)
def test_quantize_linear_dequantizes_within_half_a_step(bits):
    w = torch.randn(64, 256, generator=torch.Generator().manual_seed(bits))
    t = c.quantize_linear(w, bits, 128)
    assert t["weight_scale"].shape == (64, 2) and t["weight_scale"].dtype == torch.float16
    assert t["weight_shape"].tolist() == [64, 256]
    signed = unpack_rows(t["weight_packed"], bits) - 2 ** (bits - 1)  # stored with a 2^(bits-1) offset
    step = t["weight_scale"].float().repeat_interleave(128, dim=1)
    assert ((signed * step - w).abs() <= step * 0.51 + 1e-3).all()
    q, _ = symmetric_grid(w, bits, 128)  # the same grid as the GPTQ export
    assert torch.equal(unpack_rows(t["weight_packed"], bits), q.long())


def test_quant_config_gives_each_layer_its_own_pair():
    qc = c.quant_config(wplan([4, 8, 4, 16]), aplan([4, 8, 16, 16]), 128, 64)
    assert qc["quant_method"] == "compressed-tensors" and qc["format"] == "pack-quantized"
    g = qc["config_groups"]
    assert set(g) == {"layer_0", "layer_1", "layer_2"}
    assert g["layer_0"]["targets"] == [r"re:.*layers\.0\..*"]
    assert (g["layer_0"]["weights"]["num_bits"], g["layer_0"]["input_activations"]["num_bits"]) == (4, 4)
    assert (g["layer_1"]["weights"]["num_bits"], g["layer_1"]["input_activations"]["num_bits"]) == (8, 8)
    assert g["layer_2"]["input_activations"] is None  # 16-bit activations: weight-only layer
    a = g["layer_0"]["input_activations"]
    assert a["dynamic"] and a["symmetric"] and a["strategy"] == "group" and a["group_size"] == 64
    assert g["layer_0"]["weights"]["group_size"] == 128 and g["layer_0"]["weights"]["symmetric"]
    assert qc["ignore"] == ["lm_head", r"re:.*layers\.3\..*"]


def test_per_channel_weights_and_unsupported_activation_bits():
    qc = c.quant_config(wplan([4, 4]), aplan([8, 4]), -1, 128)
    assert qc["config_groups"]["layer_0"]["weights"] == {"num_bits": 4, "type": "int", "symmetric": True,
                                                         "dynamic": False, "strategy": "channel"}
    with pytest.raises(ValueError, match="only to \\(4, 8\\) bits, not 6"):
        c.quant_config(wplan([4, 4]), aplan([6, 8]), 128, 128)


def test_layer_regex_does_not_match_a_longer_layer_number():
    import re

    pat = c._layer_regex(3)[3:]
    assert re.match(pat, "model.layers.3.mlp.down_proj")
    assert not re.match(pat, "model.layers.13.mlp.down_proj")


def test_quant_config_refuses_what_vllm_cannot_run():
    with pytest.raises(ValueError, match="16-bit weights with 8-bit activations"):
        c.quant_config(wplan([4, 16]), aplan([4, 8]), 128, 128)
    with pytest.raises(ValueError, match="2 weight layers but 3 activation layers"):
        c.quant_config(wplan([4, 4]), aplan([4, 4, 4]), 128, 128)


class Tok:
    def save_pretrained(self, d):
        pass


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 64, bias=False)


class M(nn.Module):
    config = type("C", (), {"to_dict": lambda self: {"model_type": "llama"}})()

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block() for _ in range(3)])
        self.lm_head = nn.Linear(128, 64, bias=False)


def test_export_writes_packed_layers_and_the_config(tmp_path):
    from safetensors.torch import load_file

    size = c.export(M(), Tok(), wplan([4, 8, 16]), aplan([8, 4, 16]), 128, 128, tmp_path)
    t = load_file(tmp_path / "model.safetensors")
    assert t["model.layers.0.proj.weight_packed"].shape == (64, 128 * 4 // 32)
    assert t["model.layers.1.proj.weight_packed"].shape == (64, 128 * 8 // 32)
    assert t["model.layers.2.proj.weight"].dtype == torch.float16  # 16-bit layer stays as it is
    assert "model.layers.0.proj.weight" not in t and t["lm_head.weight"].dtype == torch.float16 and size > 0
    qc = json.loads((tmp_path / "config.json").read_text())["quantization_config"]
    assert qc["config_groups"]["layer_1"]["input_activations"]["num_bits"] == 4


def test_a_rotated_model_is_not_exported(tmp_path):
    m = M()
    m.model.layers[0].proj.register_buffer("online_rotation", torch.eye(128), persistent=False)
    assert c.has_online_rotation(m) and not c.has_online_rotation(M())
    with pytest.raises(ValueError, match="QuaRot/SpinQuant"):
        c.export(m, Tok(), wplan([4, 4, 4]), aplan([8, 8, 8]), 128, 128, tmp_path)


def _capture_worker(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["args"] = json.loads(open(cmd[-1]).read())
        seen["config"] = json.loads(open(seen["args"]["model_dir"] + "/config.json").read())
        return SimpleNamespace(returncode=0, stdout=json.dumps({"vllm_ppl_val": 1.0}), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


def test_vllm_runs_a_wxay_plan_on_humming_kernels(monkeypatch):
    seen = _capture_worker(monkeypatch)
    cfg = EvalConfig(vllm_python="/opt/vllm/bin/python")
    w = torch.zeros(2, 4, dtype=torch.long)
    out = vllm_backend.measure(M(), Tok(), wplan([4, 8, 4]), w, w, cfg, 16, 128, acts=aplan([8, 4, 16]),
                               act_group_size=128)
    assert seen["args"]["linear_backend"] == "humming"
    assert seen["config"]["quantization_config"]["quant_method"] == "compressed-tensors"
    assert out["vllm_ppl_val"] == 1.0 and out["vllm_size_gb"] > 0


def test_vllm_keeps_gptq_for_weight_plans_and_humming_for_any_wxay_plan(monkeypatch):
    seen = _capture_worker(monkeypatch)
    cfg = EvalConfig(vllm_python="/opt/vllm/bin/python")
    w = torch.zeros(2, 4, dtype=torch.long)
    vllm_backend.measure(M(), Tok(), wplan([4, 8, 4]), w, w, cfg, 16, 128)
    assert seen["args"]["linear_backend"] is None and seen["args"]["act_layers"] == []
    assert seen["config"]["quantization_config"]["quant_method"] == "gptq"
    # 16-bit activations everywhere still take the WxAy format, so A16 and A8/A4 rows differ only in their bits
    vllm_backend.measure(M(), Tok(), wplan([4, 8, 4]), w, w, cfg, 16, 128, acts=aplan([16, 16, 16]))
    assert seen["args"]["linear_backend"] == "humming" and seen["args"]["act_layers"] == []
    assert seen["config"]["quantization_config"]["quant_method"] == "compressed-tensors"
    vllm_backend.measure(M(), Tok(), wplan([4, 16, 4]), w, w, cfg, 16, 128, acts=aplan([8, 16, 16]))
    assert seen["args"]["act_layers"] == [0]  # the layers the worker must find rounding activations


def test_vllm_reports_a_pair_it_cannot_run_as_its_own_error(monkeypatch):
    _capture_worker(monkeypatch)
    w = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(vllm_backend.VLLMError, match="16-bit weights with 8-bit activations"):
        vllm_backend.measure(M(), Tok(), wplan([4, 16, 4]), w, w, EvalConfig(vllm_python="p"), 16, 128,
                             acts=aplan([8, 8, 8]))


@pytest.fixture
def stage0_dir(tiny_llama, tokenizer, small_cfg):
    ctx = start_run(small_cfg)
    run_stage0(ctx, SEARCH_SPACE.make({"calib_samples": 16}), model=copy.deepcopy(tiny_llama), tokenizer=tokenizer,
               text_loader=fake_texts)
    return ctx.run_dir / "stage_0"


def test_runner_sends_wxay_models_to_every_backend(stage0_dir, tiny_llama, tokenizer, small_cfg, monkeypatch):
    """Weight rows as before; a WxAy row's weights to llama.cpp and TensorRT-LLM (labelled 16-bit activations)
    and the whole plan to vLLM; a rotated (QuaRot) model to none of them, with the reason."""
    calls = []

    def fake(name, keys):
        def measure(model, tokenizer, plan, val, held, cfg, baseline_bits, *rest, **kw):
            calls.append((name, plan, kw))
            return {k: 1.0 for k in keys}
        return measure

    monkeypatch.setattr(llamacpp, "measure", fake("llamacpp", llamacpp.MAIN_METRICS))
    monkeypatch.setattr(vllm_backend, "measure", fake("vllm", vllm_backend.MAIN_METRICS))
    monkeypatch.setattr(trtllm_backend, "measure", fake("trtllm", trtllm_backend.MAIN_METRICS))
    cfg = small_cfg.with_overrides({"eval.llamacpp_dir": "lc", "eval.llamacpp_convert": "conv.py",
                                    "eval.vllm_python": "vp", "eval.trtllm_python": "tp",
                                    "stage0.act_group_size": 8})  # not the weight group size, so a mix-up shows
    ctx = start_run(cfg)
    cand = SEARCH_SPACE.make({"calib_samples": 16})
    out = run_stage(ctx, 1, ["rtn", "rtn_with_rtn_act", "rtn_with_quarot"], cand, stage0_dir,
                    model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer, text_loader=fake_texts)
    rows = {f"{r['method']}/{r['variant']}": r for r in json.loads(out["json"].read_text())["rows"]}
    for k, r in rows.items():
        assert r["status"] == "ok", (k, r.get("error"))

    weight_calls = [x for x in calls if isinstance(x[1], CompressionPlan) and not x[2]]
    assert {n for n, *_ in weight_calls} == {"llamacpp", "vllm", "trtllm"}
    joint_vllm = [kw for n, _, kw in calls if n == "vllm" and kw]
    assert joint_vllm and all(isinstance(kw["acts"], ActivationPlan) and kw["act_group_size"] == 8
                              for kw in joint_vllm)
    assert not any(kw for n, _, kw in calls if n != "vllm")  # the others get the weight plan only
    assert all(isinstance(p, CompressionPlan) for _, p, _ in calls if p is not None)

    joint = rows["rtn_with_rtn_act/original"]
    assert joint["info"]["vllm_activations"].startswith("the plan's bits")
    assert "group of 8 inputs" in joint["info"]["vllm_activations"]
    assert joint["info"]["llamacpp_activations"].startswith("16-bit")
    assert joint["info"]["trtllm_activations"].startswith("16-bit")
    assert joint["metrics"]["vllm_ppl_val"] == 1.0 and joint["metrics"]["llamacpp_ppl_val"] == 1.0
    assert "llamacpp_activations" not in rows["rtn/original"]["info"]
    rotated = rows["rtn_with_quarot/original"]
    for name in ("llamacpp", "vllm", "trtllm"):
        assert "QuaRot/SpinQuant" in rotated["info"][f"{name}_error"]
    assert "vllm_ppl_val" not in rotated["metrics"]

