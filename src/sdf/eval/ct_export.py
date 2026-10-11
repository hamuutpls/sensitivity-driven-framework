"""Write a WxAy (weights and activations) Stage 1 model as a compressed-tensors checkpoint, the format vLLM reads
for whole-number weights with whole-number activations: every decoder Linear is stored as packed integers
("pack-quantized", 2/4/8 bits, one FP16 scale per group of `group_size` inputs), and the config tells vLLM to
round that Linear's inputs to the layer's activation bits while the model runs (dynamic, symmetric, one scale per
token and group of `act_group_size` inputs). Each decoder layer gets its own config group, so every layer keeps
its own (weight bits, activation bits) pair; a layer with 16-bit activations is weight-only. Layers at 16-bit
weights, embeddings, norms and the output head stay FP16.

The weights are re-rounded onto the same symmetric grids as `gptq_export` (scale = 2 * max|w| / (2^bits - 1)).
vLLM rounds activations its own way (symmetric, scale from each group's largest magnitude), not the asymmetric
min/max grid the HF model uses (`round_to_nearest`), nor a method's own static scales (RPTQ), so its perplexity is
comparable between rows, not with the HF perplexity. vLLM runs these layers only on its Humming kernels
(`linear_backend="humming"`); its default kernels would skip the activation rounding without a word.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sdf.eval.gptq_export import pack, symmetric_grid
from sdf.stage0.activation import ActivationPlan
from sdf.stage0.planner import CompressionPlan, require_bits_only
from sdf.stage0.sensitivity import find_decoder_layers

BITS = (2, 4, 8)  # pack-quantized widths that fill an int32 exactly


def pack_rows(q: torch.Tensor, bits: int) -> torch.Tensor:
    """q (out, in), values 0..2^bits-1 -> int32 (out, in * bits / 32), packed along the input dimension, low bits
    first: compressed-tensors' pack_to_int32 layout (it stores signed values plus 2^(bits-1), i.e. these). The
    same words as the GPTQ packing, laid along the other axis."""
    if bits not in BITS:
        raise ValueError(f"{bits}-bit values have no compressed-tensors packing here (only {BITS})")
    return torch.from_numpy(pack(q.T.numpy(), bits).T.copy())


def quantize_linear(weight: torch.Tensor, bits: int, group_size: int) -> dict[str, torch.Tensor]:
    """compressed-tensors tensors of one Linear weight (out, in): weight_packed (out, in*bits/32) int32,
    weight_scale (out, groups) FP16, weight_shape [out, in]."""
    q, scale = symmetric_grid(weight, bits, group_size)
    return {"weight_packed": pack_rows(q.cpu(), bits), "weight_scale": scale.half().cpu().contiguous(),
            "weight_shape": torch.tensor(weight.shape, dtype=torch.int64)}


def _layer_regex(i: int) -> str:
    return rf"re:.*layers\.{i}\..*"


def quant_config(weights: CompressionPlan, acts: ActivationPlan, group_size: int, act_group_size: int,
                 baseline_bits: int = 16) -> dict[str, Any]:
    """The `quantization_config` vLLM reads: one config group per decoder layer below `baseline_bits` weights,
    every other layer and the output head in `ignore`."""
    if len(weights.layers) != len(acts.layers):
        raise ValueError(f"{len(weights.layers)} weight layers but {len(acts.layers)} activation layers")
    groups: dict[str, Any] = {}
    ignore = ["lm_head"]
    for w, a in zip(weights.layers, acts.layers):
        if w.bit_width >= baseline_bits:
            if a.act_bits < baseline_bits:
                raise ValueError(f"layer {w.layer}: 16-bit weights with {a.act_bits}-bit activations have no "
                                 "vLLM scheme (its activation rounding needs whole-number weights)")
            ignore.append(_layer_regex(w.layer))
            continue
        group: dict[str, Any] = {
            "targets": [_layer_regex(w.layer)],
            "weights": {"num_bits": w.bit_width, "type": "int", "symmetric": True, "dynamic": False,
                        "strategy": "group", "group_size": group_size},
            "input_activations": None}
        if a.act_bits < baseline_bits:
            group["input_activations"] = {"num_bits": a.act_bits, "type": "int", "symmetric": True, "dynamic": True,
                                          "strategy": "group", "group_size": act_group_size}
        groups[f"layer_{w.layer}"] = group
    return {"quant_method": "compressed-tensors", "format": "pack-quantized", "config_groups": groups,
            "ignore": ignore, "quantization_status": "compressed"}


def has_online_rotation(model: nn.Module) -> bool:
    """True when the model rotates some Linear input while it runs (QuaRot / SpinQuant before down_proj): a saved
    checkpoint holds only the weights, so a backend would run it without the rotation and get it wrong."""
    return any(hasattr(m, "online_rotation") for m in model.modules())


def export(model: nn.Module, tokenizer: Any, weights: CompressionPlan, acts: ActivationPlan, group_size: int,
           act_group_size: int, out_dir: Path, baseline_bits: int = 16) -> float:
    """Save `model` under the joint plan (weights, acts) to out_dir; returns the checkpoint size in GB."""
    from safetensors.torch import save_file

    require_bits_only(weights, "the compressed-tensors export")
    if has_online_rotation(model):
        raise ValueError("the model rotates inputs while it runs (QuaRot/SpinQuant); a saved checkpoint cannot hold "
                         "that rotation")
    qc = quant_config(weights, acts, group_size, act_group_size, baseline_bits)  # validates before writing
    if not qc["config_groups"]:
        raise ValueError(f"no layer below {baseline_bits} bits; nothing to export in compressed-tensors format")
    layers = find_decoder_layers(model)
    bits_of = {lp.layer: lp.bit_width for lp in weights.layers}
    index = {id(m): i for i, layer in enumerate(layers) for m in layer.modules()}
    quantized: set[str] = set()
    tensors: dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and id(m) in index and bits_of[index[id(m)]] < baseline_bits:
            tensors.update({f"{name}.{k}": v.contiguous() for k, v in
                            quantize_linear(m.weight, bits_of[index[id(m)]], group_size).items()})
            if m.bias is not None:
                tensors[f"{name}.bias"] = m.bias.detach().half().cpu()
            quantized.add(name + ".")
    for name, p in model.state_dict().items():
        if not any(name.startswith(q) for q in quantized):
            tensors[name] = p.detach().half().cpu().contiguous() if p.is_floating_point() else p.cpu()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out_dir / "model.safetensors", metadata={"format": "pt"})
    cfg = model.config.to_dict()
    cfg["quantization_config"] = qc
    cfg["torch_dtype"] = "float16"
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    tokenizer.save_pretrained(out_dir)
    return (out_dir / "model.safetensors").stat().st_size / 1e9
