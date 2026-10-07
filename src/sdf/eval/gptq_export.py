"""Write a bits-only plan's model as a GPTQ-format checkpoint, the packed format vLLM (and TensorRT-LLM's GPTQ
converter) read: every decoder Linear is stored as integers, 2/3/4/8 bits per weight, one scale and zero point per
group of `group_size` inputs. Each layer keeps the bits its plan gives it (vLLM's `dynamic` overrides carry the
per-layer widths); embeddings, norms and the output head stay FP16, as in the plan.

The weights are re-rounded onto symmetric grids of their own groups (scale = 2 * max|w| / (2^bits - 1), zero point
2^(bits-1), as AutoGPTQ's sym=True), like llama.cpp re-rounds: the grids a method used while rounding (GPTQ's updated
weights, AWQ's column scales) are not kept in the model, so the file's weights differ slightly from the HF model's.
Symmetric because vLLM loads only sym=True GPTQ checkpoints ("Unsupported quantization config: bits=4, sym=False" on
vLLM 0.31). The v1 GPTQ layout stores zero - 1.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from sdf.stage0.planner import CompressionPlan, require_bits_only
from sdf.stage0.sensitivity import find_decoder_layers

BITS = (2, 3, 4, 8)


def pack(q: np.ndarray, bits: int) -> np.ndarray:
    """q (N, M), values in 0..2^bits-1, N a multiple of 32 -> int32 (N * bits / 32, M), packed down axis 0 in the
    GPTQ layout (low bits first; 3-bit values spill across three words per 32 values)."""
    n, m = q.shape
    if bits not in BITS or n % 32:
        raise ValueError(f"cannot pack {bits}-bit values in rows of {n}")
    q = q.astype(np.uint64)
    if bits != 3:
        per = 32 // bits
        shifts = (np.arange(per, dtype=np.uint64) * bits)[None, :, None]
        words = np.bitwise_or.reduce(q.reshape(n // per, per, m) << shifts, axis=1)
    else:
        v = q.reshape(n // 32, 32, m)
        sh = lambda lo, hi, off: np.bitwise_or.reduce(
            np.stack([v[:, k] << np.uint64(3 * (k - lo) + off) for k in range(lo, hi)]), axis=0)
        w0 = sh(0, 10, 0) | (v[:, 10] << np.uint64(30))
        w1 = (v[:, 10] >> np.uint64(2)) & np.uint64(1) | sh(11, 21, 1) | (v[:, 21] << np.uint64(31))
        w2 = (v[:, 21] >> np.uint64(1)) & np.uint64(3) | sh(22, 32, 2)
        words = np.stack([w0, w1, w2], axis=1).reshape(n // 32 * 3, m)
    return (words & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.int32)


def quantize_linear(weight: torch.Tensor, bits: int, group_size: int) -> dict[str, torch.Tensor]:
    """GPTQ-format tensors of one Linear weight (out, in): qweight (in*bits/32, out), qzeros (groups, out*bits/32),
    scales (groups, out) FP16, g_idx (in)."""
    out_f, in_f = weight.shape
    if in_f % group_size or in_f % 32 or out_f % 32:
        raise ValueError(f"shape {tuple(weight.shape)} does not fit group size {group_size} and 32-wide packing")
    top = 2 ** bits - 1
    w = weight.detach().float().reshape(out_f, in_f // group_size, group_size)
    scale = (2 * w.abs().amax(-1, keepdim=True) / top).clamp(min=1e-8)
    zero = torch.full_like(scale, 2 ** (bits - 1))
    q = (w / scale + zero).round().clamp(0, top)
    qw = q.reshape(out_f, in_f).T.cpu().numpy()  # (in, out)
    qz = zero[..., 0].T.cpu().numpy()  # (groups, out)
    return {"qweight": torch.from_numpy(pack(qw, bits)),
            "qzeros": torch.from_numpy(pack((qz - 1).T, bits).T.copy()),  # zero - 1, packed along out
            "scales": scale[..., 0].T.half().cpu().contiguous(),
            "g_idx": (torch.arange(in_f) // group_size).int()}


def export(model: nn.Module, tokenizer: Any, plan: CompressionPlan, group_size: int, out_dir: Path) -> float:
    """Save `model` under `plan` (bits only) to out_dir; returns the checkpoint size in GB."""
    from safetensors.torch import save_file

    require_bits_only(plan, "the GPTQ-format export")
    layers = find_decoder_layers(model)
    bits_of = {lp.layer: lp.bit_width for lp in plan.layers}
    index = {id(m): i for i, layer in enumerate(layers) for m in layer.modules()}
    quantized: set[str] = set()
    tensors: dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and id(m) in index and bits_of[index[id(m)]] < 16:
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
    common = Counter(bits_of.values()).most_common(1)[0][0]
    dynamic = {rf"+:.*layers\.{i}\..*": {"bits": b} for i, b in bits_of.items() if b != common and b < 16}
    cfg = model.config.to_dict()
    cfg["quantization_config"] = {"quant_method": "gptq", "bits": common, "group_size": group_size, "desc_act": False,
                                  "sym": True, "true_sequential": True, "dynamic": dynamic}
    cfg["torch_dtype"] = "float16"
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    tokenizer.save_pretrained(out_dir)
    return sum(f.stat().st_size for f in out_dir.glob("model.safetensors")) / 1e9
