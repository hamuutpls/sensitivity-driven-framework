"""Stage 1 weight method `bitsandbytes`: NF4 (4-bit) and LLM.int8-style row-wise absmax (8-bit) weights.

On CUDA the weights are rounded with bitsandbytes' own functions (imported lazily); that path needs a GPU and is
not exercised by the CPU tests. On CPU a pure-torch version of the same maths is used (tests only): NF4 is a fixed
16-value code book applied to each block of 64 weights after dividing by the block's absmax, int8 is round(w /
absmax * 127) per output row.

Fork hamuutpls/bitsandbytes @ 833649043 (clone /tmp/forks/bitsandbytes): bitsandbytes/functional.py quantize_4bit,
dequantize_4bit, int8_vectorwise_quant.
"""

from __future__ import annotations

import torch
from torch import nn

from sdf.stage0.planner import CompressionPlan, require_bits_only
from sdf.stage0.sensitivity import find_decoder_layers

NF4 = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
       -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141506195,
       0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)
BLOCK = 64


def _nf4_torch(w: torch.Tensor) -> torch.Tensor:
    blocks = w.float().reshape(-1, BLOCK)
    absmax = blocks.abs().amax(1, keepdim=True).clamp(min=1e-12)
    code = torch.tensor(NF4, device=w.device)
    idx = ((blocks / absmax)[..., None] - code).abs().argmin(-1)
    return (code[idx] * absmax).reshape(w.shape).to(w.dtype)


def _int8_torch(w: torch.Tensor) -> torch.Tensor:
    absmax = w.float().abs().amax(1, keepdim=True).clamp(min=1e-12)
    return ((w.float() / absmax * 127).round() * absmax / 127).to(w.dtype)


def _round(w: torch.Tensor, bits: int) -> torch.Tensor:
    if not w.is_cuda:
        return _nf4_torch(w) if bits == 4 else _int8_torch(w)
    import bitsandbytes.functional as F
    if bits == 4:
        packed, state = F.quantize_4bit(w, blocksize=BLOCK, quant_type="nf4")
        return F.dequantize_4bit(packed, state).to(w.dtype)
    q, stats, _ = F.int8_vectorwise_quant(w.half())
    return F.int8_vectorwise_dequant(q, stats).to(w.dtype)


@torch.no_grad()
def bnb_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
         baseline_bits: int, int_zero: bool = True) -> None:
    """Round every decoder Linear of layers planned below `baseline_bits` to NF4 (4 bits) or int8 (8 bits), in
    place. `batches`, `group_size` and `int_zero` are unused: bitsandbytes has its own fixed grids."""
    require_bits_only(plan, "bitsandbytes")
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        if lp.bit_width >= baseline_bits:
            continue
        if lp.bit_width not in (4, 8):
            raise ValueError("bitsandbytes has only 4-bit and 8-bit weights")
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                m.weight.copy_(_round(m.weight, lp.bit_width))


# name: (fn, label, search-space params, library note, what differs from the fork)
WEIGHT_METHODS_BNB = {
    "bitsandbytes": (bnb_, "bitsandbytes (NF4 / int8)", (), "bitsandbytes (CUDA GPU; torch fallback on CPU)",
                     "int8 is row-wise absmax without LLM.int8's outlier columns; 4/8 bits only, no per-group size"),
}
