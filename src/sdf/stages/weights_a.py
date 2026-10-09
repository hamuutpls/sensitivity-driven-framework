"""More Stage 1 weight quantization methods: OmniQuant (LWC), SqueezeLLM, SpQR and EfficientQAT (Block-AP).
Same contract as `gptq_` in weights.py: simulated quantization in place, per-layer bits from the plan, nothing
removed (`require_bits_only`). Each method is a plain-torch port of the core of the user's fork of the original repo
(file paths and commits in the comment above each method); every simplification is listed in its docstring.

The two trained methods (OmniQuant, EfficientQAT) minimise a Linear's output error on the calibration inputs
through its input Hessian: mean ||X (Wq - W)^T||^2 = tr(dW H dW^T) / 2 for H = 2/n X^T X, which is exactly
the squared output error but needs no stored activations and costs one matmul per step.
"""

from __future__ import annotations

from typing import Iterator

import math

import torch
from torch import nn

from sdf.stage0.planner import CompressionPlan, require_bits_only
from sdf.stage0.sensitivity import find_decoder_layers, minmax_grid, round_to_nearest, snap
from sdf.stages.weights import input_hessians


def _hessian_layers(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], baseline_bits: int,
                    name: str) -> Iterator[tuple[int, dict[nn.Linear, torch.Tensor]]]:
    """(bits, input Hessians) of each layer below baseline, in order; the caller rounds the layer before the next
    one is asked for, so later layers see the inputs of the rounded model."""
    require_bits_only(plan, name)
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        if lp.bit_width < baseline_bits:
            yield lp.bit_width, input_hessians(model, layer, batches)


def _grouped(w: torch.Tensor, group_size: int) -> torch.Tensor:
    n = w.shape[1]
    g = group_size if 0 < group_size < n and n % group_size == 0 else n
    return w.float().reshape(w.shape[0], n // g, g)


def _ste_quant(w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int, int_zero: bool,
               zero_grad: bool = True) -> torch.Tensor:
    """Fake-quantize grouped `w` (out, groups, g) with a straight-through rounding, so scale/zero/w get gradients
    (zero_grad=False: the zero point is rounded without a straight-through, so it gets none, as in OmniQuant)."""
    ste = lambda x: x + (x.round() - x).detach()  # noqa: E731
    if int_zero:
        zero = (ste(zero) if zero_grad else zero.round()).clamp(0, 2 ** bits - 1)
    return (ste(w / scale + zero).clamp(0, 2 ** bits - 1) - zero) * scale


def _output_error(wq: torch.Tensor, w0: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    d = (wq - w0).reshape(w0.shape[0], -1)
    return ((d @ H) * d).sum() / d.shape[0]


def _store_(m: nn.Linear, w: torch.Tensor) -> None:
    m.weight.data.copy_(w.reshape(m.weight.shape).to(m.weight.dtype))


# OmniQuant: fork /tmp/forks/OmniQuant (quantize/quantizer.py UniformAffineQuantizer LWC, quantize/omniquant.py
# AdamW lwc_lr 1e-2 wd 0, MSE of block output), commit feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14
def omniquant_quantize_(m: nn.Linear, H: torch.Tensor, bits: int, group_size: int, int_zero: bool, steps: int,
                        lr: float = 1e-2) -> None:
    W = _grouped(m.weight.data, group_size)
    hi, lo = W.amax(-1, keepdim=True), W.amin(-1, keepdim=True)
    g = torch.full_like(hi, 4.0).requires_grad_()  # sigmoid(4) = 0.98 of the range, as in the repo
    b = torch.full_like(lo, 4.0).requires_grad_()
    top = 2 ** bits - 1

    def quant() -> torch.Tensor:
        h, l = torch.sigmoid(g) * hi, torch.sigmoid(b) * lo
        scale = ((h - l) / top).clamp(min=1e-5)
        return _ste_quant(W, scale, -l / scale, bits, int_zero, zero_grad=False)

    opt = torch.optim.AdamW([g, b], lr=lr, weight_decay=0)
    for _ in range(steps):
        opt.zero_grad()
        _output_error(quant(), W, H).backward()
        opt.step()
    with torch.no_grad():
        _store_(m, quant())


def omniquant_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
               baseline_bits: int, int_zero: bool = True, steps: int = 100) -> None:
    """OmniQuant (Shao et al., 2024), learnable weight clipping (LWC) only: per output channel and group, the
    max and min of the rounding grid are shrunk by learned factors sigmoid(gamma), sigmoid(beta) (start 0.98),
    trained with Adam (lr 1e-2) and a straight-through rounding to minimise the Linear's output error against
    the FP output on the calibration inputs; the grid is then fixed and the weights rounded.
    Simplified: no learnable equivalent transformation (LET); the loss is per Linear (via the input Hessian,
    full batch, `steps` steps) instead of per transformer block over epochs of mini-batches."""
    for bits, hess in _hessian_layers(model, plan, batches, baseline_bits, "OmniQuant"):
        for m, H in hess.items():
            omniquant_quantize_(m, H, bits, group_size, int_zero, steps)


def efficientqat_quantize_(m: nn.Linear, H: torch.Tensor, bits: int, group_size: int, int_zero: bool, steps: int,
                           lr_w: float, lr_q: float) -> None:
    W0 = _grouped(m.weight.data, group_size)
    scale, zero = minmax_grid(W0, bits, int_zero)
    W, scale, zero = (t.clone().requires_grad_() for t in (W0, scale, zero))
    opt = torch.optim.AdamW([{"params": [W], "lr": lr_w}, {"params": [scale, zero], "lr": lr_q}], weight_decay=0)
    # cosine decay to lr / 20 (the repo's --min_lr_factor)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: 0.05 + 0.95 * (1 + math.cos(math.pi * t / steps)) / 2)

    def quant() -> torch.Tensor:
        return _ste_quant(W, scale.clamp(min=1e-4), zero, bits, int_zero)

    for _ in range(steps):
        opt.zero_grad()
        _output_error(quant(), W0, H).backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        _store_(m, quant())


# EfficientQAT: fork /tmp/forks/EfficientQAT (quantize/block_ap.py, quantize/quantizer.py), commit 39f37f3b6053681c9b1cd4c9dcaf692d8999459e
def efficientqat_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
                  baseline_bits: int, int_zero: bool = True, steps: int = 200, lr_w: float = 1e-5,
                  lr_q: float = 1e-4) -> None:
    """EfficientQAT (Chen et al., 2024), Block-AP only: the weights, scales and zero points of every Linear are
    trained together with AdamW (weight decay 0) and a straight-through quantizer to minimise the output error
    against the FP weights on the calibration inputs, then rounded with the learned grid. The learning rates are
    the repo's for 3-/4-bit: 1e-5 for weights, 1e-4 for quantization parameters, cosine-decayed to 1/20. Simplified: no E2E-QP; the loss is per Linear (via the input Hessian) with full-batch steps
    instead of per block with mini-batches over 2 epochs."""
    for bits, hess in _hessian_layers(model, plan, batches, baseline_bits, "EfficientQAT"):
        for m, H in hess.items():
            efficientqat_quantize_(m, H, bits, group_size, int_zero, steps, lr_w, lr_q)


def _fisher(model: nn.Module, targets: list[nn.Linear], batches: list[torch.Tensor]) -> dict[nn.Linear, torch.Tensor]:
    """Sum over `batches` of the squared gradient of the LM loss w.r.t. each target weight (diagonal Fisher)."""
    fisher = {m: torch.zeros_like(m.weight, dtype=torch.float32) for m in targets}
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for m in targets:
        m.weight.requires_grad_(True)
    device = next(model.parameters()).device
    try:
        with torch.enable_grad():
            for b in batches:
                model(input_ids=b.to(device), labels=b.to(device), use_cache=False).loss.backward()
                for m in targets:
                    fisher[m] += m.weight.grad.float().square()
                    m.weight.grad = None
    finally:
        for p, f in flags:
            p.requires_grad_(f)
    return fisher


def _top_share(x: torch.Tensor, share: float) -> torch.Tensor:
    k = int(share * x.numel())
    return x > x.flatten().kthvalue(x.numel() - k).values if k else torch.zeros_like(x, dtype=torch.bool)


def _kmeans_rows(w: torch.Tensor, f: torch.Tensor, outlier: torch.Tensor, k: int, iters: int = 50) -> torch.Tensor:
    """Per-row weighted 1-D k-means (rows, n) -> each weight replaced by its row's nearest of k centroids
    (outliers get weight 0 and keep their value)."""
    f = f / f.mean(-1, keepdim=True).clamp(min=1e-30) + 1e-6
    f = f.masked_fill(outlier, 0)
    lo = w.masked_fill(outlier, float("inf")).amin(-1, keepdim=True)
    hi = w.masked_fill(outlier, float("-inf")).amax(-1, keepdim=True)
    c = lo + (hi - lo) * torch.linspace(0, 1, k, device=w.device)
    for _ in range(iters):
        idx = torch.searchsorted(((c[:, 1:] + c[:, :-1]) / 2).contiguous(), w.contiguous())
        num = torch.zeros_like(c).scatter_add_(1, idx, f * w)
        den = torch.zeros_like(c).scatter_add_(1, idx, f)
        c = torch.where(den > 0, num / den.clamp(min=1e-30), c).sort(1).values
    idx = torch.searchsorted(((c[:, 1:] + c[:, :-1]) / 2).contiguous(), w.contiguous())
    return torch.where(outlier, w, c.gather(1, idx))


# SqueezeLLM: fork /tmp/forks/SqueezeLLM (quantization/nuq.py k-means with Fisher sample weights, max_iter 50;
# squeezellm/outliers.py sensitivity and magnitude outliers), commit a5fd71f353bf569feb7b55737c1c1493e78e8f31
@torch.no_grad()
def squeezellm_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
                baseline_bits: int, int_zero: bool = True, sensitive_share: float = 0.0005,
                magnitude_share: float = 0.004) -> None:
    """SqueezeLLM (Kim et al., 2024): non-uniform quantization. Every row of a Linear gets its own 2**bits-entry
    codebook, fit by k-means weighted with the Fisher information (squared gradient of the calibration LM loss
    w.r.t. each weight, summed over `batches` on the FP model; like the paper, all layers are fit from one
    gradient pass, not sequentially). The dense-and-sparse part keeps two kinds of weights of each Linear in FP16
    and fits the codebook to the rest: the `sensitive_share` (0.05%) with the largest Fisher value and the
    `magnitude_share` (0.4%) with the largest magnitude, 0.45% together as in the paper. They are stored sparse and
    are not pruning, so the stored size is slightly bigger than the bits suggest (about +0.45% x (16 + index bits)
    per weight). `group_size` and `int_zero` do not apply (the codebook is per row). Simplified: the magnitude
    outliers are the top share instead of the repo's inter-quartile-range threshold; k-means starts from a uniform
    codebook (sklearn: k-means++) and runs on the GPU for 50 Lloyd iterations; no CUDA lookup-table kernels."""
    require_bits_only(plan, "SqueezeLLM")
    todo = [(lp.bit_width, [m for m in layer.modules() if isinstance(m, nn.Linear)])
            for lp, layer in zip(plan.layers, find_decoder_layers(model)) if lp.bit_width < baseline_bits]
    fisher = _fisher(model, [m for _, ms in todo for m in ms], batches)
    for bits, linears in todo:
        for m in linears:
            w = m.weight.data.float()
            f = fisher.pop(m)
            outlier = _top_share(w.abs(), magnitude_share) | _top_share(f, sensitive_share)
            m.weight.data.copy_(_kmeans_rows(w, f, outlier, 2 ** bits).to(m.weight.dtype))


def _spqr_hinv(W: torch.Tensor, H: torch.Tensor, damp: float) -> torch.Tensor:
    H = H.clone()
    H.diagonal().add_(damp * H.diagonal().abs().mean())
    dead = H.diagonal() == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)


def _loo_gain(w: torch.Tensor, d: torch.Tensor, bits: int) -> torch.Tensor:
    """(rows, g): how much the group's squared quantization error (each entry scaled by 1/d) drops when entry j is
    left out of the min/max that sets the grid and of the sum."""
    def error(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:  # x, keep (..., g)
        lo = x.masked_fill(~keep, float("inf")).amin(-1, keepdim=True)
        hi = x.masked_fill(~keep, float("-inf")).amax(-1, keepdim=True)
        scale, zero = minmax_grid(torch.cat([lo, hi], -1), bits, False)
        return (((x - snap(x, scale, zero, bits)) / d).square() * keep).sum(-1)

    g = w.shape[1]
    keep = ~torch.eye(g, dtype=torch.bool, device=w.device)
    full = error(w, torch.ones_like(w, dtype=torch.bool))
    return full[:, None] - error(w[:, None, :].expand(-1, g, -1), keep)


def _second_level(v: torch.Tensor, bits: int, size: int) -> torch.Tensor:
    return round_to_nearest(v.reshape(-1, size), bits, size).reshape(v.shape) if v.numel() % size == 0 else v


@torch.no_grad()
def spqr_quantize_(weight: torch.Tensor, hessian: torch.Tensor, bits: int, group: int, threshold: float,
                   stat_bits: int, damp: float = 1.0, block: int = 128) -> None:
    W = weight.data.float().clone()
    rows, cols = W.shape
    perm = hessian.diagonal().argsort(descending=True)  # act_order: biggest-input columns first
    W, hessian = W[:, perm], hessian[perm][:, perm]
    Hinv = _spqr_hinv(W, hessian, damp)
    d_all = Hinv.diagonal()
    limit = threshold * (W.var(0) / d_all.square()).mean()  # SpQR repo: relative threshold x typical error
    Q = torch.empty_like(W)
    block = group * max(1, block // group)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        err1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            c = i1 + i
            if c % group == 0:  # grid of this group: from the weights as updated so far, minus likely outliers
                gw = W1[:, i:i + group]
                likely = _loo_gain(gw, d_all[c:c + group], bits) > limit
                lo = gw.masked_fill(likely, float("inf")).amin(1)
                hi = gw.masked_fill(likely, float("-inf")).amax(1)
                none = likely.all(1)  # a row of nothing but likely outliers: use all its weights
                lo, hi = torch.where(none, gw.amin(1), lo), torch.where(none, gw.amax(1), hi)
                scale, zero = minmax_grid(torch.stack([lo, hi], -1), bits, False)
                scale, zero = (_second_level(t[:, 0], stat_bits, 16) for t in (scale, zero))
                scale = scale.clamp(min=1e-12)
            w = W1[:, i]
            q = snap(w, scale, zero, bits)
            q = torch.where(((w - q) / d_all[c]).square() > limit, w, q)  # outlier: keep in FP16
            Q[:, c] = q
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            err1[:, i] = err
        W[:, i2:] -= err1 @ Hinv[i1:i2, i2:]
    weight.data.copy_(Q[:, perm.argsort()].to(weight.dtype))


# SpQR: fork /tmp/forks/SpQR (spqr_engine.py SPQRUtil.quantize and get_leave_one_out_error, quant_groups.py
# Quantizer qq_scale/qq_zero; README settings wbits 4, groupsize 16, qq bits 3, outlier_threshold 0.2,
# permutation act_order, percdamp 1.0), commit 543d10e56f921e5eeb10d5f0aaaccd30b6662d0e
def spqr_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
          baseline_bits: int, int_zero: bool = True, threshold: float = 0.2, stat_bits: int = 3) -> None:
    """SpQR (Dettmers et al., 2023): GPTQ error feedback with small groups (16 weights, whatever `group_size`
    says) and unstructured FP16 outliers. A weight is an outlier, kept exactly and not fed back, when its
    squared quantization error (scaled by 1/Hinv_ii) exceeds `threshold` x the layer's typical value
    mean(var(W[:, j]) / Hinv_jj^2) (the repo's relative threshold, 0.2). Weights whose leave-one-out gain
    in group error exceeds the same limit are left out when the group's min/max grid is chosen. The grid's
    scales and zeros are themselves rounded to `stat_bits` (3) bits in blocks of 16 output rows, as in the paper.
    Columns are taken in act-order (largest Hessian diagonal first) and the damping is 1.0, the README settings.
    Simplified: no integer zero point (`int_zero` unused: zeros are floats,
    second-level quantized), outliers are not stored in the paper's CSR layout (only counted: ~1%).
    Layer inputs with width not divisible by 16 are not supported."""
    for bits, hess in _hessian_layers(model, plan, batches, baseline_bits, "SpQR"):
        for m, H in hess.items():
            spqr_quantize_(m.weight, H, bits, 16, threshold, stat_bits)


# (fn, label, search-space params read, library note, simplified vs the fork)
WEIGHT_METHODS_A = {
    "omniquant": (omniquant_, "OmniQuant (LWC)", ("gptq_groupsize",), "own torch port of the OmniQuant fork, LWC only",
                  "no LET; per-Linear Hessian loss with 100 full-batch AdamW steps instead of block-output MSE over 20 epochs"),
    "squeezellm": (squeezellm_, "SqueezeLLM", ("gptq_groupsize",), "own torch port of the SqueezeLLM fork",
                   "magnitude outliers by top share not IQR threshold; GPU Lloyd k-means from uniform init, not sklearn k-means++; no CUDA kernels"),
    "spqr": (spqr_, "SpQR", ("gptq_groupsize",), "own torch port of the SpQR fork",
             "no integer-zero option, outliers kept dense (not CSR); group size fixed at 16; no Spearman permutation"),
    "efficientqat": (efficientqat_, "EfficientQAT (Block-AP)", ("gptq_groupsize",), "own torch port of the EfficientQAT fork, Block-AP only",
                     "no E2E-QP; per-Linear Hessian loss with 200 full-batch steps instead of block-output MSE over 2 epochs of mini-batches"),
}
