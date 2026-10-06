"""Stage 1 weight methods that learn from calibration data: GPTQ, AWQ, Wanda, structured pruning, low-rank.

GPTQ (Frantar et al., 2022, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"):
round a layer's weights one input column at a time and, after each column, spread its rounding error over the
columns not rounded yet, weighted by how the layer's inputs correlate (the inverse Hessian of the layer's squared
output error, H = 2 X^T X). The rounding grid is the same as round-to-nearest (`minmax_grid`), so GPTQ and RTN
differ only in the error feedback. Layers are done in order, each with inputs from the already-rounded layers
before it (the paper's sequential mode). Pruned weights (the plan's magnitude mask) are kept at exactly 0 and
their error is fed back like rounding error, as SparseGPT does with a fixed mask.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from sdf.stage0.planner import CompressionPlan
from sdf.stage0.prune_sweep import magnitude_mask
from sdf.stage0.sensitivity import find_decoder_layers, minmax_grid, round_to_nearest, snap


class _Stop(Exception):
    pass


@torch.no_grad()
def run_to(model: nn.Module, layer: nn.Module, batches: list[torch.Tensor],
           on_input: Callable[[nn.Linear, torch.Tensor], None]) -> None:
    """Run `batches` through the model up to the end of `layer`, calling on_input(linear, its input) for every
    Linear in `layer`. Earlier layers are already compressed, so the inputs are the ones the compressed model sees."""
    handles = [m.register_forward_pre_hook(lambda m, args: on_input(m, args[0]))
               for m in layer.modules() if isinstance(m, nn.Linear)]

    def stop(*_):
        raise _Stop

    handles.append(layer.register_forward_hook(stop))
    device = next(model.parameters()).device
    try:
        for b in batches:
            try:
                model(input_ids=b.to(device), use_cache=False)
            except _Stop:
                pass
    finally:
        for h in handles:
            h.remove()


def input_hessians(model: nn.Module, layer: nn.Module, batches: list[torch.Tensor]) -> dict[nn.Linear, torch.Tensor]:
    """H = 2/n X^T X (float32) of the inputs of every Linear in `layer`, over the tokens of `batches`."""
    hess: dict[nn.Linear, torch.Tensor] = {}
    count: dict[nn.Linear, int] = {}

    def collect(m: nn.Linear, x: torch.Tensor) -> None:
        x = x.reshape(-1, m.in_features).float()
        if m not in hess:
            hess[m], count[m] = torch.zeros(m.in_features, m.in_features, device=x.device), 0
        hess[m].addmm_(x.T, x)
        count[m] += x.shape[0]

    run_to(model, layer, batches, collect)
    return {m: h.mul_(2 / count[m]) for m, h in hess.items()}


@torch.no_grad()
def gptq_quantize_(weight: torch.Tensor, hessian: torch.Tensor, bits: int | None, group_size: int,
                   mask: torch.Tensor | None = None, int_zero: bool = True, damp: float = 0.01,
                   block: int = 128) -> None:
    """Round `weight` (out, in) in place with GPTQ. bits=None: no rounding, only the `mask` (True = kept) with
    error feedback. Each group of `group_size` inputs gets its grid when GPTQ reaches it, from the weights as
    updated so far (group_size <= 0 or not dividing the inputs: one grid per row, from the start)."""
    W = weight.data.float().clone()  # not masked: a pruned weight's value is error to feed back
    cols = W.shape[1]
    H = hessian.clone()
    dead = H.diagonal() == 0  # inputs that are always 0 carry no information
    H[dead, dead] = 1
    W[:, dead] = 0
    H.diagonal().add_(damp * H.diagonal().mean())  # the paper's 1% damping keeps H invertible
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    gs = group_size if 0 < group_size < cols and cols % group_size == 0 else cols
    if gs < cols:
        block = gs * max(1, block // gs)  # whole groups per block, so each grid sees its updated weights
    scale = zero = None
    if bits is not None and gs == cols:
        scale, zero = minmax_grid(W if mask is None else W * mask, bits, int_zero)
        scale, zero = scale[:, 0], zero[:, 0]
    Q = torch.empty_like(W)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        err1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            c = i1 + i
            if bits is not None and gs < cols and c % gs == 0:
                g = W1[:, i:i + gs] if mask is None else W1[:, i:i + gs] * mask[:, c:c + gs]
                scale, zero = (t[:, 0] for t in minmax_grid(g, bits, int_zero))
            w = W1[:, i]
            q = w if bits is None else snap(w, scale, zero, bits)
            if mask is not None:
                q = q * mask[:, c]
            Q[:, c] = q
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            err1[:, i] = err
        W[:, i2:] -= err1 @ Hinv[i1:i2, i2:]
    weight.data.copy_(Q.to(weight.dtype))


def gptq_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
          baseline_bits: int, int_zero: bool = True) -> None:
    """Apply `plan` to every decoder Linear with GPTQ, layer by layer (in place, not restored)."""
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        rounds = lp.bit_width < baseline_bits
        if not rounds and lp.pruning_ratio == 0:
            continue
        for m, H in input_hessians(model, layer, batches).items():
            mask = magnitude_mask(m.weight.data, lp.pruning_ratio) if lp.pruning_ratio else None
            gptq_quantize_(m.weight, H, lp.bit_width if rounds else None, group_size, mask, int_zero)


def _scaled_rtn(w: torch.Tensor, s: torch.Tensor, bits: int, group_size: int, mask: torch.Tensor | None,
                int_zero: bool) -> torch.Tensor:
    """round(W diag(s)) diag(1/s): what the model computes once x / s is folded into the op before the Linear."""
    q = round_to_nearest(w * s, bits, group_size, int_zero) / s
    return q if mask is None else q * mask


@torch.no_grad()
def awq_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
         baseline_bits: int, int_zero: bool = True, grid: int = 20) -> None:
    """Apply `plan` to every decoder Linear with AWQ (Lin et al., 2024, "AWQ: Activation-aware Weight Quantization"),
    layer by layer, in place.

    Input channels that carry large activations matter most, so before rounding each weight column is scaled up by
    s = mean|x|^a (and the input down by the same s, which the real model folds into the op that feeds the Linear;
    here W is replaced by round(W s) / s, which computes the same). a is searched over `grid` steps in [0, 1)
    for the smallest output error on the calibration inputs; a = 0 is plain round-to-nearest, so AWQ is never
    worse than RTN there. Linears fed the same input (attention q/k/v, MLP gate/up) share one s, as they must
    when it is folded into the one op before them. Pruned weights (the plan's magnitude mask) stay 0.
    ponytail: no weight-clipping search (the paper's second step); add it if AWQ trails GPTQ by much."""
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        rounds = lp.bit_width < baseline_bits
        if not rounds and lp.pruning_ratio == 0:
            continue
        inputs: dict[nn.Linear, list[torch.Tensor]] = {}
        run_to(model, layer, batches, lambda m, x: inputs.setdefault(m, []).append(x.reshape(-1, m.in_features)))
        groups: dict[tuple, list[nn.Linear]] = {}
        for m, xs in inputs.items():  # same input tensor -> same group
            groups.setdefault((xs[0].data_ptr(), xs[0].shape), []).append(m)
        for linears in groups.values():
            X = torch.cat(inputs[linears[0]])
            masks = [magnitude_mask(m.weight.data, lp.pruning_ratio) if lp.pruning_ratio else None for m in linears]
            if not rounds:
                for m, mask in zip(linears, masks):
                    m.weight.data.mul_(mask)
                continue
            x_mean = X.abs().float().mean(0)
            ref = [(X @ m.weight.T).float() for m in linears]
            best_err, best = float("inf"), None
            for i in range(grid):
                s = x_mean.pow(i / grid).clamp(min=1e-4)
                s = (s / (s.max() * s.min()).sqrt()).to(X.dtype)
                qs = [_scaled_rtn(m.weight.data, s, lp.bit_width, group_size, mask, int_zero)
                      for m, mask in zip(linears, masks)]
                err = sum(((X @ q.T).float() - r).pow(2).mean().item() for q, r in zip(qs, ref))
                if err < best_err:
                    best_err, best = err, qs
            for m, q in zip(linears, best):
                m.weight.data.copy_(q)


def input_sq_norms(model: nn.Module, layer: nn.Module, batches: list[torch.Tensor]) -> dict[nn.Linear, torch.Tensor]:
    """Sum over calibration tokens of x_j^2, per input channel j, for every Linear in `layer` (float32)."""
    out: dict[nn.Linear, torch.Tensor] = {}

    def collect(m: nn.Linear, x: torch.Tensor) -> None:
        s = x.reshape(-1, m.in_features).float().pow(2).sum(0)
        out[m] = out[m] + s if m in out else s

    run_to(model, layer, batches, collect)
    return out


def _round_(m: nn.Linear, bits: int, group_size: int, baseline_bits: int, int_zero: bool) -> None:
    if bits < baseline_bits:
        m.weight.data.copy_(round_to_nearest(m.weight.data, bits, group_size, int_zero))


@torch.no_grad()
def wanda_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
           baseline_bits: int, int_zero: bool = True) -> None:
    """Unstructured pruning with Wanda (Sun et al., 2024, "A Simple and Effective Pruning Approach for LLMs"):
    each output row drops its planned share of weights with the smallest |w_ij| * ||x_j|| (weight size times how
    large its input is on the calibration data), then the rest is rounded to the planned bits (RTN grid; 0 stays
    on it). Layer by layer, on inputs from the already-compressed layers."""
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        if lp.pruning_ratio == 0 and lp.bit_width >= baseline_bits:
            continue
        norms = input_sq_norms(model, layer, batches) if lp.pruning_ratio else {}
        for m in (m for m in layer.modules() if isinstance(m, nn.Linear)):
            if lp.pruning_ratio:
                m.weight.data.mul_(magnitude_mask(m.weight.data.abs() * norms[m].sqrt().to(m.weight.dtype),
                                                  lp.pruning_ratio))
            _round_(m, lp.bit_width, group_size, baseline_bits, int_zero)


def mlp_linears(layer: nn.Module, hidden: int) -> tuple[list[nn.Linear], nn.Linear] | None:
    """(the Linears into the feed-forward inner width, the one out of it), found by shape so any decoder works:
    the output Linear maps inner -> hidden, the input Linears hidden -> inner (Llama gate/up, OPT fc1)."""
    linears = [m for m in layer.modules() if isinstance(m, nn.Linear)]
    down = [m for m in linears if m.out_features == hidden and m.in_features != hidden]
    if len(down) != 1:
        return None
    ups = [m for m in linears if m.in_features == hidden and m.out_features == down[0].in_features]
    return (ups, down[0]) if ups else None


@torch.no_grad()
def structured_prune_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
                      baseline_bits: int, int_zero: bool = True) -> None:
    """Structured pruning of feed-forward channels (Wanda-sp / FLAP style): a layer drops whole inner channels
    (a row of each input Linear and the matching column of the output Linear), as many as remove the planned share
    of the layer's weights, lowest ||x_j|| * ||W_out[:, j]|| first; then the rest is rounded to the planned bits.
    Removed channels are zeroed here (same accuracy); Stage 4 can cut them out for real size and speed.
    ponytail: attention heads are never removed; add head pruning if feed-forward channels run out."""
    hidden = model.config.hidden_size
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        if lp.pruning_ratio == 0 and lp.bit_width >= baseline_bits:
            continue
        found = mlp_linears(layer, hidden) if lp.pruning_ratio else None
        if lp.pruning_ratio and found is None:
            raise ValueError(f"layer {lp.layer}: no feed-forward block found by shape (hidden size {hidden})")
        if found:
            ups, down = found
            total = sum(m.weight.numel() for m in layer.modules() if isinstance(m, nn.Linear))
            per_channel = sum(m.in_features for m in ups) + down.out_features
            k = min(down.in_features, round(lp.pruning_ratio * total / per_channel))
            score = input_sq_norms(model, layer, batches)[down].sqrt() * down.weight.float().norm(dim=0)
            drop = score.argsort()[:k]
            down.weight.data[:, drop] = 0
            for m in ups:
                m.weight.data[drop] = 0
                if m.bias is not None:
                    m.bias.data[drop] = 0
        for m in (m for m in layer.modules() if isinstance(m, nn.Linear)):
            _round_(m, lp.bit_width, group_size, baseline_bits, int_zero)


@torch.no_grad()
def low_rank_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
              baseline_bits: int, int_zero: bool = True) -> None:
    """Low-rank factorisation, activation-aware (ASVD, Yuan et al., 2023): W ~ A B with A = U_k S_k and
    B = V_k^T D^-1, from the SVD of W D, where D = diag(||x_j||) weighs input channels by how large their inputs
    are. The rank k keeps (1 - planned share) of the layer's numbers: k (out + in) = (1 - r) out * in. Both
    factors are rounded to the planned bits. A layer planned with no share removed is only rounded."""
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        if lp.pruning_ratio == 0 and lp.bit_width >= baseline_bits:
            continue
        norms = input_sq_norms(model, layer, batches) if lp.pruning_ratio else {}
        for m in (m for m in layer.modules() if isinstance(m, nn.Linear)):
            if not lp.pruning_ratio:
                _round_(m, lp.bit_width, group_size, baseline_bits, int_zero)
                continue
            out, inp = m.weight.shape
            k = max(1, int((1 - lp.pruning_ratio) * out * inp / (out + inp)))
            d = norms[m].sqrt().clamp(min=1e-6)
            U, S, Vh = torch.linalg.svd(m.weight.float() * d, full_matrices=False)
            A, B = U[:, :k] * S[:k], Vh[:k] / d
            if lp.bit_width < baseline_bits:
                A = round_to_nearest(A, lp.bit_width, group_size, int_zero)
                B = round_to_nearest(B, lp.bit_width, group_size, int_zero)
            m.weight.data.copy_((A @ B).to(m.weight.dtype))
