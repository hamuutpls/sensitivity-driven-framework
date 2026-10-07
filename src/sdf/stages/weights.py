"""Stage 1 weight quantization methods that learn from calibration data: GPTQ and AWQ. Nothing is removed here:
plans handed to Stage 1 carry bits only (pruning is Stage 2, see stages/pruning.py).

GPTQ (Frantar et al., 2022, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"):
round a layer's weights one input column at a time and, after each column, spread its rounding error over the
columns not rounded yet, weighted by how the layer's inputs correlate (the inverse Hessian of the layer's squared
output error, H = 2 X^T X). The rounding grid is the same as round-to-nearest (`minmax_grid`), so GPTQ and RTN
differ only in the error feedback. Layers are done in order, each with inputs from the already-rounded layers
before it (the paper's sequential mode).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from sdf.stage0.planner import CompressionPlan, require_bits_only
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
def gptq_quantize_(weight: torch.Tensor, hessian: torch.Tensor, bits: int, group_size: int,
                   int_zero: bool = True, damp: float = 0.01, block: int = 128) -> None:
    """Round `weight` (out, in) in place with GPTQ. Each group of `group_size` inputs gets its grid when GPTQ reaches it, from the weights as
    updated so far (group_size <= 0 or not dividing the inputs: one grid per row, from the start)."""
    W = weight.data.float().clone()
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
    if gs == cols:
        scale, zero = minmax_grid(W, bits, int_zero)
        scale, zero = scale[:, 0], zero[:, 0]
    Q = torch.empty_like(W)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        err1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            c = i1 + i
            if gs < cols and c % gs == 0:
                scale, zero = (t[:, 0] for t in minmax_grid(W1[:, i:i + gs], bits, int_zero))
            w = W1[:, i]
            q = snap(w, scale, zero, bits)
            Q[:, c] = q
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            err1[:, i] = err
        W[:, i2:] -= err1 @ Hinv[i1:i2, i2:]
    weight.data.copy_(Q.to(weight.dtype))


def gptq_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
          baseline_bits: int, int_zero: bool = True) -> None:
    """Apply `plan` (bits only) to every decoder Linear with GPTQ, layer by layer (in place, not restored)."""
    require_bits_only(plan, "GPTQ")
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        if lp.bit_width >= baseline_bits:
            continue
        for m, H in input_hessians(model, layer, batches).items():
            gptq_quantize_(m.weight, H, lp.bit_width, group_size, int_zero)


def _scaled_rtn(w: torch.Tensor, s: torch.Tensor, bits: int, group_size: int, int_zero: bool) -> torch.Tensor:
    """round(W diag(s)) diag(1/s): what the model computes once x / s is folded into the op before the Linear."""
    return round_to_nearest(w * s, bits, group_size, int_zero) / s


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
    when it is folded into the one op before them.
    ponytail: no weight-clipping search (the paper's second step); add it if AWQ trails GPTQ by much."""
    require_bits_only(plan, "AWQ")
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        if lp.bit_width >= baseline_bits:
            continue
        inputs: dict[nn.Linear, list[torch.Tensor]] = {}
        run_to(model, layer, batches, lambda m, x: inputs.setdefault(m, []).append(x.reshape(-1, m.in_features)))
        groups: dict[tuple, list[nn.Linear]] = {}
        for m, xs in inputs.items():  # same input tensor -> same group
            groups.setdefault((xs[0].data_ptr(), xs[0].shape), []).append(m)
        for linears in groups.values():
            X = torch.cat(inputs[linears[0]])
            x_mean = X.abs().float().mean(0)
            ref = [(X @ m.weight.T).float() for m in linears]
            best_err, best = float("inf"), None
            for i in range(grid):
                s = x_mean.pow(i / grid).clamp(min=1e-4)
                s = (s / (s.max() * s.min()).sqrt()).to(X.dtype)
                qs = [_scaled_rtn(m.weight.data, s, lp.bit_width, group_size, int_zero) for m in linears]
                err = sum(((X @ q.T).float() - r).pow(2).mean().item() for q, r in zip(qs, ref))
                if err < best_err:
                    best_err, best = err, qs
            for m, q in zip(linears, best):
                m.weight.data.copy_(q)
