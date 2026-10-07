"""Stage 2 pruning methods: Wanda (unstructured), structured pruning of feed-forward channels, low-rank.

Pruning only: a method removes what the plan's pruning ratios say and rounds nothing, except low-rank, whose two
factors are stored at the plan's bits (a plan at the baseline bits leaves them in FP16). Weights are quantized
by Stage 1 when the two stages run in series (0 -> 1 -> 2 -> 4), and left in FP16 when Stage 2 runs alone
(0 -> 2 -> 4).
"""

from __future__ import annotations

import torch
from torch import nn

from sdf.stage0.planner import CompressionPlan
from sdf.stage0.prune_sweep import magnitude_mask
from sdf.stage0.sensitivity import find_decoder_layers, round_to_nearest
from sdf.stages.weights import run_to


def input_sq_norms(model: nn.Module, layer: nn.Module, batches: list[torch.Tensor]) -> dict[nn.Linear, torch.Tensor]:
    """Sum over calibration tokens of x_j^2, per input channel j, for every Linear in `layer` (float32)."""
    out: dict[nn.Linear, torch.Tensor] = {}

    def collect(m: nn.Linear, x: torch.Tensor) -> None:
        s = x.reshape(-1, m.in_features).float().pow(2).sum(0)
        out[m] = out[m] + s if m in out else s

    run_to(model, layer, batches, collect)
    return out


@torch.no_grad()
def wanda_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
           baseline_bits: int, int_zero: bool = True) -> None:
    """Unstructured pruning with Wanda (Sun et al., 2024, "A Simple and Effective Pruning Approach for LLMs"):
    each output row drops its planned share of weights with the smallest |w_ij| * ||x_j|| (weight size times how
    large its input is on the calibration data). Layer by layer, on inputs from the already-pruned layers."""
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        if lp.pruning_ratio == 0:
            continue
        norms = input_sq_norms(model, layer, batches)
        for m in (m for m in layer.modules() if isinstance(m, nn.Linear)):
            # float32: sqrt of a channel's summed squares can pass the FP16 maximum (65504) -> inf / nan scores
            m.weight.data.mul_(magnitude_mask(m.weight.data.float().abs() * norms[m].sqrt(), lp.pruning_ratio))


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
    of the layer's weights, lowest ||x_j|| * ||W_out[:, j]|| first.
    Removed channels are zeroed here (same accuracy); Stage 4 can cut them out for real size and speed.
    ponytail: attention heads are never removed; add head pruning if feed-forward channels run out."""
    hidden = model.config.hidden_size
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        if lp.pruning_ratio == 0:
            continue
        found = mlp_linears(layer, hidden)
        if found is None:
            raise ValueError(f"layer {lp.layer}: no feed-forward block found by shape (hidden size {hidden})")
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


@torch.no_grad()
def low_rank_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
              baseline_bits: int, int_zero: bool = True) -> None:
    """Low-rank factorisation, activation-aware (ASVD, Yuan et al., 2023): W ~ A B with A = U_k S_k and
    B = V_k^T D^-1, from the SVD of W D, where D = diag(||x_j||) weighs input channels by how large their inputs
    are. The rank k keeps (1 - planned share) of the layer's numbers: k (out + in) = (1 - r) out * in. Both
    factors are rounded to the planned bits (nothing is rounded at the baseline bits)."""
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        if lp.pruning_ratio == 0:
            continue
        norms = input_sq_norms(model, layer, batches)
        for m in (m for m in layer.modules() if isinstance(m, nn.Linear)):
            out, inp = m.weight.shape
            k = max(1, int((1 - lp.pruning_ratio) * out * inp / (out + inp)))
            d = norms[m].sqrt().clamp(min=1e-6)
            U, S, Vh = torch.linalg.svd(m.weight.float() * d, full_matrices=False)
            A, B = U[:, :k] * S[:k], Vh[:k] / d
            if lp.bit_width < baseline_bits:
                A = round_to_nearest(A, lp.bit_width, group_size, int_zero)
                B = round_to_nearest(B, lp.bit_width, group_size, int_zero)
            m.weight.data.copy_((A @ B).to(m.weight.dtype))
