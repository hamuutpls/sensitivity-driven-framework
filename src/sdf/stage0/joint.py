"""Stage 0e: joint weight x activation sensitivity and plan for Stage 1 (W<bits>A<bits> per decoder layer).

The weight plan and the activation plan are each made from their own measurement, as if rounding a layer's
weights and rounding its inputs hurt independently. They do not: both errors meet in the same matrix product
(W + dW)(x + dx) = Wx + W dx + dW x + dW dx, and how much each one costs depends on the other. So here every
decoder layer is measured at every (weight bits, activation bits) pair: the layer's Linear weights are rounded to
w bits (per group, as Stage 1 stores them) and their inputs to a bits (per token), all other layers untouched, and
the calibration perplexity rise is recorded.

`plan_joint` then picks one pair per layer that minimises the summed measured rise under two budgets at once: the
average weight bits (memory) and the average activation bits (the width the multiplications run at). The choice
is exact for that model (dynamic programming over both budgets), not a greedy one. 16 bits is one of the options,
so a layer stays unrounded in either dimension only when the measurement says it is worth the budget; no layer is
protected by a fixed rule.

`interaction` reports how far the measured pairs are from the separate-measurement assumption.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import torch
from torch import nn

from sdf.stage0.activation import ActivationLayerPlan, ActivationPlan, quantize_inputs
from sdf.stage0.planner import CompressionPlan
from sdf.stage0.sensitivity import _mean_loss, find_decoder_layers, profiling, quantize_layer
from sdf.utils.cache import JsonFile


@dataclass
class JointProfile(JsonFile):
    w_options: list[int]  # ascending
    a_options: list[int]  # ascending
    rise: list[list[list[float]]]  # [layer][weight option][activation option] calibration perplexity rise
    layer_numel: list[int]  # weights per decoder layer (Linear weights only)
    cost: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        return len(self.rise)

    def monotone(self, layer: int) -> list[list[float]]:
        """The layer's measured rises, clipped at 0 and made non-increasing in both bit widths (more bits never
        predicts more damage; the rest is measurement noise)."""
        r = [[max(v, 0.0) for v in row] for row in self.rise[layer]]
        for i in range(len(r)):
            for j in range(len(r[i])):
                if i:
                    r[i][j] = min(r[i][j], r[i - 1][j])
                if j:
                    r[i][j] = min(r[i][j], r[i][j - 1])
        return r

    def rise_at(self, layer: int, w: int, a: int) -> float:
        return self.monotone(layer)[self.w_options.index(w)][self.a_options.index(a)]


@torch.no_grad()
def profile_joint(model: nn.Module, batches: Iterable[torch.Tensor], w_options: list[int], a_options: list[int],
                  group_size: int, act_group_size: int, int_zero: bool = True, baseline_bits: int = 16,
                  device: torch.device | str | None = None, meta: dict[str, Any] | None = None) -> JointProfile:
    """Perplexity rise of each decoder layer at each (weight bits, activation bits) pair, one layer at a time."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    batches = list(batches)
    if not batches:
        raise ValueError("no calibration batches given")
    layers = find_decoder_layers(model)
    numel = [sum(m.weight.numel() for m in l.modules() if isinstance(m, nn.Linear)) for l in layers]
    with profiling(model, device) as cost_so_far:
        base = math.exp(_mean_loss(model, batches, device))
        rise = []
        for layer in layers:
            grid = []
            for w in w_options:
                row = []
                for a in a_options:
                    if w >= baseline_bits and a >= baseline_bits:
                        row.append(0.0)
                        continue
                    with ExitStack() as stack:
                        if w < baseline_bits:
                            stack.enter_context(quantize_layer(layer, w, group_size, int_zero))
                        if a < baseline_bits:
                            stack.enter_context(quantize_inputs(layer, a, act_group_size))
                        row.append(math.exp(_mean_loss(model, batches, device)) - base)
                grid.append(row)
            rise.append(grid)
    cost = cost_so_far(len(batches) * len(layers) * (len(w_options) * len(a_options) - 1),
                       sum(b.numel() for b in batches))
    return JointProfile(list(w_options), list(a_options), rise, numel, cost,
                        {**(meta or {}), "baseline_ppl": base})


def plan_joint(profile: JointProfile, avg_weight_bits: float, avg_act_bits: float) -> list[tuple[int, int]]:
    """(weight bits, activation bits) per layer with the smallest summed measured rise such that the weights
    average at most `avg_weight_bits` (weighted by each layer's weight count) and the activations at most
    `avg_act_bits` (plain mean over layers, as ActivationPlan.avg_bits). Exact under the assumption that layers'
    rises add up: dynamic programming over (weight-bit units used, activation-bit units used)."""
    wo, ao = profile.w_options, profile.a_options
    step = math.gcd(*wo, *ao)
    small = min(profile.layer_numel)
    size = [max(1, round(n / small)) for n in profile.layer_numel]  # relative weight count, in units
    w_cap = math.floor(avg_weight_bits / step * sum(size) + 1e-9)
    a_cap = math.floor(avg_act_bits / step * profile.num_layers + 1e-9)
    if min(wo) / step * sum(size) > w_cap or min(ao) / step * profile.num_layers > a_cap:
        raise ValueError(f"budget below the smallest option: weights {avg_weight_bits} < {min(wo)} or "
                         f"activations {avg_act_bits} < {min(ao)}")
    # state (weight units, activation units) -> (total rise, choices so far)
    states: dict[tuple[int, int], tuple[float, tuple[tuple[int, int], ...]]] = {(0, 0): (0.0, ())}
    for i in range(profile.num_layers):
        r = profile.monotone(i)
        nxt: dict[tuple[int, int], tuple[float, tuple[tuple[int, int], ...]]] = {}
        for (wu, au), (cost, picks) in states.items():
            for wi, w in enumerate(wo):
                for ai, a in enumerate(ao):
                    key = (wu + w // step * size[i], au + a // step)
                    if key[0] > w_cap or key[1] > a_cap:
                        continue
                    c = cost + r[wi][ai]
                    if key not in nxt or c < nxt[key][0] - 1e-12:
                        nxt[key] = (c, picks + ((w, a),))
        states = _pareto(nxt)
    return list(min(states.values(), key=lambda t: (t[0], -sum(w for w, _ in t[1])))[1])


def _pareto(states: dict[tuple[int, int], tuple[float, Any]]) -> dict[tuple[int, int], tuple[float, Any]]:
    """Drop states that use at least as many units of both budgets for at least as much rise."""
    keep: dict[tuple[int, int], tuple[float, Any]] = {}
    best_by_a: dict[int, float] = {}
    for (wu, au), v in sorted(states.items()):  # ascending weight units
        if any(c <= v[0] for a2, c in best_by_a.items() if a2 <= au):
            continue
        keep[(wu, au)] = v
        best_by_a[au] = min(v[0], best_by_a.get(au, math.inf))
    return keep


def joint_plans(picks: list[tuple[int, int]], template: CompressionPlan) -> tuple[CompressionPlan, ActivationPlan]:
    """The picks as a weight plan (the Stage 1 quant plan with its bits replaced, nothing pruned) and an
    activation plan. A layer is marked protected in a plan when it got the highest bits of the picks there."""
    top_w, top_a = max(w for w, _ in picks), max(a for _, a in picks)
    weights = CompressionPlan(
        tuple(replace(lp, bit_width=w, pruning_ratio=0.0, protected=w == top_w)
              for lp, (w, _) in zip(template.layers, picks)),
        "joint", template.sensitive_threshold, 0.0)
    acts = ActivationPlan(tuple(ActivationLayerPlan(i, a, a == top_a) for i, (_, a) in enumerate(picks)), "joint")
    return weights, acts


def predicted_joint_rise(profile: JointProfile, picks: list[tuple[int, int]]) -> float | None:
    """Sum of the measured rises at the given pairs; None when a pair was not measured."""
    try:
        return sum(profile.rise_at(i, w, a) for i, (w, a) in enumerate(picks))
    except ValueError:
        return None


def interaction(profile: JointProfile, w: int, a: int) -> dict[str, float]:
    """At (w, a): measured rise of rounding both, against the sum of rounding each alone (the assumption behind
    separate plans), summed over layers. ratio > 1: the two errors make each other worse."""
    hi_w, hi_a = max(profile.w_options), max(profile.a_options)
    both = sum(profile.rise[i][profile.w_options.index(w)][profile.a_options.index(a)]
               for i in range(profile.num_layers))
    alone = sum(profile.rise[i][profile.w_options.index(w)][profile.a_options.index(hi_a)]
                + profile.rise[i][profile.w_options.index(hi_w)][profile.a_options.index(a)]
                for i in range(profile.num_layers))
    return {"joint_rise": both, "sum_of_separate_rises": alone, "ratio": both / alone if alone > 0 else math.nan}


def combos(picks: list[tuple[int, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for w, a in picks:
        out[f"W{w}A{a}"] = out.get(f"W{w}A{a}", 0) + 1
    return dict(sorted(out.items()))

