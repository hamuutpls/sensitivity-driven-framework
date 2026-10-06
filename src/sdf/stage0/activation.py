"""Stage 0d: activation sensitivity and plan for Stage 2 (SmoothQuant, QuaRot, RPTQ, SpinQuant).

Activations are the numbers flowing into each Linear weight while the model runs. Per decoder layer, Stage 0
plans how many bits those inputs get.

`profile_activations` rounds the inputs of every Linear in one layer at a time to each candidate bit width (per
token, `group_size` channels share a scale) and measures the calibration perplexity rise, the same way the KV
cache is profiled. `plan_activations` then spends a fixed average bit budget where the measured damage is
largest (`allocate_bits`, shared with the KV plan).

`activation_plan_from_weights` is the cheaper alternative kept for comparison: it gives protected and
guarded layers of the weight plan the high bit width and every other layer the low one, which assumes a layer
fragile for weights is fragile for activations too.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator

import torch
from torch import nn

from sdf.stage0.kv_cache import _monotone, allocate_bits
from sdf.stage0.planner import CompressionPlan
from sdf.stage0.sensitivity import _mean_loss, find_decoder_layers, profiling, round_to_nearest
from sdf.utils.cache import JsonFile


@dataclass
class ActivationProfile(JsonFile):
    bits_options: list[int]
    rise: list[list[float]]  # [layer][bits option] calibration perplexity rise
    cost: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        return len(self.rise)


@dataclass(frozen=True)
class ActivationLayerPlan:
    layer: int
    act_bits: int
    protected: bool  # kept at the highest bit width


@dataclass(frozen=True)
class ActivationPlan(JsonFile):
    layers: tuple[ActivationLayerPlan, ...]
    kind: str  # "uniform" | "measured" | "from_weight_plan"

    @property
    def avg_bits(self) -> float:
        # Decoder layers of one model share their activation shapes, so the plain mean is the average.
        return sum(lp.act_bits for lp in self.layers) / len(self.layers)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "avg_bits": self.avg_bits, "layers": [asdict(lp) for lp in self.layers]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActivationPlan":
        return cls(tuple(ActivationLayerPlan(**lp) for lp in d["layers"]), d["kind"])


@contextmanager
def quantize_inputs(layer: nn.Module, bits: int, group_size: int) -> Iterator[None]:
    """Round the input of every Linear in `layer` to `bits` (per token) while the context is open."""
    def hook(mod, args):
        return (round_to_nearest(args[0], bits, group_size),) + tuple(args[1:])

    handles = [m.register_forward_pre_hook(hook) for m in layer.modules() if isinstance(m, nn.Linear)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def profile_activations(model: nn.Module, batches: Iterable[torch.Tensor], bits_options: list[int],
                        group_size: int, device: torch.device | str | None = None,
                        meta: dict[str, Any] | None = None) -> ActivationProfile:
    device = torch.device(device) if device is not None else next(model.parameters()).device
    batches = list(batches)
    if not batches:
        raise ValueError("no calibration batches given")
    with profiling(model, device) as cost_so_far:
        base = math.exp(_mean_loss(model, batches, device))
        rise = []
        for layer in find_decoder_layers(model):
            row = []
            for bits in bits_options:
                with quantize_inputs(layer, bits, group_size):
                    row.append(math.exp(_mean_loss(model, batches, device)) - base)
            rise.append(row)
    cost = cost_so_far(len(batches), sum(b.numel() for b in batches))
    return ActivationProfile(list(bits_options), rise, cost, {**(meta or {}), "baseline_ppl": base})


def plan_activations(profile: ActivationProfile, avg_bits: float) -> ActivationPlan:
    """Measured plan: `avg_bits` per layer on average, spent where rounding hurts most."""
    bits = allocate_bits({i: (1, r) for i, r in enumerate(profile.rise)}, profile.bits_options, avg_bits)
    top = max(profile.bits_options)
    return ActivationPlan(tuple(ActivationLayerPlan(i, bits[i], bits[i] == top) for i in range(profile.num_layers)),
                          "measured")


def activation_plan_from_weights(weights: CompressionPlan, protected_bits: int, compressed_bits: int) -> ActivationPlan:
    return ActivationPlan(tuple(
        ActivationLayerPlan(lp.layer, protected_bits if lp.protected or lp.guarded else compressed_bits,
                            lp.protected or lp.guarded)
        for lp in weights.layers), "from_weight_plan")


def uniform_activation_plan(num_layers: int, bits: int) -> ActivationPlan:
    return ActivationPlan(tuple(ActivationLayerPlan(i, bits, False) for i in range(num_layers)), "uniform")


def predicted_rise(plan: ActivationPlan, profile: ActivationProfile) -> float | None:
    """Sum of the measured per-layer rises at the planned bits (assumes they add up). None when the plan uses a
    bit width that was not measured."""
    total = 0.0
    for lp, r in zip(plan.layers, profile.rise):
        if lp.act_bits not in profile.bits_options:
            return None
        total += _monotone(r)[profile.bits_options.index(lp.act_bits)]
    return total
