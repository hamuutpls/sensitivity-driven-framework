"""Every compression method of Stages 1-3 in one table, and the ones implemented so far.

A method is applied as a context manager on a fresh copy of the model:

    with method.apply(MethodCall(model, plan, ...)) as extra_metrics:
        measure the model

`plan` is the per-layer plan the method follows: the uniform plan built from the Stage 0 "original method"
settings for the standard variant, a Stage 0 plan for the framework variant. So "original" and "framework"
run the same code and differ only in the plan, which is the comparison the thesis makes.

Methods with `apply=None` are listed (with the library they need) but not written yet; the runner records
them as a failed row saying so, so a report never silently drops a method. See
docs/stage-methods-feasibility.md for which library each one needs on Windows and Colab.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Iterator

import torch
from torch import nn

from sdf.stage0.activation import ActivationPlan, quantize_inputs
from sdf.stage0.kv_cache import KVPlan, kv_projections, quantize_output
from sdf.stage0.planner import CompressionPlan
from sdf.stage0.prune_sweep import apply_plan
from sdf.stage0.sensitivity import int_zero
from sdf.stage0.sensitivity import find_decoder_layers
from sdf.stages.weights import awq_, gptq_


@dataclass
class MethodCall:
    model: nn.Module
    plan: Any  # CompressionPlan (Stage 1), ActivationPlan (Stage 2) or KVPlan (Stage 3)
    candidate: dict[str, Any]
    cfg: Any  # FrameworkConfig
    batches: Callable[[], list[torch.Tensor]]  # calibration batches, built on first call


@dataclass(frozen=True)
class Method:
    name: str
    stage: int
    label: str
    # Stage 0 plans the framework variant runs with, one framework row each (keys of runner.PLAN_FILES).
    plans: tuple[str, ...]
    apply: Callable[[MethodCall], ContextManager[dict[str, Any]]] | None = None
    version: int = 1  # bump when the implementation changes, so cached original-method results are redone
    params: tuple[str, ...] = ()  # search-space parameters the method reads (part of its cache key)
    calibrated: bool = False  # reads calibration batches (calibration settings join its cache key)
    simulated: bool = True  # numbers are rounded in place (FP16 storage): speed and file size are not real
    library: str = ""  # where the implementation comes from, for the feasibility table


# ----------------------------------------------------------------------------------------------- baselines
# Round-to-nearest (RTN) is the standard "no cleverness" baseline in the GPTQ / AWQ / SmoothQuant papers. Here
# it proves the runner end to end on every stage, and it is the floor every real method must beat.

@contextmanager
def _rtn_weights(call: MethodCall) -> Iterator[dict[str, Any]]:
    plan: CompressionPlan = call.plan
    with apply_plan(call.model, plan, call.candidate["gptq_groupsize"], quantize=True,
                    baseline_bits=call.cfg.stage0.baseline_bits,
                    int_zero=int_zero(call.cfg.stage0)):
        yield {}


@contextmanager
def _gptq(call: MethodCall) -> Iterator[dict[str, Any]]:
    s0 = call.cfg.stage0
    gptq_(call.model, call.plan, call.batches(), call.candidate["gptq_groupsize"], s0.baseline_bits, int_zero(s0))
    yield {}


@contextmanager
def _awq(call: MethodCall) -> Iterator[dict[str, Any]]:
    s0 = call.cfg.stage0
    awq_(call.model, call.plan, call.batches(), call.candidate["gptq_groupsize"], s0.baseline_bits, int_zero(s0))
    yield {}


@contextmanager
def _rtn_activations(call: MethodCall) -> Iterator[dict[str, Any]]:
    plan: ActivationPlan = call.plan
    gs, base = call.cfg.stage0.act_group_size, call.cfg.stage0.baseline_bits
    with ExitStack() as stack:
        for lp, layer in zip(plan.layers, find_decoder_layers(call.model)):
            if lp.act_bits < base:
                stack.enter_context(quantize_inputs(layer, lp.act_bits, gs))
        yield {}


@contextmanager
def _rtn_kv(call: MethodCall) -> Iterator[dict[str, Any]]:
    """Keys rounded per channel, values per token (as Stage 0 measured them), at each layer's planned bits.
    Eviction is not applied here: rtn_kv runs on the bits-only plan; H2O / SnapKV carry out the token budget."""
    plan: KVPlan = call.plan
    s0 = call.cfg.stage0
    names = tuple(s0.kv_module_names)
    with ExitStack() as stack:
        for lp, layer in zip(plan.layers, find_decoder_layers(call.model)):
            k, v = kv_projections(layer, names)
            if lp.key_bits < s0.baseline_bits:
                stack.enter_context(quantize_output(k, lp.key_bits, s0.kv_group_size, per_channel=True))
            if lp.value_bits < s0.baseline_bits:
                stack.enter_context(quantize_output(v, lp.value_bits, s0.kv_group_size, per_channel=False))
        yield {}


_WEIGHT_PLANS = ("weights", "weights_same_size")

METHODS: dict[str, Method] = {m.name: m for m in [
    # Stage 1: weights
    # baseline; same rounding and pruning as the Stage 0 pruning-levels study
    Method("rtn", 1, "Round-to-nearest + magnitude pruning", _WEIGHT_PLANS, _rtn_weights,
           params=("gptq_groupsize",), library="in repo (torch)", version=2),  # 2: integer zero point
    # per-layer bits from the plan; Hessian of layer inputs, column-by-column error feedback
    Method("gptq", 1, "GPTQ", _WEIGHT_PLANS, _gptq, params=("gptq_groupsize",), calibrated=True,
           library="in repo (torch)"),
    Method("awq", 1, "AWQ", _WEIGHT_PLANS, _awq, params=("gptq_groupsize",), calibrated=True,
           library="in repo (torch); autoawq is deprecated"),  # activation-aware channel scaling, then RTN
    Method("structured_prune", 1, "Structured pruning", _WEIGHT_PLANS, calibrated=True,
           library="in repo (torch) or torch-pruning"),  # removes whole channels; real speed and size gains
    Method("unstructured_prune", 1, "Unstructured pruning (Wanda)", _WEIGHT_PLANS, calibrated=True,
           library="in repo (torch)"),  # |w| x input norm per output row
    # rank chosen so each layer matches its planned size
    Method("low_rank", 1, "Low-rank (SVD)", _WEIGHT_PLANS, library="in repo (torch)"),
    # Stage 2: activations
    Method("rtn_act", 2, "Round-to-nearest activations", ("activations",), _rtn_activations,
           library="in repo (torch)"),  # baseline; same rounding as the Stage 0 activation measurement
    Method("smoothquant", 2, "SmoothQuant", ("activations",), params=("smoothquant_alpha",), calibrated=True,
           library="in repo (torch)"),  # moves outliers from activations into weights
    # fast-hadamard-transform is a CUDA source build; a torch matmul Hadamard is fast enough at 1B
    Method("quarot", 2, "QuaRot", ("activations",), library="in repo (torch Hadamard)"),
    # reorder channels into clusters, one scale per cluster
    Method("rptq", 2, "RPTQ", ("activations",), calibrated=True, library="in repo (research code only)"),
    # learned rotations: a short optimisation run, the most expensive Stage 2 method
    Method("spinquant", 2, "SpinQuant", ("activations",), calibrated=True, library="in repo (research code only)"),
    # Stage 3: KV cache
    # baseline; same rounding as the Stage 0 KV measurement, no eviction
    Method("rtn_kv", 3, "Round-to-nearest KV cache", ("kv_bits_only",), _rtn_kv, library="in repo (torch)"),
    Method("quarot_kv", 3, "QuaRot KV", ("kv_bits_only",), params=("quarot_k_bits",),
           library="in repo (torch Hadamard)"),  # rotated keys/values, then rounding
    # pre-RoPE per-channel keys, dense-and-sparse outliers
    Method("kvquant", 3, "KVQuant", ("kv_bits_only",), calibrated=True, library="in repo (research code only)"),
    # keeps recent + heavy-hitter tokens; framework keeps each layer's planned share
    Method("h2o", 3, "H2O", ("kv",), library="in repo (attention hook)"),
    # picks tokens from an observation window at the end of the prompt
    Method("snapkv", 3, "SnapKV", ("kv",), library="in repo (attention hook)"),
    # offloads the cache to CPU and prefetches; latency results depend on PCIe
    Method("infinigen", 3, "InfiniGen", ("kv",), library="research code only (custom offloading)"),
]}


def methods_for(stage: int, names: list[str]) -> list[Method]:
    """The listed methods of one stage, in the given order; unknown names or wrong stages are an error."""
    out = []
    for n in names:
        m = METHODS.get(n)
        if m is None or m.stage != stage:
            known = sorted(k for k, v in METHODS.items() if v.stage == stage)
            raise ValueError(f"{n!r} is not a Stage {stage} method; known: {known}")
        out.append(m)
    return out
