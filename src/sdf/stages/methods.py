"""Every compression method of Stages 1-3 in one table, and the ones implemented so far.

Stage 1 is quantization only (weights: RTN, GPTQ, AWQ; activations: RTN, SmoothQuant, QuaRot, ...), Stage 2 is
pruning only (Wanda, structured, low-rank) and Stage 3 is KV-cache compression only. Paths: 0 -> 1 -> 4,
0 -> 2 -> 4 (prunes the FP16 model), 0 -> 3 -> 4, and 0 -> 1 -> 2 -> 4 ("<pruning>_after_<quantization>": a Stage 1
method, then a Stage 2 one on its result).

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
from dataclasses import dataclass, replace
from typing import Any, Callable, ContextManager, Iterator

import torch
from torch import nn

from sdf.stage0.activation import ActivationPlan, quantize_inputs
from sdf.stage0.kv_cache import KVPlan, kv_projections, quantize_output
from sdf.stage0.planner import CompressionPlan, bits_only
from sdf.stage0.prune_sweep import apply_plan
from sdf.stage0.sensitivity import int_zero
from sdf.stage0.sensitivity import find_decoder_layers
from sdf.stages.pruning import low_rank_, structured_prune_, wanda_
from sdf.stages.activation_methods import ACTIVATION_METHODS
from sdf.stages.bnb import WEIGHT_METHODS_BNB
from sdf.stages.weights import awq_, gptq_
from sdf.stages.weights_a import WEIGHT_METHODS_A
from sdf.stages.weights_b import WEIGHT_METHODS_B


@dataclass
class MethodCall:
    model: nn.Module
    plan: Any  # CompressionPlan (weights, pruning), ActivationPlan or KVPlan
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
    # a Stage 1 method run before this Stage 2 method: its plan keys, one per entry of `plans`
    quant_plans: tuple[str, ...] = ()
    # how removed weights are stored in the predicted size, when not stage0.sparse_storage ("free": whole channels or
    # low-rank factors are removed, so no mask of kept weights is needed)
    storage: str | None = None
    # why the method cannot follow a per-layer bit plan (its bit width is fixed); its standard row still runs
    fixed_bits: str = ""
    # why a method without `apply` cannot run here at all (shown on its failed row)
    unavailable: str = ""
    # shown on the method's rows: what is a simplified port of the fork's algorithm, and how its size is counted
    note: str = ""


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


def _calibrated(fn: Callable) -> Callable[[MethodCall], ContextManager[dict[str, Any]]]:
    """apply() for a weights.py method fn(model, plan, batches, group size, baseline bits, int_zero): in place."""
    @contextmanager
    def apply(call: MethodCall) -> Iterator[dict[str, Any]]:
        s0 = call.cfg.stage0
        fn(call.model, call.plan, call.batches(), call.candidate["gptq_groupsize"], s0.baseline_bits, int_zero(s0))
        yield {}
    return apply


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


_WEIGHT_PLANS = ("quant",)
_PRUNE_PLANS = ("prune", "prune_same_size")


def _transformed_activations(fn: Callable, extra: Callable[[MethodCall], dict[str, Any]] = lambda call: {}):
    """apply() for an activation_methods.py fn(model, plan, batches, group_size, **extra): it rewrites the model into an
    equivalent one, then returns the context manager that rounds the activations at the plan's bits."""
    @contextmanager
    def apply(call: MethodCall) -> Iterator[dict[str, Any]]:
        with fn(call.model, call.plan, call.batches(), call.cfg.stage0.act_group_size, **extra(call)):
            yield {}
    return apply


_SIMPLIFIED = "simplified port of the fork: "
_SIZE_NOTE = "size column is the plan's nominal bits; codebooks, outliers and scale overhead are not counted"
_UNAVAILABLE = {
    "qtip": "QTIP (trellis-coded quantization with a bitshift trellis and tail-biting Viterbi) is a CUDA pipeline that "
            "writes its own packed checkpoint; it is not ported to this simulated, per-layer-bit setting",
    "abqllm": "ABQ-LLM's gain is its arbitrary-bit CUDA kernels (W2A8-type); its quantizer is OmniQuant-style "
              "training and is not ported separately (see the omniquant row)",
}


def _fork_methods() -> list[Method]:
    out = []
    for registry in (WEIGHT_METHODS_A, WEIGHT_METHODS_B, WEIGHT_METHODS_BNB):
        for name, (fn, label, params, library, *rest) in registry.items():
            takes_bits, simplified = (rest[0], rest[1]) if len(rest) == 2 else (True, rest[0])
            out.append(Method(
                name, 1, label, _WEIGHT_PLANS, _calibrated(fn), params=params, calibrated=name != "bitsandbytes",
                library=library, note=f"{_SIMPLIFIED}{simplified}; {_SIZE_NOTE}" if simplified else _SIZE_NOTE,
                fixed_bits="" if takes_bits else "it binarizes every layer the same way (about 1 bit), so its "
                "standard row runs at its own width and the plan-following row is not possible"))
    out += [Method(name, 1, label, _WEIGHT_PLANS, library=reason, unavailable=reason)
            for name, label, reason in (("qtip", "QTIP", _UNAVAILABLE["qtip"]), ("abqllm", "ABQ-LLM", _UNAVAILABLE["abqllm"]))]
    extra = {"smoothquant": lambda c: {"alpha": c.candidate["smoothquant_alpha"]},
             "quarot": lambda c: {"seed": c.cfg.run.seed}, "spinquant": lambda c: {"seed": c.cfg.run.seed},
             "rptq": lambda c: {"seed": c.cfg.run.seed}}
    for name, (fn, label, library, simplified) in ACTIVATION_METHODS.items():
        out.append(Method(name, 1, label, ("activations",), _transformed_activations(fn, extra[name]),
                          params=("smoothquant_alpha",) if name == "smoothquant" else (), calibrated=True,
                          library=library, note=_SIMPLIFIED + simplified + "; weights stay unquantized in this stage"))
    return out

METHODS: dict[str, Method] = {m.name: m for m in [
    # Stage 1: quantization, weights (plans carry bits only)
    # baseline; same rounding as the Stage 0 pruning-levels study
    Method("rtn", 1, "Round-to-nearest", _WEIGHT_PLANS, _rtn_weights,
           params=("gptq_groupsize",), library="in repo (torch)", version=3),  # 3: bits-only plans
    # per-layer bits from the plan; Hessian of layer inputs, column-by-column error feedback
    Method("gptq", 1, "GPTQ", _WEIGHT_PLANS, _calibrated(gptq_), params=("gptq_groupsize",), calibrated=True,
           library="in repo (torch)", version=2),
    Method("awq", 1, "AWQ", _WEIGHT_PLANS, _calibrated(awq_), params=("gptq_groupsize",), calibrated=True,
           library="in repo (torch); autoawq is deprecated", version=2),  # activation-aware channel scaling, then RTN
    # Stage 1: quantization, activations
    Method("rtn_act", 1, "Round-to-nearest activations", ("activations",), _rtn_activations,
           library="in repo (torch)"),  # baseline; same rounding as the Stage 0 activation measurement
    # Stage 2: pruning (plans carry pruning ratios; weights stay FP16 unless run after Stage 1)
    Method("unstructured_prune", 2, "Unstructured pruning (Wanda)", _PRUNE_PLANS, _calibrated(wanda_),
           calibrated=True, library="in repo (torch)", version=3),  # |w| x input norm per output row; 3: prunes only
    # removes whole feed-forward channels; real speed and size gains once a backend cuts them out
    Method("structured_prune", 2, "Structured pruning (feed-forward channels)", _PRUNE_PLANS,
           _calibrated(structured_prune_), calibrated=True, storage="free",
           library="in repo (torch)", version=2),
    # rank chosen so each layer keeps (1 - planned share) of its numbers; activation-aware SVD
    Method("low_rank", 2, "Low-rank (activation-aware SVD)", _PRUNE_PLANS, _calibrated(low_rank_),
           params=("gptq_groupsize",), calibrated=True, storage="free", library="in repo (torch)", version=2),
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
] + _fork_methods()}

AFTER = "_after_"  # "<Stage 2 method>_after_<Stage 1 weight method>": the series path 0 -> 1 -> 2 -> 4


def series(quant: Method, prune: Method) -> Method:
    """`prune` (Stage 2) applied to the model `quant` (Stage 1, weights) has just quantized. The quantization
    follows the plan's bits, the pruning its ratios; both plans are the two halves of one combined plan."""
    if (quant.stage, prune.stage) != (1, 2) or quant.plans != _WEIGHT_PLANS:
        raise ValueError(f"{prune.name} (Stage {prune.stage}) cannot follow {quant.name} (Stage {quant.stage}): "
                         "only a Stage 1 weight method can precede a Stage 2 method")

    @contextmanager
    def apply(call: MethodCall) -> Iterator[dict[str, Any]]:
        with quant.apply(replace(call, plan=bits_only(call.plan))) as first, prune.apply(call) as second:
            yield {**first, **second}

    return Method(prune.name + AFTER + quant.name, 2, f"{prune.label} after {quant.label}", prune.plans, apply,
                  version=quant.version * 1000 + prune.version, params=tuple(dict.fromkeys(quant.params + prune.params)),
                  calibrated=quant.calibrated or prune.calibrated, library=prune.library, storage=prune.storage,
                  quant_plans=quant.plans * len(prune.plans))  # every pruning plan follows the one bits plan


def methods_for(stage: int, names: list[str]) -> list[Method]:
    """The listed methods of one stage, in the given order; unknown names or wrong stages are an error."""
    out = []
    for n in names:
        if AFTER in n and stage == 2:
            prune, quant = n.split(AFTER)
            if prune in METHODS and quant in METHODS:
                out.append(series(METHODS[quant], METHODS[prune]))
                continue
        m = METHODS.get(n)
        if m is None or m.stage != stage:
            known = sorted(k for k, v in METHODS.items() if v.stage == stage)
            raise ValueError(f"{n!r} is not a Stage {stage} method; known: {known}")
        out.append(m)
    return out
