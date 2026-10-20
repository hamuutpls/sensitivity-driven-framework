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

from sdf.stage0.activation import ActivationPlan
from sdf.stage0.kv_cache import KVPlan, kv_projections, quantize_output
from sdf.stage0.planner import CompressionPlan, bits_only
from sdf.stage0.prune_sweep import apply_plan
from sdf.stage0.sensitivity import int_zero
from sdf.stage0.sensitivity import find_decoder_layers
from sdf.stages.kv_methods import LayerKV, calibrate_kvquant, kv_attention, kvquant_hooks
from sdf.stages.pruning import low_rank_, structured_prune_, wanda_
from sdf.stages.activation_methods import ACTIVATION_METHODS, _round_inputs
from sdf.stages.bnb import WEIGHT_METHODS_BNB
from sdf.stages.weights import awq_, gptq_
from sdf.stages.weights_a import WEIGHT_METHODS_A
from sdf.stages import weights_b
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
    # shown only on the method's Our method (framework) rows, after `note`
    framework_note: str = ""
    # weights and activations quantized together (JointPlan): the standard row's (weight bits, activation bits),
    # None for the Stage 0 "original method" settings; None for a method that is not joint
    joint: tuple[int | None, int | None] | None = None
    # most a Stage 2 method removes from one layer; plans asking for more are capped (structured: 75%, Mohammad
    # 2026-10-13), and the row says so
    max_ratio: float | None = None
    # a Stage 3 technique that forgets words: its standard row keeps stage0.kv_standard_keep_ratio of them at 16 bits
    forgets: bool = False
    # what the method stores per rounded layer besides its bits, in bits (planner.predict_cost `overhead`);
    # None: one scale and zero per gptq_groupsize weights
    overhead: Callable[[int, int, int], float] | None = None
    # a Stage 2 method after a WxAy (joint) Stage 1 method: the activation plan key its Our method rows follow
    # ("activations" or "joint_acts"); its plans are then JointPlan, the weight half carrying the pruning
    act_plan: str = ""
    joint_bits: tuple[int | None, int | None] | None = None  # and that method's standard (weight, activation) bits


@dataclass(frozen=True)
class JointPlan:
    """A weight plan and an activation plan applied to one model: layer i gets weights at weights.layers[i]
    bits and its Linear inputs at acts.layers[i] bits (W8A8, W8A4, W4A8 or W4A4 per layer)."""
    weights: CompressionPlan
    acts: ActivationPlan

    def to_dict(self) -> dict[str, Any]:
        return {"weights": self.weights.to_dict(), "activations": self.acts.to_dict()}

    def combos(self) -> dict[str, int]:
        """How many layers get each W<bits>A<bits> combination."""
        out: dict[str, int] = {}
        for w, a in zip(self.weights.layers, self.acts.layers):
            k = f"W{w.bit_width}A{a.act_bits}"
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))


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
        if fn.__module__ == weights_b.__name__:  # its random draws follow the run seed
            weights_b.set_seed(call.cfg.run.seed)
        fn(call.model, call.plan, call.batches(), call.candidate["gptq_groupsize"], s0.baseline_bits, int_zero(s0))
        yield {}
    return apply


@contextmanager
def _rtn_activations(call: MethodCall) -> Iterator[dict[str, Any]]:
    with _round_inputs(call.model, call.plan, call.cfg.stage0.act_group_size, call.cfg.stage0.baseline_bits):
        yield {}


@contextmanager
def _rtn_kv_hooks(model: nn.Module, plan: KVPlan, s0) -> Iterator[None]:
    """Keys rounded per channel, values per token (as Stage 0 measured them), at each layer's planned bits."""
    names = tuple(s0.kv_module_names)
    with ExitStack() as stack:
        for lp, layer in zip(plan.layers, find_decoder_layers(model)):
            k, v = kv_projections(layer, names)
            if lp.key_bits < s0.baseline_bits:
                stack.enter_context(quantize_output(k, lp.key_bits, s0.kv_group_size, per_channel=True))
            if lp.value_bits < s0.baseline_bits:
                stack.enter_context(quantize_output(v, lp.value_bits, s0.kv_group_size, per_channel=False))
        yield


def _kv_method(rounding: str, forget: str) -> Callable[[MethodCall], ContextManager[dict[str, Any]]]:
    """apply() for a Stage 3 technique: keys and values rounded at each layer's planned bits (`rounding`: rtn,
    quarot or kvquant), and each layer's planned share of words kept with the `forget` rule (h2o, snapkv,
    infinigen). Every technique follows the whole Stage 0 plan, bits and shares (Mohammad, 2026-10-18); rounding
    techniques forget with H2O's rule."""
    @contextmanager
    def apply(call: MethodCall) -> Iterator[dict[str, Any]]:
        plan: KVPlan = call.plan
        s0 = call.cfg.stage0
        names, base = tuple(s0.kv_module_names), s0.baseline_bits
        keys, values = [lp.key_bits for lp in plan.layers], [lp.value_bits for lp in plan.layers]
        specs = [LayerKV(kb if rounding == "quarot" and kb < base else 16, vb if rounding == "quarot" and vb < base
                         else 16, forget if lp.keep_ratio < 1 else None, lp.keep_ratio)
                 for lp, kb, vb in zip(plan.layers, keys, values)]
        with ExitStack() as stack:
            if rounding == "rtn":
                stack.enter_context(_rtn_kv_hooks(call.model, plan, s0))
            elif rounding == "kvquant" and min(keys + values) < base:
                layers = calibrate_kvquant(call.model, call.batches(), keys, values, names, base)
                stack.enter_context(kvquant_hooks(call.model, layers, keys, values, names, base))
            if any(sp != LayerKV() for sp in specs):  # else attention stays the model's own (SDPA)
                stack.enter_context(kv_attention(call.model, specs))
            yield {}
    return apply


_KV_NOTE = ("Simulated: keys and values are rounded in place and forgotten words are masked out of attention for "
            "each word, so accuracy is real but memory is the plan's prediction, speed is not a smaller cache's, and peak "
            "memory includes the simulation's masks.")
_WEIGHT_PLANS = ("quant",)
_PRUNE_PLANS = ("prune", "prune_same_size")


def _transformed_activations(fn: Callable, extra: Callable[[MethodCall], dict[str, Any]] = lambda call: {}):
    """apply() for an activation_methods.py fn(model, plan, batches, group_size, below, **extra): it rewrites the
    model into an equivalent one, then returns the context manager that rounds the activations at the plan's bits."""
    @contextmanager
    def apply(call: MethodCall) -> Iterator[dict[str, Any]]:
        with fn(call.model, call.plan, call.batches(), call.cfg.stage0.act_group_size,
                below=call.cfg.stage0.baseline_bits, **extra(call)):
            yield {}
    return apply


_SIMPLIFIED = "simplified port of the fork: "
_SIZE_NOTE = ("size column: the plan's bits plus what the method stores besides them (SqueezeLLM, SpQR, AQLM, QuIP# "
              "and bitsandbytes: their codebooks, outliers and scales; outliers assumed at 0.45% for SqueezeLLM and "
              "about 1% for SpQR); "
              "for the others it is not modelled: one scale and zero per gptq_groupsize weights is assumed. A "
              "2**b-entry codebook per row makes SqueezeLLM's 8-bit layers about 1.6 bits per weight larger")
_LINEARS = 7  # shortcut: the Linears of a Llama decoder layer (q, k, v, o, gate, up, down); count them for other models
# Extra bits a method stores for one rounded decoder layer that keeps `kept` weights, in Linears with `rows` outputs in
# all, besides `b` bits per weight. An fp16 number is 16 bits; an outlier is stored as in CSR, a 16-bit value and a
# 16-bit column index (every input width here is below 65536).
SIZE_OVERHEAD: dict[str, Callable[[int, int, int], float]] = {
    # a 2**b-entry fp16 codebook per output row; 0.45% outliers (the paper's share)
    "squeezellm": lambda b, kept, rows: rows * 2 ** b * 16 + 0.0045 * kept * 32,
    # 3-bit scale and zero per 16 weights; an fp16 scale and zero for each of them per 16 rows x 16 weights
    # (spqr_'s defaults in weights_a.py: 3-bit statistics, groups of 16);
    # outliers counted at about 1% (assumed: SpQR's share depends on its threshold)
    "spqr": lambda b, kept, rows: kept * (2 * 3 / 16 + 2 * 2 * 16 / (16 * 16) + 0.01 * 32),
    # b codebooks of 256 fp16 codewords of 8 numbers per Linear; one fp16 scale per output row
    "aqlm": lambda b, kept, rows: _LINEARS * b * 256 * 8 * 16 + rows * 16,
    # the E8 lattice codebook is fixed, not stored; one scale per Linear (its random-sign vectors, about 0.003 bits
    # per weight, are not counted)
    "quipsharp": lambda b, kept, rows: _LINEARS * 16,
    # NF4: an fp32 absmax per 64 weights; int8: an fp32 absmax per output row
    "bitsandbytes": lambda b, kept, rows: kept * 32 / 64 if b == 4 else rows * 32,
}
_UNAVAILABLE = {
    "qtip": "QTIP (trellis-coded quantization with a bitshift trellis and tail-biting Viterbi) is a CUDA pipeline that "
            "writes its own packed checkpoint; it is not ported to this simulated, per-layer-bit setting",
    "abqllm": "ABQ-LLM's gain is its arbitrary-bit CUDA kernels (W2A8-type); its quantizer is OmniQuant-style "
              "training and is not ported separately (see the omniquant row)",
}


_ACT_EXTRA: dict[str, Callable[[MethodCall], dict[str, Any]]] = {
    "smoothquant": lambda c: {"alpha": c.candidate["smoothquant_alpha"]},
    "quarot": lambda c: {"seed": c.cfg.run.seed}, "spinquant": lambda c: {"seed": c.cfg.run.seed},
    "rptq": lambda c: {"seed": c.cfg.run.seed}}


def _fork_methods() -> list[Method]:
    out = []
    for registry in (WEIGHT_METHODS_A, WEIGHT_METHODS_B, WEIGHT_METHODS_BNB):
        for name, (fn, label, params, library, takes_bits, simplified) in registry.items():
            out.append(Method(
                name, 1, label, _WEIGHT_PLANS, _calibrated(fn), params=params, calibrated=name != "bitsandbytes",
                library=library, note=f"{_SIMPLIFIED}{simplified}; {_SIZE_NOTE}" if simplified else _SIZE_NOTE,
                fixed_bits="" if takes_bits else "it binarizes every layer the same way (about 1 bit), so its "
                "standard row runs at its own width and the plan-following row is not possible",
                overhead=SIZE_OVERHEAD.get(name)))
    out += [Method(name, 1, label, _WEIGHT_PLANS, library=reason, unavailable=reason)
            for name, label, reason in (("qtip", "QTIP", _UNAVAILABLE["qtip"]), ("abqllm", "ABQ-LLM", _UNAVAILABLE["abqllm"]))]
    for name, (fn, label, library, simplified) in ACTIVATION_METHODS.items():
        out.append(Method(name, 1, label, ("activations",), _transformed_activations(fn, _ACT_EXTRA[name]),
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
           library="in repo (torch); autoawq is deprecated", version=3),  # activation-aware channel scaling, then RTN; 3: o_proj unscaled under GQA
    # Stage 1: quantization, activations
    Method("rtn_act", 1, "Round-to-nearest activations", ("activations",), _rtn_activations,
           library="in repo (torch)"),  # baseline; same rounding as the Stage 0 activation measurement
    # Stage 2: pruning (plans carry pruning ratios; weights stay FP16 unless run after Stage 1)
    Method("unstructured_prune", 2, "Unstructured pruning (Wanda)", _PRUNE_PLANS, _calibrated(wanda_),
           calibrated=True, library="in repo (torch)", version=3),  # |w| x input norm per output row; 3: prunes only
    # removes whole feed-forward channels; real speed and size gains once a backend cuts them out
    Method("structured_prune", 2, "Structured pruning (feed-forward channels)", _PRUNE_PLANS,
           _calibrated(structured_prune_), calibrated=True, storage="free", max_ratio=0.75,
           library="in repo (torch)", version=2),
    # rank chosen so each layer keeps (1 - planned share) of its numbers; activation-aware SVD
    Method("low_rank", 2, "Low-rank (activation-aware SVD)", _PRUNE_PLANS, _calibrated(low_rank_),
           params=("gptq_groupsize",), calibrated=True, storage="free", library="in repo (torch)", version=2),
    # Stage 3: KV cache (stages/kv_methods.py). Standard rows: rounding techniques at stage0.kv_uniform_bits keeping
    # every word, forgetting techniques at 16 bits keeping stage0.kv_standard_keep_ratio (H2O's 20%; SnapKV and
    # InfiniGen set no share: SnapKV a fixed word count, InfiniGen an attention threshold).
    # baseline; same rounding as the Stage 0 KV measurement. Its Our method row follows the whole plan, so it is the
    # same computation as h2o's Our method row (Mohammad, 2026-10-18: "Whole plan")
    Method("rtn_kv", 3, "Round-to-nearest KV cache", ("kv",), _kv_method("rtn", "h2o"), library="in repo (torch)",
           note=_KV_NOTE, framework_note="This row forgets with H2O's rule, so it matches H2O's Our method row."),
    # keys after RoPE and values rotated by a Hadamard matrix, then rounded per token and head
    # versions 2: each word keeps its share of the words it can see; fixes after the first review (2026-10-18)
    Method("quarot_kv", 3, "QuaRot KV", ("kv",), _kv_method("quarot", "h2o"), params=("quarot_k_bits",),
           library="in repo (torch Hadamard)", note=_KV_NOTE, version=2),
    # pre-RoPE per-channel keys, per-token values, non-uniform levels, 1% outliers and the first word unrounded
    Method("kvquant", 3, "KVQuant", ("kv",), _kv_method("kvquant", "h2o"), calibrated=True, version=2,
           library="in repo (simplified port)", note=_KV_NOTE + " KVQuant: levels from plain k-means, not "
           "sensitivity-weighted; the 1% unrounded outliers are not counted in the predicted memory."),
    # keeps recent + heavy-hitter words; 3: fixed cache, filled before anything is forgotten (keeps the first word)
    Method("h2o", 3, "H2O", ("kv",), _kv_method("rtn", "h2o"), forgets=True, library="in repo (attention function)",
           note=_KV_NOTE + " H2O: a fixed cache of ceil(share x window) words per layer, filled first, then half "
           "recent and half heavy hitters; scores are averaged over the query heads sharing a key/value head. SnapKV "
           "and InfiniGen use the same cache size.", version=3),
    # keeps an observation window and the earlier words it attends to most; 3: H2O's fixed cache
    Method("snapkv", 3, "SnapKV", ("kv",), _kv_method("rtn", "snapkv"), forgets=True, version=3,
           library="in repo (attention function)", note=_KV_NOTE + " SnapKV: its paper keeps a fixed number of words; "
           "the standard row keeps H2O's 20% instead, in the same fixed cache as H2O (ceil(share x window) words, "
           "filled first), and the choice is redone at every word."),
    # guesses attention from 30% of the channels and fetches only the top words; the CPU offloading is not run;
    # 3: H2O's fixed cache
    Method("infinigen", 3, "InfiniGen", ("kv",), _kv_method("rtn", "infinigen"), forgets=True, version=3,
           library="in repo (simplified port, no CPU offloading)",
           note=_KV_NOTE + " InfiniGen: its paper fetches the words above an attention threshold (a varying "
           "count); the standard row keeps H2O's 20% instead, in the same fixed cache as H2O. Channels are picked "
           "from the words so far, without the offline skewing or the previous layer's guess; the full cache would "
           "sit in CPU memory, not run here."),
] + _fork_methods()}

AFTER = "_after_"  # "<Stage 2 method>_after_<Stage 1 weight method>": the series path 0 -> 1 -> 2 -> 4
WITH = "_with_"  # "<Stage 1 weight method>_with_<Stage 1 activation method>": both on one model (JointPlan)
_JOINT_PLANS = ("quant", "joint_weights")  # separate weight + activation plans; the joint plan (stage0/joint.py)
W8A8 = "_w8a8"  # suffix of a joint method whose only row is the standard with every layer at W8A8


def joint(weights: Method, acts: str, standard_w8a8: bool = False) -> Method:
    """`weights` (Stage 1 weight method) and the activation method `acts` (rtn_act or an activation_methods.py one)
    on one model. The activation method first rewrites the model (smoothing or rotation, nothing for rtn_act), the
    weight method then quantizes the rewritten weights from unrounded calibration inputs (as QuaRot runs GPTQ),
    and the activations are rounded while the model is measured. Standard row: W4A8 on every layer (Stage 0
    uniform_bits / act_uniform_bits), or W8A8 for the `_w8a8` method, which has no framework row. Framework rows:
    the separate Stage 0 weight and activation plans together, and the joint plan ("<name>_joint")."""
    if weights.stage != 1 or weights.plans != _WEIGHT_PLANS:
        raise ValueError(f"{weights.name} is not a Stage 1 weight method")
    if acts == "rtn_act":
        fn, extra, label = (lambda model, plan, batches, gs, below: _round_inputs(model, plan, gs, below),
                            (lambda c: {}), "round-to-nearest activations")
    elif acts in ACTIVATION_METHODS:
        fn, extra, label = ACTIVATION_METHODS[acts][0], _ACT_EXTRA[acts], ACTIVATION_METHODS[acts][1]
    else:
        raise ValueError(f"{acts!r} is not a Stage 1 activation method")

    @contextmanager
    def apply(call: MethodCall) -> Iterator[dict[str, Any]]:
        plan: JointPlan = call.plan
        rounding = fn(call.model, plan.acts, call.batches(), call.cfg.stage0.act_group_size,
                      below=call.cfg.stage0.baseline_bits, **extra(call))
        with weights.apply(replace(call, plan=plan.weights)) as first, rounding:
            yield first

    name = weights.name + WITH + acts + (W8A8 if standard_w8a8 else "")
    return Method(name, 1, f"{weights.label} + {label}" + (", every layer W8A8" if standard_w8a8 else ""),
                  () if standard_w8a8 else _JOINT_PLANS,
                  apply if weights.apply else None,
                  version=weights.version,
                  params=tuple(dict.fromkeys(weights.params + (("smoothquant_alpha",) if acts == "smoothquant" else ()))),
                  calibrated=True, library=weights.library, note=weights.note,
                  fixed_bits=weights.fixed_bits, unavailable=weights.unavailable, overhead=weights.overhead,
                  joint=(8, 8) if standard_w8a8 else (None, None))


def series(quant: Method, prune: Method, quant_plan: str = "quant") -> Method:
    """`prune` (Stage 2) applied to the model `quant` (Stage 1, weights or WxAy) has just quantized. The quantization
    follows the plan's bits, the pruning its ratios; both plans are the two halves of one combined plan. After a WxAy
    method the plan is a JointPlan whose weight half carries the pruning, and the activations stay rounded while the
    pruning calibrates and the model is measured. `quant_plan`: the Stage 1 weight plan key Our method rows follow
    ("joint_weights" for a WxAy method's joint plan)."""
    joint = quant.joint is not None
    if (quant.stage, prune.stage) != (1, 2) or not (quant.plans == _WEIGHT_PLANS or joint):
        raise ValueError(f"{prune.name} (Stage {prune.stage}) cannot follow {quant.name} (Stage {quant.stage}): "
                         "only a Stage 1 weight or WxAy method can precede a Stage 2 method")

    @contextmanager
    def apply(call: MethodCall) -> Iterator[dict[str, Any]]:
        weights = call.plan.weights if joint else call.plan
        first_plan = JointPlan(bits_only(weights), call.plan.acts) if joint else bits_only(weights)
        with quant.apply(replace(call, plan=first_plan)) as first, \
                prune.apply(replace(call, plan=weights)) as second:
            yield {**first, **second}

    # low-rank rounds its own factors with round-to-nearest, replacing what the Stage 1 method did to the weights
    unavailable = quant.unavailable or (f"{prune.label} cannot follow a Stage 1 method: it rounds its factors with "
                                        "round-to-nearest, which replaces the Stage 1 rounding"
                                        if prune.name == "low_rank" else "")
    act_plan = {"quant": "activations", "joint_weights": "joint_acts"}[quant_plan] if joint else ""
    return Method(prune.name + AFTER + quant.name, 2, f"{prune.label} after {quant.label}", prune.plans,
                  apply if quant.apply and prune.apply and not unavailable else None,
                  version=quant.version * 1000 + prune.version, params=tuple(dict.fromkeys(quant.params + prune.params)),
                  calibrated=quant.calibrated or prune.calibrated, library=prune.library, storage=prune.storage,
                  quant_plans=(quant_plan,) * len(prune.plans),  # every pruning plan follows the one bits plan
                  note=quant.note, fixed_bits=quant.fixed_bits, unavailable=unavailable,
                  max_ratio=prune.max_ratio, overhead=quant.overhead, act_plan=act_plan,
                  joint_bits=quant.joint if joint else None)


def methods_for(stage: int, names: list[str]) -> list[Method]:
    """The listed methods of one stage, in the given order; unknown names or wrong stages are an error."""
    out = []
    for n in names:
        if WITH in n and stage == 1:
            w, a = n.split(WITH)
            w8 = a.endswith(W8A8)
            a = a.removesuffix(W8A8)
            if w in METHODS and (a == "rtn_act" or a in ACTIVATION_METHODS):
                out.append(joint(METHODS[w], a, w8))
                continue
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
