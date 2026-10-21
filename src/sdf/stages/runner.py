"""The one runner for Stages 1-3 (1: quantization, 2: pruning, 3: KV cache): load the Stage 0 plans, then for each
method build and measure

    fp16       the uncompressed model (cached, the same entry Stage 0 measured)
    original   the method on the uniform plan from the Stage 0 "original method" settings (cached by config)
    framework  the method on each Stage 0 plan it accepts (one row per plan)

under identical conditions (calibration data and samples, seed, evaluation windows, device, backend), and
report through StageReporter to <run_dir>/stage_<N>/. Every variant starts from a fresh FP16 model, so the
stages stay independent of each other (each applies on top of the Stage 0 plan, not on top of another stage).
"""

from __future__ import annotations

import functools
import gc
import json
import math
import time
import re
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from sdf.data import eval_data_label, eval_windows
from sdf.eval import downstream, llamacpp, trtllm_backend, vllm_backend
from sdf.eval.ct_export import has_online_rotation
from sdf.eval.metrics import measure_model
from sdf.run import RunContext
from sdf.stage0.activation import ActivationPlan, uniform_activation_plan
from sdf.stage0.kv_cache import KVPlan, KVProfile, predict_kv, uniform_kv_plan
from sdf.stage0.planner import (CompressionPlan, baseline_cost, cap_pruning, combine_plans, decoder_share_removed,
                                level_tag, same_size_pruning_plan, uniform_plan)
from sdf.stage0.pruning_levels import layer_caps
from sdf.stage0.run import (_cost_metrics, _kv_metrics, calib_batches, fp16_key, load_fp16, source_key,
                            setup, stage_reporter, weight_cost)
from sdf.stage0.sensitivity import SensitivityProfile, find_decoder_layers, normalize
from sdf.stages.methods import AFTER, WITH, JointPlan, Method, MethodCall, methods_for, series
from sdf.utils.logging import get_logger

log = get_logger(__name__)

# Stage 0 output files the later stages read (see stage_0/handoff.md), by plan key.
PLAN_FILES: dict[str, tuple[str, type]] = {
    "quant": ("quant_plan.json", CompressionPlan),  # Stage 1: bits only
    "prune": ("prune_plan.json", CompressionPlan),  # Stage 2: pruning ratios only (bits at the baseline)
    "prune_same_size": ("prune_plan_same_size.json", CompressionPlan),
    "activations": ("activation_plan.json", ActivationPlan),
    "kv": ("kv_cache_plan.json", KVPlan),
    "kv_bits_only": ("kv_cache_plan_bits_only.json", KVPlan),
    "joint_weights": ("joint_weight_plan.json", CompressionPlan),  # Stage 1 joint: weight half of the joint plan
    "joint_acts": ("joint_activation_plan.json", ActivationPlan),  # and its activation half
}

# Framework rows after the first plan get their own method name and are compared with the method's original row.
# Values: (method-name suffix, label, plain description) as in the Stage 0 report.
_PLAN_ROWS = {
    "joint_weights": (
        "_joint", "Our method, weights and activations planned together",
        "our method with each layer's weight bits and activation bits chosen together, from a measurement of "
        "both rounded at once, at the same average bits as the separate plans."),
    "prune_same_size": (
        "_same_size", "Our method, aiming at the standard method's amount removed",
        "our method aiming to remove as many numbers as the standard method, but taking them from the least "
        "sensitive layers instead of evenly, so the two can be compared at a similar size. The share it really "
        "removes is in its name: lower than the standard method's when the layers it may prune cannot hold that "
        "much."),
}

TITLES = {1: "Quantization (weights and activations)", 2: "Pruning", 3: "KV-cache compression"}

MAIN_METRICS = {
    1: ["ppl_val", "ppl_heldout", "predicted_weight_memory_gb", "avg_bits_per_weight", "avg_activation_bits",
        "zero_weight_share", "peak_memory_gb", "prefill_ms_mean", "decode_ms_per_token_mean", "build_time_s"],
    2: ["ppl_val", "ppl_heldout", "predicted_weight_memory_gb", "sparsity", "decoder_weights_removed",
        "peak_memory_gb", "prefill_ms_mean", "decode_ms_per_token_mean", "build_time_s"],
    3: ["ppl_val", "ppl_heldout", "predicted_kv_memory_gb", "avg_kv_bits", "kv_kept_share", "peak_memory_gb",
        "prefill_ms_mean", "decode_ms_per_token_mean", "build_time_s"],
}

INTROS = {
    1: "A model is a huge store of numbers (its \"weights\"), and while it runs every layer passes numbers to the "
       "next one (its \"activations\"). This stage keeps those numbers with fewer bits, like rounding prices to the "
       "nearest dollar, and removes nothing. Each technique is tried as standard quantization, which gives every "
       "weight the same bit length, and as our method, which mixes bit lengths following the Stage 0 plan, then the model is tested for accuracy, memory "
       "and speed.",
    2: "This stage makes the model smaller by removing numbers that barely matter (single weights, whole "
       "channels, or the small part of a weight table that a low-rank copy can drop), and rounds nothing. The "
       "standard way removes the same share from every layer; our method removes more from the layers Stage "
       "0 found robust and nothing from the fragile ones. Methods run on the uncompressed model, or after a "
       "Stage 1 method (named \"<pruning>_after_<quantization>\").",
    3: "While writing a reply, the model keeps notes on every earlier word (the \"KV cache\"); for long texts these "
       "notes can take more memory than the model itself. This stage stores the notes with fewer bits or forgets "
       "the least-used ones. The standard way treats every layer the same; our method follows the Stage 0 "
       "plan for each layer.",
}


@dataclass
class Stage0Plans:
    """Everything a later stage reads from a finished Stage 0 run's stage_0/ folder."""

    dir: Path
    plans: dict[str, Any]  # plan key (PLAN_FILES) -> plan; keys whose file is missing are absent
    profile: SensitivityProfile  # layer sizes, for the predicted weight memory
    kv_profile: KVProfile | None  # layer key/value sizes, for the predicted KV memory
    # Stage 2 pruning levels Stage 0 saved (prune_plan_<level>.json): share -> {"prune": ..., "prune_same_size": ...}
    levels: dict[float, dict[str, CompressionPlan]] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=lambda: {k: f for k, (f, _) in PLAN_FILES.items()})

    @property
    def num_layers(self) -> int:
        return self.profile.num_layers

    @classmethod
    def load(cls, stage0_dir: str | Path) -> "Stage0Plans":
        d = Path(stage0_dir)
        if not (d / "sensitivity_profile.json").exists():
            raise FileNotFoundError(f"{d} is not a Stage 0 output folder (no sensitivity_profile.json); point "
                                    "stages.stage0_dir at <run>/stage_0")
        plans = {k: cls_.load(d / f) for k, (f, cls_) in PLAN_FILES.items() if (d / f).exists()}
        kv_prof = KVProfile.load(d / "kv_profile.json") if (d / "kv_profile.json").exists() else None
        by_tag: dict[str, dict[str, CompressionPlan]] = {}
        loaded = list(plans.items())
        for f in sorted(d.glob("prune_plan_*.json")):
            if m := re.fullmatch(r"prune_plan_(same_size_)?(\d{3})\.json", f.name):
                p = CompressionPlan.load(f)
                by_tag.setdefault(m[2], {})["prune_same_size" if m[1] else "prune"] = p
                loaded.append((f.name, p))
        levels = {int(t) / 100: ps for t, ps in by_tag.items()}
        out = cls(d, plans, SensitivityProfile.load(d / "sensitivity_profile.json"), kv_prof, levels)
        for k, p in loaded:
            if len(p.layers) != out.num_layers:
                raise ValueError(f"{out.files.get(k, k)} has {len(p.layers)} layers, the profile {out.num_layers}")
        return out

    def at_level(self, ratio: float) -> "Stage0Plans":
        """These plans with the pruning plans of one level in place of the default ones."""
        tag = level_tag(ratio)
        files = {"prune": f"prune_plan_{tag}.json", "prune_same_size": f"prune_plan_same_size_{tag}.json"}
        plans = {k: p for k, p in self.plans.items() if k not in files}
        return replace(self, plans={**plans, **self.levels[ratio]}, files={**self.files, **files})


def sweep(m: Method, plans: Stage0Plans) -> list[tuple[Method, Stage0Plans, float | None]]:
    """The row groups of one method: (method, plans, level). A Stage 2 method runs once per pruning level Stage 0
    saved (Mohammad, 2026-10-13), named <method>_<level>, e.g. low_rank_030; any other method, or a Stage 0 folder
    without level plans, once (level None: the standard row prunes prune_ratio_aggressive)."""
    if kind_of(m) != "prune" or not plans.levels:
        return [(m, plans, None)]
    return [(replace(m, name=f"{m.name}_{level_tag(r)}"), plans.at_level(r), r)
            for r in sorted(plans.levels)]


# What a plan is about, by plan key: the standard method and the cost prediction differ by kind.
PLAN_KIND = {"quant": "weights", "prune": "prune", "prune_same_size": "prune",
             "activations": "activations", "kv": "kv", "kv_bits_only": "kv",
             "joint_weights": "joint", "joint_acts": "joint"}


def kind_of(m: Method) -> str:
    """weights | activations | joint (both, Stage 1), prune (Stage 2, alone or after a Stage 1 method), kv
    (Stage 3)."""
    return "joint" if m.joint is not None else PLAN_KIND[m.plans[0]]


def original_plan(m: Method, plans: Stage0Plans, s0, prune_ratio: float, quarot_k_bits: int | None = None) -> Any:
    """The uniform plan of the standard method, from the Stage 0 "original method" settings: `uniform_bits` and
    nothing removed (Stage 1), `prune_ratio` removed from every layer and nothing rounded (Stage 2), both for a
    Stage 2 method run after Stage 1."""
    kind = kind_of(m)
    scores = lambda: normalize(plans.profile.raw_scores, s0.normalization)  # noqa: E731
    if kind == "weights":
        return uniform_plan(scores(), s0.uniform_bits, 0.0)
    if kind == "prune" and m.joint_bits is not None:  # after a WxAy method: its standard bits (W4A8 or W8A8)
        w, a = m.joint_bits
        return JointPlan(uniform_plan(scores(), w or s0.uniform_bits, prune_ratio),
                         uniform_activation_plan(plans.num_layers, a or s0.act_uniform_bits))
    if kind == "prune":
        return uniform_plan(scores(), s0.uniform_bits if m.quant_plans else s0.baseline_bits, prune_ratio)
    if kind == "activations":
        return uniform_activation_plan(plans.num_layers, s0.act_uniform_bits)
    if kind == "joint":
        w, a = m.joint
        return JointPlan(uniform_plan(scores(), w or s0.uniform_bits, 0.0),
                         uniform_activation_plan(plans.num_layers, a or s0.act_uniform_bits))
    if m.forgets:
        return uniform_kv_plan(plans.num_layers, s0.baseline_bits, s0.kv_standard_keep_ratio)
    plan = uniform_kv_plan(plans.num_layers, s0.kv_uniform_bits)
    if "quarot_k_bits" in m.params and quarot_k_bits is not None:  # QUAROT_K_BITS: QuaRot KV's standard key bits
        plan = replace(plan, layers=tuple(replace(lp, key_bits=quarot_k_bits) for lp in plan.layers))
    return plan


def _plan_keys(m: Method, plan_key: str) -> list[str]:
    """The plan files a framework row needs: its plan, plus the Stage 1 plan it follows in the series path, or the
    activation plan of a joint method."""
    if m.joint is not None:  # each weight plan with the activation plan made with it
        return [plan_key, {"quant": "activations", "joint_weights": "joint_acts"}[plan_key]]
    if not m.quant_plans:
        return [plan_key]
    return [plan_key, m.quant_plans[m.plans.index(plan_key)]] + ([m.act_plan] if m.act_plan else [])


def framework_plan(m: Method, plans: Stage0Plans, plan_key: str) -> Any:
    """The Stage 0 plan a method's framework row follows; after a Stage 1 method, its bits join the pruning plan."""
    plan, *other = (plans.plans[k] for k in _plan_keys(m, plan_key))
    if m.joint is not None:
        return JointPlan(plan, other[0])
    if m.act_plan:  # Stage 2 after a WxAy method: the Stage 1 bits and activation plan with this pruning
        return JointPlan(combine_plans(other[0], plan), other[1])
    return combine_plans(other[0], plan) if other else plan


# Stage 0 settings each kind of method reads that a plan does not carry; they join the cache key when not default
# (so entries made with the defaults stay valid).
_KIND_SETTINGS = {"weights": ("weight_zero_point", "baseline_bits"), "activations": ("act_group_size", "baseline_bits"),
                  "prune": ("weight_zero_point", "baseline_bits"),
                  "kv": ("kv_group_size", "kv_module_names", "baseline_bits"),
                  "joint": ("weight_zero_point", "act_group_size", "baseline_bits")}


def _setting_key(kind: str, s0) -> dict[str, Any]:
    default = type(s0)()
    return {k: getattr(s0, k) for k in _KIND_SETTINGS[kind] if getattr(s0, k) != getattr(default, k)}


def plan_metrics(plan: Any, plans: Stage0Plans, cfg, candidate: dict[str, Any],
                 storage: str | None = None, overhead: Callable[[int, int, int], float] | None = None,
                 ) -> dict[str, Any]:
    """What the plan is predicted to cost (the same predictions Stage 0 reports), next to the measured metrics.
    `storage`: how removed weights are stored (Method.storage), when not the run's stage0.sparse_storage;
    `overhead`: what the method stores besides its bits (Method.overhead)."""
    s0 = cfg.stage0
    if isinstance(plan, CompressionPlan):
        return _cost_metrics(weight_cost(plans.profile, candidate["gptq_groupsize"], s0, storage=storage,
                                         overhead=overhead)(plan))
    if isinstance(plan, ActivationPlan):
        return {"avg_activation_bits": plan.avg_bits}
    if isinstance(plan, JointPlan):
        return {**plan_metrics(plan.weights, plans, cfg, candidate, storage, overhead),
                **plan_metrics(plan.acts, plans, cfg, candidate)}
    if isinstance(plan, KVPlan) and plans.kv_profile is not None:
        m = _kv_metrics(predict_kv(plan, plans.kv_profile, s0.kv_context_len, s0.kv_batch_size, s0.kv_group_size,
                                   s0.group_overhead_bits, s0.baseline_bits))
        return {k: m[k] for k in ("predicted_kv_memory_gb", "avg_kv_bits", "kv_kept_share")}
    return {}


def _fp16_plan_metrics(stage: int, plans: Stage0Plans, cfg, candidate) -> dict[str, Any]:
    base = cfg.stage0.baseline_bits
    weights = _cost_metrics(baseline_cost(plans.profile, base))
    if stage == 1:
        return {**weights, **plan_metrics(uniform_activation_plan(plans.num_layers, base), plans, cfg, candidate)}
    if stage == 2:
        return {**weights, "decoder_weights_removed": 0.0}
    return plan_metrics(uniform_kv_plan(plans.num_layers, base), plans, cfg, candidate)


def _describe(plan: Any) -> str:
    if isinstance(plan, JointPlan):
        prune = sorted({lp.pruning_ratio for lp in plan.weights.layers})
        return (f"weights and activations, layers per combination {plan.combos()}"
                + (f", prune ratios {prune}" if prune != [0.0] else ""))
    if isinstance(plan, CompressionPlan):
        bits = sorted({lp.bit_width for lp in plan.layers})
        prune = sorted({lp.pruning_ratio for lp in plan.layers})
        return f"{plan.kind} plan, bits {bits}, prune ratios {prune}"
    if isinstance(plan, ActivationPlan):
        return f"{plan.kind} plan, activation bits avg {plan.avg_bits:.3g}"
    bits = sorted({b for lp in plan.layers for b in (lp.key_bits, lp.value_bits)})
    return f"{plan.kind} plan, KV bits {bits}, kept share min {min(lp.keep_ratio for lp in plan.layers):.2g}"


def zero_weight_share(model: nn.Module) -> float:
    """Share of the decoder Linear weights that are exactly 0 (evidence that a Stage 1 method removed nothing: rounding
    onto a grid that contains 0 makes some, so compare each row with the FP16 row and the round-to-nearest row)."""
    ws = [m.weight for layer in find_decoder_layers(model) for m in layer.modules() if isinstance(m, nn.Linear)]
    return sum(int((w == 0).sum()) for w in ws) / sum(w.numel() for w in ws)


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_stage(
    ctx: RunContext,
    stage: int,
    method_names: list[str],
    candidate: dict[str, Any],
    stage0_dir: str | Path,
    model_factory: Callable[[], nn.Module] | None = None,
    tokenizer=None,
    text_loader: Callable[[str, str], list[str]] | None = None,
    measure_fp16: bool = True,
    after: dict[str, str] | None = None,
    before: dict[str, tuple[Method, Any]] | None = None,
    subdir: str | None = None,
    used: dict[tuple[str, str], tuple[Method, Any, float | None]] | None = None,
    only: str | None = None,
) -> dict[str, Path]:
    """Run one of Stages 1-3 with `method_names` (see sdf.stages.methods.METHODS). `model_factory` returns a
    fresh FP16 model on the right device (default: from_pretrained); tests pass a tiny model's deepcopy.

    Chained runs (run_chain), by variant ("original" = Standard rows, "framework" = Our method rows):
    `after` (Stage 2): the Stage 1 weight method each pruning row runs after, as "<pruning>_after_<it>" does, but
    the rows keep the pruning method's name so Our method is compared with Standard. `before` (Stage 3): a
    (method, plan) built on the fresh model before the row's method, e.g. the best Stage 2 row. `subdir`: the
    report folder instead of stage_<N>/. `used`, when given, is filled with (row name, variant) -> (method, plan,
    pruning level) of every row, so the next stage can rebuild a row exactly. `only`: "original" or "framework" to
    run only that side's rows (the chain follows each side's rows on its own)."""
    if stage not in TITLES:
        raise ValueError(f"run_stage runs Stages 1-3, not {stage}")
    if after and stage != 2 or before and stage != 3:
        raise ValueError("after= is for Stage 2 (after a Stage 1 method), before= for Stage 3 (after a Stage 2 row)")
    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    methods = methods_for(stage, method_names)
    plans = all_plans = Stage0Plans.load(stage0_dir)
    device, handle, text_loader = setup(ctx, None, tokenizer, text_loader)
    dtype = getattr(torch, cfg.model.dtype)
    if model_factory is None:
        def model_factory() -> nn.Module:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(cfg.model.name, torch_dtype=dtype).to(device)
    handle.factory = model_factory  # an FP16 cache miss builds the model like every row

    # weight, WxAy, pruning rows; in Stage 3 only the uncompressed row (see _KV_NOT_ON_BACKENDS)
    use_llamacpp = cfg.eval.llamacpp_dir is not None
    lc_key = [cfg.eval.llamacpp_dir, cfg.eval.llamacpp_convert, llamacpp.VERSION] if use_llamacpp else None
    # Serving backends run in a Python of their own (Linux only) on the model saved at the plan's bits; weight and
    # WxAy rows of Stage 1, pruning rows of Stage 2 (see be_on). name -> (module, its error for a plan it cannot
    # load, cache key part)
    serving = {name: (mod, err, [py, mod.VERSION]) for name, mod, err, py in (
        ("vllm", vllm_backend, vllm_backend.VLLMError, cfg.eval.vllm_python),
        ("trtllm", trtllm_backend, trtllm_backend.TRTLLMError, cfg.eval.trtllm_python)) if py}
    calib = (f"{candidate['calib_dataset']}, {candidate['calib_samples']} x {cfg.calibration.seq_len} tokens")
    start = "the uncompressed model, for every row (stages are independent)"
    side = {"original": "Standard", "framework": "Our method"}
    if after and only:
        start = f"{side[only]} rows only, each after the Stage 1 row {after[only]} ({stage1_row(after[only])[0].label})"
    elif after:
        start = (f"Standard rows: the best Standard Stage 1 row ({stage1_row(after['original'])[0].label}); Our "
                 f"method rows: the best Our method Stage 1 row {after['framework']} "
                 f"({stage1_row(after['framework'])[0].label}); each then pruned")
    if before and only:
        start = (f"{side[only]} rows only, each on the Stage 2 row {before[only][0].name} "
                 f"({before[only][0].label})")
    elif before:
        start = (f"Standard rows: the Stage 2 row {before['original'][0].name} ({before['original'][0].label}, "
                 f"Standard); Our method rows: the Stage 2 row {before['framework'][0].name} "
                 f"({before['framework'][0].label}, Our method); each then given this stage's method")
    rep = stage_reporter(
        ctx, candidate, handle, plans.profile, ("stages", "stage0", "calibration", "eval", "model", "run"),
        stage=stage, title=TITLES[stage], subdir=subdir,
        conditions={"model": cfg.model.name, "calibration": calib, "seed": cfg.run.seed,
                    "evaluation data": eval_data_label(cfg.eval),
                    "device": str(device), "backend": "HF Transformers", "Stage 0 plans": str(plans.dir),
                    "starting point": start},
        main_metrics=MAIN_METRICS[stage] + (["predicted_weight_memory_gb", "decoder_weights_removed"] if before else [])
        + (llamacpp.MAIN_METRICS if use_llamacpp else [])
        + [k for mod, _, _ in serving.values() for k in mod.MAIN_METRICS],
    )
    rep.plain_intro = INTROS[stage]

    # Evaluation windows and calibration batches: built once, identical for every row.
    @functools.cache
    def windows() -> tuple[torch.Tensor, torch.Tensor]:
        return eval_windows(text_loader(cfg.eval.dataset, "test"), handle.tokenizer, cfg.eval.seq_len,
                            cfg.eval.max_windows)

    @functools.cache
    def batches() -> list[torch.Tensor]:
        return calib_batches(ctx, candidate, handle, text_loader, candidate["calib_samples"])

    for t in cfg.eval.downstream_tasks:
        downstream.task_metric(t)  # register before any cached row is reported

    def tasks(model: nn.Module) -> dict[str, float]:
        ev = cfg.eval
        if not ev.downstream_tasks:
            return {}
        return downstream.downstream_accuracy(model, handle.tokenizer, ev.downstream_tasks, ev.downstream_limit,
                                              ev.downstream_batch_size, cfg.run.seed)

    # A WxAy row's Stage 1 model goes to every backend: vLLM runs its activation bits too (ACTIVATIONS); llama.cpp
    # and TensorRT-LLM have no kernels for them on this hardware, so they run its weights at the plan's bits with
    # 16-bit activations, and the row says so (Mohammad's choice, 2026-10-11).
    def lc_on(method: Method) -> bool:
        return use_llamacpp and kind_of(method) in ("weights", "joint", "prune")

    def be_on(method: Method) -> bool:
        return bool(serving) and kind_of(method) in ("weights", "joint", "prune")

    def not_on_backends(method: Method) -> dict[str, str]:
        """Why a Stage 3 row has no backend numbers; set on the row, not in the cached result (the reason
        depends on which backends this run has, which a KV row's cache key does not hold)."""
        if kind_of(method) != "kv" or before:  # a chained row's backends measured its Stage 2 model
            return {}
        return {f"{name}_error": _KV_NOT_ON_BACKENDS for name in ["llamacpp"] * use_llamacpp + list(serving)}

    def on_backends(model: nn.Module, method: Method, plan: Any, metrics: dict[str, Any], info: dict[str, str],
                    val: torch.Tensor, held: torch.Tensor) -> None:
        """llama.cpp and the serving backends on `model` (built with `method` and `plan`), into metrics and info."""
        joint = isinstance(plan, JointPlan)
        weights, acts = (plan.weights, plan.acts) if joint else (plan, None)
        rotated = has_online_rotation(model)  # QuaRot/SpinQuant: no saved file can hold the rotation
        if lc_on(method):
            if rotated:
                info["llamacpp_error"] = _ROTATED
            else:  # a row keeps its HF numbers if llama.cpp fails, as with vLLM (Mohammad, 2026-10-13)
                try:
                    metrics.update(llamacpp.measure(model, handle.tokenizer, weights, val, held, cfg.eval,
                                                    s0.baseline_bits))
                    if joint:
                        info["llamacpp_activations"] = _WEIGHTS_ONLY
                except Exception as e:  # noqa: BLE001 - a missing program or file too, not only a failed run
                    log.warning("llama.cpp could not measure %s: %s", method.name, e)
                    info["llamacpp_error"] = str(e)
        for name, (mod, err, _) in serving.items() if be_on(method) else ():
            if rotated:
                info[f"{name}_error"] = _ROTATED
                continue
            extra_args = ({"acts": acts, "act_group_size": cfg.stage0.act_group_size}
                          if joint and getattr(mod, "ACTIVATIONS", False) else {})
            try:
                metrics.update(mod.measure(model, handle.tokenizer, weights, val, held, cfg.eval,
                                           s0.baseline_bits, candidate["gptq_groupsize"], **extra_args))
            except err as e:  # a plan the backend cannot load (e.g. 3-bit GPTQ in vLLM): keep the row,
                log.warning("%s could not measure %s: %s", name, method.name, e)  # record why
                info[f"{name}_error"] = str(e)
                continue
            if joint:
                info[f"{name}_activations"] = (_rounded_by_backend(name, method, cfg.stage0.act_group_size)
                                               if extra_args else _WEIGHTS_ONLY)

    def build_and_measure(method: Method, plan: Any, variant: str) -> dict[str, Any]:
        model = model_factory()
        info: dict[str, str] = {}
        try:
            t0 = time.perf_counter()  # a chained row's build time includes the earlier stage's build
            with ExitStack() as stack:
                prior: dict[str, Any] = {}
                if before:
                    first, first_plan = before[variant]
                    stack.enter_context(first.apply(MethodCall(model, first_plan, candidate, cfg, batches)))
                    if lc_on(first) or be_on(first):  # the backends cannot run the KV technique: they measure the
                        on_backends(model, first, first_plan, prior, info, *windows())  # model it was built on
                        info["backends"] = _BACKENDS_BEFORE
                extra = stack.enter_context(method.apply(MethodCall(model, plan, candidate, cfg, batches)))
                build_s = time.perf_counter() - t0
                val, held = windows()
                metrics, raw = measure_model(model, val, held, cfg.eval, device, seed=cfg.run.seed)
                metrics.update(tasks(model))
                if kind_of(method) in ("weights", "joint"):
                    metrics["zero_weight_share"] = zero_weight_share(model)
                on_backends(model, method, plan, metrics, info, val, held)
                metrics.update(prior)
        finally:
            del model
            _free_memory()
        metrics.update(extra)
        metrics["build_time_s"] = build_s
        if (method.simulated and stage in (1, 2)) or (before and before[variant][0].simulated):
            metrics.pop("model_size_gb", None)  # still FP16 in memory; predicted_weight_memory_gb is the size
        return {"metrics": metrics, "raw": raw, "info": info}

    # --- FP16: the same cache entry Stage 0 measured ------------------------------------------------------------
    with rep.method("baseline", "fp16", description="uncompressed model") as row:
        row.metrics.update(_fp16_plan_metrics(stage, plans, cfg, candidate))
        if before:  # the weight columns the chained rows carry
            row.metrics.update(_fp16_plan_metrics(2, plans, cfg, candidate))
        row.metrics["build_time_s"] = 0.0
        if measure_fp16:
            fp16, cached = load_fp16(ctx, handle, text_loader)
            row.metrics.update(fp16["metrics"])
            if stage == 1:
                row.metrics["zero_weight_share"] = zero_weight_share(model_factory())
            row.info["cached"] = cached
            rep.add_raw("baseline", "fp16", fp16["raw"])
            if cfg.eval.downstream_tasks:
                key = {**fp16_key(ctx, device), "tasks": list(cfg.eval.downstream_tasks),
                       "limit": cfg.eval.downstream_limit}
                acc, _ = ctx.cache.get_or_compute("fp16_downstream", key, lambda: tasks(model_factory()))
                row.metrics.update(acc)
            if use_llamacpp:
                lc, _ = ctx.cache.get_or_compute(
                    "fp16_llamacpp", {**fp16_key(ctx, device), "llamacpp": lc_key},
                    lambda: llamacpp.measure(model_factory(), handle.tokenizer, None, *windows(), cfg.eval,
                                             s0.baseline_bits))
                row.metrics.update(lc)
            for name, (mod, _, be_key) in serving.items():
                be, _ = ctx.cache.get_or_compute(
                    f"fp16_{name}", {**fp16_key(ctx, device), name: be_key},
                    lambda mod=mod: mod.measure(model_factory(), handle.tokenizer, None, *windows(), cfg.eval,
                                                s0.baseline_bits, candidate["gptq_groupsize"]))
                row.metrics.update(be)

    @functools.cache
    def caps(max_ratio: float) -> tuple[float, ...]:  # per layer: the model is built once, only when needed
        model = model_factory()
        try:
            return tuple(layer_caps(model, max_ratio))
        finally:
            del model
            _free_memory()

    simulated_any = False
    if stage == 2 and not all_plans.levels:
        log.warning("Stage 2: no pruning level plans in %s (stage0.pruning_levels was empty, or an older Stage 0 "
                    "run); each method runs once, its standard row at %.0f%%", all_plans.dir,
                    100 * candidate["prune_ratio_aggressive"])
    def pair(method: Method) -> list[tuple[tuple[Method, Stage0Plans, float | None], Method]]:
        """The row groups of one method, each with the method its Our method rows run (another Stage 1 method
        before it when `after` names different ones for Standard and Our method)."""
        if not after:
            return [(row, row[0]) for row in sweep(method, all_plans)]
        std, ours = (replace(series(q, method, key), name=method.name)
                     for q, key in (stage1_row(after.get(v) or after[only]) for v in ("original", "framework")))
        return [(row, theirs) for row, (theirs, _, _) in zip(sweep(std, all_plans), sweep(ours, all_plans))]

    for (m, plans, level), mf in (row for method in methods for row in pair(method)):
        simulated_any |= m.simulated or mf.simulated
        orig = original_plan(m, plans, s0, candidate["prune_ratio_aggressive"] if level is None else level,
                             candidate.get("quarot_k_bits"))
        if only != "framework":
            with rep.method(m.name, "original", description=f"{m.label}, standard settings: {_describe(orig)}",
                            simulated=m.simulated) as row:
                orig, cap_note = _capped(m, orig, caps, plans.profile.layer_numel)
                row.info["description"] = f"{m.label}, standard settings: {_describe(orig)}"
                _label_level(row, m, level, orig, plans.profile.layer_numel)
                _require(m)
                if note := "; ".join(n for n in (m.note, m.fixed_bits, cap_note) if n):
                    row.info["note"] = note
                key = {"stage": stage, "method": m.name, "version": m.version, "plan": orig.to_dict(),
                       "params": {p: candidate[p] for p in m.params}, **fp16_key(ctx, device),
                       "downstream": [list(cfg.eval.downstream_tasks), cfg.eval.downstream_limit],
                       "llamacpp": lc_key if lc_on(m) else None}
                if be_on(m):  # absent when off, so cached rows from before stay valid
                    key.update({name: be_key for name, (_, _, be_key) in serving.items()})
                    if "gptq_groupsize" not in m.params:  # the serving export still uses it (e.g. bitsandbytes)
                        key["export_group_size"] = candidate["gptq_groupsize"]
                if after:  # the rows keep the pruning method's name; the Stage 1 method is in the key
                    key["after"] = after["original"]
                if before:
                    first, first_plan = before["original"]
                    if lc_on(first) or be_on(first):  # the backends measure the Stage 2 model under the KV row
                        key["before_backends"] = {"llamacpp": lc_key if lc_on(first) else None,
                                                  **({n: k for n, (_, _, k) in serving.items()} if be_on(first) else {})}
                    key["before"] = {"method": first.name, "label": first.label,  # the label names the Stage 1 method
                                     "version": first.version, "plan": first_plan.to_dict(),
                                     "params": {p: candidate[p] for p in first.params},
                                     "settings": _setting_key(kind_of(first), s0)}
                if settings := _setting_key(kind_of(m), s0):
                    key["settings"] = settings
                if m.calibrated or before and before["original"][0].calibrated:
                    key["calibration"] = {"dataset": candidate["calib_dataset"], "samples": candidate["calib_samples"],
                                          "seq_len": cfg.calibration.seq_len, "batch_size": cfg.calibration.batch_size,
                                          **source_key(cfg, candidate["calib_dataset"])}
                result, cached = ctx.cache.get_or_compute(f"stage{stage}_original", key,
                                                          lambda: build_and_measure(m, orig, "original"))
                if used is not None:
                    used[(m.name, "original")] = (m, orig, level)
                _before_metrics(row, before, "original", plans, cfg, candidate)
                row.metrics.update(plan_metrics(orig, plans, cfg, candidate, m.storage, m.overhead))
                if stage == 2:
                    row.metrics["decoder_weights_removed"] = decoder_share_removed(_weights(orig), plans.profile.layer_numel)
                row.metrics.update(result["metrics"])
                row.info.update(result.get("info", {}), cached=cached)  # .get: rows cached before "info" existed
                row.info.update(not_on_backends(m))
                rep.add_raw(m.name, "original", result["raw"])

        for plan_key in mf.plans if only != "original" else ():
            suffix, *plain = _PLAN_ROWS.get(plan_key, ("",))
            info = {"compare_to": mf.name, "plan_file": plans.files[plan_key], "simulated": mf.simulated}
            if plain:
                info["label"], info["plain_desc"] = plain
            with rep.method(mf.name + suffix, "framework", **info) as row:
                _label_level(row, mf, level, None, plans.profile.layer_numel)  # until the plan is known
                if gone := [plans.files[k] for k in _plan_keys(mf, plan_key) if k not in plans.plans]:
                    raise FileNotFoundError(f"{', '.join(gone)} not in {plans.dir}; re-run Stage 0 with this plan "
                                            "enabled")
                plan, cap_note = _capped(mf, framework_plan(mf, plans, plan_key), caps, plans.profile.layer_numel)
                _label_level(row, mf, level, plan, plans.profile.layer_numel)
                _require(mf)
                if note := " ".join(n for n in (mf.note, mf.framework_note) if n):
                    row.info["note"] = note
                if mf.fixed_bits:
                    raise ValueError(f"{mf.label} cannot follow a per-layer bit plan: {mf.fixed_bits}")
                if cap_note:
                    row.info["note"] = "; ".join(n for n in (row.info.get("note"), cap_note) if n)
                row.info["description"] = f"{mf.label}, Stage 0 {_describe(plan)}"
                result = build_and_measure(mf, plan, "framework")
                if used is not None:
                    used[(mf.name + suffix, "framework")] = (mf, plan, level)
                _before_metrics(row, before, "framework", plans, cfg, candidate)
                row.metrics.update(plan_metrics(plan, plans, cfg, candidate, mf.storage, mf.overhead))
                if stage == 2:
                    row.metrics["decoder_weights_removed"] = decoder_share_removed(_weights(plan),
                                                                                  plans.profile.layer_numel)
                row.metrics.update(result["metrics"])
                row.info.update(result["info"])
                row.info.update(not_on_backends(mf))
                rep.add_raw(mf.name + suffix, "framework", result["raw"])

    if any(m.storage for m in methods):
        rep.plain_why.append(
            "Structured pruning and low-rank remove whole channels or store smaller factors, so their predicted size "
            "has no extra record of which numbers were kept; unstructured pruning does (one bit per number).")
    if simulated_any:
        rep.plain_why.append(
            "Some techniques here are simulated: the numbers are rounded as the compressed model would store them, "
            "but kept in the uncompressed format. Their accuracy is real; their memory is the plan's prediction, "
            "and their speed is not the speed a real compressed model would have.")
    return rep.finalize()


def _before_metrics(row, before: dict[str, tuple[Method, Any]] | None, variant: str, plans: Stage0Plans, cfg,
                    candidate: dict[str, Any]) -> None:
    """A chained Stage 3 row: which Stage 2 row it was built on, and that row's predicted weight size and share
    removed (the KV metrics follow from this stage's own plan)."""
    if not before:
        return
    first, first_plan = before[variant]
    row.info["built_on"] = first.name
    row.metrics.update(plan_metrics(first_plan, plans, cfg, candidate, first.storage, first.overhead))
    if isinstance(first_plan, (CompressionPlan, JointPlan)):
        row.metrics["decoder_weights_removed"] = decoder_share_removed(_weights(first_plan), plans.profile.layer_numel)


def stage1_row(name: str) -> tuple[Method, str]:
    """The Stage 1 method of a Stage 1 row name and the weight plan key the row follows (a WxAy method's "_joint"
    row follows the joint plan)."""
    joint = name.endswith(_PLAN_ROWS["joint_weights"][0])
    (m,) = methods_for(1, [name.removesuffix(_PLAN_ROWS["joint_weights"][0]) if joint else name])
    return m, "joint_weights" if joint else "quant"


def _weights(plan: Any) -> CompressionPlan:
    return plan.weights if isinstance(plan, JointPlan) else plan


def _capped(m: Method, plan: Any, caps: Callable[[float], tuple[float, ...]],
            layer_numel: list[int]) -> tuple[Any, str]:
    """`plan` capped at the method's most per layer (`caps(Method.max_ratio)`: 75%, or the layer's feed-forward
    share when lower), and the row's note when it was ("" if not). A same-size plan is rebuilt with the caps, so
    what a capped layer cannot take moves to the other unguarded layers and the total still matches the standard
    row while they can hold it (Mohammad, 2026-10-13)."""
    if isinstance(plan, JointPlan) and m.act_plan:  # Stage 2 after a WxAy method: cap the weight half
        weights, note = _capped(m, plan.weights, caps, layer_numel)
        return replace(plan, weights=weights), note
    if m.max_ratio is None or not isinstance(plan, CompressionPlan):
        return plan, ""
    layer_cap = list(caps(m.max_ratio))
    if all(lp.pruning_ratio <= c for lp, c in zip(plan.layers, layer_cap)):
        return plan, ""
    asked = max(lp.pruning_ratio for lp in plan.layers)
    note = (f"capped at {m.max_ratio:.0%} of each layer, or the layer's feed-forward share when lower (lowest cap "
            f"{min(layer_cap):.1%}; the plan asked for up to {asked:.0%})")
    if plan.kind != "same_size_pruning":
        return cap_pruning(plan, layer_cap), note
    # the same total as the capped standard row: the level on every layer, each at most its cap
    level = plan.prune_ratio_aggressive
    target = sum(min(level, c) * n for c, n in zip(layer_cap, layer_numel)) / sum(layer_numel)
    refit = same_size_pruning_plan([lp.sensitivity for lp in plan.layers], plan.layers[0].bit_width, target,
                                   layer_numel, frozenset(lp.layer for lp in plan.layers if lp.guarded), layer_cap)
    return combine_plans(plan, refit), note + "; what the capped layers could not take moved to the other layers"


def _label_level(row, m: Method, level: float | None, plan: CompressionPlan | None, layer_numel: list[int]) -> None:
    """A pruning level's row label: the technique, the row's kind, the level and the share of the decoder layers'
    weights the row really removes (Mohammad, 2026-10-13), which differs when structured pruning is capped or the
    same-size plan runs out of layers. Without a plan (it failed to load) the share is left out."""
    if level is None:
        return
    kind = row.info.get("kind_label") or row.plain_name
    row.info["kind_label"] = kind  # the plain name before this label replaces it
    row.info["label"] = f"{m.label}: {kind}, level {level:.0%}"
    if plan is not None:
        row.info["label"] += f", {decoder_share_removed(_weights(plan), layer_numel):.1%} of the layer weights removed"


_ROTATED = ("not measured: the model rotates inputs while it runs (QuaRot/SpinQuant), and a saved model file "
            "cannot hold that rotation")
_KV_NOT_ON_BACKENDS = ("not run: this backend cannot forget words or give each layer its own key/value bits, and does not "
                       "round keys and values this technique's way; backends run only rows they can run exactly "
                       "(Mohammad, 2026-10-18)")
_BACKENDS_BEFORE = ("the backend columns are the Stage 2 model this row was built on: the backends cannot run the "
                    "KV technique, so they measured the model before it")
_WEIGHTS_ONLY = ("16-bit: no kernel for the plan's activation bits here, so the weights run at the plan's bits and "
                 "the activations are not rounded")


def _rounded_by_backend(name: str, m: Method, group: int) -> str:
    out = (f"the plan's bits, rounded the backend's way ({name}: on the fly, symmetric, one scale per token and "
           f"group of {group} inputs)")
    if "rptq" in m.name:
        out += "; RPTQ's own static scales are not in the saved model, so this is plain rounding"
    return out


def _require(m: Method) -> None:
    if m.apply is None:
        raise NotImplementedError(m.unavailable or f"{m.label} is not implemented yet ({m.library})")


def run_stages(ctx: RunContext, candidate: dict[str, Any], stage0_dir: str | Path, **kwargs: Any) -> dict[str, Path]:
    """Stages 1-3 with the methods in cfg.stages, each on top of the same Stage 0 plans. Stage 2 runs every
    pruning method alone, then after each Stage 1 weight method in cfg.stages.stage2_after."""
    st = ctx.cfg.stages
    stage2 = st.stage2_methods + [f"{p}{AFTER}{q}" for q in st.stage2_after for p in st.stage2_methods]
    outputs = {}
    for stage, names in ((1, st.stage1_methods), (2, stage2), (3, st.stage3_methods)):
        if names:
            for name, path in run_stage(ctx, stage, names, candidate, stage0_dir, **kwargs).items():
                outputs[f"stage_{stage}_{name}"] = path
    return outputs



def trade_off_rows(rows: list[dict[str, Any]], memory: str = "predicted_weight_memory_gb") -> list[dict[str, Any]]:
    """The finished rows on the best trade-off line between memory and validation perplexity: no other row is
    both as small and as accurate and better in one (Mohammad, 2026-10-19: "the highest memory reduction and lowest
    perplexity"). Held-out perplexity is never used to choose."""
    ok = [r for r in rows if r["status"] == "ok" and all(
        isinstance(r["metrics"].get(k), (int, float)) and math.isfinite(r["metrics"][k]) for k in ("ppl_val", memory))]
    point = lambda r: (r["metrics"][memory], r["metrics"]["ppl_val"])  # noqa: E731
    return [r for r in ok if not any(point(o) != point(r) and all(a <= b for a, b in zip(point(o), point(r)))
                                     for o in ok)]


def run_chain(ctx: RunContext, candidate: dict[str, Any], stage0_dir: str | Path, **kwargs: Any) -> dict[str, Path]:
    """The sequential path 0 -> 1 -> 2 -> 3 -> 4 (Mohammad, 2026-10-19), Standard and Our method each on their own:
    Stage 1 runs cfg.stages.stage1_methods; every WxAy row of a side on its memory vs perplexity trade-off line
    (trade_off_rows) is pruned by every cfg.stages.stage2_methods method at every share, in
    stage_2_<side>_after_<row>/; every Stage 2 row of the side on the trade-off line (all shares and Stage 1 rows
    together) gets every cfg.stages.stage3_methods method, in stage_3_<side>_after_<Stage 1 row>_<Stage 2 row>/.
    Stage 4 (evaluation) runs inside every row as in the other modes: Hugging Face always, the backends set in
    cfg.eval for Stage 1 and 2 rows; Stage 3 rows run on Hugging Face only, as in the 0 -> 3 -> 4 path (backends
    cannot run the KV techniques)."""
    st = ctx.cfg.stages
    if not any(WITH in n for n in st.stage1_methods):
        raise ValueError(f"the chain picks WxAy Stage 1 rows; stages.stage1_methods {st.stage1_methods} has none "
                         "(<weights>_with_<activations>)")
    out: dict[str, Path] = {}

    def run(prefix: str, *args: Any, **kw: Any) -> list[dict[str, Any]]:
        paths = run_stage(ctx, *args, candidate, stage0_dir, **kw, **kwargs)
        out.update({f"{prefix}_{k}": p for k, p in paths.items()})
        return json.loads(paths["json"].read_text())["rows"]

    stage1 = run("stage_1", 1, st.stage1_methods)
    for variant, side in (("original", "standard"), ("framework", "ours")):
        first = trade_off_rows([r for r in stage1 if r["variant"] == variant and WITH in r["method"]])
        if not first:
            raise RuntimeError(f"Stage 1 has no finished WxAy {side} row (STAGE1_METHODS needs <weights>_with_<acts>); "
                               "the chain cannot go on")
        log.info("chain: %s Stage 1 rows going on: %s", side, [r["method"] for r in first])
        second: list[tuple[str, dict[str, Any], tuple[Method, Any]]] = []  # (Stage 1 row, Stage 2 row, built as)
        for r1 in first:
            used: dict[tuple[str, str], tuple[Method, Any, float | None]] = {}
            sub = f"stage_2_{side}_after_{r1['method']}"
            for r2 in run(sub, 2, st.stage2_methods, after={variant: r1["method"]}, only=variant, used=used, subdir=sub):
                if r2["variant"] == variant and r2["status"] == "ok":  # failed rows are never picked
                    m, plan, _ = used[(r2["method"], variant)]
                    second.append((r1["method"], r2, (replace(m, name=r2["method"]), plan)))
        if not st.stage3_methods:
            continue
        line = trade_off_rows([r2 for _, r2, _ in second])
        log.info("chain: %s Stage 2 rows going on: %s", side,
                 [f"{a}/{r2['method']}" for a, r2, _ in second if r2 in line])
        for r1_name, r2, built in second:
            if r2 in line:
                sub = f"stage_3_{side}_after_{r1_name}_{r2['method']}"
                run(sub, 3, st.stage3_methods, before={variant: built}, only=variant, subdir=sub)
    return out
