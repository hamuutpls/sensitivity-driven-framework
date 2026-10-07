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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from sdf.data import eval_windows
from sdf.eval import downstream, llamacpp
from sdf.eval.metrics import measure_model
from sdf.run import RunContext
from sdf.stage0.activation import ActivationPlan, uniform_activation_plan
from sdf.stage0.kv_cache import KVPlan, KVProfile, predict_kv, uniform_kv_plan
from sdf.stage0.planner import CompressionPlan, baseline_cost, combine_plans, uniform_plan
from sdf.stage0.run import (_cost_metrics, _kv_metrics, calib_batches, fp16_key, load_fp16,
                            setup, stage_reporter, weight_cost)
from sdf.stage0.sensitivity import SensitivityProfile, normalize
from sdf.stages.methods import AFTER, Method, MethodCall, methods_for
from sdf.utils.logging import get_logger

log = get_logger(__name__)

# Stage 0 output files the later stages read (see stage_0/handoff.md), by plan key.
PLAN_FILES: dict[str, tuple[str, type]] = {
    "quant": ("quant_plan.json", CompressionPlan),  # Stage 1: bits only
    "quant_same_size": ("quant_plan_budget_matched.json", CompressionPlan),
    "prune": ("prune_plan.json", CompressionPlan),  # Stage 2: pruning ratios only (bits at the baseline)
    "prune_same_size": ("prune_plan_same_size.json", CompressionPlan),
    "activations": ("activation_plan.json", ActivationPlan),
    "kv": ("kv_cache_plan.json", KVPlan),
    "kv_bits_only": ("kv_cache_plan_bits_only.json", KVPlan),
}

# Framework rows after the first plan get their own method name and are compared with the method's original row.
# Values: (method-name suffix, label, plain description) as in the Stage 0 report.
_PLAN_ROWS = {
    "quant_same_size": (
        "_same_size", "Sensitivity-guided framework, budget plan (fits in the standard method's memory)",
        "the framework limited to the memory the standard method uses: the most sensitive layers keep more bits "
        "and the least sensitive ones drop to fewer, so the two can be compared fairly, size for size."),
    "prune_same_size": (
        "_same_size", "Sensitivity-guided framework, same amount removed as the standard method",
        "the framework removing exactly as many numbers as the standard method, but taking them from the least "
        "sensitive layers instead of evenly, so the two can be compared fairly, size for size."),
}

TITLES = {1: "Quantization (weights and activations)", 2: "Pruning", 3: "KV-cache compression"}

MAIN_METRICS = {
    1: ["ppl_val", "ppl_heldout", "predicted_weight_memory_gb", "avg_bits_per_weight", "avg_activation_bits",
        "peak_memory_gb", "prefill_ms_mean", "decode_ms_per_token_mean", "build_time_s"],
    2: ["ppl_val", "ppl_heldout", "predicted_weight_memory_gb", "sparsity", "peak_memory_gb", "prefill_ms_mean",
        "decode_ms_per_token_mean", "build_time_s"],
    3: ["ppl_val", "ppl_heldout", "predicted_kv_memory_gb", "avg_kv_bits", "kv_kept_share", "peak_memory_gb",
        "prefill_ms_mean", "decode_ms_per_token_mean", "build_time_s"],
}

INTROS = {
    1: "A model is a huge store of numbers (its \"weights\"), and while it runs every layer passes numbers to the "
       "next one (its \"activations\"). This stage keeps those numbers with fewer bits, like rounding prices to the "
       "nearest dollar, and removes nothing. Each technique is tried the standard way, treating every layer the "
       "same, and the framework way, following the Stage 0 plan, then the model is tested for accuracy, memory "
       "and speed.",
    2: "This stage makes the model smaller by removing numbers that barely matter (single weights, whole "
       "channels, or the small part of a weight table that a low-rank copy can drop), and rounds nothing. The "
       "standard way removes the same share from every layer; the framework removes more from the layers Stage "
       "0 found robust and nothing from the fragile ones. Methods run on the uncompressed model, or after a "
       "Stage 1 method (named \"<pruning>_after_<quantization>\").",
    3: "While writing a reply, the model keeps notes on every earlier word (the \"KV cache\"); for long texts these "
       "notes can take more memory than the model itself. This stage stores the notes with fewer bits or forgets "
       "the least-used ones. The standard way treats every layer the same; the framework follows the Stage 0 "
       "plan for each layer.",
}


@dataclass
class Stage0Plans:
    """Everything a later stage reads from a finished Stage 0 run's stage_0/ folder."""

    dir: Path
    plans: dict[str, Any]  # plan key (PLAN_FILES) -> plan; keys whose file is missing are absent
    profile: SensitivityProfile  # layer sizes, for the predicted weight memory
    kv_profile: KVProfile | None  # layer key/value sizes, for the predicted KV memory

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
        out = cls(d, plans, SensitivityProfile.load(d / "sensitivity_profile.json"), kv_prof)
        for k, p in plans.items():
            if len(p.layers) != out.num_layers:
                raise ValueError(f"{PLAN_FILES[k][0]} has {len(p.layers)} layers, the profile {out.num_layers}")
        return out


# What a plan is about, by plan key: the standard method and the cost prediction differ by kind.
PLAN_KIND = {"quant": "weights", "quant_same_size": "weights", "prune": "prune", "prune_same_size": "prune",
             "activations": "activations", "kv": "kv", "kv_bits_only": "kv"}


def kind_of(m: Method) -> str:
    """weights | activations (Stage 1), prune (Stage 2, alone or after a Stage 1 method), kv (Stage 3)."""
    return PLAN_KIND[m.plans[0]]


def original_plan(m: Method, plans: Stage0Plans, s0, prune_ratio: float) -> Any:
    """The uniform plan of the standard method, from the Stage 0 "original method" settings: `uniform_bits` and
    nothing removed (Stage 1), `prune_ratio` removed from every layer and nothing rounded (Stage 2), both for a
    Stage 2 method run after Stage 1."""
    kind = kind_of(m)
    scores = lambda: normalize(plans.profile.raw_scores, s0.normalization)  # noqa: E731
    if kind == "weights":
        return uniform_plan(scores(), s0.uniform_bits, 0.0)
    if kind == "prune":
        return uniform_plan(scores(), s0.uniform_bits if m.quant_plans else s0.baseline_bits, prune_ratio)
    if kind == "activations":
        return uniform_activation_plan(plans.num_layers, s0.act_uniform_bits)
    return uniform_kv_plan(plans.num_layers, s0.kv_uniform_bits)


def _plan_keys(m: Method, plan_key: str) -> list[str]:
    """The plan files a framework row needs: its plan, plus the Stage 1 plan it follows in the series path."""
    return [plan_key, m.quant_plans[m.plans.index(plan_key)]] if m.quant_plans else [plan_key]


def framework_plan(m: Method, plans: Stage0Plans, plan_key: str) -> Any:
    """The Stage 0 plan a method's framework row follows; after a Stage 1 method, its bits join the pruning plan."""
    plan, *quant = (plans.plans[k] for k in _plan_keys(m, plan_key))
    return combine_plans(quant[0], plan) if quant else plan


# Stage 0 settings each kind of method reads that a plan does not carry; they join the cache key when not default
# (so entries made with the defaults stay valid).
_KIND_SETTINGS = {"weights": ("weight_zero_point", "baseline_bits"), "activations": ("act_group_size", "baseline_bits"),
                  "prune": ("weight_zero_point", "baseline_bits"),
                  "kv": ("kv_group_size", "kv_module_names", "baseline_bits")}


def _setting_key(kind: str, s0) -> dict[str, Any]:
    default = type(s0)()
    return {k: getattr(s0, k) for k in _KIND_SETTINGS[kind] if getattr(s0, k) != getattr(default, k)}


def plan_metrics(plan: Any, plans: Stage0Plans, cfg, candidate: dict[str, Any],
                 storage: str | None = None) -> dict[str, Any]:
    """What the plan is predicted to cost (the same predictions Stage 0 reports), next to the measured metrics.
    `storage`: how removed weights are stored (Method.storage), when not the run's stage0.sparse_storage."""
    s0 = cfg.stage0
    if isinstance(plan, CompressionPlan):
        return _cost_metrics(weight_cost(plans.profile, candidate["gptq_groupsize"], s0, storage=storage)(plan))
    if isinstance(plan, ActivationPlan):
        return {"avg_activation_bits": plan.avg_bits}
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
        return weights
    return plan_metrics(uniform_kv_plan(plans.num_layers, base), plans, cfg, candidate)


def _describe(plan: Any) -> str:
    if isinstance(plan, CompressionPlan):
        bits = sorted({lp.bit_width for lp in plan.layers})
        prune = sorted({lp.pruning_ratio for lp in plan.layers})
        return f"{plan.kind} plan, bits {bits}, prune ratios {prune}"
    if isinstance(plan, ActivationPlan):
        return f"{plan.kind} plan, activation bits avg {plan.avg_bits:.3g}"
    bits = sorted({b for lp in plan.layers for b in (lp.key_bits, lp.value_bits)})
    return f"{plan.kind} plan, KV bits {bits}, kept share min {min(lp.keep_ratio for lp in plan.layers):.2g}"


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
) -> dict[str, Path]:
    """Run one of Stages 1-3 with `method_names` (see sdf.stages.methods.METHODS). `model_factory` returns a
    fresh FP16 model on the right device (default: from_pretrained); tests pass a tiny model's deepcopy."""
    if stage not in TITLES:
        raise ValueError(f"run_stage runs Stages 1-3, not {stage}")
    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    methods = methods_for(stage, method_names)
    plans = Stage0Plans.load(stage0_dir)
    device, handle, text_loader = setup(ctx, None, tokenizer, text_loader)
    dtype = getattr(torch, cfg.model.dtype)
    if model_factory is None:
        def model_factory() -> nn.Module:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(cfg.model.name, torch_dtype=dtype).to(device)
    handle.factory = model_factory  # an FP16 cache miss builds the model like every row

    use_llamacpp = stage == 1 and cfg.eval.llamacpp_dir is not None  # weight rows only (see lc_on)
    lc_key = [cfg.eval.llamacpp_dir, cfg.eval.llamacpp_convert, llamacpp.VERSION] if use_llamacpp else None
    calib = (f"{candidate['calib_dataset']}, {candidate['calib_samples']} x {cfg.calibration.seq_len} tokens")
    rep = stage_reporter(
        ctx, candidate, handle, plans.profile, ("stages", "stage0", "calibration", "eval", "model", "run"),
        stage=stage, title=TITLES[stage],
        conditions={"model": cfg.model.name, "calibration": calib, "seed": cfg.run.seed,
                    "evaluation data": f"wikitext-2 test, {cfg.eval.seq_len}-token windows, validation/held-out halves",
                    "device": str(device), "backend": "HF Transformers", "Stage 0 plans": str(plans.dir),
                    "starting point": "the uncompressed model, for every row (stages are independent)"},
        main_metrics=MAIN_METRICS[stage] + (llamacpp.MAIN_METRICS if use_llamacpp else []),
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

    def lc_on(method: Method) -> bool:
        return use_llamacpp and kind_of(method) == "weights"

    def build_and_measure(method: Method, plan: Any) -> dict[str, Any]:
        model = model_factory()
        try:
            t0 = time.perf_counter()
            with method.apply(MethodCall(model, plan, candidate, cfg, batches)) as extra:
                build_s = time.perf_counter() - t0
                val, held = windows()
                metrics, raw = measure_model(model, val, held, cfg.eval, device, seed=cfg.run.seed)
                metrics.update(tasks(model))
                if lc_on(method):
                    metrics.update(llamacpp.measure(model, handle.tokenizer, plan, val, held, cfg.eval,
                                                    s0.baseline_bits))
        finally:
            del model
            _free_memory()
        metrics.update(extra)
        metrics["build_time_s"] = build_s
        if method.simulated and stage in (1, 2):
            metrics.pop("model_size_gb", None)  # still FP16 in memory; predicted_weight_memory_gb is the size
        return {"metrics": metrics, "raw": raw}

    # --- FP16: the same cache entry Stage 0 measured ------------------------------------------------------------
    with rep.method("baseline", "fp16", description="uncompressed model") as row:
        row.metrics.update(_fp16_plan_metrics(stage, plans, cfg, candidate))
        row.metrics["build_time_s"] = 0.0
        if measure_fp16:
            fp16, cached = load_fp16(ctx, handle, text_loader)
            row.metrics.update(fp16["metrics"])
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

    simulated_any = False
    for m in methods:
        simulated_any |= m.simulated
        orig = original_plan(m, plans, s0, candidate["prune_ratio_aggressive"])
        with rep.method(m.name, "original", description=f"{m.label}, standard settings: {_describe(orig)}",
                        simulated=m.simulated) as row:
            _require(m)
            key = {"stage": stage, "method": m.name, "version": m.version, "plan": orig.to_dict(),
                   "params": {p: candidate[p] for p in m.params}, **fp16_key(ctx, device),
                   "downstream": [list(cfg.eval.downstream_tasks), cfg.eval.downstream_limit],
                   "llamacpp": lc_key if lc_on(m) else None}
            if settings := _setting_key(kind_of(m), s0):
                key["settings"] = settings
            if m.calibrated:
                key["calibration"] = {"dataset": candidate["calib_dataset"], "samples": candidate["calib_samples"],
                                      "seq_len": cfg.calibration.seq_len, "batch_size": cfg.calibration.batch_size}
            result, cached = ctx.cache.get_or_compute(f"stage{stage}_original", key,
                                                      lambda: build_and_measure(m, orig))
            row.metrics.update(plan_metrics(orig, plans, cfg, candidate, m.storage))
            row.metrics.update(result["metrics"])
            row.info["cached"] = cached
            rep.add_raw(m.name, "original", result["raw"])

        for plan_key in m.plans:
            suffix, *plain = _PLAN_ROWS.get(plan_key, ("",))
            info = {"compare_to": m.name, "plan_file": PLAN_FILES[plan_key][0], "simulated": m.simulated}
            if plain:
                info["label"], info["plain_desc"] = plain
            with rep.method(m.name + suffix, "framework", **info) as row:
                _require(m)
                if gone := [PLAN_FILES[k][0] for k in _plan_keys(m, plan_key) if k not in plans.plans]:
                    raise FileNotFoundError(f"{', '.join(gone)} not in {plans.dir}; re-run Stage 0 with this plan "
                                            "enabled")
                plan = framework_plan(m, plans, plan_key)
                row.info["description"] = f"{m.label}, Stage 0 {_describe(plan)}"
                result = build_and_measure(m, plan)
                row.metrics.update(plan_metrics(plan, plans, cfg, candidate, m.storage))
                row.metrics.update(result["metrics"])
                rep.add_raw(m.name + suffix, "framework", result["raw"])

    if any(m.storage for m in methods):
        rep.plain_why.append(
            "Structured pruning and low-rank remove whole channels or store smaller factors, so their predicted size "
            "has no extra record of which numbers were kept; unstructured pruning does (one bit per number).")
    if stage == 1 and any(kind_of(m) == "weights" for m in methods):
        rep.plain_why.append(
            "The budget plan here removes nothing: it keeps the sensitive layers at more bits and pays for them by "
            "dropping the robust layers to fewer. It replaces the earlier budget plan, which paid by pruning; that "
            "one now belongs to Stage 2 as the same-size pruning plan. Do not compare numbers across the two.")
    if simulated_any:
        rep.plain_why.append(
            "Some techniques here are simulated: the numbers are rounded as the compressed model would store them, "
            "but kept in the uncompressed format. Their accuracy is real; their memory is the plan's prediction, "
            "and their speed is not the speed a real compressed model would have.")
    return rep.finalize()


def _require(m: Method) -> None:
    if m.apply is None:
        raise NotImplementedError(f"{m.label} is not implemented yet ({m.library})")


def run_stages(ctx: RunContext, candidate: dict[str, Any], stage0_dir: str | Path, **kwargs: Any) -> dict[str, Path]:
    """Stages 1-3 with the methods in cfg.stages, each on top of the same Stage 0 plans. Stage 2 runs every
    pruning method alone, then after each Stage 1 weight method in cfg.stages.stage2_after."""
    st = ctx.cfg.stages
    stage2 = st.stage2_methods + [f"{p}{AFTER}{q}" for q in st.stage2_after for p in st.stage2_methods]
    outputs = {}
    for stage, names in ((1, st.stage1_methods), (2, stage2 if st.stage2_methods else []), (3, st.stage3_methods)):
        if names:
            for name, path in run_stage(ctx, stage, names, candidate, stage0_dir, **kwargs).items():
                outputs[f"stage_{stage}_{name}"] = path
    return outputs
