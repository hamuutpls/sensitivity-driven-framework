"""Stage 0 end to end: profile (cached), plan, compare against FP16 and a uniform allocation, report."""

from __future__ import annotations

import functools
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch

from sdf.data import calibration_batches, eval_windows, load_texts
from sdf.eval.metrics import measure_model
from sdf.reporting.reporter import StageReporter
from sdf.run import RunContext
from sdf.stage0.planner import (
    CompressionPlan,
    budget_matched_plan,
    guarded_layers,
    PlanCost,
    baseline_cost,
    plan_compression,
    predict_cost,
    uniform_plan,
)
from sdf.stage0.activation import (
    ActivationPlan,
    ActivationProfile,
    activation_plan_from_weights,
    plan_activations,
    predicted_rise,
    profile_activations,
    uniform_activation_plan,
)
from sdf.stage0.handoff import write_handoff
from sdf.stage0.kv_cache import KVCost, KVPlan, KVProfile, plan_kv, predict_kv, profile_kv, uniform_kv_plan
from sdf.stage0.sensitivity import (
    GRADIENT_SCORES,
    SCORES,
    SensitivityProfile,
    normalize,
    outlier_layers,
    profile_by_ablation,
    profile_sensitivity,
)
from sdf.utils.env import environment_info, resolve_device
from sdf.utils.model_info import count_parameters, describe_model
from sdf.utils.logging import get_logger

log = get_logger(__name__)

METHOD = "allocation"
METHOD_BUDGET = "allocation_same_size"
METHOD_NO_PRUNE = "allocation_same_size_no_prune"
METHOD_KV = "kv_cache"
METHOD_KV_BITS = "kv_cache_bits_only"
METHOD_ACT = "activations"
METHOD_ACT_FROM_WEIGHTS = "activations_from_weights"
MAIN_METRICS = ["ppl_val", "ppl_heldout", "predicted_weight_memory_gb", "peak_memory_gb", "prefill_ms_mean",
                "decode_ms_per_token_mean",
                "avg_bits_per_weight", "sparsity", "sensitivity_exposure",
                "predicted_kv_memory_gb", "avg_kv_bits", "kv_kept_share", "predicted_kv_ppl_rise",
                "kv_attention_kept", "avg_activation_bits", "predicted_act_ppl_rise", "build_time_s"]


# What the per-layer "Raw score" column holds, per sensitivity score.
_RAW_SCORE_MEANING = {
    "layer_removal": "How much the prediction error (perplexity) on the calibration text rises when this layer "
                     "is skipped. Bigger means the model depends on the layer more.",
    "layer_quant": "How much the prediction error (perplexity) on the calibration text rises when only this "
                   "layer is compressed. Bigger means the layer is more fragile.",
    "grad_x_weight": "Size of each number times how much the model's error would change if it were nudged, "
                     "added up over the layer. Bigger means more sensitive.",
}


@dataclass
class Stage0Result:
    """Everything later stages take from Stage 0. Each plan is also saved as JSON next to the report and can be
    read back with its class's `load` (CompressionPlan, KVPlan, ActivationPlan)."""

    plan: CompressionPlan  # Stage 1: bits and pruning per layer (threshold plan)
    profile: SensitivityProfile
    outputs: dict[str, Path]
    budget_plan: CompressionPlan | None = None  # same size as the uniform plan
    no_prune_plan: CompressionPlan | None = None  # same size, nothing pruned
    activation_plan: ActivationPlan | None = None  # Stage 2: activation bits per layer
    activation_profile: ActivationProfile | None = None  # None when stage0.act_plan = "from_weights"
    kv_plan: KVPlan | None = None  # Stage 3: key/value bits and token budget per layer
    kv_plan_bits_only: KVPlan | None = None


class _ModelHandle:
    """Loads the model and tokenizer only when a cache miss needs them."""

    def __init__(self, ctx: RunContext, device: torch.device, model=None, tokenizer=None):
        self.ctx, self.device, self._model, self._tokenizer = ctx, device, model, tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.ctx.cfg.model.name)
        return self._tokenizer

    def model(self, dtype: str):
        if self._model is None:
            from transformers import AutoModelForCausalLM

            log.info("loading %s", self.ctx.cfg.model.name)
            self._model = AutoModelForCausalLM.from_pretrained(self.ctx.cfg.model.name)
        return self._model.to(device=self.device, dtype=getattr(torch, dtype))


def original_model_info(ctx: RunContext, handle: _ModelHandle,
                        profile: SensitivityProfile | None = None) -> dict[str, Any]:
    """The uncompressed model's parameters for the report. Reads only the config file when the weights are
    not loaded (everything came from cache); the parameter count then comes from the sensitivity profile."""
    name = ctx.cfg.model.name
    try:
        if handle._model is not None:
            return describe_model(handle._model.config, name, count_parameters(handle._model))
        from transformers import AutoConfig

        n = sum(profile.layer_numel) + profile.other_numel if profile is not None else None
        return describe_model(AutoConfig.from_pretrained(name), name, n)
    except Exception:  # noqa: BLE001 - the report must still be written
        log.exception("could not read the configuration of %s", name)
        return {"name": name}


def profile_key(ctx: RunContext, cand: dict[str, Any], score: str | None = None) -> dict[str, Any]:
    cfg = ctx.cfg
    score = score or cfg.stage0.score
    key = {"model": cfg.model.name, "profile_dtype": cfg.stage0.profile_dtype, "score": score,
           "calib_dataset": cand["calib_dataset"], "calib_samples": cand["calib_samples"],
           "seq_len": cfg.calibration.seq_len, "batch_size": cfg.calibration.batch_size, "seed": cfg.run.seed}
    if score == "layer_quant":  # the per-layer compression depends on these too
        key.update(bits=cfg.stage0.compressed_bits, group_size=cand["gptq_groupsize"])
    extra = {"taylor_ema": {"ema_beta": cfg.stage0.taylor_ema_beta}, "movement": {"lr": cfg.stage0.movement_lr},
             "hessian": {"eps": cfg.stage0.hessian_eps, "probes": cfg.stage0.hessian_probes, "clamp": True}}
    key.update(extra.get(score, {}))
    return key


def fp16_key(ctx: RunContext, device: torch.device) -> dict[str, Any]:
    cfg = ctx.cfg
    hw = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    # Downstream settings are left out: the FP16 perplexity and latency do not depend on them (and existing cache
    # entries stay valid); downstream results are cached under their own key.
    ev = {k: v for k, v in asdict(cfg.eval).items() if not k.startswith("downstream")}
    return {"model": cfg.model.name, "dtype": cfg.model.dtype, "eval": ev, "hardware": hw,
            "backend": "hf-transformers", "seed": cfg.run.seed}


def load_fp16(ctx: RunContext, handle: _ModelHandle,
              text_loader: Callable[[str, str], list[str]]) -> tuple[dict[str, Any], bool]:
    """The uncompressed model's measured metrics and raw measurements, cached per (model, eval settings,
    hardware)."""
    cfg = ctx.cfg

    def compute() -> dict[str, Any]:
        val, held = eval_windows(text_loader(cfg.eval.dataset, "test"), handle.tokenizer, cfg.eval.seq_len,
                                 cfg.eval.max_windows)
        metrics, raw = measure_model(handle.model(cfg.model.dtype), val, held, cfg.eval, handle.device,
                                     seed=cfg.run.seed)
        return {"metrics": metrics, "raw": raw}

    return ctx.cache.get_or_compute("fp16_baseline", fp16_key(ctx, handle.device), compute)


def _cost_metrics(cost: PlanCost) -> dict[str, float]:
    return {"predicted_weight_memory_gb": cost.weight_memory_gb, "avg_bits_per_weight": cost.avg_bits_per_weight,
            "sparsity": cost.sparsity, "sensitivity_exposure": cost.sensitivity_exposure}


def kv_profile_key(ctx: RunContext, cand: dict[str, Any]) -> dict[str, Any]:
    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    return {"model": cfg.model.name, "profile_dtype": s0.profile_dtype, "calib_dataset": cand["calib_dataset"],
            "calib_samples": s0.kv_calib_samples, "seq_len": cfg.calibration.seq_len,
            "batch_size": cfg.calibration.batch_size, "seed": cfg.run.seed, "bits": s0.kv_bits_options,
            "group_size": s0.kv_group_size, "keep_ratios": s0.kv_keep_ratios, "modules": s0.kv_module_names}


def load_kv_profile(ctx: RunContext, candidate: dict[str, Any], handle: _ModelHandle,
                    text_loader: Callable[[str, str], list[str]]) -> tuple[KVProfile, bool]:
    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    key = kv_profile_key(ctx, candidate)

    def compute() -> dict[str, Any]:
        batches = calibration_batches(text_loader(candidate["calib_dataset"], "train"), handle.tokenizer,
                                      s0.kv_calib_samples, cfg.calibration.seq_len, cfg.calibration.batch_size,
                                      cfg.run.seed)
        return profile_kv(handle.model(s0.profile_dtype), batches, s0.kv_bits_options, s0.kv_group_size,
                          s0.kv_keep_ratios, tuple(s0.kv_module_names), device=handle.device, meta=key).to_dict()

    d, cached = ctx.cache.get_or_compute("kv_profile", key, compute)
    return KVProfile.from_dict(d), cached


def _kv_metrics(cost: KVCost) -> dict[str, float]:
    return {"predicted_kv_memory_gb": cost.memory_gb, "avg_kv_bits": cost.avg_bits, "kv_kept_share": cost.kept_share,
            "predicted_kv_ppl_rise": cost.ppl_rise, "kv_attention_kept": cost.attention_kept}


def load_profile(ctx: RunContext, candidate: dict[str, Any], handle: _ModelHandle,
                 text_loader: Callable[[str, str], list[str]],
                 score: str | None = None) -> tuple[SensitivityProfile, bool]:
    """The sensitivity profile for this calibration setting; it depends on nothing else, so it is cached.
    `score` defaults to stage0.score."""
    cfg = ctx.cfg
    score = score or cfg.stage0.score

    def compute() -> dict[str, Any]:
        batches = calibration_batches(text_loader(candidate["calib_dataset"], "train"), handle.tokenizer,
                                      candidate["calib_samples"], cfg.calibration.seq_len,
                                      cfg.calibration.batch_size, cfg.run.seed)
        s0 = cfg.stage0
        if score in GRADIENT_SCORES:
            prof = profile_sensitivity(handle.model(s0.profile_dtype), batches, device=handle.device,
                                       meta=profile_key(ctx, candidate, score), method=score,
                                       ema_beta=s0.taylor_ema_beta, movement_lr=s0.movement_lr,
                                       hessian_eps=s0.hessian_eps,
                                       hessian_probes=s0.hessian_probes, seed=cfg.run.seed)
        elif score in ("layer_removal", "layer_quant"):
            prof = profile_by_ablation(handle.model(s0.profile_dtype), batches, score, device=handle.device,
                                       bits=s0.compressed_bits, group_size=candidate["gptq_groupsize"],
                                       meta=profile_key(ctx, candidate, score))
        else:
            raise ValueError(f"unknown stage0.score {score!r}; choose from {SCORES}")
        return prof.to_dict()

    prof_dict, cached = ctx.cache.get_or_compute("sensitivity_profile", profile_key(ctx, candidate, score), compute)
    return SensitivityProfile.from_dict(prof_dict), cached


def load_guard(ctx: RunContext, candidate: dict[str, Any], handle: _ModelHandle,
               text_loader: Callable[[str, str], list[str]],
               profile: SensitivityProfile) -> tuple[frozenset[int], float]:
    """Layers no plan may prune, and the extra profiling seconds it took. The guard always ranks by layer
    removal; when that is not the score picking the bits, a removal profile is measured (and cached) too."""
    k = ctx.cfg.stage0.guard_top_k
    if k == 0:
        return frozenset(), 0.0
    if profile.method == "layer_removal":
        return guarded_layers(profile.raw_scores, k), 0.0
    removal, _ = load_profile(ctx, candidate, handle, text_loader, score="layer_removal")
    return guarded_layers(removal.raw_scores, k), removal.cost.get("wall_clock_s", 0.0)


def run_stage0(
    ctx: RunContext,
    candidate: dict[str, Any],
    model=None,
    tokenizer=None,
    text_loader: Callable[[str, str], list[str]] | None = None,
    measure_fp16: bool = True,
) -> Stage0Result:
    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    device = resolve_device(cfg.model.device)
    handle = _ModelHandle(ctx, device, model, tokenizer)
    text_loader = text_loader or functools.partial(load_texts, cfg.data.sources)

    profile, prof_cached = load_profile(ctx, candidate, handle, text_loader)
    # The only place scores are normalised: cheap, so not cached, and changing the method reuses the profile.
    scores = normalize(profile.raw_scores, s0.normalization)
    guarded, guard_s = load_guard(ctx, candidate, handle, text_loader, profile)
    profiling_s = profile.cost["wall_clock_s"] + guard_s  # true cost of a plan, even when cached this time

    rep = StageReporter(
        stage=0, run_dir=ctx.run_dir, title="Sensitivity profiling and compression planning",
        config={"hyperparams": dict(candidate), "stage0": asdict(s0), "calibration": asdict(cfg.calibration),
                "eval": asdict(cfg.eval), "model": asdict(cfg.model), "run": asdict(cfg.run)},
        environment=environment_info(),
        conditions={"model": cfg.model.name, "calibration": f"{candidate['calib_dataset']}, "
                    f"{candidate['calib_samples']} x {cfg.calibration.seq_len} tokens", "seed": cfg.run.seed,
                    "evaluation data": f"wikitext-2 test, {cfg.eval.seq_len}-token windows, validation/held-out halves",
                    "device": str(device), "backend": "HF Transformers",
                    "group size for memory prediction": candidate["gptq_groupsize"]},
        requirement=cfg.requirement,
        main_metrics=MAIN_METRICS,
        original_model=original_model_info(ctx, handle, profile),
    )
    predict = lambda plan: predict_cost(plan, profile, candidate["gptq_groupsize"],  # noqa: E731
                                        s0.group_overhead_bits, s0.baseline_bits)

    # --- FP16 baseline: measured once per (model, eval settings, hardware) and cached ----------------------
    with rep.method("baseline", "fp16", description="uncompressed model") as row:
        row.metrics.update(_cost_metrics(baseline_cost(profile, s0.baseline_bits)))
        row.metrics["build_time_s"] = 0.0
        if measure_fp16:
            fp16, fp16_cached = load_fp16(ctx, handle, text_loader)
            row.metrics.update(fp16["metrics"])
            row.info["cached"] = fp16_cached
            rep.add_raw("baseline", "fp16", fp16["raw"])

    # --- original method: uniform allocation, no Stage 0 guidance ------------------------------------------
    uniform = uniform_plan(scores, s0.uniform_bits, s0.uniform_prune_ratio)
    with rep.method(METHOD, "original",
                    description=f"uniform {s0.uniform_bits}-bit, prune {s0.uniform_prune_ratio} on every layer") as row:
        row.metrics.update(_cost_metrics(predict(uniform)))
        row.metrics["protected_layers"] = 0
        row.metrics["build_time_s"] = 0.0  # needs no profiling

    # --- framework: sensitivity-driven plan ------------------------------------------------------------------
    plan = None
    with rep.method(METHOD, "framework",
                    description=f"sensitivity plan (threshold {candidate['sensitive_threshold']:.3g}, "
                                f"prune {candidate['prune_ratio_aggressive']:.3g} on robust layers)") as row:
        t0 = time.perf_counter()
        plan = plan_compression(scores, candidate["sensitive_threshold"], candidate["prune_ratio_aggressive"],
                                s0.protected_bits, s0.compressed_bits, guarded)
        planning_s = time.perf_counter() - t0
        row.metrics.update(_cost_metrics(predict(plan)))
        row.metrics["protected_layers"] = len(plan.protected_layers)
        row.metrics["build_time_s"] = profiling_s + planning_s
        row.info.update(profile_cached=prof_cached, profiling_cost=profile.cost)

    # --- framework at the same size as the original method --------------------------------------------------
    # The threshold plan usually spends more bits than uniform, so its accuracy edge would be partly bought
    # with memory. This plan protects as many of the most sensitive layers as fit in the uniform plan's size,
    # making the comparison size-for-size fair.
    budget = None
    with rep.method(METHOD_BUDGET, "framework", compare_to=METHOD,
                    label="Sensitivity-guided framework, same size as the standard method",
                    plain_desc="the same approach, but limited to exactly the memory the standard method uses. "
                               "It protects as many of the most sensitive layers as fit in that budget, so the "
                               "two can be compared fairly, size for size.",
                    description="sensitivity plan protecting as many top layers as fit in the uniform plan's "
                                "predicted memory") as row:
        t0 = time.perf_counter()
        budget = budget_matched_plan(scores, predict(uniform).weight_memory_gb, candidate["prune_ratio_aggressive"],
                                     s0.protected_bits, s0.compressed_bits, predict, guarded)
        row.metrics.update(_cost_metrics(predict(budget)))
        row.metrics["protected_layers"] = len(budget.protected_layers)
        row.metrics["build_time_s"] = profiling_s + time.perf_counter() - t0
        row.info["budget_gb"] = predict(uniform).weight_memory_gb

    # --- same size, no pruning ---------------------------------------------------------------------------------
    # The same-size plan pays for its protected layers by pruning the rest, which uniform does not do, so its
    # comparison mixes two effects. Here the robust layers pay with fewer bits instead and nothing is pruned.
    no_prune = None
    with rep.method(METHOD_NO_PRUNE, "framework", compare_to=METHOD,
                    label="Sensitivity-guided framework, same size, nothing removed",
                    plain_desc=f"the same-size approach without removing any numbers: the less sensitive layers "
                               f"drop to {s0.no_prune_compressed_bits} bits per number instead, to pay for "
                               "protecting the sensitive ones. This separates the effect of choosing where to "
                               "spend the bits from the effect of removing numbers.",
                    description=f"sensitivity plan in the uniform plan's predicted memory, no pruning, robust "
                                f"layers at {s0.no_prune_compressed_bits} bits") as row:
        t0 = time.perf_counter()
        no_prune = budget_matched_plan(scores, predict(uniform).weight_memory_gb, 0.0, s0.protected_bits,
                                       s0.no_prune_compressed_bits, predict, guarded)
        row.metrics.update(_cost_metrics(predict(no_prune)))
        row.metrics["protected_layers"] = len(no_prune.protected_layers)
        row.metrics["build_time_s"] = profiling_s + time.perf_counter() - t0
        row.info["budget_gb"] = predict(uniform).weight_memory_gb

    act_prof, act = _act_rows(ctx, candidate, handle, text_loader, rep, plan, len(scores))

    kv = _kv_rows(ctx, candidate, handle, text_loader, rep) if s0.kv_cache else None

    if plan is None:
        rep.finalize()
        raise RuntimeError(f"Stage 0 planning failed; see {rep.report_path}")

    _add_stage0_details(rep, profile, plan, uniform, budget, no_prune, predict, s0.baseline_bits, guarded, act,
                        act_prof)
    if budget is not None:
        budget.save(rep.dir / "compression_plan_budget_matched.json")
    if no_prune is not None:
        no_prune.save(rep.dir / "compression_plan_budget_matched_no_prune.json")
    if kv is not None:
        _add_kv_details(rep, *kv)
        kv[0].save(rep.dir / "kv_profile.json")
        kv[1].save(rep.dir / "kv_cache_plan.json")
        kv[2].save(rep.dir / "kv_cache_plan_bits_only.json")
    if act is not None:
        act.save(rep.dir / "activation_plan.json")
    if act_prof is not None:
        act_prof.save(rep.dir / "activation_profile.json")
    plan.save(rep.dir / "compression_plan.json")
    profile.save(rep.dir / "sensitivity_profile.json")
    fp16_row = next((r for r in rep.rows if r.variant == "fp16"), None)
    handoff = write_handoff(rep.dir / "handoff.md", cfg=cfg, candidate=candidate, original_model=rep.original_model,
                            profile=profile, plan=plan, budget=budget, no_prune=no_prune, uniform=uniform,
                            predict=predict, fp16=fp16_row.metrics if fp16_row else {}, guarded=guarded, act=act,
                            act_prof=act_prof, kv=kv)
    rep.sections.append(("What later stages receive", f"See {handoff.name}: the plan each stage loads, layer by "
                                                      "layer, with its predicted cost."))
    return Stage0Result(plan=plan, profile=profile, outputs={**rep.finalize(), "handoff": handoff}, budget_plan=budget,
                        no_prune_plan=no_prune, activation_plan=act, activation_profile=act_prof,
                        kv_plan=kv[1] if kv is not None else None, kv_plan_bits_only=kv[2] if kv is not None else None)


def activation_profile_key(ctx: RunContext, cand: dict[str, Any]) -> dict[str, Any]:
    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    return {"model": cfg.model.name, "profile_dtype": s0.profile_dtype, "calib_dataset": cand["calib_dataset"],
            "calib_samples": s0.act_calib_samples, "seq_len": cfg.calibration.seq_len,
            "batch_size": cfg.calibration.batch_size, "seed": cfg.run.seed, "bits": s0.act_bits_options,
            "group_size": s0.act_group_size}


def _act_rows(ctx: RunContext, candidate: dict[str, Any], handle: _ModelHandle,
              text_loader: Callable[[str, str], list[str]], rep: StageReporter, plan: CompressionPlan | None,
              num_layers: int) -> tuple[ActivationProfile | None, ActivationPlan | None]:
    """Activation rows: uniform (original), the plan Stage 2 gets (framework) and, when that plan is measured,
    the weight-derived plan at its own average for comparison."""
    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    if s0.act_plan not in ("measured", "from_weights"):
        raise ValueError(f"unknown stage0.act_plan {s0.act_plan!r}; choose measured or from_weights")
    lo, hi = min(s0.act_bits_options), max(s0.act_bits_options)
    prof: ActivationProfile | None = None
    act: ActivationPlan | None = None

    def metrics(p: ActivationPlan) -> dict[str, Any]:
        out = {"avg_activation_bits": p.avg_bits, "protected_layers": sum(lp.protected for lp in p.layers)}
        if prof is not None:
            out["predicted_act_ppl_rise"] = predicted_rise(p, prof)
        return out

    uniform = uniform_activation_plan(num_layers, s0.act_uniform_bits)
    with rep.method(METHOD_ACT, "original", label="Standard method: activations",
                    plain_desc=f"every layer's activations at {s0.act_uniform_bits} bits.",
                    description=f"uniform {s0.act_uniform_bits}-bit activations") as orig_row:
        orig_row.metrics.update(metrics(uniform))
        orig_row.metrics["build_time_s"] = 0.0

    if s0.act_plan == "measured":
        with rep.method(METHOD_ACT, "framework", label="Sensitivity-guided framework: activations",
                        plain_desc=f"each layer was tested for how much rounding its activations hurts; the "
                                   f"average of {s0.act_avg_bits:g} bits goes to the layers that suffer most.",
                        description=f"measured plan, {s0.act_avg_bits:g} average bits over {s0.act_bits_options}"
                        ) as row:
            key = activation_profile_key(ctx, candidate)

            def compute() -> dict[str, Any]:
                batches = calibration_batches(text_loader(candidate["calib_dataset"], "train"), handle.tokenizer,
                                              s0.act_calib_samples, cfg.calibration.seq_len,
                                              cfg.calibration.batch_size, cfg.run.seed)
                return profile_activations(handle.model(s0.profile_dtype), batches, s0.act_bits_options,
                                           s0.act_group_size, device=handle.device, meta=key).to_dict()

            d, cached = ctx.cache.get_or_compute("activation_profile", key, compute)
            prof = ActivationProfile.from_dict(d)
            act = plan_activations(prof, s0.act_avg_bits)
            row.metrics.update(metrics(act))
            row.metrics["build_time_s"] = prof.cost.get("wall_clock_s", 0.0)
            row.info.update(profile_cached=cached, profiling_cost=prof.cost)
        if prof is not None:  # the original row could only be scored once the profile existed
            orig_row.metrics.update(metrics(uniform))

    if plan is not None:
        with rep.method(METHOD_ACT_FROM_WEIGHTS if s0.act_plan == "measured" else METHOD_ACT, "framework",
                        compare_to=METHOD_ACT,
                        label="Sensitivity-guided framework: activations from the weight plan",
                        plain_desc="no test of the activations: layers the weight plan protects or never prunes "
                                   f"keep {hi} bits, the rest drop to {lo}.",
                        description=f"{hi}-bit on layers the weight plan protects or never prunes, "
                                    f"{lo}-bit elsewhere") as row:
            derived = activation_plan_from_weights(plan, hi, lo)
            row.metrics.update(metrics(derived))
            row.metrics["build_time_s"] = 0.0
            if s0.act_plan == "from_weights":
                act = derived
    return prof, act


def _add_stage0_details(rep: StageReporter, profile: SensitivityProfile, plan: CompressionPlan,
                        uniform: CompressionPlan, budget: CompressionPlan | None,
                        no_prune: CompressionPlan | None,
                        predict: Callable[[CompressionPlan], PlanCost], baseline_bits: int,
                        guarded: frozenset[int], act: ActivationPlan | None,
                        act_prof: ActivationProfile | None) -> None:
    fw_cost, un_cost = predict(plan), predict(uniform)
    outliers = outlier_layers(profile.raw_scores)
    budget_layers = budget.layers if budget is not None else [None] * len(plan.layers)
    no_prune_layers = no_prune.layers if no_prune is not None else [None] * len(plan.layers)
    act_layers = act.layers if act is not None else [None] * len(plan.layers)
    act_rise = [r[0] for r in act_prof.rise] if act_prof is not None else [None] * len(plan.layers)
    for lp, ul, bl, nl, al, ar, raw, numel, fw_mb, un_mb in zip(plan.layers, uniform.layers, budget_layers,
                                                            no_prune_layers, act_layers, act_rise,
                                                            profile.raw_scores,
                                                            profile.layer_numel, fw_cost.per_layer_mb,
                                                            un_cost.per_layer_mb):
        rep.per_layer.append({
            "layer": lp.layer, "sensitivity": lp.sensitivity, "raw_score": raw, "outlier": lp.layer in outliers,
            "weights": numel, "guarded": lp.guarded,
            "activation_bits": None if al is None else al.act_bits,
            "activation_rise_low_bits": ar,
            "framework_protected": lp.protected, "framework_bits": lp.bit_width,
            "framework_prune_ratio": lp.pruning_ratio, "framework_predicted_mb": fw_mb,
            "same_size_protected": None if bl is None else bl.protected,
            "same_size_bits": None if bl is None else bl.bit_width,
            "same_size_prune_ratio": None if bl is None else bl.pruning_ratio,
            "no_prune_protected": None if nl is None else nl.protected,
            "no_prune_bits": None if nl is None else nl.bit_width,
            "uniform_bits": ul.bit_width, "uniform_prune_ratio": ul.pruning_ratio, "uniform_predicted_mb": un_mb,
        })

    n = len(plan.layers)
    scores = [lp.sensitivity for lp in plan.layers]
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    rep.sections.append(("Sensitivity profile and plan", "\n".join([
        f"Score: {profile.method}, {rep.config['stage0']['normalization']}-normalised to [0, 1] over {n} decoder layers.",
        f"Most sensitive layers: {', '.join(f'{i} ({scores[i]:.2f})' for i in ranked[:3])}. "
        f"Least sensitive: {', '.join(f'{i} ({scores[i]:.2f})' for i in ranked[-3:])}.",
        f"Protected ({len(plan.protected_layers)}/{n}): {plan.protected_layers or 'none'}.",
        f"Profiling cost: {profile.cost.get('calibration_batches')} batches, "
        f"{profile.cost.get('calibration_tokens')} tokens, {profile.cost.get('wall_clock_s', 0):.1f}s, "
        "peak GPU memory " + (f"{profile.cost['peak_memory_gb']:.2f} GB." if profile.cost.get("peak_memory_gb")
                              else "n/a (CPU)."),
        "",
        "Stage 0 only allocates precision, so memory here is predicted from the plan. Accuracy and latency of "
        "the two allocations are measured once Stage 1 applies them.",
    ])))

    k = rep.config["stage0"]["guard_top_k"]
    rep.sections.append(("Pruning guard", (
        f"Never pruned in any framework plan: layers {sorted(guarded)}, the {k} whose removal raises perplexity "
        "most (layer-removal score). They keep the bits their plan gives them; only pruning is blocked."
        if guarded else "Off (stage0.guard_top_k = 0): any layer may be pruned.")))
    if act is not None:
        how = ("measured: each layer's Linear inputs were rounded to each of "
               f"{act_prof.bits_options} bits on its own and the perplexity rise recorded; the average budget "
               "goes where the rise is largest" if act_prof is not None else
               "derived from the weight plan, not measured: layers it protects or guards get the most bits, the "
               "rest the fewest. This assumes a layer fragile for weights is fragile for activations too")
        top = [lp.layer for lp in act.layers if lp.protected]
        rep.sections.append(("Activation plan (for Stage 2)", (
            f"Activation bits per layer are {how}. Layers at the highest width: {top or 'none'}; average "
            f"{act.avg_bits:.2f} bits. Saved as activation_plan.json"
            + (" (measurements in activation_profile.json)." if act_prof is not None else "."))))

    fixed = fw_cost.fixed_memory_gb
    rep.sections.append(("Size floor", (
        f"Embeddings, the LM head and norms ({profile.other_numel:,} parameters) stay at {baseline_bits} bits in "
        f"every plan: {fixed:.3f} GB, {fixed / un_cost.weight_memory_gb:.0%} of the uniform plan's predicted "
        "size. No layer allocation can go below this floor; compressing the embeddings / LM head is the only "
        "way past it.")))

    for i in outliers:
        others = [r for j, r in enumerate(profile.raw_scores) if j != i]
        side = "below" if profile.raw_scores[i] < min(others) else "above" if profile.raw_scores[i] > max(others) \
            else "far from"
        fate = ("protected" if i in plan.protected_layers else
                f"compressed and pruned at {plan.layers[i].pruning_ratio:.0%}")
        rep.anomalies.append(
            f"Layer {i} is an outlier: raw sensitivity {profile.raw_scores[i]:.4g}, {side} every other layer "
            f"({min(others):.4g}–{max(others):.4g}). The plan has it {fate}. Check this layer's quality "
            "separately before trusting the plan for it.")

    if budget is not None and not budget.protected_layers:
        rep.anomalies.append("The same-size plan could not protect any layer within the uniform plan's memory; "
                             "raise prune_ratio_aggressive to free room for protection.")

    # sanity checks
    if len(plan.protected_layers) == n:
        rep.anomalies.append("Every layer is protected: the plan compresses nothing beyond 8-bit. "
                             "Lower sensitive_threshold.")
    elif not plan.protected_layers:
        rep.anomalies.append("No layer is protected: the plan is uniform 4-bit plus pruning. "
                             "Raise sensitive_threshold.")
    if len(set(profile.raw_scores)) == 1:
        rep.anomalies.append("All raw sensitivity scores are identical; the profile carries no signal.")
    if fw_cost.weight_memory_gb >= baseline_cost(profile, baseline_bits).weight_memory_gb:
        rep.anomalies.append("The framework plan is predicted to use no less memory than FP16.")

    if budget is not None:
        pruned = budget.compressed_layers
        ratio = budget.prune_ratio_aggressive
        if pruned and ratio > 0 and uniform.prune_ratio_aggressive == 0:
            bits = budget.layers[pruned[0]].bit_width
            rep.sections.append(("Same-size plan: pruning caveat", (
                f"The same-size plan protects layers {budget.protected_layers or 'none'} at full protected precision "
                f"and prunes the other {len(pruned)} layers by {ratio:.0%}, while the uniform plan prunes nothing: "
                "the size match is bought with pruning. Sensitivity exposure treats pruning as a linear loss of "
                f"bits ({bits} x {1 - ratio:.2g} = {bits * (1 - ratio):.3g} effective bits), which likely understates "
                "the damage of removing weights outright, so its exposure advantage is optimistic. The "
                f"`{METHOD_NO_PRUNE}` row matches the size without pruning to isolate the effect of the "
                "sensitivity guidance itself.")))

    for name, bp in (("same-size plan", budget), ("same-size plan without pruning", no_prune)):
        if bp is not None:
            rep.sections.append((f"Unused budget: {name}", _unused_budget(bp, un_cost.weight_memory_gb,
                                                                               rep.config["stage0"]["protected_bits"], predict)))

    _add_plain_explanation(rep, profile, plan, fw_cost, un_cost,
                           predict(budget) if budget is not None else None, budget,
                           predict(no_prune) if no_prune is not None else None, no_prune, outliers)

    rep.next_steps += [
        "Run Stage 1 (GPTQ) with both allocations to measure their perplexity and latency.",
        "If sensitivity exposure is high, raise sensitive_threshold or lower prune_ratio_aggressive.",
    ]



def _add_plain_explanation(rep: StageReporter, profile: SensitivityProfile, plan: CompressionPlan,
                           fw_cost: PlanCost, un_cost: PlanCost, budget_cost: PlanCost | None,
                           budget: CompressionPlan | None, np_cost: PlanCost | None,
                           no_prune: CompressionPlan | None, outliers: list[int]) -> None:
    """The plain-language part of the Stage 0 report, for readers with no AI background."""
    s0 = rep.config["stage0"]
    hp = rep.config["hyperparams"]
    n, n_prot = len(plan.layers), len(plan.protected_layers)
    rep.plain_intro = (
        "Compressing a language model is like shrinking a photo: done carefully, you save a lot of space and "
        "barely notice the difference; done carelessly, the picture turns to mush. The catch is that not every "
        "part of the model is equally delicate. Some layers can be squeezed hard with no visible effect, while "
        "others fall apart at the slightest change.\n\n"
        "Stage 0 finds out which is which, in four steps.\n\n"
        f"1. **Measure.** The model reads {hp['calib_samples']} short passages of ordinary text "
        f"({hp['calib_dataset']}) and tries to predict each next word. " + _MEASURE_PLAIN[profile.method]
        + f" This is done for each of the {n} layers.\n"
        f"2. **Rank.** The {n} raw scores are put in order and turned into a 0-to-1 scale: the least "
        "sensitive layer gets 0, the most sensitive gets 1, and the rest are spaced evenly between by their "
        "position in the order. Using the order rather than the raw values stops one unusual layer from "
        "squashing all the others together.\n"
        f"3. **Choose.** Every layer at or above the threshold ({hp['sensitive_threshold']:.2f}) is "
        f"**protected**; the rest are **compressed**. A threshold of {hp['sensitive_threshold']:.2f} "
        f"protects roughly the most sensitive {1 - hp['sensitive_threshold']:.0%} of layers.\n"
        f"4. **Allocate.** Protected layers keep {s0['protected_bits']} bits per number and lose nothing. "
        f"Compressed layers get {s0['compressed_bits']} bits per number and have "
        f"{hp['prune_ratio_aggressive']:.0%} of their numbers removed. For a fair comparison, two same-size "
        "versions are also made: they protect as many of the top-ranked layers as fit in the standard "
        "method's memory, paying for it either by removing numbers from the other layers or, in the version "
        f"that removes nothing, by storing the other layers with {s0['no_prune_compressed_bits']} bits.\n\n"
        "The idea being tested is simple: spend the memory where damage hurts most. Whether it works is "
        "decided by accuracy, which Stage 1 measures.\n\n"
        "Nothing is actually compressed yet. Stage 0 only makes the plan and predicts how big the model would "
        "be. Later stages carry out the plan and measure the real accuracy and speed.")

    rep.glossary.update({
        "Sensitivity": "How much the model's predictions would suffer if a layer were changed. It is measured "
                       "by checking how strongly each layer's numbers influence the model's mistakes on "
                       "ordinary text. Here it is shown on a 0-to-1 scale, where 1 is the most sensitive layer "
                       "and 0 the least.",
        "Protected layer": f"A sensitive layer that the plan keeps at {s0['protected_bits']} bits and does not "
                           "trim.",
        "Pruning": "Deleting the numbers that matter least, like cutting unimportant words from an essay.",
        "Threshold": f"The cut-off for protecting a layer. With {hp['sensitive_threshold']:.2f}, roughly the "
                     f"most sensitive {1 - hp['sensitive_threshold']:.0%} of layers are protected.",
        "Calibration text": "A small sample of ordinary text used only to measure sensitivity. It is kept "
                            "separate from the text used to test accuracy.",
    })

    exp_fw, exp_un = fw_cost.sensitivity_exposure, un_cost.sensitivity_exposure
    mem_fw, mem_un = fw_cost.weight_memory_gb, un_cost.weight_memory_gb
    why: list[str] = []
    if mem_fw > mem_un:
        why.append(f"Keeping the sensitive layers at higher precision costs space, so the framework's model is "
                   f"predicted to be larger ({mem_fw:.3g} GB against {mem_un:.3g} GB for the standard "
                   "method).")
    else:
        why.append(f"Trimming the robust layers more than pays for the protected ones, so the framework's "
                   f"model is predicted to be smaller ({mem_fw:.3g} GB against {mem_un:.3g} GB).")
    if exp_fw < exp_un:
        why.append(f"In return, far less of the compression lands on the fragile layers (a score of "
                   f"{exp_fw:.2f} against {exp_un:.2f}, where lower is safer). Damage to the fragile layers "
                   "is what hurts accuracy, so the framework's model is expected to stay closer to the "
                   "original.")
    else:
        why.append(f"It also does not shield the fragile layers better than the standard method ({exp_fw:.2f} "
                   f"against {exp_un:.2f}, lower is safer), so this plan has no expected accuracy advantage. "
                   "Try a lower threshold or less pruning.")
    if mem_fw > mem_un:
        why.append("Because the framework's plan is bigger, comparing its accuracy with the standard method "
                   "would not be fair: some of any gain would come simply from using more memory.")
    if budget is not None and budget_cost is not None:
        why.append(f"That is why the report also includes a same-size version of the framework. It gets exactly "
                   f"the standard method's memory budget ({budget_cost.weight_memory_gb:.3g} GB against "
                   f"{mem_un:.3g} GB) and spends it on protecting {_top(len(budget.protected_layers))}, "
                   f"paid for by trimming the others. Its fragile-parts score is "
                   f"{budget_cost.sensitivity_exposure:.2f} against {exp_un:.2f} for the standard method. This "
                   "is the fair head-to-head: same size, different choice of where to spend the bits.")
        if budget.compressed_layers and budget.prune_ratio_aggressive > 0 and un_cost.sparsity == 0:
            bits = s0["compressed_bits"]
            ratio = budget.prune_ratio_aggressive
            why.append(f"There is a catch. The same-size plan saves its space by removing {ratio:.0%} of the numbers "
                       f"in the other {len(budget.compressed_layers)} layers, while the standard method removes "
                       "nothing. The fragile-parts score counts removing numbers as if it were just a milder "
                       f"form of rounding ({bits} bits with {ratio:.0%} removed is scored like "
                       f"{bits * (1 - ratio):.3g} bits). In practice, deleting numbers outright probably does more "
                       "harm than that, so this score likely makes the same-size plan look safer than it is.")
    if no_prune is not None and np_cost is not None:
        why.append(f"To check the guidance on its own, the report also includes a same-size version that removes "
                   f"nothing. It pays for protecting {_top(len(no_prune.protected_layers))} by storing the other "
                   f"layers with {s0['no_prune_compressed_bits']} bits instead of {s0['uniform_bits']}. It needs "
                   f"{np_cost.weight_memory_gb:.3g} GB and its fragile-parts score is "
                   f"{np_cost.sensitivity_exposure:.2f} against {exp_un:.2f} for the standard method. If it "
                   "matches or beats the standard method once Stage 1 measures accuracy, the gain comes from "
                   "choosing where to spend the bits, not from removing numbers.")
    why.append(f"Some parts of the model (the word dictionary at its input and output, called embeddings and the "
               f"LM head) stay uncompressed in every plan. They alone take {fw_cost.fixed_memory_gb:.2f} GB, "
               f"about {fw_cost.fixed_memory_gb / mem_un:.0%} of the standard method's size, so no plan here "
               "can get the model smaller than that.")
    median = sorted(profile.raw_scores)[len(profile.raw_scores) // 2]
    for i in outliers:
        verb = "protects" if i in plan.protected_layers else "squeezes and trims"
        level = "low" if profile.raw_scores[i] < median else "high"
        why.append(f"Warning: layer {i} behaves very differently from all the others (its sensitivity score is "
                   f"unusually {level}). "
                   f"The plan {verb} it based on that score. Unusual layers, especially the first, can "
                   "matter more than the score suggests, so this layer should be checked before relying on "
                   "the plan.")
    why.append("Whether the trade pays off (similar accuracy at a similar or smaller size) is only proven once "
               "Stage 1 applies the plans and measures their actual accuracy.")
    rep.plain_why = why

    fp16_row = next((r for r in rep.rows if r.variant == "fp16" and r.status == "ok"), None)
    mem_full = fp16_row.metrics.get("predicted_weight_memory_gb") if fp16_row else None
    summary = (
        f"Stage 0 measured how sensitive each of the model's {n} layers is, using {hp['calib_samples']} short "
        "passages of ordinary text, and planned how to compress it. "
        f"The sensitivity-guided plan protects {_top(n_prot)} and compresses the other "
        f"{n - n_prot}. It is predicted to need {mem_fw:.2f} GB, against {mem_un:.2f} GB for the standard method "
        f"(every layer at {s0['uniform_bits']} bits)"
        + (f" and {mem_full:.2f} GB for the uncompressed model. " if mem_full else ". "))
    if budget is not None and budget_cost is not None:
        summary += (
            f"Because {'that plan is bigger than' if mem_fw > mem_un else 'plans of different sizes are hard to compare with'} "
            f"the standard method, a same-size version was also made: at {budget_cost.weight_memory_gb:.2f} GB it "
            f"protects {_top(len(budget.protected_layers))} and puts "
            f"{'less' if budget_cost.sensitivity_exposure < exp_un else 'no less'} of the compression on fragile "
            f"layers than the standard method ({budget_cost.sensitivity_exposure:.2f} against {exp_un:.2f}, "
            "lower is better). ")
        if budget.prune_ratio_aggressive > 0 and un_cost.sparsity == 0:
            summary += ("It makes room by removing numbers from the other layers, which the "
                        "standard method does not do, so a version that removes nothing was added as well "
                        + (f"({_top(len(no_prune.protected_layers))} protected, the rest at "
                           f"{s0['no_prune_compressed_bits']} bits, fragile-parts score "
                           f"{np_cost.sensitivity_exposure:.2f}). " if no_prune is not None and np_cost else ". "))
    if outliers:
        summary += (f"Layer{'s' if len(outliers) > 1 else ''} {', '.join(map(str, outliers))} behave"
                    f"{'' if len(outliers) > 1 else 's'} unusually and should be checked. ")
    summary += "Nothing has been compressed yet: the real accuracy and speed of these plans are measured in Stage 1."
    rep.plain_summary = summary

    rep.plain_layer_columns = (
        "Sensitivity of every layer",
        "Sensitivity runs from 0 (least sensitive layer) to 1 (most sensitive). The raw score is the measurement "
        "before it is put on that scale. \"Protected\" layers keep high precision; the others are compressed "
        "and have the listed share of their numbers removed. The last column gives the bits per number in the "
        "same-size plan that removes nothing.",
        [("layer", "Layer", "The layer's position in the model, counting from 0 at the input end."),
         ("sensitivity", "Sensitivity (0 to 1)", "The layer's rank among all layers: 0 is the least sensitive "
          "layer, 1 the most. The plans compare this with the threshold."),
         ("raw_score", "Raw score", _RAW_SCORE_MEANING.get(profile.method, "The measurement before it is put on "
          "the 0 to 1 scale.")),
         ("framework_protected", "Protected (framework plan)", "\"yes\" if the framework plan keeps this layer "
          "at high precision because its sensitivity is at or above the threshold."),
         ("framework_prune_ratio", "Share removed (framework plan)", "Share of the layer's numbers the "
          "framework plan deletes (0.3 means 30%); protected and guarded layers lose nothing."),
         ("guarded", "Never pruned", "\"yes\" for the layers the model depends on most (removing one alone "
          "hurts the most), which no plan may trim, whatever their sensitivity score."),
         ("activation_bits", "Activation bits (framework)", "Bits for the numbers flowing into this layer, "
          "planned for Stage 2."),
         ("activation_rise_low_bits", "Activation damage at fewest bits", "How much the prediction error "
          "(perplexity) rose when only this layer's incoming numbers were rounded to the fewest bits allowed. "
          "Bigger means the layer needs more bits. Empty when activations were not measured."),
         ("same_size_protected", "Protected (same-size plan)", "\"yes\" if the plan that fits in the standard "
          "method's memory protects this layer. It protects the most sensitive layers first, as many as fit."),
         ("same_size_prune_ratio", "Share removed (same-size plan)", "Share of the layer's numbers the "
          "same-size plan deletes."),
         ("no_prune_bits", "Bits (same-size, nothing removed)", "Bits per number for this layer in the "
          "same-size plan that deletes nothing: protected layers keep more bits, the rest drop to fewer."),
         ("outlier", "Unusual layer", "\"yes\" if the raw score is far from the other layers' (robust "
          "z-score above 3.5), so the layer stands out as much more (or less) sensitive than the rest.")],
    )
    rep.glossary["Embeddings and LM head"] = ("The model's word dictionary: the table that turns words into "
                                              "numbers at the start, and numbers back into words at the end.")



def _unused_budget(bp: CompressionPlan, budget_gb: float, protected_bits: int,
                   predict: Callable[[CompressionPlan], PlanCost]) -> str:
    """Why budget packing stopped: the protected set is always the top-k layers by sensitivity."""
    left = budget_gb - predict(bp).weight_memory_gb
    if not bp.compressed_layers:
        return f"{left:.3f} GB of the budget is unused; every layer is already protected."
    nxt = max(bp.compressed_layers, key=lambda i: bp.layers[i].sensitivity)
    grown = replace(bp, layers=tuple(replace(lp, bit_width=protected_bits, pruning_ratio=0.0, protected=True)
                                     if lp.layer == nxt else lp for lp in bp.layers))
    need = predict(grown).weight_memory_gb - predict(bp).weight_memory_gb
    return (f"{left:.3f} GB of the budget is unused. Protecting the next most sensitive layer ({nxt}) would add "
            f"{need:.3f} GB. The protected set is kept to the top-k layers by sensitivity, so packing stops "
            "here rather than skipping to a less sensitive layer; in a model whose decoder layers are all the "
            "same size (TinyLlama), no other layer would fit either.")


_MEASURE_PLAIN = {
    "grad_x_weight": "For every number inside the model we ask two things: how big is this number, and how much "
                     "would the model's mistakes change if this number were nudged (the *gradient*)? Multiplying "
                     "the two gives a rough estimate of how much damage changing that number would do. Adding "
                     "these up over every number in a layer, and over all the passages, gives that layer's raw "
                     "sensitivity score.",
    "layer_removal": "Then one layer at a time is switched off (skipped, so the text passes straight through it) "
                     "and the passages are read again. How much the prediction error (perplexity) rises is that "
                     "layer's raw sensitivity score: the bigger the rise, the more the model depends on it.",
    "layer_quant": "Then one layer at a time is compressed on its own, the way the plan would compress it (fewer "
                   "bits per number), and the passages are read again. How much the prediction error "
                   "(perplexity) rises is that layer's raw sensitivity score.",
    "fisher": "For every number inside the model we multiply its size by how much the model's mistakes would "
              "change if it were nudged (the *gradient*), and square the result. Half of that, added up over a "
              "layer's numbers and averaged over the passages, is the layer's raw sensitivity score (an estimate "
              "of the damage from removing those numbers).",
    "taylor_ema": "For every passage we multiply each number by its gradient and add the results up over the "
                  "whole layer, keeping the size of the total. A running average of that total over the passages "
                  "is the layer's raw sensitivity score.",
    "hessian": "For every number inside the model we estimate how sharply the model's mistakes curve upwards if "
               "that number is moved (the *curvature*), using a few random test nudges per passage. Half the "
               "curvature times the number squared, added up over a layer, is that layer's raw sensitivity "
               "score; negative curvature estimates are treated as noise and set to zero.",
    "movement": "The model is briefly fine-tuned on the passages, and for every number we record how strongly that "
                "training pushes it towards or away from zero. Adding the size of these pushes up over a layer "
                "gives its raw sensitivity score: layers whose numbers training keeps moving are the sensitive ones.",
}


def _top(k: int) -> str:
    return "no layers" if k == 0 else "the most sensitive layer" if k == 1 else f"the {k} most sensitive layers"


def _kv_rows(ctx: RunContext, candidate: dict[str, Any], handle: _ModelHandle,
             text_loader: Callable[[str, str], list[str]], rep: StageReporter):
    """KV cache rows: FP16 cache (added to the baseline row), uniform bits (original), the sensitivity plan
    with token budgets (framework) and the same plan without eviction (to separate the two effects)."""
    s0 = ctx.cfg.stage0
    avg_bits = s0.kv_avg_bits if s0.kv_avg_bits is not None else s0.kv_uniform_bits
    prof = predict = uniform = None
    with rep.method(METHOD_KV, "original", label="Standard KV cache compression",
                    plain_desc=f"every layer's notes stored at {s0.kv_uniform_bits} bits per number, and every "
                               "earlier word kept.",
                    description=f"uniform {s0.kv_uniform_bits}-bit keys and values, no eviction") as row:
        prof, cached = load_kv_profile(ctx, candidate, handle, text_loader)
        predict = lambda p: predict_kv(p, prof, s0.kv_context_len, s0.kv_batch_size,  # noqa: E731
                                       s0.kv_group_size, s0.group_overhead_bits, s0.baseline_bits)
        uniform = uniform_kv_plan(prof.num_layers, s0.kv_uniform_bits)
        row.metrics.update(_kv_metrics(predict(uniform)))
        row.metrics["build_time_s"] = 0.0
        row.info["kv_profile_cached"] = cached
        base = next((r for r in rep.rows if r.variant == "fp16"), None)
        if base is not None:
            base.metrics.update(_kv_metrics(predict(uniform_kv_plan(prof.num_layers, s0.baseline_bits))))
    if prof is None:
        return None

    plans = {}
    for method, target, label, plain, desc in (
        (METHOD_KV, s0.kv_attention_coverage, "Sensitivity-guided KV cache plan",
         "each layer's keys and values get their own bits, spending the same average as the standard method "
         "where rounding hurts most, and layers that focus on few earlier words forget the rest.",
         f"per-layer key/value bits at {avg_bits:g} average bits, per-layer token budget keeping "
         f"{s0.kv_attention_coverage:.0%} of attention"),
        (METHOD_KV_BITS, None, "Sensitivity-guided KV cache plan, bits only",
         "the same per-layer bits, but no layer forgets anything. Comparing it with the full plan separates "
         "the effect of choosing bits from the effect of forgetting.",
         f"per-layer key/value bits at {avg_bits:g} average bits, no eviction"),
    ):
        with rep.method(method, "framework", compare_to=METHOD_KV, label=label, plain_desc=plain,
                        description=desc) as row:
            t0 = time.perf_counter()
            plans[method] = plan_kv(prof, avg_bits, target)
            row.metrics.update(_kv_metrics(predict(plans[method])))
            row.metrics["build_time_s"] = prof.cost["wall_clock_s"] + time.perf_counter() - t0
            row.info["kv_profiling_cost"] = prof.cost
    if len(plans) < 2:
        return None
    return prof, plans[METHOD_KV], plans[METHOD_KV_BITS], uniform, predict


def _add_kv_details(rep: StageReporter, prof: KVProfile, plan: KVPlan, bits_only: KVPlan, uniform: KVPlan,
                    predict: Callable[[KVPlan], KVCost]) -> None:
    s0 = rep.config["stage0"]
    low, top = prof.bits_options[0], prof.keep_ratios[0]
    fw, bo, un = predict(plan), predict(bits_only), predict(uniform)
    fp16 = predict(uniform_kv_plan(prof.num_layers, s0["baseline_bits"]))
    for row, lp, kr, vr, cov, mb in zip(rep.per_layer, plan.layers, prof.key_rise, prof.value_rise, prof.coverage,
                                        fw.per_layer_mb):
        row.update({"kv_key_bits": lp.key_bits, "kv_value_bits": lp.value_bits, "kv_keep_ratio": lp.keep_ratio,
                    f"kv_key_rise_{low}bit": kr[0], f"kv_value_rise_{low}bit": vr[0],
                    f"kv_attention_on_top_{top:.0%}": cov[0], "kv_predicted_mb": mb})

    n = prof.num_layers
    key_mean = sum(r[0] for r in prof.key_rise) / n
    val_mean = sum(r[0] for r in prof.value_rise) / n
    evicting = [lp.layer for lp in plan.layers if lp.keep_ratio < 1.0]
    ctx_desc = f"{s0['kv_context_len']} tokens x {s0['kv_batch_size']} sequence(s)"
    rep.conditions["KV cache memory prediction"] = ctx_desc
    rep.sections.append(("KV cache plan", "\n".join([
        f"Measured on {prof.cost.get('calibration_batches')} batches ({prof.cost.get('calibration_tokens')} tokens) "
        f"in {prof.cost.get('wall_clock_s', 0):.1f}s. Keys rounded per channel, values per token, groups of "
        f"{s0['kv_group_size']}; keys before RoPE.",
        f"Mean perplexity rise with one layer's cache at {low} bits: keys {key_mean:.4g}, values {val_mean:.4g}.",
        f"Key bits per layer: {[lp.key_bits for lp in plan.layers]}.",
        f"Value bits per layer: {[lp.value_bits for lp in plan.layers]}.",
        f"Layers that evict tokens ({len(evicting)}/{n}): "
        + (", ".join(f"{lp.layer} (keep {lp.keep_ratio:.0%})" for lp in plan.layers if lp.keep_ratio < 1.0)
           or "none") + f"; target: kept tokens receive {s0['kv_attention_coverage']:.0%} of attention.",
        f"Predicted cache memory at {ctx_desc}: FP16 {fp16.memory_gb:.4g} GB, uniform {s0['kv_uniform_bits']}-bit "
        f"{un.memory_gb:.4g} GB, plan {fw.memory_gb:.4g} GB, plan without eviction {bo.memory_gb:.4g} GB.",
        "",
        "Attention coverage takes each query's top tokens (an oracle); H2O / SnapKV choose tokens from past "
        "attention and keep a little less. Perplexity rises are summed over layers, assuming they add up. "
        "Stage 3 measures both for real.",
    ])))
    if key_mean <= val_mean:
        rep.anomalies.append(f"Keys were no more fragile than values at {low} bits (mean rise {key_mean:.4g} vs "
                             f"{val_mean:.4g}), the opposite of what KIVI / KVQuant report.")

    rep.plain_intro += (
        "\n\nStage 0 also plans the model's **short-term memory** (the KV cache). While writing, the model keeps "
        "two notes (a *key* and a *value*) about every earlier word in every layer, so it doesn't have to "
        "reread the whole text for each new word. For long texts these notes can take more memory than the "
        "model itself. The same idea is applied to them: each layer's keys, then its values, are rounded to "
        f"fewer bits one at a time to see how much that hurts, and each layer's attention is checked to see "
        "whether it only looks at a few earlier words. The plan then gives fragile notes more bits, robust "
        "ones fewer, and lets layers that focus on few words forget the rest.")
    rep.glossary.update({
        "KV cache": "The model's short-term memory while writing: notes on every earlier word, kept so they don't "
                    "have to be recomputed. It grows with the length of the text.",
        "Keys and values": "The two notes each layer keeps per word. The key says what a word is about (used to "
                           "decide where to look), the value holds what it contributes once looked at.",
        "Attention": "How much the model looks at each earlier word when choosing the next one.",
        "Forgetting (token eviction)": "Dropping the notes on earlier words that a layer hardly looks at, "
                                       "to save memory.",
    })
    rep.plain_why.insert(-1, (
        f"For the short-term memory, rounding one layer's keys to {low} bits raised the prediction error by "
        f"{key_mean:.3g} on average, against {val_mean:.3g} for values, so "
        f"{'keys are the more fragile notes' if key_mean > val_mean else 'values were at least as fragile as keys here'}. "
        f"With the same average bits as the standard method ({un.avg_bits:.3g}), the plan's predicted accuracy "
        f"loss from rounding is {fw.ppl_rise:.3g} against {un.ppl_rise:.3g} for the standard method (lower is "
        f"better). {len(evicting)} of {n} layers can forget part of the text while keeping "
        f"{s0['kv_attention_coverage']:.0%} of their attention, which brings the notes for a "
        f"{s0['kv_context_len']}-token text (a token is about three quarters of a word) from {un.memory_gb:.3g} GB to {fw.memory_gb:.3g} GB "
        f"(uncompressed: {fp16.memory_gb:.3g} GB)."))
    rep.plain_summary = rep.plain_summary.replace(
        " Nothing has been compressed yet",
        f" For the short-term memory (KV cache) of a {s0['kv_context_len']}-token text, the plan predicts "
        f"{fw.memory_gb:.3g} GB against {un.memory_gb:.3g} GB for standard {s0['kv_uniform_bits']}-bit notes and "
        f"{fp16.memory_gb:.3g} GB uncompressed. Nothing has been compressed yet")
    if rep.plain_layer_columns:
        head, intro, cols = rep.plain_layer_columns
        rep.plain_layer_columns = (head, intro + " The last three columns are the short-term memory (KV cache) "
                                   "plan: bits for keys, bits for values, and the share of earlier words kept.",
                                   cols + [("kv_key_bits", "Key bits (cache)", "Bits per number for this layer's "
                                            "stored keys (the notes used to decide which earlier words matter)."),
                                           ("kv_value_bits", "Value bits (cache)", "Bits per number for this layer's "
                                            "stored values (the content taken from earlier words)."),
                                           ("kv_keep_ratio", "Share of words kept (cache)", "Share of earlier "
                                            "words this layer keeps notes on; 1.00 keeps all, 0.10 keeps the "
                                            "10% that get the most attention.")])
    rep.next_steps.append("Run Stage 3 (KVQuant, H2O, SnapKV) with the KV cache plan to measure its real accuracy.")
