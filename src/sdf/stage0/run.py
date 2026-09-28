"""Stage 0 end to end: profile (cached), plan, compare against FP16 and a uniform allocation, report."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from sdf.data import calibration_batches, eval_windows, make_text_loader
from sdf.eval import measure_model
from sdf.reporting import StageReporter
from sdf.run import RunContext
from sdf.search_space import Candidate
from sdf.stage0.planner import (
    CompressionPlan,
    PlanCost,
    baseline_cost,
    plan_compression,
    predict_cost,
    uniform_plan,
)
from sdf.stage0.sensitivity import SensitivityProfile, normalize, profile_sensitivity
from sdf.utils.env import environment_info, resolve_device
from sdf.utils.logging import get_logger

log = get_logger(__name__)

METHOD = "allocation"
MAIN_METRICS = ["ppl_val", "predicted_weight_memory_gb", "avg_bits_per_weight", "sensitivity_exposure", "build_time_s"]


@dataclass
class Stage0Result:
    plan: CompressionPlan
    profile: SensitivityProfile
    outputs: dict[str, Path]


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


def profile_key(ctx: RunContext, cand: Candidate) -> dict[str, Any]:
    cfg = ctx.cfg
    return {"model": cfg.model.name, "profile_dtype": cfg.stage0.profile_dtype, "score": cfg.stage0.score,
            "calib_dataset": cand.calib_dataset, "calib_samples": cand.calib_samples,
            "seq_len": cfg.calibration.seq_len, "batch_size": cfg.calibration.batch_size, "seed": cfg.run.seed}


def fp16_key(ctx: RunContext, device: torch.device) -> dict[str, Any]:
    cfg = ctx.cfg
    hw = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    return {"model": cfg.model.name, "dtype": cfg.model.dtype, "eval": asdict(cfg.eval), "hardware": hw,
            "backend": "hf-transformers", "seed": cfg.run.seed}


def _cost_metrics(cost: PlanCost) -> dict[str, float]:
    return {"predicted_weight_memory_gb": cost.weight_memory_gb, "avg_bits_per_weight": cost.avg_bits_per_weight,
            "sparsity": cost.sparsity, "sensitivity_exposure": cost.sensitivity_exposure}


def run_stage0(
    ctx: RunContext,
    candidate: Candidate,
    model=None,
    tokenizer=None,
    text_loader: Callable[[str, str], list[str]] | None = None,
    measure_fp16: bool = True,
) -> Stage0Result:
    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    device = resolve_device(cfg.model.device)
    handle = _ModelHandle(ctx, device, model, tokenizer)
    text_loader = text_loader or make_text_loader(cfg.data.sources)

    # --- sensitivity profile (depends only on model + calibration, so cached across trials) -------------
    def compute_profile() -> dict[str, Any]:
        batches = calibration_batches(text_loader(candidate.calib_dataset, "train"), handle.tokenizer,
                                      candidate.calib_samples, cfg.calibration.seq_len,
                                      cfg.calibration.batch_size, cfg.run.seed)
        prof = profile_sensitivity(handle.model(s0.profile_dtype), batches, device=device,
                                   meta=profile_key(ctx, candidate))
        return prof.to_dict()

    prof_dict, prof_cached = ctx.cache.get_or_compute("sensitivity_profile", profile_key(ctx, candidate),
                                                      compute_profile)
    # Normalisation is cheap and not part of the cache key: re-derive the scores from the cached raw scores.
    profile = dataclasses.replace(SensitivityProfile.from_dict(prof_dict),
                                  scores=normalize(prof_dict["raw_scores"], s0.normalization))

    rep = StageReporter(
        stage=0, run_dir=ctx.run_dir, title="Sensitivity profiling and compression planning",
        config={"hyperparams": candidate.to_dict(), "stage0": asdict(s0), "calibration": asdict(cfg.calibration),
                "eval": asdict(cfg.eval), "model": asdict(cfg.model), "run": asdict(cfg.run)},
        environment=environment_info(),
        conditions={"model": cfg.model.name, "calibration": f"{candidate.calib_dataset}, "
                    f"{candidate.calib_samples} x {cfg.calibration.seq_len} tokens", "seed": cfg.run.seed,
                    "evaluation data": f"wikitext-2 test, {cfg.eval.seq_len}-token windows, validation/held-out halves",
                    "device": str(device), "backend": "HF Transformers",
                    "group size for memory prediction": candidate.gptq_groupsize},
        requirement=cfg.requirement,
        main_metrics=MAIN_METRICS,
    )
    predict = lambda plan: predict_cost(plan, profile, candidate.gptq_groupsize,  # noqa: E731
                                        s0.group_overhead_bits, s0.baseline_bits)

    # --- FP16 baseline: measured once per (model, eval settings, hardware) and cached ----------------------
    with rep.method("baseline", "fp16", description="uncompressed model") as row:
        row.metrics.update(_cost_metrics(baseline_cost(profile, s0.baseline_bits)))
        row.metrics["build_time_s"] = 0.0
        if measure_fp16:
            def compute_fp16() -> dict[str, Any]:
                texts = text_loader(cfg.eval.dataset, "test")
                val, held = eval_windows(texts, handle.tokenizer, cfg.eval.seq_len, cfg.eval.max_windows)
                metrics, raw = measure_model(handle.model(cfg.model.dtype), val, held, cfg.eval, device,
                                             seed=cfg.run.seed)
                return {"metrics": metrics, "raw": raw}

            fp16, fp16_cached = ctx.cache.get_or_compute("fp16_baseline", fp16_key(ctx, device), compute_fp16)
            row.metrics.update(fp16["metrics"])
            row.info["cached"] = fp16_cached
            rep.add_raw("baseline", "fp16", fp16["raw"])

    # --- original method: uniform allocation, no Stage 0 guidance ------------------------------------------
    uniform = uniform_plan(profile, s0.uniform_bits, s0.uniform_prune_ratio)
    with rep.method(METHOD, "original",
                    description=f"uniform {s0.uniform_bits}-bit, prune {s0.uniform_prune_ratio} on every layer") as row:
        row.metrics.update(_cost_metrics(predict(uniform)))
        row.metrics["protected_layers"] = 0
        row.metrics["build_time_s"] = 0.0  # needs no profiling

    # --- framework: sensitivity-driven plan ------------------------------------------------------------------
    plan = None
    with rep.method(METHOD, "framework",
                    description=f"sensitivity plan (threshold {candidate.sensitive_threshold:.3g}, "
                                f"prune {candidate.prune_ratio_aggressive:.3g} on robust layers)") as row:
        t0 = time.perf_counter()
        plan = plan_compression(profile, candidate.sensitive_threshold, candidate.prune_ratio_aggressive,
                                s0.protected_bits, s0.compressed_bits)
        planning_s = time.perf_counter() - t0
        row.metrics.update(_cost_metrics(predict(plan)))
        row.metrics["protected_layers"] = len(plan.protected_layers)
        # True cost of producing the plan, even when the profile came from cache this time.
        row.metrics["build_time_s"] = profile.cost["wall_clock_s"] + planning_s
        row.info.update(profile_cached=prof_cached, profiling_cost=profile.cost)

    if plan is None:
        rep.finalize()
        raise RuntimeError(f"Stage 0 planning failed; see {rep.report_path}")

    _add_stage0_details(rep, profile, plan, uniform, predict, s0.baseline_bits)
    plan.save(rep.dir / "compression_plan.json")
    profile.save(rep.dir / "sensitivity_profile.json")
    return Stage0Result(plan=plan, profile=profile, outputs=rep.finalize())


def _add_stage0_details(rep: StageReporter, profile: SensitivityProfile, plan: CompressionPlan,
                        uniform: CompressionPlan, predict: Callable[[CompressionPlan], PlanCost],
                        baseline_bits: int) -> None:
    fw_cost, un_cost = predict(plan), predict(uniform)
    for lp, ul, raw, numel, fw_mb, un_mb in zip(plan.layers, uniform.layers, profile.raw_scores,
                                                profile.layer_numel, fw_cost.per_layer_mb, un_cost.per_layer_mb):
        rep.per_layer.append({
            "layer": lp.layer, "sensitivity": lp.sensitivity, "raw_score": raw, "weights": numel,
            "framework_protected": lp.protected, "framework_bits": lp.bit_width,
            "framework_prune_ratio": lp.pruning_ratio, "framework_predicted_mb": fw_mb,
            "uniform_bits": ul.bit_width, "uniform_prune_ratio": ul.pruning_ratio, "uniform_predicted_mb": un_mb,
        })

    n = len(plan)
    ranked = sorted(range(n), key=lambda i: profile.scores[i], reverse=True)
    rep.sections.append(("Sensitivity profile and plan", "\n".join([
        f"Score: {profile.method}, {rep.config['stage0']['normalization']}-normalised to [0, 1] over {n} decoder layers.",
        f"Most sensitive layers: {', '.join(f'{i} ({profile.scores[i]:.2f})' for i in ranked[:3])}. "
        f"Least sensitive: {', '.join(f'{i} ({profile.scores[i]:.2f})' for i in ranked[-3:])}.",
        f"Protected ({len(plan.protected_layers)}/{n}): {plan.protected_layers or 'none'}.",
        f"Profiling cost: {profile.cost.get('calibration_batches')} batches, "
        f"{profile.cost.get('calibration_tokens')} tokens, {profile.cost.get('wall_clock_s', 0):.1f}s, "
        "peak GPU memory " + (f"{profile.cost['peak_memory_gb']:.2f} GB." if profile.cost.get("peak_memory_gb")
                              else "n/a (CPU)."),
        "",
        "Stage 0 only allocates precision, so memory here is predicted from the plan. Accuracy and latency of "
        "the two allocations are measured once Stage 1 applies them.",
    ])))

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

    rep.next_steps += [
        "Run Stage 1 (GPTQ) with both allocations to measure their perplexity and latency.",
        "If sensitivity exposure is high, raise sensitive_threshold or lower prune_ratio_aggressive.",
    ]

