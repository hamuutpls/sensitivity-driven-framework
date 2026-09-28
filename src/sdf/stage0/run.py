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
    budget_matched_plan,
    PlanCost,
    baseline_cost,
    plan_compression,
    predict_cost,
    uniform_plan,
)
from sdf.stage0.sensitivity import SensitivityProfile, normalize, outlier_layers, profile_sensitivity
from sdf.utils.env import environment_info, resolve_device
from sdf.utils.logging import get_logger

log = get_logger(__name__)

METHOD = "allocation"
METHOD_BUDGET = "allocation_same_size"
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

    # --- framework at the same size as the original method --------------------------------------------------
    # The threshold plan usually spends more bits than uniform, so its accuracy edge would be partly bought
    # with memory. This plan protects as many of the most sensitive layers as fit in the uniform plan's size,
    # making the comparison size-for-size fair.
    budget = None
    with rep.method(METHOD_BUDGET, "framework", compare_to=METHOD,
                    label="Sensitivity-guided framework, same size as the standard method",
                    description="sensitivity plan protecting as many top layers as fit in the uniform plan's "
                                "predicted memory") as row:
        t0 = time.perf_counter()
        budget = budget_matched_plan(profile, predict(uniform).weight_memory_gb, candidate.prune_ratio_aggressive,
                                     s0.protected_bits, s0.compressed_bits, predict)
        row.metrics.update(_cost_metrics(predict(budget)))
        row.metrics["protected_layers"] = len(budget.protected_layers)
        row.metrics["build_time_s"] = profile.cost["wall_clock_s"] + time.perf_counter() - t0
        row.info["budget_gb"] = predict(uniform).weight_memory_gb

    if plan is None:
        rep.finalize()
        raise RuntimeError(f"Stage 0 planning failed; see {rep.report_path}")

    _add_stage0_details(rep, profile, plan, uniform, budget, predict, s0.baseline_bits)
    if budget is not None:
        budget.save(rep.dir / "compression_plan_budget_matched.json")
    plan.save(rep.dir / "compression_plan.json")
    profile.save(rep.dir / "sensitivity_profile.json")
    return Stage0Result(plan=plan, profile=profile, outputs=rep.finalize())


def _add_stage0_details(rep: StageReporter, profile: SensitivityProfile, plan: CompressionPlan,
                        uniform: CompressionPlan, budget: CompressionPlan | None,
                        predict: Callable[[CompressionPlan], PlanCost], baseline_bits: int) -> None:
    fw_cost, un_cost = predict(plan), predict(uniform)
    outliers = outlier_layers(profile.raw_scores)
    budget_layers = budget.layers if budget is not None else [None] * len(plan)
    for lp, ul, bl, raw, numel, fw_mb, un_mb in zip(plan.layers, uniform.layers, budget_layers,
                                                    profile.raw_scores, profile.layer_numel,
                                                    fw_cost.per_layer_mb, un_cost.per_layer_mb):
        rep.per_layer.append({
            "layer": lp.layer, "sensitivity": lp.sensitivity, "raw_score": raw, "outlier": lp.layer in outliers,
            "weights": numel,
            "framework_protected": lp.protected, "framework_bits": lp.bit_width,
            "framework_prune_ratio": lp.pruning_ratio, "framework_predicted_mb": fw_mb,
            "same_size_protected": None if bl is None else bl.protected,
            "same_size_bits": None if bl is None else bl.bit_width,
            "same_size_prune_ratio": None if bl is None else bl.pruning_ratio,
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
                f"compressed and pruned at {plan[i].pruning_ratio:.0%}")
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

    _add_plain_explanation(rep, profile, plan, fw_cost, un_cost,
                           predict(budget) if budget is not None else None, budget, outliers)

    rep.next_steps += [
        "Run Stage 1 (GPTQ) with both allocations to measure their perplexity and latency.",
        "If sensitivity exposure is high, raise sensitive_threshold or lower prune_ratio_aggressive.",
    ]



def _add_plain_explanation(rep: StageReporter, profile: SensitivityProfile, plan: CompressionPlan,
                           fw_cost: PlanCost, un_cost: PlanCost, budget_cost: PlanCost | None,
                           budget: CompressionPlan | None, outliers: list[int]) -> None:
    """The plain-language part of the Stage 0 report, for readers with no AI background."""
    s0 = rep.config["stage0"]
    hp = rep.config["hyperparams"]
    n, n_prot = len(plan), len(plan.protected_layers)
    rep.plain_intro = (
        "Compressing a language model is like shrinking a photo: done carefully, you save a lot of space and "
        "barely notice the difference; done carelessly, the picture turns to mush. The catch is that not every "
        "part of the model is equally delicate. Some layers can be squeezed hard with no visible effect, while "
        "others fall apart at the slightest change.\n\n"
        "Stage 0 finds out which is which. It feeds the model some ordinary text "
        f"({hp['calib_samples']} short passages) and measures, for each of its {n} layers, how much the "
        "model's predictions would suffer if that layer were changed. This is the layer's *sensitivity*. "
        "It then writes a compression plan: the most sensitive layers are **protected** (kept at "
        f"{s0['protected_bits']} bits per number and never trimmed), and the rest are **compressed** "
        f"({s0['compressed_bits']} bits per number, with {hp['prune_ratio_aggressive']:.0%} of their numbers "
        "removed).\n\n"
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
        "Perplexity": "A standard score for how well a model predicts real text. Lower is better.",
    })

    exp_fw, exp_un = fw_cost.sensitivity_exposure, un_cost.sensitivity_exposure
    mem_fw, mem_un = fw_cost.weight_memory_gb, un_cost.weight_memory_gb
    why = [
        f"The framework protected {n_prot} of the {n} layers, the most sensitive ones, and compressed the "
        f"other {n - n_prot}. The standard method compresses all {n} layers equally to "
        f"{s0['uniform_bits']} bits.",
    ]
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
                   f"{mem_un:.3g} GB) and spends it on protecting the {len(budget.protected_layers)} most "
                   f"sensitive layers, paid for by trimming the others. Its fragile-parts score is "
                   f"{budget_cost.sensitivity_exposure:.2f} against {exp_un:.2f} for the standard method. This "
                   "is the fair head-to-head: same size, different choice of where to spend the bits.")
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
    rep.glossary["Embeddings and LM head"] = ("The model's word dictionary: the table that turns words into "
                                              "numbers at the start, and numbers back into words at the end.")
    rep.glossary["Same-size plan"] = ("A version of the framework's plan limited to exactly the memory the "
                                      "standard method uses, so the two can be compared fairly.")
