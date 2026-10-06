"""The one runner for Stages 1-3: load the Stage 0 plans, then for each method build and measure

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
from sdf.eval import downstream
from sdf.eval.metrics import measure_model
from sdf.run import RunContext
from sdf.stage0.activation import ActivationPlan, uniform_activation_plan
from sdf.stage0.kv_cache import KVPlan, KVProfile, predict_kv, uniform_kv_plan
from sdf.stage0.planner import CompressionPlan, baseline_cost, uniform_plan
from sdf.stage0.run import (_cost_metrics, _kv_metrics, calib_batches, fp16_key, load_fp16,
                            setup, stage_reporter, weight_cost)
from sdf.stage0.sensitivity import SensitivityProfile, normalize
from sdf.stages.methods import Method, MethodCall, methods_for
from sdf.utils.logging import get_logger

log = get_logger(__name__)

# Stage 0 output files the later stages read (see stage_0/handoff.md), by plan key.
PLAN_FILES: dict[str, tuple[str, type]] = {
    "weights": ("compression_plan.json", CompressionPlan),
    "weights_same_size": ("compression_plan_budget_matched.json", CompressionPlan),
    "activations": ("activation_plan.json", ActivationPlan),
    "kv": ("kv_cache_plan.json", KVPlan),
    "kv_bits_only": ("kv_cache_plan_bits_only.json", KVPlan),
}

# Framework rows after the first plan get their own method name and are compared with the method's original row.
# Values: (method-name suffix, label, plain description) as in the Stage 0 report.
_PLAN_ROWS = {
    "weights_same_size": (
        "_same_size", "Sensitivity-guided framework, budget plan (fits in the standard method's memory)",
        "the framework limited to the memory the standard method uses: it protects as many of the most sensitive "
        "layers as fit in that budget, so the two can be compared fairly, size for size."),
}

TITLES = {1: "Weight compression", 2: "Activation compression", 3: "KV-cache compression", 4: "Evaluation on {}"}

MAIN_METRICS = {
    1: ["ppl_val", "ppl_heldout", "predicted_weight_memory_gb", "avg_bits_per_weight", "sparsity", "peak_memory_gb",
        "prefill_ms_mean", "decode_ms_per_token_mean", "build_time_s"],
    2: ["ppl_val", "ppl_heldout", "avg_activation_bits", "peak_memory_gb", "prefill_ms_mean",
        "decode_ms_per_token_mean", "build_time_s"],
    3: ["ppl_val", "ppl_heldout", "predicted_kv_memory_gb", "avg_kv_bits", "kv_kept_share", "peak_memory_gb",
        "prefill_ms_mean", "decode_ms_per_token_mean", "build_time_s"],
    4: ["ppl_val", "ppl_heldout", "model_size_gb", "predicted_weight_memory_gb", "peak_memory_gb",
        "prefill_tokens_per_s", "decode_tokens_per_s", "build_time_s"],
}

INTROS = {
    1: "A model is a huge store of numbers (its \"weights\"). This stage makes that store smaller: it keeps each "
       "number with fewer bits (like rounding prices to the nearest dollar) and removes numbers that barely "
       "matter. Each technique is tried the standard way, treating every layer the same, and the framework way, "
       "following the Stage 0 plan, then the compressed model is tested for accuracy, memory and speed.",
    2: "While the model runs, every layer passes numbers to the next one (its \"activations\"). This stage stores "
       "those numbers with fewer bits so the model needs less memory and can use faster arithmetic. The standard "
       "way rounds every layer the same; the framework gives more bits to the layers Stage 0 found fragile.",
    3: "While writing a reply, the model keeps notes on every earlier word (the \"KV cache\"); for long texts these "
       "notes can take more memory than the model itself. This stage stores the notes with fewer bits or forgets "
       "the least-used ones. The standard way treats every layer the same; the framework follows the Stage 0 "
       "plan for each layer.",
    4: "The earlier stages measure accuracy inside the research software, where compressed numbers are only "
       "simulated. This stage saves each Stage 1 model in the format of {}, a program people use to run models "
       "on their own computers, and measures there what a user would get: the real file size, the memory used "
       "on the graphics card, accuracy, and how fast it reads a prompt and writes a reply.",
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


def original_plan(stage: int, plans: Stage0Plans, s0) -> Any:
    """The uniform plan of the standard method, from the Stage 0 "original method" settings."""
    if stage == 1:
        return uniform_plan(normalize(plans.profile.raw_scores, s0.normalization), s0.uniform_bits,
                            s0.uniform_prune_ratio)
    if stage == 2:
        return uniform_activation_plan(plans.num_layers, s0.act_uniform_bits)
    return uniform_kv_plan(plans.num_layers, s0.kv_uniform_bits)


def plan_metrics(plan: Any, plans: Stage0Plans, cfg, candidate: dict[str, Any]) -> dict[str, Any]:
    """What the plan is predicted to cost (the same predictions Stage 0 reports), next to the measured metrics."""
    s0 = cfg.stage0
    if isinstance(plan, CompressionPlan):
        return _cost_metrics(weight_cost(plans.profile, candidate["gptq_groupsize"], s0)(plan))
    if isinstance(plan, ActivationPlan):
        return {"avg_activation_bits": plan.avg_bits}
    if isinstance(plan, KVPlan) and plans.kv_profile is not None:
        m = _kv_metrics(predict_kv(plan, plans.kv_profile, s0.kv_context_len, s0.kv_batch_size, s0.kv_group_size,
                                   s0.group_overhead_bits, s0.baseline_bits))
        return {k: m[k] for k in ("predicted_kv_memory_gb", "avg_kv_bits", "kv_kept_share")}
    return {}


def _fp16_plan_metrics(stage: int, plans: Stage0Plans, cfg, candidate) -> dict[str, Any]:
    base = cfg.stage0.baseline_bits
    if stage == 1:
        return _cost_metrics(baseline_cost(plans.profile, base))
    if stage == 2:
        return plan_metrics(uniform_activation_plan(plans.num_layers, base), plans, cfg, candidate)
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
    backend=None,
) -> dict[str, Path]:
    """Run one of Stages 1-3 with `method_names` (see sdf.stages.methods.METHODS). `model_factory` returns a
    fresh FP16 model on the right device (default: from_pretrained); tests pass a tiny model's deepcopy.

    With a Stage 4 `backend` (e.g. llama_cpp.LlamaCpp) the same rows are built and measured on that backend
    instead of HF Transformers, and reported as Stage 4 (stage_4/)."""
    if stage not in (1, 2, 3) or (backend is not None and stage != 1):
        raise ValueError(f"run_stage runs Stages 1-3 (Stage 4 backends: Stage 1 rows), not {stage}")
    report_stage = 4 if backend else stage
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

    calib = (f"{candidate['calib_dataset']}, {candidate['calib_samples']} x {cfg.calibration.seq_len} tokens")
    label = backend.label if backend else "HF Transformers"
    conditions = {"model": cfg.model.name, "calibration": calib, "seed": cfg.run.seed,
                  "evaluation data": f"wikitext-2 test, {cfg.eval.seq_len}-token windows, validation/held-out halves",
                  "device": str(device), "backend": label, "Stage 0 plans": str(plans.dir),
                  "starting point": "the uncompressed model, for every row (stages are independent)"}
    if backend:
        conditions |= {"backend version": backend.version, "methods": "Stage 1 rows, rebuilt the same way"}
    rep = stage_reporter(
        ctx, candidate, handle, plans.profile, ("stages", "stage0", "calibration", "eval", "model", "run"),
        stage=report_stage, title=TITLES[report_stage].format(label), conditions=conditions,
        main_metrics=MAIN_METRICS[report_stage],
    )
    rep.plain_intro = INTROS[report_stage].format(label)
    if backend:
        rep.plain_why += backend.notes

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
        if not ev.downstream_tasks or backend:  # lm-eval runs the HF model only
            return {}
        return downstream.downstream_accuracy(model, handle.tokenizer, ev.downstream_tasks, ev.downstream_limit,
                                              ev.downstream_batch_size, cfg.run.seed)

    def build_and_measure(method: Method, plan: Any) -> dict[str, Any]:
        model = model_factory()
        try:
            t0 = time.perf_counter()
            with method.apply(MethodCall(model, plan, candidate, cfg, batches)) as extra:
                build_s = time.perf_counter() - t0
                if backend:
                    metrics, raw = backend.measure(model, plan, windows(), handle.tokenizer)
                else:
                    metrics, raw = measure_model(model, *windows(), cfg.eval, device, seed=cfg.run.seed)
                metrics.update(tasks(model))
        finally:
            del model
            _free_memory()
        metrics.update(extra)
        metrics["build_time_s"] = build_s
        if method.simulated and stage == 1 and not backend:
            metrics.pop("model_size_gb", None)  # still FP16 in memory; predicted_weight_memory_gb is the size
        return {"metrics": metrics, "raw": raw}

    # --- FP16: the same cache entry Stage 0 measured ------------------------------------------------------------
    with rep.method("baseline", "fp16", description="uncompressed model") as row:
        row.metrics.update(_fp16_plan_metrics(stage, plans, cfg, candidate))
        row.metrics["build_time_s"] = 0.0
        if backend:
            key = {**fp16_key(ctx, device), **backend.key}
            result, row.info["cached"] = ctx.cache.get_or_compute(
                "stage4_fp16", key, lambda: dict(zip(("metrics", "raw"), backend.measure(
                    model_factory(), None, windows(), handle.tokenizer))))
            row.metrics.update(result["metrics"])
            rep.add_raw("baseline", "fp16", result["raw"])
        elif measure_fp16:
            fp16, cached = load_fp16(ctx, handle, text_loader)
            row.metrics.update(fp16["metrics"])
            row.info["cached"] = cached
            rep.add_raw("baseline", "fp16", fp16["raw"])
            if cfg.eval.downstream_tasks:
                key = {**fp16_key(ctx, device), "tasks": list(cfg.eval.downstream_tasks),
                       "limit": cfg.eval.downstream_limit}
                acc, _ = ctx.cache.get_or_compute("fp16_downstream", key, lambda: tasks(model_factory()))
                row.metrics.update(acc)

    simulated_any = False
    for m in methods:
        simulated_any |= m.simulated
        orig = original_plan(stage, plans, s0)
        with rep.method(m.name, "original", description=f"{m.label}, standard settings: {_describe(orig)}",
                        simulated=m.simulated) as row:
            _require(m)
            key = {"stage": stage, "method": m.name, "version": m.version, "plan": orig.to_dict(),
                   "params": {p: candidate[p] for p in m.params}, **fp16_key(ctx, device),
                   "downstream": [list(cfg.eval.downstream_tasks), cfg.eval.downstream_limit]}
            if backend:
                key["backend"] = backend.key
            if m.calibrated:
                key["calibration"] = {"dataset": candidate["calib_dataset"], "samples": candidate["calib_samples"],
                                      "seq_len": cfg.calibration.seq_len, "batch_size": cfg.calibration.batch_size}
            result, cached = ctx.cache.get_or_compute(f"stage{report_stage}_original", key,
                                                      lambda: build_and_measure(m, orig))
            row.metrics.update(plan_metrics(orig, plans, cfg, candidate))
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
                if plan_key not in plans.plans:
                    raise FileNotFoundError(f"{PLAN_FILES[plan_key][0]} not in {plans.dir}; re-run Stage 0 with "
                                            "this plan enabled")
                plan = plans.plans[plan_key]
                row.info["description"] = f"{m.label}, Stage 0 {_describe(plan)}"
                result = build_and_measure(m, plan)
                row.metrics.update(plan_metrics(plan, plans, cfg, candidate))
                row.metrics.update(result["metrics"])
                rep.add_raw(m.name + suffix, "framework", result["raw"])

    if simulated_any and not backend:
        rep.plain_why.append(
            "Some techniques here are simulated: the numbers are rounded as the compressed model would store them, "
            "but kept in the uncompressed format. Their accuracy is real; their memory is the plan's prediction, "
            "and their speed is not the speed a real compressed model would have.")
    return rep.finalize()


def _require(m: Method) -> None:
    if m.apply is None:
        raise NotImplementedError(f"{m.label} is not implemented yet ({m.library})")


def make_backend(name: str, ctx: RunContext):
    st = ctx.cfg.stages
    if name == "llama_cpp":
        from sdf.stages.llama_cpp import LlamaCpp

        if not (st.llama_cpp_dir and st.llama_cpp_convert):
            raise ValueError("Stage 4 llama_cpp needs stages.llama_cpp_dir and stages.llama_cpp_convert")
        return LlamaCpp(st.llama_cpp_dir, st.llama_cpp_convert, ctx.cfg.eval, ctx.run_dir / "stage_4" / "work",
                        st.llama_cpp_gpu_layers)
    raise ValueError(f"unknown Stage 4 backend {name!r}; known: ['llama_cpp']")


def run_stages(ctx: RunContext, candidate: dict[str, Any], stage0_dir: str | Path, backends: dict[str, Any] | None = None,
               **kwargs: Any) -> dict[str, Path]:
    """Stages 1-3 with the methods in cfg.stages, each on top of the same Stage 0 plans, then Stage 4: the
    Stage 1 rows on every backend in cfg.stages.stage4_backends (`backends` maps names to ready backends, for tests)."""
    st = ctx.cfg.stages
    outputs = {}
    for stage, names in ((1, st.stage1_methods), (2, st.stage2_methods), (3, st.stage3_methods)):
        if names:
            for name, path in run_stage(ctx, stage, names, candidate, stage0_dir, **kwargs).items():
                outputs[f"stage_{stage}_{name}"] = path
    for b in st.stage4_backends if st.stage1_methods else []:
        try:
            backend = (backends or {}).get(b) or make_backend(b, ctx)
        except (ValueError, OSError, RuntimeError) as e:  # stages 1-3 are done; keep them and the master report
            log.error("Stage 4 %s not run: %s", b, e)
            continue
        for name, path in run_stage(ctx, 1, st.stage1_methods, candidate, stage0_dir, backend=backend,
                                    **kwargs).items():
            outputs[f"stage_4_{b}_{name}"] = path
    return outputs
