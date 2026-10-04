"""Pruning levels: actually prune the model at several ratios and measure what it does to accuracy.

Stage 0 normally only *predicts* what pruning costs. This study applies each plan to the real weights and
measures perplexity, for every ratio in `stage0.prune_sweep_ratios` (10% to 100% by default):

- original (standard method): every layer at `uniform_bits`, every layer pruned at the ratio.
- framework: the Stage 0 plan at the run's threshold, pruned at the ratio. Protected layers keep
  `protected_bits` and are not pruned; guarded layers are not pruned. It removes less and is bigger than
  the original at the same ratio, so this pair is not a fair comparison.
- same_size (the fair test, `prune_sweep_same_size`): `uniform_bits` everywhere and the same number of weights
  removed as the original, so the same predicted size and share removed, but placed by sensitivity
  (`same_size_pruning_plan`): robust layers lose more, sensitive ones less, guarded ones nothing.

Pruning is unstructured magnitude pruning per output row (each row loses its smallest |w|), the usual
baseline; rounding is round-to-nearest at the plan's bits and group size (`prune_sweep_quantize`). Both are
simulated on FP16 weights, so speed and real file size are not measured; memory is the plan's prediction.
Stage 1 replaces both with the real methods (GPTQ, AWQ, structured pruning).

Outputs under <run_dir>/stage_0_prune_sweep/: report.md, stage_0_comparison.xlsx, results.json.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Callable, Iterator

import torch
from torch import nn

from sdf.data import eval_windows, load_texts
from sdf.eval.metrics import perplexity
from sdf.reporting.reporter import StageReporter
from sdf.run import RunContext
from sdf.stage0.planner import (CompressionPlan, baseline_cost, plan_compression, predict_cost,
                                same_size_pruning_plan, uniform_plan)
from sdf.stage0.run import (_cost_metrics, _ModelHandle, fp16_key, load_fp16, load_guard, load_profile,
                            original_model_info)
from sdf.stage0.sensitivity import fake_quantize_, find_decoder_layers, normalize
from sdf.utils.env import environment_info, resolve_device
from sdf.utils.logging import get_logger

log = get_logger(__name__)

METRICS = ["ppl_val", "ppl_heldout", "predicted_weight_memory_gb", "avg_bits_per_weight", "sparsity",
           "protected_layers"]
BROKEN = 2.0  # a version counts as broken once its perplexity is this many times the uncompressed model's


def magnitude_mask(weight: torch.Tensor, ratio: float) -> torch.Tensor:
    """True for the weights kept: each output row drops its round(ratio * inputs) smallest |w|."""
    rows, cols = weight.shape
    k = round(ratio * cols)
    mask = torch.ones_like(weight, dtype=torch.bool)
    if k >= cols:
        return ~mask
    if k > 0:
        mask.scatter_(1, weight.abs().float().argsort(dim=1)[:, :k], False)
    return mask


@contextmanager
def apply_plan(model: nn.Module, plan: CompressionPlan, group_size: int, quantize: bool,
               baseline_bits: int = 16) -> Iterator[None]:
    """Prune (and round, if `quantize`) every decoder Linear weight as `plan` says; restore on exit.
    Originals are kept on the CPU, so this needs no extra GPU memory."""
    saved = []
    try:
        for lp, layer in zip(plan.layers, find_decoder_layers(model)):
            rounds = quantize and lp.bit_width < baseline_bits
            if lp.pruning_ratio == 0 and not rounds:
                continue
            for m in (m for m in layer.modules() if isinstance(m, nn.Linear)):
                saved.append((m, m.weight.data.to("cpu", copy=True)))
                mask = magnitude_mask(m.weight.data, lp.pruning_ratio) if lp.pruning_ratio else None
                if mask is not None:
                    m.weight.data.mul_(mask)
                if rounds:
                    fake_quantize_(m.weight, lp.bit_width, group_size)
                    if mask is not None:  # rounding moves zeros off zero (min/max scale); keep them pruned
                        m.weight.data.mul_(mask)
        yield
    finally:
        for m, w in saved:
            m.weight.data.copy_(w)


def _validated(ratios: list[float]) -> list[float]:
    out = sorted({float(r) for r in ratios})
    if not out or not all(0.0 <= r <= 1.0 for r in out):
        raise ValueError(f"stage0.prune_sweep_ratios must be shares in [0, 1], got {ratios}")
    return out


def run_prune_sweep(ctx: RunContext, candidate: dict[str, Any], model=None, tokenizer=None,
                    text_loader: Callable[[str, str], list[str]] | None = None) -> dict[str, Any]:
    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    device = resolve_device(cfg.model.device)
    handle = _ModelHandle(ctx, device, model, tokenizer)
    text_loader = text_loader or functools.partial(load_texts, cfg.data.sources)
    ratios = _validated(s0.prune_sweep_ratios)
    gs, threshold = candidate["gptq_groupsize"], candidate["sensitive_threshold"]

    profile, _ = load_profile(ctx, candidate, handle, text_loader)
    scores = normalize(profile.raw_scores, s0.normalization)
    guarded, _ = load_guard(ctx, candidate, handle, text_loader, profile)
    predict = functools.partial(predict_cost, profile=profile, group_size=gs,
                                group_overhead_bits=s0.group_overhead_bits, baseline_bits=s0.baseline_bits)
    rounding = (f"weights rounded to the plan's bits (group size {gs})" if s0.prune_sweep_quantize
                else "no rounding (pruning only)")
    rep = StageReporter(
        stage=0, run_dir=ctx.run_dir, subdir="stage_0_prune_sweep",
        title="Pruning levels: how much of the model can be removed",
        config={"hyperparams": dict(candidate), "stage0": asdict(s0), "eval": asdict(cfg.eval),
                "model": asdict(cfg.model), "run": asdict(cfg.run)},
        environment=environment_info(),
        conditions={"model": cfg.model.name, "seed": cfg.run.seed, "device": str(device),
                    "backend": "HF Transformers", "pruning": "magnitude, per output row (simulated)",
                    "rounding": rounding, "levels": ", ".join(f"{r:.0%}" for r in ratios),
                    "framework threshold": threshold, "never pruned (guard)": sorted(guarded),
                    "evaluation data": f"wikitext-2 test, {cfg.eval.seq_len}-token windows, validation/held-out halves"},
        requirement=cfg.requirement, main_metrics=METRICS,
        original_model=original_model_info(ctx, handle, profile),
    )

    with rep.method("baseline", "fp16", description="uncompressed model") as row:
        fp16, cached = load_fp16(ctx, handle, text_loader)
        row.metrics.update({k: fp16["metrics"][k] for k in ("ppl_val", "ppl_heldout")})
        row.metrics.update(_cost_metrics(baseline_cost(profile, s0.baseline_bits)), protected_layers=0)
        row.info["cached"] = cached

    windows: list[torch.Tensor] = []  # (validation, held-out), tokenised on the first cache miss

    def measure(plan: CompressionPlan) -> dict[str, Any]:
        if not windows:
            windows.extend(eval_windows(text_loader(cfg.eval.dataset, "test"), handle.tokenizer,
                                        cfg.eval.seq_len, cfg.eval.max_windows))
        m = handle.model(cfg.model.dtype)
        with apply_plan(m, plan, gs, s0.prune_sweep_quantize, s0.baseline_bits):
            return {"ppl_val": perplexity(m, windows[0], device)[0],
                    "ppl_heldout": perplexity(m, windows[1], device)[0]}

    for r in ratios:
        pct = round(r * 100)
        plans = {"original": (uniform_plan(scores, s0.uniform_bits, r),
                              f"Standard method, {pct}% removed",
                              f"every layer at {s0.uniform_bits} bits, {pct}% pruned"),
                 "framework": (plan_compression(scores, threshold, r, s0.protected_bits, s0.compressed_bits, guarded),
                               f"Sensitivity-guided framework, {pct}% removed",
                               f"Stage 0 plan (threshold {threshold:g}), {pct}% pruned on unprotected layers")}
        if s0.prune_sweep_same_size:
            plans["same_size"] = (same_size_pruning_plan(scores, s0.uniform_bits, r, profile.layer_numel, guarded),
                                  f"Framework, same size, {pct}% removed",
                                  f"every layer at {s0.uniform_bits} bits, {pct}% of all weights pruned, "
                                  "placed by sensitivity")
        for key_, (plan, name, desc) in plans.items():
            method, variant = (f"same_{pct:03d}", "framework") if key_ == "same_size" else (f"prune_{pct:03d}", key_)
            with rep.method(method, variant, label=name, key_term=False, description=desc) as row:
                row.metrics.update(_cost_metrics(predict(plan)), protected_layers=len(plan.protected_layers))
                key = {**fp16_key(ctx, device), "group_size": gs, "quantize": s0.prune_sweep_quantize,
                       "pruning": "magnitude_per_row", "layers": [[lp.bit_width, lp.pruning_ratio] for lp in plan.layers]}
                result, row.info["cached"] = ctx.cache.get_or_compute("prune_eval", key, lambda: measure(plan))
                row.metrics.update(result)
                log.info("prune %d%% %s: ppl_val %.3f", pct, key_, result["ppl_val"])

    _write_plain(rep, ratios, s0, threshold, guarded, rounding)
    return rep.finalize()


# --- report text -------------------------------------------------------------------------------------------------

def _fair(curves) -> list[tuple[int, dict, dict, bool]]:
    """(level, standard, same size, sizes equal) for each level both measured."""
    return [(p, a, f, abs(f["predicted_weight_memory_gb"] - a["predicted_weight_memory_gb"]) < 1e-6)
            for (p, a), (_, f) in zip(curves["original"], curves["same_size"]) if a and f]


def _curve(rep: StageReporter, variant: str) -> list[tuple[int, dict[str, Any] | None]]:
    """(level, metrics or None if failed) per level; variant "same_size" is the framework's same_NNN rows."""
    prefix, variant = ("same_", "framework") if variant == "same_size" else ("prune_", variant)
    rows = {r.method: r for r in rep.rows if r.variant == variant and r.method.startswith(prefix)}
    return [(int(m[len(prefix):]), rows[m].metrics if rows[m].status == "ok" else None) for m in sorted(rows)]


def _write_plain(rep: StageReporter, ratios: list[float], s0, threshold: float, guarded, rounding: str) -> None:
    fp16 = next((r.metrics for r in rep.rows if r.variant == "fp16" and r.status == "ok"), {})
    base = fp16.get("ppl_val")
    curves = {v: c for v in ("original", "framework", "same_size") if (c := _curve(rep, v))}
    names = {"original": "the standard method", "framework": "the framework",
             "same_size": "the framework at the same size"}

    def breaks(v: str) -> int | None:
        return next((p for p, m in curves[v] if m is None or (base and m["ppl_val"] > BROKEN * base)), None)

    summary = [f"We deleted between {ratios[0]:.0%} and {ratios[-1]:.0%} of the numbers in each pruned layer, in "
               f"{len(ratios)} levels, and measured how well each version still predicts text."]
    if base:
        summary.append(f"The uncompressed model scores {base:.2f} (lower is better).")
        for v in curves:
            b = breaks(v)
            summary.append(f"With {names[v]}, the error first more than doubles at {b}% removed." if b is not None
                           else f"With {names[v]}, the error never doubles, even at {ratios[-1]:.0%} removed.")
    both = [(p, a, f) for (p, a), (_, f) in zip(curves["original"], curves["framework"]) if a and f]
    wins = [p for p, a, f in both if f["ppl_val"] < a["ppl_val"]]
    if both:
        summary.append(f"The framework has the lower error at {len(wins)} of {len(both)} levels, but it removes "
                       "less and is bigger at each level.")
    fair = _fair(curves) if "same_size" in curves else []
    fair_wins = [p for p, a, f, _ in fair if f["ppl_val"] < a["ppl_val"]]
    if fair:
        summary.append(f"In the fair test (same size, same share removed) the framework has the lower error at "
                       f"{len(fair_wins)} of {len(fair)} levels.")
    rep.plain_summary = " ".join(summary)

    rep.plain_intro = (
        "Pruning means deleting the numbers in the model that matter least, so the model takes less memory. "
        "Earlier, Stage 0 only predicted what pruning would cost. Here we actually delete the numbers, at "
        f"every level from {ratios[0]:.0%} to {ratios[-1]:.0%}, and test the result on real text. In each "
        "row of each layer we delete the numbers closest to zero, since they change the output least. "
        f"Both versions also store the remaining numbers with fewer bits ({rounding}). "
        f"The standard method treats every layer the same: {s0.uniform_bits} bits and the same share deleted "
        f"everywhere. The framework protects the most sensitive layers (threshold {threshold:g}): they keep "
        f"{s0.protected_bits} bits and nothing is deleted from them, and the layers whose removal breaks the "
        f"model ({', '.join(map(str, sorted(guarded))) or 'none'}) are never pruned.")
    rep.glossary.update({
        "Standard method, N% removed": f"Every layer stored at {s0.uniform_bits} bits with N% of its numbers "
                                       "deleted.",
        "Sensitivity-guided framework, N% removed": f"The Stage 0 plan: the fragile layers kept at "
                                                    f"{s0.protected_bits} bits and untouched, the others at "
                                                    f"{s0.compressed_bits} bits with N% of their numbers deleted.",
        "Pruning": "Deleting numbers from the model. 100% removed means a layer keeps none of its numbers and "
                   "simply passes its input on unchanged.",
    })
    if both:
        rep.plain_why.append(
            "The framework deletes numbers only from its unprotected layers, so at the same level it removes "
            "less of the model overall and needs more memory than the standard method. Compare the share "
            "removed and memory columns, not just the level.")
    if base and wins:
        rep.plain_why.append(f"The framework is more accurate at {', '.join(f'{p}%' for p in wins)} removed.")
    if fair:
        rep.glossary["Framework, same size, N% removed"] = (
            f"The fair comparison: every layer at {s0.uniform_bits} bits like the standard method, and N% of all "
            "the numbers deleted, so the same size and the same share removed. Only where the deleting happens "
            "differs: the layers that matter least lose the most, the fragile ones less, the never-pruned ones "
            "nothing.")
        unequal = [p for p, _, _, same in fair if not same]
        rep.plain_why.append(
            "The same-size version is the fair test: it has exactly the standard method's size and share removed, "
            "so any difference in error comes only from where the numbers are deleted."
            + (f" At {', '.join(f'{p}%' for p in unequal)} the never-pruned layers make it impossible to delete "
               "as much, so there it stays bigger than the standard method." if unequal else ""))

    header = ["Removed", "Standard: error", "Framework: error", "Standard: error (held-out)",
              "Framework: error (held-out)", "Standard: memory (GB)", "Framework: memory (GB)",
              "Standard: share removed", "Framework: share removed"]
    fmt = lambda m, k: "failed" if m is None else f"{m[k]:.4g}"  # noqa: E731
    table = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for (p, a), (_, f) in zip(curves["original"], curves["framework"]):
        table.append(f"| {p}% | " + " | ".join(fmt(m, k) for k in ("ppl_val",) for m in (a, f)) + " | "
                     + " | ".join(fmt(m, "ppl_heldout") for m in (a, f)) + " | "
                     + " | ".join(fmt(m, "predicted_weight_memory_gb") for m in (a, f)) + " | "
                     + " | ".join(fmt(m, "sparsity") for m in (a, f)) + " |")
    notes = [
        "- **Removed**: the share of numbers deleted from each pruned layer.",
        "- **Error**: perplexity on the validation half of the test text; lower is better"
        + (f" (uncompressed model: {base:.4g})." if base else "."),
        "- **Error (held-out)**: the same on the held-out half, which nothing is tuned on.",
        "- **Memory (GB)**: predicted memory to store the model, assuming deleted numbers take no space.",
        "- **Share removed**: the share of all the model's numbers deleted (0.1 means 10%); the framework's is "
        "lower because it leaves protected and guarded layers whole.",
    ]
    rep.sections.append(("Pruning curve", "\n".join(table + ["", "**What each column means**", ""] + notes)))
    if fair:
        header = ["Removed", "Standard: error", "Same size: error", "Standard: error (held-out)",
                  "Same size: error (held-out)", "Memory (GB), standard / same size", "Lower error"]
        t = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for p, a, f, _ in fair:
            t.append(f"| {p}% | {a['ppl_val']:.4g} | {f['ppl_val']:.4g} | {a['ppl_heldout']:.4g} | "
                     f"{f['ppl_heldout']:.4g} | {a['predicted_weight_memory_gb']:.4g} / "
                     f"{f['predicted_weight_memory_gb']:.4g} | "
                     + ("same size" if f["ppl_val"] < a["ppl_val"] else "standard") + " |")
        fair_notes = [
            "- **Removed**: the share of all the decoder's numbers deleted; the same for both versions unless the "
            "never-pruned layers stop the same-size version from deleting as much (then its memory is higher).",
            "- **Error**: perplexity on the validation half; lower is better.",
            "- **Error (held-out)**: the same on the held-out half, which nothing is tuned on; lower is better.",
            "- **Memory (GB)**: predicted memory of each version; equal unless the never-pruned layers stop the "
            "same-size version from deleting as much.",
            "- **Lower error**: which version predicts the text better at that level.",
        ]
        rep.sections.append(("Fair test: same size, same share removed",
                             "\n".join(t + ["", "**What each column means**", ""] + fair_notes)))
    rep.next_steps += [
        "Pick the pruning range for the search (prune_ratio_aggressive, now 0 to 0.6) from where the error "
        "starts to climb.",
        "Stage 1 repeats this with real methods (GPTQ/AWQ rounding, structured pruning) and real file sizes.",
    ]
