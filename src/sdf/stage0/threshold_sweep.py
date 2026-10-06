"""Protection thresholds: build the Stage 0 plan at several thresholds (and guard sizes) and measure each one.

The threshold decides how many layers are protected (kept at `protected_bits`, nothing removed); the others get
`compressed_bits` with `prune_ratio_aggressive` removed. A low threshold protects many layers (accurate, big),
a high one few (small, riskier). `MODE "sweep"` only predicts this trade-off; here every plan is applied to the
real weights (as in prune_sweep: magnitude pruning per row + round-to-nearest, simulated) and its perplexity
measured, for every threshold in `stage0.threshold_sweep` and guard size in `stage0.guard_sweep`.

Reference rows: the uncompressed model, the standard method (every layer at `uniform_bits`) and, per guard size,
the two budget plans (fit in the standard method's memory) (they do not depend on the threshold). Measurements are cached per plan and shared with
prune_sweep, so identical plans (e.g. guard layers already protected) are measured once.

Outputs under <run_dir>/stage_0_threshold_sweep/: report.md, stage_0_comparison.xlsx, results.json.
"""

from __future__ import annotations

from typing import Any, Callable

from sdf.reporting.markdown import _table
from sdf.reporting.reporter import StageReporter
from sdf.run import RunContext
from sdf.stage0.planner import baseline_cost, budget_matched_plan, plan_compression, uniform_plan
from sdf.stage0.prune_sweep import METRICS, make_evaluator
from sdf.stage0.run import _cost_metrics, load_fp16, load_guard, load_profile, setup, stage_reporter, weight_cost
from sdf.stage0.sensitivity import normalize


def run_threshold_sweep(ctx: RunContext, candidate: dict[str, Any], model=None, tokenizer=None,
                        text_loader: Callable[[str, str], list[str]] | None = None) -> dict[str, Any]:
    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    device, handle, text_loader = setup(ctx, model, tokenizer, text_loader)
    thresholds, guards = sorted(set(s0.threshold_sweep)), sorted(set(s0.guard_sweep))
    gs, pr = candidate["gptq_groupsize"], candidate["prune_ratio_aggressive"]

    profile, _ = load_profile(ctx, candidate, handle, text_loader)
    scores = normalize(profile.raw_scores, s0.normalization)
    predict = weight_cost(profile, gs, s0)
    rep = stage_reporter(
        ctx, candidate, handle, profile, ("stage0", "eval", "model", "run"), stage=0,
        subdir="stage_0_threshold_sweep", title="Protection thresholds: how many layers to protect",
        conditions={"model": cfg.model.name, "seed": cfg.run.seed, "device": str(device),
                    "backend": "HF Transformers", "pruning": "magnitude, per output row (simulated)",
                    "rounding": f"round-to-nearest at the plan's bits (group size {gs})",
                    "thresholds": ", ".join(f"{t:g}" for t in thresholds),
                    "never-pruned layers (guard sizes)": ", ".join(map(str, guards)),
                    "share removed from unprotected layers": pr, "sensitivity score": profile.method,
                    "evaluation data": f"wikitext-2 test, {cfg.eval.seq_len}-token windows, validation/held-out halves"},
        main_metrics=METRICS,
    )
    evaluate = make_evaluator(ctx, handle, text_loader, device)

    def row(method, variant, plan, **info):
        with rep.method(method, variant, key_term=False, compare_to="standard", **info) as r:
            r.metrics.update(_cost_metrics(predict(plan)), protected_layers=len(plan.protected_layers))
            result, r.info["cached"] = evaluate(plan, gs)
            r.metrics.update(result)

    with rep.method("baseline", "fp16", description="uncompressed model") as r:
        fp16, r.info["cached"] = load_fp16(ctx, handle, text_loader)
        r.metrics.update({k: fp16["metrics"][k] for k in ("ppl_val", "ppl_heldout")})
        r.metrics.update(_cost_metrics(baseline_cost(profile, s0.baseline_bits)), protected_layers=0)
    uniform = uniform_plan(scores, s0.uniform_bits, s0.uniform_prune_ratio)
    row("standard", "original", uniform, label="Standard method",
        description=f"every layer at {s0.uniform_bits} bits, {s0.uniform_prune_ratio:.0%} removed")
    budget = predict(uniform).weight_memory_gb
    for k in guards:
        guarded, _ = load_guard(ctx, candidate, handle, text_loader, profile, k)
        row(f"same_k{k}", "framework", budget_matched_plan(scores, budget, pr, s0.protected_bits, s0.compressed_bits,
                                                           predict, guarded),
            label=f"Budget plan, guard {k}", guard=k, description="as many top layers protected as fit in the "
                                                                 "standard method's size")
        row(f"noprune_k{k}", "framework", budget_matched_plan(scores, budget, 0.0, s0.protected_bits,
                                                              s0.no_prune_compressed_bits, predict, guarded),
            label=f"Benchmark: budget size, nothing removed, guard {k}", guard=k,
            description=f"benchmark, robust layers at {s0.no_prune_compressed_bits} bits, nothing removed")
        for t in thresholds:
            row(f"t{round(t * 100):03d}_k{k}", "framework",
                plan_compression(scores, t, pr, s0.protected_bits, s0.compressed_bits, guarded),
                label=f"Threshold {t:g}, guard {k}", threshold=t, guard=k,
                description=f"Stage 0 plan, threshold {t:g}, {pr:.0%} removed from unprotected layers, "
                            f"{k} layers never pruned")

    _write_plain(rep, s0, pr)
    return rep.finalize()


def frontier(points: list[tuple[str, float, float]]) -> list[str]:
    """Names of the (name, size, error) points no other point beats on both size and error."""
    return [n for n, gb, e in points
            if not any(g2 <= gb and e2 <= e and (g2, e2) != (gb, e) for _, g2, e2 in points)]


def _write_plain(rep: StageReporter, s0, pr: float) -> None:
    ok = [r for r in rep.rows if r.status == "ok"]
    fp16 = next((r.metrics for r in ok if r.variant == "fp16"), {})
    std = next((r.metrics for r in ok if r.variant == "original"), None)
    sweep = [r for r in ok if "threshold" in r.info]
    best = frontier([(r.plain_name, r.metrics["predicted_weight_memory_gb"], r.metrics["ppl_val"])
                     for r in ok if r.variant != "fp16"])
    rep.plain_intro = (
        "The protection threshold decides how many layers the framework protects: protected layers keep "
        f"{s0.protected_bits} bits per number and lose nothing; the others get {s0.compressed_bits} bits and lose "
        f"{pr:.0%} of their numbers. A low threshold protects many layers, so the model stays accurate but big; a "
        "high threshold protects few, so it is small but riskier. The guard is the number of layers whose "
        "removal hurts most, which are never pruned whatever the threshold. Here every combination is applied "
        "to the real model and tested on real text.")
    if std and sweep:
        wins = [r.plain_name for r in sweep if r.metrics["predicted_weight_memory_gb"] <= std["predicted_weight_memory_gb"]
                and r.metrics["ppl_val"] < std["ppl_val"]]
        rep.plain_summary = (
            f"We tried {len(sweep)} threshold and guard settings. The uncompressed model scores "
            f"{fp16.get('ppl_val', float('nan')):.2f} and the standard method {std['ppl_val']:.2f} at "
            f"{std['predicted_weight_memory_gb']:.3f} GB (lower is better for both). "
            + (f"{len(wins)} settings are both smaller and more accurate than the standard method: "
               f"{', '.join(wins)}. " if wins else
               "No setting is both smaller and more accurate than the standard method. ")
            + f"The best trade-offs (no other version is both smaller and more accurate) are: {', '.join(best)}.")
    rep.glossary.update({
        "Threshold": "The cut-off on the 0-to-1 sensitivity score; layers at or above it are protected.",
        "Guard": "How many of the layers whose removal hurts most are never pruned.",
        "Best trade-off": "A version no other version beats on both size and error.",
    })
    header = ["Threshold", "Guard", "Protected layers", "Memory (GB)", "Error", "Error (held-out)",
              "Best trade-off"]
    t = [_table(header, [[f"{r.info['threshold']:g}", r.info["guard"]]
                         + [r.metrics[k] for k in ("protected_layers", "predicted_weight_memory_gb", "ppl_val",
                                                   "ppl_heldout")] + ["yes" if r.plain_name in best else ""]
                         for r in sorted(sweep, key=lambda r: (r.info["guard"], r.info["threshold"]))])]
    notes = [
        "- **Threshold**: layers with a sensitivity score at or above it are protected.",
        "- **Guard**: how many of the layers whose removal hurts most are never pruned.",
        "- **Protected layers**: how many of the layers keep "
        f"{s0.protected_bits} bits with nothing removed.",
        "- **Memory (GB)**: predicted memory to store the model; lower is better"
        + (f" (standard method: {std['predicted_weight_memory_gb']:.4g})." if std else "."),
        "- **Error**: perplexity on the validation half of the test text; lower is better"
        + (f" (standard method: {std['ppl_val']:.4g})." if std else "."),
        "- **Error (held-out)**: the same on the held-out half, which nothing is tuned on; lower is better.",
        "- **Best trade-off**: yes when no other version is both smaller and more accurate.",
    ]
    rep.sections.append(("Threshold sweep", "\n".join(t + ["", "**What each column means**", ""] + notes)))
    rep.next_steps.append("Set the search range for sensitive_threshold around the best trade-offs, and pick the "
                          "default threshold for Stage 1 from them.")
