"""Pruning levels test, part of every Stage 0 run (Mohammad, 2026-10-13).

Each Stage 2 pruning technique (Wanda, structured pruning of feed-forward channels, low-rank) prunes the FP16
model at every level in `stage0.pruning_levels` (30% to 90% by default) on three sets of layers, and the
perplexity of each pruned model is measured:

- all layers: every decoder layer loses the level's share of its weights.
- non-guarded layers: every layer except the guarded ones (the top `guard_top_k` by layer-removal score).
- guarded layers only: only the guarded layers.

Nothing is rounded (pruning only). Mohammad picks the pruning share for Stage 2 after seeing the results.
Structured pruning removes only feed-forward channels, so it cannot remove more of a layer than the layer's
feed-forward share (78.6% in TinyLlama); a higher level is capped at that share, and the row records the share
really removed.

Outputs: one row per (technique, layer set, level) in the Stage 0 report, and <stage_0>/pruning_levels.csv.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch import nn

from sdf.data import eval_windows
from sdf.eval.metrics import perplexity
from sdf.reporting.markdown import _table
from sdf.reporting.reporter import StageReporter
from sdf.stage0.planner import CompressionPlan, LayerPlan
from sdf.stage0.sensitivity import find_decoder_layers
from sdf.stages.pruning import low_rank_, mlp_linears, structured_prune_, wanda_
from sdf.utils.logging import get_logger

log = get_logger(__name__)

# name -> (pruning function from stages/pruning.py, label)
TECHNIQUES: dict[str, tuple[Callable, str]] = {
    "wanda": (wanda_, "Wanda (single weights)"),
    "structured": (structured_prune_, "Structured (feed-forward channels)"),
    "low_rank": (low_rank_, "Low-rank (activation-aware SVD)"),
}
LAYER_SETS = {"all_layers": "All layers", "non_guarded_layers": "Non-guarded layers",
              "guarded_layers_only": "Guarded layers only"}
CSV_NAME = "pruning_levels.csv"


def layer_set(name: str, num_layers: int, guarded: set[int] | list[int]) -> list[int]:
    """Decoder layer indices the layer set `name` prunes."""
    guarded = set(guarded)
    if name == "all_layers":
        return list(range(num_layers))
    if name == "non_guarded_layers":
        return [i for i in range(num_layers) if i not in guarded]
    if name == "guarded_layers_only":
        return sorted(i for i in guarded if 0 <= i < num_layers)
    raise ValueError(f"unknown layer set {name!r}; choose from {list(LAYER_SETS)}")


def level_plan(num_layers: int, layers: list[int], ratio: float, bits: int) -> CompressionPlan:
    """Pruning-only plan: `ratio` removed from each layer in `layers`, nothing from the others, all at `bits`."""
    chosen = set(layers)
    return CompressionPlan(tuple(LayerPlan(layer=i, bit_width=bits, pruning_ratio=ratio if i in chosen else 0.0,
                                           protected=False, sensitivity=0.0) for i in range(num_layers)),
                           "pruning_level", None, ratio)


def feed_forward_share(model: nn.Module, layer: nn.Module) -> float:
    """The most structured pruning can remove from `layer`: its feed-forward weights over all its Linear weights."""
    found = mlp_linears(layer, model.config.hidden_size)
    if found is None:
        return 0.0
    ups, down = found
    total = sum(m.weight.numel() for m in layer.modules() if isinstance(m, nn.Linear))
    return (sum(m.weight.numel() for m in ups) + down.weight.numel()) / total


def capped_plan(model: nn.Module, plan: CompressionPlan) -> CompressionPlan:
    """`plan` with each layer's ratio capped at the share structured pruning can remove from it."""
    caps = [feed_forward_share(model, layer) for layer in find_decoder_layers(model)]
    return replace(plan, layers=tuple(replace(lp, pruning_ratio=min(lp.pruning_ratio, cap))
                                      for lp, cap in zip(plan.layers, caps)))


def share_removed(model: nn.Module, plan: CompressionPlan) -> float:
    """Share of all decoder Linear weights the plan removes (each layer weighted by its size)."""
    sizes = [sum(m.weight.numel() for m in layer.modules() if isinstance(m, nn.Linear))
             for layer in find_decoder_layers(model)]
    return sum(lp.pruning_ratio * n for lp, n in zip(plan.layers, sizes)) / sum(sizes)


@contextmanager
def restored(model: nn.Module) -> Iterator[None]:
    """Keep a CPU copy of every decoder Linear weight and bias, and put them back on exit."""
    saved = [(p, p.data.to("cpu", copy=True)) for layer in find_decoder_layers(model)
             for m in layer.modules() if isinstance(m, nn.Linear)
             for p in (m.weight, m.bias) if p is not None]
    try:
        yield
    finally:
        for p, data in saved:
            p.data.copy_(data)


def run_pruning_levels(ctx, candidate: dict[str, Any], handle, text_loader: Callable[[str, str], list[str]],
                       rep: StageReporter, guarded: set[int] | list[int], out_dir: Path) -> list[dict[str, Any]]:
    """Add one Stage 0 row per (technique, layer set, level), write pruning_levels.csv; return the CSV rows."""
    from sdf.stage0.run import calib_batches, fp16_key, source_key  # run.py calls this module

    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    levels = sorted({float(r) for r in s0.pruning_levels})
    if not all(0.0 < r <= 1.0 for r in levels):
        raise ValueError(f"stage0.pruning_levels must be shares in (0, 1], got {s0.pruning_levels}")
    batches: list[torch.Tensor] = []
    windows: list[torch.Tensor] = []

    def measure(fn: Callable, plan: CompressionPlan) -> dict[str, float]:
        if not windows:
            windows.extend(eval_windows(text_loader(cfg.eval.dataset, "test"), handle.tokenizer,
                                        cfg.eval.seq_len, cfg.eval.max_windows))
        if not batches:
            batches.extend(calib_batches(ctx, candidate, handle, text_loader, candidate["calib_samples"]))
        model = handle.model(cfg.model.dtype)
        with restored(model):
            fn(model, plan, batches, candidate["gptq_groupsize"], s0.baseline_bits)
            return {"ppl_val": perplexity(model, windows[0], handle.device)[0],
                    "ppl_heldout": perplexity(model, windows[1], handle.device)[0]}

    model = handle.model(cfg.model.dtype)
    num_layers = len(find_decoder_layers(model))
    key_base = {**fp16_key(ctx, handle.device), "calib_dataset": candidate["calib_dataset"],
                "calib_samples": candidate["calib_samples"], "calib_seq_len": cfg.calibration.seq_len,
                "calib_batch_size": cfg.calibration.batch_size, **source_key(cfg, candidate["calib_dataset"])}
    out: list[dict[str, Any]] = []
    for tech, (fn, tech_label) in TECHNIQUES.items():
        for set_name, set_label in LAYER_SETS.items():
            layers = layer_set(set_name, num_layers, guarded)
            for r in levels:
                pct = round(r * 100)
                plan = level_plan(num_layers, layers, r, s0.baseline_bits)
                if tech == "structured":
                    plan = capped_plan(model, plan)
                removed = share_removed(model, plan)
                record = {"technique": tech, "layer_set": set_name, "level": r, "layers_pruned": len(layers),
                          "share_removed_per_pruned_layer": max((lp.pruning_ratio for lp in plan.layers), default=0.0),
                          "share_of_model_removed": removed, "ppl_val": None, "ppl_heldout": None, "status": "failed"}
                with rep.method(f"{tech}_{set_name}_{pct:03d}", "original", key_term=False,
                                label=f"{tech_label}, {set_label.lower()}, {pct}%",
                                description=f"{tech_label} removes {pct}% of each of layers {layers or 'none'}"
                                            " (pruning only, nothing rounded)") as row:
                    row.metrics.update(sparsity=removed, protected_layers=num_layers - len(layers))
                    if not layers:
                        raise ValueError(f"layer set {set_name} has no layers (guard: {sorted(guarded)})")
                    key = {**key_base, "technique": tech, "version": 1,
                           "ratios": [lp.pruning_ratio for lp in plan.layers]}
                    result, row.info["cached"] = ctx.cache.get_or_compute(
                        "prune_level_eval", key, lambda: measure(fn, plan))
                    row.metrics.update(result)
                    record.update(result, status="ok")
                    log.info("pruning level %s %s %d%%: ppl_val %.3f", tech, set_name, pct, result["ppl_val"])
                out.append(record)

    path = out_dir / CSV_NAME
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    _add_section(rep, out, levels, guarded)
    return out


def _add_section(rep: StageReporter, records: list[dict[str, Any]], levels: list[float], guarded) -> None:
    fp16 = next((r.metrics.get("ppl_val") for r in rep.rows if r.variant == "fp16" and r.status == "ok"), None)
    by = {(r["technique"], r["layer_set"], r["level"]): r for r in records}
    header = ["Technique", "Layers pruned"] + [f"{round(r * 100)}%" for r in levels]

    def cell(r: dict[str, Any]) -> str:
        if r["status"] != "ok":
            return "failed"
        cap = r["share_removed_per_pruned_layer"]
        return f"{r['ppl_val']:.4g}" + (f" (capped {cap:.1%})" if cap + 1e-9 < r["level"] else "")

    rows = [[TECHNIQUES[t][1], LAYER_SETS[s]] + [cell(by[(t, s, r)]) for r in levels]
            for t in TECHNIQUES for s in LAYER_SETS]
    notes = [
        "- **Level**: the share of weights removed from each pruned layer. Nothing is rounded.",
        "- **Cells**: perplexity on the validation half of the test text; lower is better"
        + (f" (uncompressed model: {fp16:.4g})." if fp16 else "."),
        f"- **Guarded layers**: {sorted(guarded) or 'none'} (never pruned by Our method).",
        "- **Capped**: structured pruning removes only feed-forward channels, so it stops at the layer's "
        "feed-forward share; the cell shows the share really removed.",
        f"- Every value, with the held-out half and the share of the model removed, is in {CSV_NAME}.",
    ]
    rep.sections.append(("Pruning levels test (for Stage 2)",
                         "\n".join([_table(header, rows), "", "**What each column means**", ""] + notes)))
