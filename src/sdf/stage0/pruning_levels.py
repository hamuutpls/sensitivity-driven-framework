"""Pruning levels test, part of every Stage 0 run (Mohammad, 2026-10-13).

Each Stage 2 pruning technique (Wanda, structured pruning of feed-forward channels, low-rank) prunes the FP16
model at every level in `stage0.pruning_levels` (30% to 90% by default) on three sets of layers, and the
perplexity of each pruned model is measured:

- all layers: every decoder layer loses the level's share of its weights.
- non-guarded layers: every layer except the guarded ones (the top `guard_top_k` by layer-removal score).
- guarded layers only: only the guarded layers.

Nothing is rounded (pruning only). Mohammad picks the pruning share for Stage 2 after seeing the results.
Structured pruning is capped as in Stage 2: at most 75% of a layer, or the layer's feed-forward share when that is
lower (it removes only feed-forward channels; 78.6% in TinyLlama). A higher level is capped, and the row records
the share really removed.

Outputs, in their own folder <run_dir>/stage_0_pruning_levels/ (so the 63 rows stay out of Stage 0's own tables
and charts): report.md, stage_0_comparison.xlsx and results.json with one row per (technique, layer set, level),
and pruning_levels.csv. A layer set with no layers, or the same layers as "all layers" (no guard), is skipped.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch import nn

from sdf.data import eval_windows
from sdf.eval.metrics import perplexity
from sdf.reporting.markdown import _table
from sdf.reporting.reporter import StageReporter
from sdf.stage0.planner import CompressionPlan, LayerPlan, cap_pruning, decoder_share_removed
from sdf.stage0.sensitivity import find_decoder_layers
from sdf.stages.methods import METHODS
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
SUBDIR = "stage_0_pruning_levels"
VERSION = 1  # bump when this test changes how a row is built (the pruning code's own versions are in the key too)


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
        raise ValueError(f"no feed-forward block found by shape (hidden size {model.config.hidden_size})")
    ups, down = found
    total = sum(m.weight.numel() for m in layer.modules() if isinstance(m, nn.Linear))
    return (sum(m.weight.numel() for m in ups) + down.weight.numel()) / total


def layer_caps(model: nn.Module, max_ratio: float) -> list[float]:
    """The most structured pruning removes from each decoder layer: `max_ratio` (75%, Mohammad 2026-10-13), or the
    layer's feed-forward share when that is lower (it can remove nothing else; Mohammad 2026-10-13)."""
    return [min(max_ratio, feed_forward_share(model, layer)) for layer in find_decoder_layers(model)]


def capped_plan(model: nn.Module, plan: CompressionPlan) -> CompressionPlan:
    """`plan` with each layer's ratio capped as Stage 2 caps structured pruning (layer_caps)."""
    return cap_pruning(plan, layer_caps(model, METHODS["structured_prune"].max_ratio))


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
                       profile, guarded: set[int] | list[int], fp16: dict[str, Any] | None) -> Path:
    """Measure every (technique, layer set, level) in its own report folder; return that folder.
    `fp16`: the Stage 0 FP16 row's metrics, repeated as this report's baseline row (None: not measured)."""
    from sdf.stage0.run import calib_batches, fp16_key, source_key, stage_reporter  # run.py calls this module

    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    levels = sorted({float(r) for r in s0.pruning_levels})
    model = handle.model(cfg.model.dtype)
    num_layers = len(find_decoder_layers(model))
    sets = {}
    for name in LAYER_SETS:
        layers = layer_set(name, num_layers, guarded)
        if layers and (name == "all_layers" or len(layers) < num_layers):
            sets[name] = layers
        else:
            log.info("pruning levels: layer set %s skipped (%d of %d layers)", name, len(layers), num_layers)
    rep = stage_reporter(
        ctx, candidate, handle, profile, ("stage0", "calibration", "eval", "model", "run"), stage=0, subdir=SUBDIR,
        title="Pruning levels test (for Stage 2)",
        conditions={"model": cfg.model.name, "seed": cfg.run.seed, "device": str(handle.device),
                    "calibration": f"{candidate['calib_dataset']}, {candidate['calib_samples']} x "
                                   f"{cfg.calibration.seq_len} tokens",
                    "levels": ", ".join(f"{r:.0%}" for r in levels), "guarded layers": sorted(guarded),
                    "layer sets": ", ".join(sets), "rounding": "none (pruning only)"},
        main_metrics=["ppl_val", "ppl_heldout", "decoder_weights_removed"])
    for old in (CSV_NAME, rep.report_path.name, rep.xlsx_path.name, rep.json_path.name):  # no stale outputs on a rerun
        (rep.dir / old).unlink(missing_ok=True)
    if fp16:
        with rep.method("baseline", "fp16", description="uncompressed model (Stage 0 measurement)") as row:
            row.metrics.update({k: fp16[k] for k in ("ppl_val", "ppl_heldout") if k in fp16}, decoder_weights_removed=0.0)
    batches: list[torch.Tensor] = []
    windows: list[torch.Tensor] = []

    def measure(fn: Callable, plan: CompressionPlan) -> dict[str, float]:
        if not windows:
            windows.extend(eval_windows(text_loader(cfg.eval.dataset, "test"), handle.tokenizer,
                                        cfg.eval.seq_len, cfg.eval.max_windows))
        if not batches:
            batches.extend(calib_batches(ctx, candidate, handle, text_loader, candidate["calib_samples"]))
        with restored(model):
            fn(model, plan, batches, candidate["gptq_groupsize"], s0.baseline_bits)
            return {"ppl_val": perplexity(model, windows[0], handle.device)[0],
                    "ppl_heldout": perplexity(model, windows[1], handle.device)[0]}

    key_base = {**fp16_key(ctx, handle.device), "calib_dataset": candidate["calib_dataset"],
                "calib_samples": candidate["calib_samples"], "calib_seq_len": cfg.calibration.seq_len,
                "calib_batch_size": cfg.calibration.batch_size, **source_key(cfg, candidate["calib_dataset"])}
    # only perplexity is measured here, so timing settings must not re-run every row
    key_base["eval"] = {k: v for k, v in key_base["eval"].items() if not k.startswith("latency_")}
    out: list[dict[str, Any]] = []
    for tech, (fn, tech_label) in TECHNIQUES.items():
        for set_name, layers in sets.items():
            for r in levels:
                plan = level_plan(num_layers, layers, r, s0.baseline_bits)
                if tech == "structured":
                    plan = capped_plan(model, plan)
                per_layer = max(lp.pruning_ratio for lp in plan.layers)  # what is really removed (capped)
                removed = decoder_share_removed(plan, profile.layer_numel)
                record = {"technique": tech, "layer_set": set_name, "level": r, "layers_pruned": len(layers),
                          "share_removed_per_pruned_layer": per_layer, "share_of_decoder_weights_removed": removed,
                          "ppl_val": None, "ppl_heldout": None, "status": "failed"}
                shown = f"{per_layer:.1%} (capped from {r:.0%})" if per_layer + 1e-9 < r else f"{r:.0%}"
                with rep.method(f"{tech}_{set_name}_{round(r * 100):03d}", "original", key_term=False,
                                label=f"{tech_label}, {LAYER_SETS[set_name].lower()}, {shown}",
                                description=f"{tech_label} removes {shown} of each of layers {layers}"
                                            " (pruning only, nothing rounded)") as row:
                    row.metrics.update(decoder_weights_removed=removed, protected_layers=num_layers - len(layers))
                    key = {**key_base, "technique": tech, "version": [VERSION, METHODS[_METHOD[tech]].version],
                           "ratios": [lp.pruning_ratio for lp in plan.layers]}
                    result, row.info["cached"] = ctx.cache.get_or_compute(
                        "prune_level_eval", key, lambda: measure(fn, plan))
                    row.metrics.update(result)
                    record.update(result, status="ok")
                    log.info("pruning level %s %s %s: ppl_val %.3f", tech, set_name, shown, result["ppl_val"])
                out.append(record)

    with (rep.dir / CSV_NAME).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    _add_section(rep, out, levels, guarded, sets, fp16)
    rep.finalize()
    return rep.dir


_METHOD = {"wanda": "unstructured_prune", "structured": "structured_prune", "low_rank": "low_rank"}  # Stage 2 names


def _add_section(rep: StageReporter, records: list[dict[str, Any]], levels: list[float], guarded,
                 sets: dict[str, list[int]], fp16: dict[str, Any] | None) -> None:
    by = {(r["technique"], r["layer_set"], r["level"]): r for r in records}
    header = ["Technique", "Layers pruned"] + [f"{round(r * 100)}%" for r in levels]

    def cell(r: dict[str, Any]) -> str:
        if r["status"] != "ok":
            return "failed"
        cap = r["share_removed_per_pruned_layer"]
        return f"{r['ppl_val']:.4g}" + (f" (capped {cap:.1%})" if cap + 1e-9 < r["level"] else "")

    rows = [[TECHNIQUES[t][1], f"{LAYER_SETS[s]} ({len(sets[s])})"] + [cell(by[(t, s, r)]) for r in levels]
            for t in TECHNIQUES for s in sets]
    base = (fp16 or {}).get("ppl_val")
    skipped = [LAYER_SETS[s] for s in LAYER_SETS if s not in sets]
    notes = [
        "- **Level**: the share of weights removed from each pruned layer. Nothing is rounded.",
        "- **Cells**: perplexity on the validation half of the test text; lower is better"
        + (f" (uncompressed model: {base:.4g})." if base else "."),
        f"- **Guarded layers**: {sorted(guarded) or 'none'} (never pruned by Our method).",
        f"- **Capped**: structured pruning removes at most {METHODS['structured_prune'].max_ratio:.0%} of a layer, "
        "or the layer's feed-forward share when that is lower (it removes only feed-forward channels), as in "
        "Stage 2; the cell shows the share really removed.",
        "- **Low-rank**: the share removed is the share of numbers the two smaller factors save; the factors are "
        "multiplied back into a full-size matrix here, so the weights are not zeros.",
        f"- Every value, with the held-out half and the share of the decoder layers' weights removed, is in {CSV_NAME}.",
    ] + ([f"- **Skipped layer sets**: {', '.join(skipped)} (no layers, or the same layers as all layers)."]
         if skipped else [])
    rep.sections.append(("Perplexity by technique, layer set and level",
                         "\n".join([_table(header, rows), "", "**What each column means**", ""] + notes)))
