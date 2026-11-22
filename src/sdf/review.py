"""Experiments for the paper review of 22 Nov 2026 (Mohammad chose "Paper fixes and new Colab runs"; spec:
/mnt/project-files/thesis/review-20261122/experiment-spec.md). scripts/review_20261122.py runs them.

Run 1, `layer_damage` / `spearman_table`: does the layer-skipping score predict compression damage? For each decoder
layer alone, the perplexity rise when it is skipped, when its weights are rounded to 8 and to 4 bits and when 25% of
its weights are pruned (magnitude, and Wanda), and the rank correlation between every pair of those columns.

Run 2, `bit_plans` / `plan_row`: the per-layer plan against other plans at the same size (the inverse plan, random
plans) and against uniform 4 bits at about the same size (the most sensitive layers at 8 bits, the rest at 3).

Run 3, `ablation_points`: the stage ablation, Baseline and E1-E6, as search points of one final model's settings with
a stage left out (no_quant: no quantization; pruning share 0: no pruning; "none": no KV method).
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import replace
from itertools import combinations
from typing import Any, Callable, Iterator

import torch
from torch import nn

from sdf.search_space import NO_QUANT
from sdf.stage0.planner import CompressionPlan
from sdf.stage0.prune_sweep import magnitude_mask
from sdf.stage0.sensitivity import _mean_loss, _ppl, find_decoder_layers, normalize, quantize_layer, skip_layer
from sdf.stages.pruning import input_sq_norms

# ---------------------------------------------------------------------------------------------------------- run 1


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation of `a` and `b`: the Pearson correlation of their ranks, tied values sharing their
    average rank. nan when either has no spread (all values equal)."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError(f"need two columns of the same length >= 2, got {len(a)} and {len(b)}")
    ra, rb = normalize(list(a), "rank"), normalize(list(b), "rank")  # rank / (n - 1), ties averaged
    if len(set(a)) == 1 or len(set(b)) == 1:
        return math.nan
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))


def spearman_table(columns: dict[str, list[float]]) -> list[dict[str, Any]]:
    """Spearman's rho between every pair of columns, with the number of layers."""
    return [{"a": x, "b": y, "spearman": spearman(columns[x], columns[y]), "layers": len(columns[x])}
            for x, y in combinations(columns, 2)]


@contextmanager
def prune_layer(model: nn.Module, layer: nn.Module, ratio: float, how: str = "magnitude",
                batches: list[torch.Tensor] | None = None) -> Iterator[None]:
    """Prune `ratio` of every Linear weight in `layer` while open (each output row drops its smallest scores):
    "magnitude" scores |w|, "wanda" |w| x the input's norm on `batches` (measured with no layer pruned). The
    original weights are restored afterwards."""
    if how not in ("magnitude", "wanda"):
        raise ValueError(f"unknown pruning {how!r}")
    if how == "wanda" and not batches:
        raise ValueError("Wanda needs calibration batches")
    linears = [m for m in layer.modules() if isinstance(m, nn.Linear)]
    norms = input_sq_norms(model, layer, batches) if how == "wanda" else {}
    saved = [m.weight.data.clone() for m in linears]
    try:
        for m in linears:
            score = m.weight.data.float().abs() * norms[m].sqrt() if how == "wanda" else m.weight.data
            m.weight.data.mul_(magnitude_mask(score, ratio))
        yield
    finally:
        for m, w in zip(linears, saved):
            m.weight.data.copy_(w)


DAMAGE = ("skip", "w8", "w4", "prune25_magnitude", "prune25_wanda")


@torch.no_grad()
def layer_damage(model: nn.Module, texts: dict[str, list[torch.Tensor]], calib: list[torch.Tensor],
                 group_size: int = 128, prune_ratio: float = 0.25, int_zero: bool = True,
                 kinds: tuple[str, ...] = DAMAGE, device: torch.device | str | None = None) -> list[dict[str, Any]]:
    """One row per decoder layer: for each text set in `texts` (name -> batches), the perplexity rise when only that
    layer is changed in each way of `kinds` (skipped; weights rounded to 8 or 4 bits, round-to-nearest with
    `group_size`; `prune_ratio` of its weights pruned by magnitude or by Wanda on `calib`). Columns are
    "<kind>_<set>"; the baseline perplexity of each set is in every row as "base_<set>"."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    if bad := [k for k in kinds if k not in DAMAGE]:
        raise ValueError(f"unknown damage kinds {bad}; choose from {DAMAGE}")
    was = model.training
    model.eval()
    layers = find_decoder_layers(model)
    changes: dict[str, Callable[[nn.Module], Any]] = {
        "skip": skip_layer,
        "w8": lambda l: quantize_layer(l, 8, group_size, int_zero),
        "w4": lambda l: quantize_layer(l, 4, group_size, int_zero),
        "prune25_magnitude": lambda l: prune_layer(model, l, prune_ratio, "magnitude"),
        "prune25_wanda": lambda l: prune_layer(model, l, prune_ratio, "wanda", calib),
    }
    try:
        base = {name: _ppl(_mean_loss(model, list(b), device)) for name, b in texts.items()}
        rows = []
        for i, layer in enumerate(layers):
            row: dict[str, Any] = {"layer": i, **{f"base_{n}": v for n, v in base.items()}}
            for kind in kinds:
                with changes[kind](layer):
                    for name, b in texts.items():
                        row[f"{kind}_{name}"] = _ppl(_mean_loss(model, list(b), device)) - base[name]
            rows.append(row)
    finally:
        model.train(was)
    return rows


# ---------------------------------------------------------------------------------------------------------- run 2


def ranked(template: CompressionPlan) -> list[int]:
    """Layers from most to least sensitive (the Stage 0 score each layer carries in the plan; ties by index)."""
    return sorted(range(len(template.layers)), key=lambda i: (-template.layers[i].sensitivity, i))


def bits_plan(template: CompressionPlan, high: set[int] | list[int], high_bits: int, low_bits: int,
              kind: str) -> CompressionPlan:
    """`template` (a Stage 0 plan, for its layers' scores) with layers in `high` at `high_bits` and the rest at
    `low_bits`, nothing pruned: weights only."""
    high = set(high)
    if bad := high - set(range(len(template.layers))):
        raise ValueError(f"layers {sorted(bad)} not in the plan")
    layers = tuple(replace(lp, bit_width=high_bits if lp.layer in high else low_bits, pruning_ratio=0.0,
                           protected=lp.layer in high, guarded=False) for lp in template.layers)
    return CompressionPlan(layers, kind, template.sensitive_threshold, 0.0)


def bit_plans(template: CompressionPlan, k: int, seeds: range | list[int] = range(5), high_bits: int = 8,
              low_bits: int = 4) -> dict[str, CompressionPlan]:
    """Plans with `k` layers at `high_bits` and the rest at `low_bits`, so all the same size: "sensitivity" (the k
    most sensitive layers), "inverse" (the k least sensitive) and "random_<seed>" (k layers drawn with each seed)."""
    n = len(template.layers)
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside 0..{n}")
    order = ranked(template)
    out = {"sensitivity": bits_plan(template, order[:k], high_bits, low_bits, "sensitivity"),
           "inverse": bits_plan(template, order[n - k:], high_bits, low_bits, "inverse")}
    for s in seeds:
        out[f"random_{s}"] = bits_plan(template, random.Random(s).sample(range(n), k), high_bits, low_bits,
                                       f"random_{s}")
    return out


def average_bits(plan: CompressionPlan, layer_numel: list[int]) -> float:
    """Average bits per decoder weight (the plan's bits weighted by each layer's weight count; no scales)."""
    return sum(lp.bit_width * n for lp, n in zip(plan.layers, layer_numel)) / sum(layer_numel)


def plan_row(method: Any, plan: CompressionPlan, model: nn.Module, call: Callable[[nn.Module, Any], Any],
             measure: Callable[[nn.Module], dict[str, float]]) -> dict[str, Any]:
    """Apply the Stage 1 weight method `method` (methods.Method) to `model` with `plan` (`call(model, plan)` makes
    its MethodCall) and return `measure(model)` while it is applied."""
    if method.stage != 1 or method.apply is None:
        raise ValueError(f"{method.name} is not a Stage 1 method that can run")
    with method.apply(call(model, plan)):
        return dict(measure(model))


def equal_ppl_size(rows: list[dict[str, Any]], target: float, size: str = "predicted_weight_memory_gb",
                   ppl: str = "ppl_val") -> dict[str, Any] | None:
    """The smallest row (by `size`) whose `ppl` is at or below `target`; None when none is."""
    ok = [r for r in rows if r.get(ppl) is not None and r[ppl] <= target and r.get(size) is not None]
    return min(ok, key=lambda r: r[size], default=None)


def equal_ppl_table(rows: list[dict[str, Any]], targets: list[str], size: str = "predicted_weight_memory_gb",
                    ppl: str = "ppl_val") -> list[dict[str, Any]]:
    """Part C of Run 2: for each method, each target plan in `targets` (its validation perplexity is the target) and
    each family of plans (row "family"), the smallest row of that family reaching the target. Rows without `ppl`
    (failed) are skipped; a family with no row at or below the target gives None."""
    out = []
    for method in dict.fromkeys(r["method"] for r in rows):
        mine = [r for r in rows if r["method"] == method and r.get(ppl) is not None]
        for target in targets:
            ref = next((r for r in mine if r["plan"] == target), None)
            for family in dict.fromkeys(r["family"] for r in mine):
                best = equal_ppl_size([r for r in mine if r["family"] == family], ref[ppl], size, ppl) if ref else None
                out.append({"method": method, "target": target, "target_ppl_val": ref and ref[ppl], "family": family,
                            "smallest_plan": best and best["plan"], "smallest_gb": best and best[size],
                            "its_ppl_val": best and best[ppl]})
    return out


# ---------------------------------------------------------------------------------------------------------- run 3

ABLATION = (  # row, (quantize, prune, KV): which stages the row keeps
    ("Baseline", (False, False, False)),
    ("E1", (True, False, False)),
    ("E2", (False, True, False)),
    ("E3", (False, False, True)),
    ("E4", (True, True, False)),
    ("E5", (True, False, True)),
    ("E6", (True, True, True)),
)


def ablation_points(point: dict[str, Any], prune_ratio: float) -> list[tuple[str, dict[str, Any]]]:
    """Baseline and E1-E6 for one final model's search `point`: each row keeps that point's Stage 1 technique,
    Stage 2 technique at `prune_ratio` and Stage 3 technique where the row keeps the stage, and leaves the stage out
    otherwise (NO_QUANT, pruning share 0, KV "none")."""
    out = []
    for row, (q, p, kv) in ABLATION:
        out.append((row, {**point, "stage1_method": point["stage1_method"] if q else NO_QUANT,
                          "prune_ratio_aggressive": prune_ratio if p else 0.0,
                          "stage3_method": point["stage3_method"] if kv else "none"}))
    return out


VLLM_SAME = {"E3": "Baseline", "E5": "E1", "E6": "E4"}  # the KV step does not run in vLLM: same vLLM model


def copy_backend(rows: list[dict[str, Any]], prefix: str = "vllm_") -> list[dict[str, Any]]:
    """`rows` (one model's ablation rows, "row" = Baseline/E1-E6) with each row of VLLM_SAME given its partner's
    `prefix` columns (measured once, on the partner) and "vllm_from" naming the partner."""
    by = {r["row"]: r for r in rows}
    out = []
    for r in rows:
        src = by.get(VLLM_SAME.get(r["row"], ""))
        if src is None:
            out.append(r)
            continue
        out.append({**{k: v for k, v in r.items() if not k.startswith(prefix)},
                    **{k: v for k, v in src.items() if k.startswith(prefix)}, "vllm_from": src["row"]})
    return out

