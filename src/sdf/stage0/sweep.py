"""Stage 0 sweep: try several values of every search-space setting and compare the plans side by side.

Calibration settings (dataset, number of samples) need a new sensitivity profile each, so they are profiled
once per combination (cached). Everything else (threshold, prune ratio, group size) only changes the plan, so
the full grid over those is cheap. Stage 0 predicts size and exposure only; accuracy comes with Stage 1.

Outputs under <run_dir>/stage_0_sweep/: report.md (plain language first), sweep.xlsx, results.json.
"""

from __future__ import annotations

import functools
import itertools
import json
from typing import Any, Callable

from openpyxl import Workbook
from openpyxl.styles import Font

from sdf.data import load_texts
from sdf.reporting.markdown import original_model_lines
from sdf.run import RunContext
from sdf.search_space import PER_CHANNEL, SEARCH_SPACE
from sdf.stage0.planner import budget_matched_plan, guarded_layers, plan_compression, predict_cost, uniform_plan
from sdf.stage0.run import _ModelHandle, load_profile, original_model_info
from sdf.stage0.sensitivity import SensitivityProfile, normalize, outlier_layers
from sdf.utils.cache import atomic_write_text
from sdf.utils.env import resolve_device
from sdf.utils.logging import get_logger

log = get_logger(__name__)

CALIB = ("calib_dataset", "calib_samples")


def default_grid(points: int = 5) -> dict[str, list[Any]]:
    """Every choice of a categorical parameter; `points` evenly spaced values of a continuous one. Only the
    parameters Stage 0 plans with (stages 0 and 1); later stages' parameters do not change a Stage 0 plan."""
    grid = {}
    for p in (p for p in SEARCH_SPACE.params if p.stage <= 1):
        if p.choices is not None:
            grid[p.name] = list(p.choices)
        else:
            grid[p.name] = [round(p.low + i * (p.high - p.low) / (points - 1), 4) for i in range(points)]
    return grid


def _gs(g: int) -> str:
    return "per row" if g == PER_CHANNEL else str(g)


def plan_rows(profile: SensitivityProfile, calib: dict[str, Any], grid: dict[str, list[Any]],
              s0) -> list[dict[str, Any]]:
    """One row per (threshold, prune ratio, group size): the framework, same-size and no-prune plans vs uniform."""
    scores = normalize(profile.raw_scores, s0.normalization)
    # The sweep guards only when the profile is itself a layer-removal one (the default score); run_stage0
    # measures a removal profile for the guard whatever the score.
    guard = guarded_layers(profile.raw_scores, s0.guard_top_k) if profile.method == "layer_removal" else frozenset()
    rows = []
    for gs in grid["gptq_groupsize"]:
        cost = functools.partial(predict_cost, profile=profile, group_size=gs,
                                 group_overhead_bits=s0.group_overhead_bits, baseline_bits=s0.baseline_bits)
        uni = cost(uniform_plan(scores, s0.uniform_bits, s0.uniform_prune_ratio))
        no_prune = budget_matched_plan(scores, uni.weight_memory_gb, 0.0, s0.protected_bits,
                                       s0.no_prune_compressed_bits, cost, guard)
        np_cost = cost(no_prune)
        for t, pr in itertools.product(grid["sensitive_threshold"], grid["prune_ratio_aggressive"]):
            fw_plan = plan_compression(scores, t, pr, s0.protected_bits, s0.compressed_bits, guard)
            fw = cost(fw_plan)
            same = budget_matched_plan(scores, uni.weight_memory_gb, pr, s0.protected_bits, s0.compressed_bits, cost,
                                       guard)
            sc = cost(same)
            rows.append({
                **calib, "sensitive_threshold": t, "prune_ratio_aggressive": pr, "gptq_groupsize": gs,
                "uniform_gb": uni.weight_memory_gb, "uniform_exposure": uni.sensitivity_exposure,
                "fw_protected": len(fw_plan.protected_layers), "fw_gb": fw.weight_memory_gb,
                "fw_bits": fw.avg_bits_per_weight, "fw_sparsity": fw.sparsity,
                "fw_exposure": fw.sensitivity_exposure,
                "fw_beats_uniform": fw.weight_memory_gb <= uni.weight_memory_gb
                and fw.sensitivity_exposure < uni.sensitivity_exposure,
                "same_protected": len(same.protected_layers), "same_gb": sc.weight_memory_gb,
                "same_sparsity": sc.sparsity, "same_exposure": sc.sensitivity_exposure,
                "noprune_protected": len(no_prune.protected_layers), "noprune_gb": np_cost.weight_memory_gb,
                "noprune_exposure": np_cost.sensitivity_exposure,
            })
    return rows


def _spearman(a: list[float], b: list[float]) -> float:
    ra, rb = normalize(a), normalize(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return cov / den if den else 1.0


def calibration_rows(profiles: dict[tuple, SensitivityProfile], reference: tuple, threshold: float,
                     normalization: str) -> list[dict[str, Any]]:
    """How much each calibration setting changes the sensitivity ranking, against the reference setting."""
    ref = profiles[reference]
    ref_prot = set(plan_compression(normalize(ref.raw_scores, normalization), threshold, 0.0, 8, 4).protected_layers)
    out = []
    for key, prof in profiles.items():
        prot = set(plan_compression(normalize(prof.raw_scores, normalization), threshold, 0.0, 8, 4).protected_layers)
        out.append({
            "calib_dataset": key[0], "calib_samples": key[1],
            "rank_agreement": _spearman(prof.raw_scores, ref.raw_scores),
            "same_protected": len(prot & ref_prot), "protected": len(prot),
            "protected_layers": sorted(prot), "outliers": outlier_layers(prof.raw_scores),
            "profiling_s": prof.cost.get("wall_clock_s"),
        })
    return out


def run_sweep(ctx: RunContext, grid: dict[str, list[Any]] | None = None, model=None, tokenizer=None,
              text_loader: Callable[[str, str], list[str]] | None = None,
              profiles: dict[tuple, SensitivityProfile] | None = None) -> dict[str, Any]:
    """Profile each calibration setting (unless `profiles` is given), then plan the whole grid and report."""
    cfg, s0 = ctx.cfg, ctx.cfg.stage0
    grid = {**default_grid(), **(grid or {})}
    defaults = SEARCH_SPACE.make(cfg.hyperparams)
    handle = _ModelHandle(ctx, resolve_device(cfg.model.device), model, tokenizer)
    if profiles is None:
        text_loader = text_loader or functools.partial(load_texts, cfg.data.sources)
        profiles = {}
        for ds, n in itertools.product(grid["calib_dataset"], grid["calib_samples"]):
            try:
                profiles[(ds, n)], _ = load_profile(ctx, {**defaults, "calib_dataset": ds, "calib_samples": n},
                                                    handle, text_loader)
            except Exception as e:  # one unavailable dataset must not sink the sweep
                log.exception("profiling %s x %s failed", ds, n)
                profiles[(ds, n)] = e
    failures = {k: repr(v) for k, v in profiles.items() if isinstance(v, Exception)}
    profiles = {k: v for k, v in profiles.items() if not isinstance(v, Exception)}
    if not profiles:
        raise RuntimeError(f"every calibration setting failed: {failures}")

    # The report's one-setting-at-a-time tables hold the others at their default, or the grid's middle value
    # when the default is not in the grid.
    focus = {k: defaults[k] if defaults[k] in v else v[len(v) // 2] for k, v in grid.items()}
    reference = (focus["calib_dataset"], focus["calib_samples"])
    if reference not in profiles:
        reference = next(iter(profiles))
    rows = [r for key, prof in profiles.items()
            for r in plan_rows(prof, {"calib_dataset": key[0], "calib_samples": key[1]}, grid, s0)]
    calib = calibration_rows(profiles, reference, focus["sensitive_threshold"], s0.normalization)
    result = {"grid": grid, "defaults": defaults, "focus": focus, "reference": list(reference), "rows": rows,
              "calibration": calib, "failures": {f"{k[0]} x {k[1]}": v for k, v in failures.items()},
              "num_layers": profiles[reference].num_layers,
              "original_model": original_model_info(ctx, handle, profiles[reference])}

    out = ctx.run_dir / "stage_0_sweep"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "results.json", json.dumps(result, indent=2, default=str))
    _write_xlsx(result, out / "sweep.xlsx")
    atomic_write_text(out / "report.md", _report(result, s0))
    log.info("sweep: %d plans over %d calibration settings -> %s", len(rows), len(profiles), out)
    return {"report": out / "report.md", "xlsx": out / "sweep.xlsx", "json": out / "results.json"}


# --- outputs ---------------------------------------------------------------------------------------------------

def _write_xlsx(result: dict[str, Any], path) -> None:
    wb = Workbook()
    for ws, rows in ((wb.active, result["rows"]), (wb.create_sheet(), result["calibration"])):
        ws.title = "All plans" if rows is result["rows"] else "Calibration"
        if not rows:
            continue
        headers = list(rows[0])
        ws.append(headers)
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in rows:
            ws.append([json.dumps(v) if isinstance(v, list) else v for v in r.values()])
        ws.freeze_panes = "A2"
    tmp = path.with_suffix(".tmp.xlsx")
    wb.save(tmp)
    tmp.replace(path)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("yes" if v is True else "no" if v is False else str(v))  # noqa: E731
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + \
           ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows] + [""]


def _pick(rows: list[dict], **fixed) -> list[dict]:
    return [r for r in rows if all(r[k] == v for k, v in fixed.items())]


def _report(res: dict[str, Any], s0) -> str:
    d, grid, rows, n = res["focus"], res["grid"], res["rows"], res["num_layers"]
    ref = {"calib_dataset": res["reference"][0], "calib_samples": res["reference"][1]}
    base = _pick(rows, **ref, gptq_groupsize=d["gptq_groupsize"])
    uni = base[0]
    L: list[str] = ["# Stage 0 sweep: trying several values of every setting", ""]

    # summary
    by_t = _pick(base, prune_ratio_aggressive=d["prune_ratio_aggressive"])
    wins = [r for r in _pick(rows, **ref) if r["fw_beats_uniform"]]
    best_same = min(base, key=lambda r: r["same_exposure"])
    calib = res["calibration"]
    worst = min(calib, key=lambda c: c["rank_agreement"])
    L += ["## Summary", "",
          f"We tried {len(rows)} combinations of settings and compared each resulting compression plan with the "
          f"standard method (every layer at {s0.uniform_bits} bits, {uni['uniform_gb']:.3f} GB, fragile-parts "
          f"score {uni['uniform_exposure']:.2f}). "
          f"The threshold decides how big the plan gets: from {min(r['fw_gb'] for r in by_t):.2f} GB to "
          f"{max(r['fw_gb'] for r in by_t):.2f} GB across the values tried. "
          + (f"{len(wins)} combinations (counting every group size) are both no bigger than the standard method "
             "and safer for the fragile layers" + (", and every one of them removes numbers. "
             if all(r["prune_ratio_aggressive"] > 0 for r in wins) else ". ") if wins else
             "No combination with the reference calibration is both smaller than the standard method and safer. ")
          + f"At exactly the standard method's size, the safest option removes {best_same['prune_ratio_aggressive']:.0%} "
          f"of the numbers in unprotected layers and protects {best_same['same_protected']} layers (score "
          f"{best_same['same_exposure']:.2f}); without removing anything, {best_same['noprune_protected']} layers "
          f"can be protected (score {best_same['noprune_exposure']:.2f}). "
          + ("Only one calibration setting was profiled, so the effect of the calibration text is not "
             "measured yet. " if len(calib) == 1 else
             f"Changing the calibration text barely changes which layers count as sensitive (lowest agreement "
             f"{worst['rank_agreement']:.2f} out of 1). " if worst["rank_agreement"] >= 0.8 else
             f"The calibration text matters: with {worst['calib_dataset']} x {worst['calib_samples']} the "
             f"ranking agrees only {worst['rank_agreement']:.2f} (out of 1) with the default. ")
          + "Nothing here measures accuracy yet; that needs Stage 1.", ""]

    L += original_model_lines(res.get("original_model", {}))
    L += ["## Key terms", "",
          "- **Threshold**: the cut-off for protecting a layer. Layers are ranked from 0 (least sensitive) to 1 "
          "(most sensitive); every layer at or above the threshold is protected.",
          f"- **Protected layer**: kept at {s0.protected_bits} bits and never trimmed.",
          f"- **Prune ratio**: the share of numbers removed from each unprotected layer, on top of storing it at "
          f"{s0.compressed_bits} bits.",
          "- **Group size**: how many numbers share one scale factor when stored with fewer bits. Smaller groups "
          "are more precise but need more room for the scale factors.",
          "- **Calibration text / samples**: the ordinary text, and how many passages of it, used to measure "
          "each layer's sensitivity.",
          "- **Memory (GB)**: predicted memory to store the compressed model. Lower is better.",
          "- **Fragile-parts score**: 0 to 1, how much of the compression lands on sensitive layers. Lower is "
          "safer. It counts removed numbers as a milder form of rounding, which probably flatters plans that "
          "remove numbers.",
          "- **Same-size plan**: protects as many of the most sensitive layers as fit in the standard method's "
          "memory. The **no-removal** version stores the other layers at "
          f"{s0.no_prune_compressed_bits} bits instead of removing numbers.",
          "- **Rank agreement**: 1 means two calibration settings rank the layers in exactly the same order; 0 "
          "means no relation.", ""]

    # 1. threshold
    L += ["## 1. Threshold", "",
          f"Prune ratio {d['prune_ratio_aggressive']}, group size {_gs(d['gptq_groupsize'])}, calibration "
          f"{ref['calib_dataset']} x {ref['calib_samples']}. Standard method: {uni['uniform_gb']:.3f} GB, score "
          f"{uni['uniform_exposure']:.3f}.", ""]
    L += _table(["Threshold", f"Protected (of {n})", "Memory (GB)", "Bits per number", "Share removed",
                 "Fragile-parts score", "Smaller and safer than standard?"],
                [[r["sensitive_threshold"], r["fw_protected"], r["fw_gb"], r["fw_bits"], r["fw_sparsity"],
                  r["fw_exposure"], r["fw_beats_uniform"]] for r in by_t])

    # 2. prune ratio
    by_p = _pick(base, sensitive_threshold=d["sensitive_threshold"])
    L += ["## 2. Prune ratio", "",
          f"Threshold {d['sensitive_threshold']}; the same-size columns do not depend on the threshold.", ""]
    L += _table(["Prune ratio", "Memory (GB)", "Fragile-parts score", "Same-size: protected",
                 "Same-size: memory (GB)", "Same-size: score"],
                [[r["prune_ratio_aggressive"], r["fw_gb"], r["fw_exposure"], r["same_protected"], r["same_gb"],
                  r["same_exposure"]] for r in by_p])

    # 3. group size
    gs_rows = [_pick(rows, **ref, sensitive_threshold=d["sensitive_threshold"],
                     prune_ratio_aggressive=d["prune_ratio_aggressive"], gptq_groupsize=g)[0]
               for g in grid["gptq_groupsize"]]
    L += ["## 3. Group size", "",
          f"Threshold {d['sensitive_threshold']}, prune ratio {d['prune_ratio_aggressive']}.", ""]
    L += _table(["Group size", "Standard method (GB)", "Framework (GB)", "Same-size: protected",
                 "No-removal: protected", "No-removal: score"],
                [[_gs(r["gptq_groupsize"]), r["uniform_gb"], r["fw_gb"], r["same_protected"],
                  r["noprune_protected"], r["noprune_exposure"]] for r in gs_rows])

    # plans that beat the standard method
    if wins:
        L += ["## Plans that are smaller and safer than the standard method", "",
              f"Calibration {ref['calib_dataset']} x {ref['calib_samples']}, every group size. Each is compared "
              "with the standard method at the same group size.", ""]
        L += _table(["Threshold", "Prune ratio", "Group size", "Protected", "Memory (GB)", "Standard (GB)",
                     "Score", "Standard score"],
                    [[r["sensitive_threshold"], r["prune_ratio_aggressive"], _gs(r["gptq_groupsize"]),
                      r["fw_protected"], r["fw_gb"], r["uniform_gb"], r["fw_exposure"], r["uniform_exposure"]]
                     for r in sorted(wins, key=lambda r: r["fw_exposure"])])

    # 4. calibration
    L += ["## 4. Calibration text and amount", "",
          f"Compared with {ref['calib_dataset']} x {ref['calib_samples']}. \"Same protected\" counts how many of "
          f"the layers protected at threshold {d['sensitive_threshold']} are also protected under the reference.", ""]
    L += _table(["Text", "Passages", "Rank agreement", "Same protected", "Unusual layers", "Profiling time (s)"],
                [[c["calib_dataset"], c["calib_samples"], c["rank_agreement"],
                  f"{c['same_protected']} of {c['protected']}", ", ".join(f"layer {i}" for i in c["outliers"]) or "none",
                  c["profiling_s"]] for c in calib])
    if res["failures"]:
        L += ["Calibration settings that could not be profiled: "
              + "; ".join(f"{k} ({v})" for k, v in res["failures"].items()), ""]
    if len(calib) == 1:
        L += ["Only one calibration setting was profiled in this run, so this section has nothing to compare "
              "yet.", ""]

    # findings
    L += ["## Findings", ""]
    lo = min(by_t, key=lambda r: r["fw_gb"])
    L.append(f"- The threshold is the biggest lever. Lower thresholds protect more layers and are safer, but "
             f"bigger. At {lo['sensitive_threshold']} the plan protects {lo['fw_protected']} layers and needs "
             f"{lo['fw_gb']:.3f} GB against {uni['uniform_gb']:.3f} GB, with a score of {lo['fw_exposure']:.3f} "
             f"against {uni['uniform_exposure']:.3f}.")
    np0 = by_p[0]
    L.append(f"- Removing numbers is what buys room for protection at the same size. With no removal the "
             f"same-size plan can only protect {np0['same_protected']} layers at 4 bits, or "
             f"{np0['noprune_protected']} if the rest drop to {s0.no_prune_compressed_bits} bits.")
    L.append("- Group size moves every plan's memory by the same amount, so it hardly changes which plan wins; "
             "it matters for accuracy in Stage 1.")
    L.append("- The best settings here are the ones to carry into Stage 1, where accuracy is measured; the "
             "fragile-parts score is only a guide.")
    L += ["", f"All {len(rows)} plans are listed in `sweep.xlsx` (sheet *All plans*) and `results.json`.", ""]
    return "\n".join(L)
