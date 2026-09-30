"""Compare sensitivity scores: profile the same model and calibration text with each score and show how
they rank the layers. Outputs under <run_dir>/stage_0_scores/: report.md, scores.xlsx, results.json."""

from __future__ import annotations

import dataclasses
import functools
import json
from typing import Any, Callable

from openpyxl import Workbook
from openpyxl.styles import Font

from sdf.data import load_texts
from sdf.reporting.markdown import original_model_lines
from sdf.run import RunContext
from sdf.stage0.planner import plan_compression
from sdf.stage0.run import _ModelHandle, load_profile, original_model_info
from sdf.stage0.sensitivity import SCORES, SensitivityProfile, normalize, outlier_layers
from sdf.stage0.sweep import _spearman
from sdf.utils.cache import atomic_write_text
from sdf.utils.env import resolve_device

PLAIN_NAME = {
    "grad_x_weight": "gradient x weight",
    "layer_removal": "remove the layer",
    "layer_quant": "compress only that layer",
}


def compare_scores(ctx: RunContext, candidate: dict[str, Any], scores: tuple[str, ...] = SCORES, model=None,
                   tokenizer=None, text_loader: Callable[[str, str], list[str]] | None = None) -> dict[str, Any]:
    cfg = ctx.cfg
    handle = _ModelHandle(ctx, resolve_device(cfg.model.device), model, tokenizer)
    text_loader = text_loader or functools.partial(load_texts, cfg.data.sources)
    profiles: dict[str, SensitivityProfile] = {}
    for score in scores:
        sub = dataclasses.replace(ctx, cfg=cfg.with_overrides({"stage0.score": score}))
        profiles[score], _ = load_profile(sub, candidate, handle, text_loader)

    t = candidate["sensitive_threshold"]
    ranks = {s: normalize(p.raw_scores) for s, p in profiles.items()}
    protected = {s: set(plan_compression(r, t, 0.0, 8, 4).protected_layers) for s, r in ranks.items()}
    names = list(profiles)
    result = {
        "candidate": candidate, "threshold": t, "scores": names,
        "agreement": {a: {b: _spearman(profiles[a].raw_scores, profiles[b].raw_scores) for b in names} for a in names},
        "overlap": {a: {b: len(protected[a] & protected[b]) for b in names} for a in names},
        "protected": {s: sorted(p) for s, p in protected.items()},
        "outliers": {s: outlier_layers(p.raw_scores) for s, p in profiles.items()},
        "time_s": {s: p.cost.get("wall_clock_s") for s, p in profiles.items()},
        "baseline_ppl": next((p.meta["baseline_ppl"] for p in profiles.values() if "baseline_ppl" in p.meta), None),
        "per_layer": [{"layer": i, **{f"{s}_raw": profiles[s].raw_scores[i] for s in names},
                       **{f"{s}_rank": ranks[s][i] for s in names},
                       **{f"{s}_protected": i in protected[s] for s in names}}
                      for i in range(profiles[names[0]].num_layers)],
    }
    result["original_model"] = original_model_info(ctx, handle, profiles[names[0]])
    out = ctx.run_dir / "stage_0_scores"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "results.json", json.dumps(result, indent=2, default=str))
    wb = Workbook()
    ws = wb.active
    ws.title = "Per-layer"
    ws.append(list(result["per_layer"][0]))
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in result["per_layer"]:
        ws.append(list(r.values()))
    tmp = out / "scores.tmp.xlsx"
    wb.save(tmp)
    tmp.replace(out / "scores.xlsx")
    atomic_write_text(out / "report.md", _report(result, candidate))
    return {"report": out / "report.md", "xlsx": out / "scores.xlsx", "json": out / "results.json"}


def _f(v: Any) -> str:
    return f"{v:.2f}" if isinstance(v, float) else "yes" if v is True else "" if v is False else str(v)


def _report(res: dict[str, Any], cand: dict[str, Any]) -> str:
    names, t = res["scores"], res["threshold"]
    n = len(res["per_layer"])
    k = len(next(iter(res["protected"].values())))
    label = lambda s: PLAIN_NAME.get(s, s)  # noqa: E731
    L = ["# Stage 0: comparing ways to measure layer sensitivity", ""]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    L += ["## Summary", "",
          f"The same model and text ({cand['calib_dataset']}, {cand['calib_samples']} passages) were scored "
          f"{len(names)} ways. At threshold {t} each way protects {k} of {n} layers. "
          + " ".join(f"*{label(a)}* and *{label(b)}* rank the layers with agreement "
                     f"{res['agreement'][a][b]:.2f} (1 = same order) and pick {res['overlap'][a][b]} of the same "
                     f"{k} layers to protect." for a, b in pairs), ""]
    L += original_model_lines(res.get("original_model", {}))
    L += ["## The ways compared", "",
          "- **Gradient x weight**: one pass over the text; for every number, its size times how much the "
          "model's mistakes would change if it were nudged, added up per layer. Fast, but an estimate.",
          "- **Remove the layer**: skip one layer at a time and measure how much the prediction error "
          "(perplexity) on the text rises. Direct, but removing a whole layer is much harsher than "
          "compressing it.",
          "- **Compress only that layer**: compress one layer at a time the way the plan would "
          f"({cand['gptq_groupsize']}-number groups, simple rounding) and measure the rise in perplexity. The "
          "closest to what compression actually does, but the rises are small and so noisier.",
          "- **Agreement**: 1 means both ways put the layers in the same order, 0 means no relation, negative "
          "means roughly opposite.", ""]
    L += ["## Agreement between the ways", "", "| | " + " | ".join(label(s) for s in names) + " |",
          "|" + "---|" * (len(names) + 1)]
    L += [f"| {label(a)} | " + " | ".join(f"{res['agreement'][a][b]:.2f}" for b in names) + " |" for a in names]
    L += ["", f"Protected at threshold {t}:", ""]
    L += [f"- **{label(s)}**: layers {', '.join(map(str, p))}" for s, p in res["protected"].items()]
    L += ["- Unusual layers: " + "; ".join(f"{label(s)}: {', '.join(map(str, o)) or 'none'}"
                                             for s, o in res["outliers"].items()), ""]
    if res["baseline_ppl"]:
        L += [f"Perplexity on the calibration text with nothing changed: {res['baseline_ppl']:.2f}. The "
              "raw scores for the two ablation ways are the rise from that value.", ""]
    L += ["## Every layer", "",
          "Rank runs from 0 (least sensitive) to 1 (most sensitive).", "",
          "| Layer | " + " | ".join(f"{label(s)}: raw | rank | protected" for s in names) + " |",
          "|---|" + "---|---|---|" * len(names)]
    for r in res["per_layer"]:
        L.append(f"| {r['layer']} | " + " | ".join(
            f"{r[s + '_raw']:.4g} | {_f(r[s + '_rank'])} | {_f(r[s + '_protected'])}" for s in names) + " |")
    L += ["", "Time to score all layers: " + ", ".join(f"{label(s)} {v:.1f} s" for s, v in res["time_s"].items()
                                                         if v is not None) + ".", ""]
    return "\n".join(L)
