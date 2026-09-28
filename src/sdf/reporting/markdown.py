"""report.md output for StageReporter.

The report opens with a plain-language part for readers with no AI background (the short version, what the
stage does, what was compared, what each number means, whether the framework won and why, a glossary) and
keeps the full technical detail below it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sdf.reporting.metrics import METRICS, MetricSpec, is_better
from sdf.utils.cache import atomic_write_text

if TYPE_CHECKING:
    from pathlib import Path

    from sdf.reporting.reporter import ComparisonRow, StageReporter


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _value(v: Any, spec: MetricSpec | None) -> str:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return _fmt(v)
    text = f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}" if abs(v) < 1 else f"{v:.2f}".rstrip("0").rstrip(".")
    unit = spec.unit if spec else ""
    return f"{text} {unit}".strip()


def _compare(v: float, ref: float, metric: str, ref_name: str) -> str | None:
    """'22% more than the standard method, which is worse' style sentence fragment."""
    if ref == 0:  # a percentage of zero means nothing
        return f"the same as the {ref_name}" if v == 0 else None
    pct = (v - ref) / abs(ref) * 100
    if abs(pct) < 0.5:
        return f"about the same as the {ref_name}"
    direction = "more" if pct > 0 else "less"
    better = is_better(metric, v - ref)
    verdict = "" if better is None else (", which is better" if better else ", which is worse")
    return f"{abs(pct):.0f}% {direction} than the {ref_name}{verdict}"


def _plain_verdicts(rep: "StageReporter") -> list[str]:
    """One plain sentence per method: did the framework beat the standard method, and on what."""
    from sdf.reporting.reporter import VARIANT_PLAIN

    out = []
    for fw in (r for r in rep.rows if r.variant == "framework" and r.status == "ok"):
        orig = next((r for r in rep.rows if r.variant == "original" and r.method == fw.method
                     and r.status == "ok"), None)
        if orig is None:
            out.append(f"There is no working {VARIANT_PLAIN['original'][0].lower()} result for {fw.method}, "
                       "so the framework could not be compared against it.")
            continue
        wins, losses = [], []
        for m in rep.main_metrics:
            spec = METRICS.get(m)
            a, b = fw.metrics.get(m), orig.metrics.get(m)
            if spec is None or not all(isinstance(x, (int, float)) for x in (a, b)):
                continue
            better = is_better(m, a - b)
            if better is not None:
                (wins if better else losses).append(spec.plain or spec.label)
        if wins and not losses:
            head = "The sensitivity-guided framework beat the standard method on every measure compared"
        elif losses and not wins:
            head = "The standard method beat the sensitivity-guided framework on every measure compared"
        elif wins:
            head = "It is a trade-off: the framework did better on some measures and worse on others"
        else:
            head = "The two came out the same on the measures compared"
        detail = []
        if wins:
            detail.append("the framework was better on " + ", ".join(wins))
        if losses:
            detail.append("worse on " + ", ".join(losses))
        out.append(head + (" (" + "; ".join(detail) + ")." if detail else "."))
    return out


def _requirement_plain(r: "ComparisonRow") -> str | None:
    met = r.requirement.get("met")
    if met is True:
        return "meets every deployment target that was set"
    if met is False:
        return "misses a deployment target (" + ", ".join(r.requirement.get("shortfall", {})) + ")"
    return None


def _plain_part(rep: "StageReporter") -> list[str]:
    from sdf.reporting.reporter import VARIANT_PLAIN

    lines = ["## The short version", ""]
    lines += [f"- {v}" for v in _plain_verdicts(rep)] or ["- No comparison could be made (see failures below)."]
    failed = [r for r in rep.rows if r.status != "ok"]
    if failed:
        lines.append(f"- {len(failed)} run(s) failed and are listed under *Anomalies and failures*.")
    lines.append("")

    if rep.plain_intro:
        lines += ["## What this stage does", "", rep.plain_intro, ""]

    lines += ["## What was compared", "",
              "Every version below was tested under exactly the same conditions (same data, same computer, "
              "same settings), so differences come from the method alone.", ""]
    present = [v for v in VARIANT_PLAIN if any(r.variant == v for r in rep.rows)]
    lines += [f"- **{VARIANT_PLAIN[v][0]}**: {VARIANT_PLAIN[v][1]}" for v in present]
    lines.append("")

    lines += ["## What the numbers mean", ""]
    ok = [r for r in rep.rows if r.status == "ok"]
    by_variant = {v: next((r for r in ok if r.variant == v), None) for v in VARIANT_PLAIN}
    for m in rep.main_metrics:
        spec = METRICS.get(m)
        rows = [r for r in ok if isinstance(r.metrics.get(m), (int, float))]
        if spec is None or not rows:
            continue
        lines += [f"### {(spec.plain or spec.label)[:1].upper() + (spec.plain or spec.label)[1:]}", "",
                  spec.meaning, ""]
        for r in rows:
            name = VARIANT_PLAIN[r.variant][0]
            text = f"- {name}: **{_value(r.metrics[m], spec)}**"
            refs = []
            orig = by_variant.get("original")
            fp16 = by_variant.get("fp16")
            if r.variant == "framework" and orig is not None and isinstance(orig.metrics.get(m), (int, float)):
                refs.append(_compare(r.metrics[m], orig.metrics[m], m, "standard method"))
            if r.variant != "fp16" and fp16 is not None and isinstance(fp16.metrics.get(m), (int, float)):
                refs.append(_compare(r.metrics[m], fp16.metrics[m], m, "uncompressed model"))
            refs = [x for x in refs if x]
            if refs:
                text += " (" + "; ".join(refs) + ")"
            lines.append(text)
        not_measured = [VARIANT_PLAIN[r.variant][0] for r in ok if m not in r.metrics]
        if not_measured:
            lines.append(f"- Not measured at this stage for: {', '.join(not_measured)}.")
        lines.append("")

    lines += ["## Why the framework won or lost", ""]
    lines += [f"- {w}" for w in rep.plain_why] or ["- See the numbers above."]
    targets = {k: v for k, v in rep.to_dict()["requirement"].items() if k != "hardware_profile"}
    if not any(v is not None for v in targets.values()):
        lines.append("- No deployment targets (maximum size, speed or error) were set for this run, so none "
                     "were checked.")
    else:
        for r in ok:
            req = _requirement_plain(r)
            if req:
                lines.append(f"- {VARIANT_PLAIN[r.variant][0]} {req}.")
    lines.append("")

    lines += ["## Words used in this report", ""]
    lines += [f"- **{term}**: {text}" for term, text in rep.glossary.items()]
    lines.append("")
    return lines


def _technical_part(rep: "StageReporter") -> list[str]:
    lines = ["---", "", "# Technical details", "", "## What was run", ""]
    lines += [f"- `{r.key}`" + (f": {r.info.get('description')}" if r.info.get("description") else "")
              for r in rep.rows]
    lines += ["", "Every row shares these conditions:", ""]
    lines += [f"- **{k}**: {_fmt(v)}" for k, v in rep.conditions.items()]

    lines += ["", "## Environment", ""]
    lines += [f"- **{k}**: {_fmt(v)}" for k, v in rep.environment.items()]

    lines += ["", "## Results", ""]
    metrics = [m for m in rep.main_metrics if any(m in r.metrics for r in rep.rows)]
    headers = ["Row", "Status"] + [METRICS[m].label if m in METRICS else m for m in metrics] + ["Requirement met"]
    rows = []
    for r in rep.rows:
        met = r.requirement.get("met")
        rows.append([f"`{r.key}`", r.status] + [r.metrics.get(m) for m in metrics]
                    + ["n/a (not all targets measurable yet)" if met is None else met])
    lines += [_table(headers, rows), ""]
    lines += [f"Full metrics, deltas and raw measurements are in `{rep.xlsx_path.name}` and `results.json`.", ""]

    lines += ["## Key findings", ""]
    lines += [f"- {f}" for f in rep.all_findings] or ["- No findings."]

    for heading, body in rep.sections:
        lines += ["", f"## {heading}", "", body]

    lines += ["", "## Anomalies and failures", ""]
    lines += [f"- {a}" for a in rep.anomalies] or ["- None."]

    lines += ["", "## Suggested next steps", ""]
    lines += [f"- {s}" for s in rep.next_steps] or ["- None."]
    return lines


def write_report(rep: "StageReporter", path: "Path") -> None:
    lines = [f"# Stage {rep.stage}: {rep.title}", "", f"Run started {rep.started}.", ""]
    lines += _plain_part(rep)
    lines += _technical_part(rep)
    atomic_write_text(path, "\n".join(lines) + "\n")
