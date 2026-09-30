"""report.md output for StageReporter.

The report opens with a plain-language part for readers with no AI background (the short version, what the
stage does, what was compared, what each number means, whether the framework won and why, a glossary) and
keeps the full technical detail below it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sdf.reporting.metrics import METRICS, MetricSpec, is_better, label
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


def _plain_verdicts(rep: "StageReporter") -> list[str]:
    """One plain sentence per method: did the framework beat the standard method, and on what."""
    from sdf.reporting.reporter import VARIANT_PLAIN

    out = []
    for fw in (r for r in rep.rows if r.variant == "framework" and r.status == "ok"):
        orig = rep.find_original(fw)
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
            head = "beat the standard method on every measure compared"
        elif losses and not wins:
            head = "lost to the standard method on every measure compared"
        elif wins:
            head = "is a trade-off against the standard method: better on some measures, worse on others"
        else:
            head = "came out the same as the standard method on the measures compared"
        detail = []
        if wins:
            detail.append("better on " + ", ".join(wins))
        if losses:
            detail.append("worse on " + ", ".join(losses))
        out.append(f"**{fw.plain_name}** {head}" + (" (" + "; ".join(detail) + ")." if detail else "."))
    return out


def _requirement_plain(r: "ComparisonRow") -> str | None:
    met = r.requirement.get("met")
    if met is True:
        return "meets every deployment target that was set"
    if met is False:
        return "misses a deployment target (" + ", ".join(r.requirement.get("shortfall", {})) + ")"
    return None


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _plain_part(rep: "StageReporter") -> list[str]:
    """Plain-language part: summary, key terms, one results table, per-layer table, findings."""
    from sdf.reporting.reporter import VARIANT_PLAIN

    ok = [r for r in rep.rows if r.status == "ok"]
    verdicts = _plain_verdicts(rep)

    # 1. one-paragraph summary
    summary = rep.plain_summary or " ".join(v.replace("**", "") for v in verdicts)
    lines = ["## Summary", "", summary or "No comparison could be made; see the failures below.", ""]

    # 2. key terms: the versions compared, the measures, then stage-specific words
    lines += ["## Key terms", ""]
    seen = set()
    for r in rep.rows:
        if r.plain_name in seen:
            continue
        seen.add(r.plain_name)
        lines.append(f"- **{r.plain_name}**: {_cap(r.info.get('plain_desc') or VARIANT_PLAIN[r.variant][1])}")
    metrics = [m for m in rep.main_metrics if m in METRICS and any(m in r.metrics for r in ok)]
    for m in metrics:
        spec = METRICS[m]
        lines.append(f"- **{_cap(spec.plain or spec.label)}**: {spec.meaning}")
    lines += [f"- **{term}**: {text}" for term, text in rep.glossary.items()]
    lines.append("")

    # 3. every version in one table, units in the headers
    lines += ["## Results", "",
              "All versions were tested under exactly the same conditions (same data, same computer, same "
              "settings), so differences come from the method alone. A dash means the measure is not available "
              "at this stage.", ""]
    headers = ["Version"]
    for m in metrics:
        spec = METRICS[m]
        bits = [spec.unit] if spec.unit else []
        if spec.better:
            bits.append(f"{spec.better} is better")
        headers.append(_cap(spec.plain or spec.label) + (f" ({', '.join(bits)})" if bits else ""))
    table = [[r.plain_name] + [_value(r.metrics.get(m), None) if m in r.metrics else "–" for m in metrics]
             for r in ok]
    lines += [_table(headers, table), ""]

    # 4. per-layer table, when the stage provides one
    if rep.plain_layer_columns and rep.per_layer:
        heading, intro, cols = rep.plain_layer_columns
        lines += [f"## {heading}", "", intro, ""]
        lines += [_table([h for _, h in cols],
                         [[_layer_cell(row.get(k)) for k, _ in cols] for row in rep.per_layer]), ""]

    # 5. findings
    lines += ["## Findings", ""]
    lines += [f"- {v}" for v in verdicts]
    lines += [f"- {w}" for w in rep.plain_why]
    targets = {k: v for k, v in rep.to_dict()["requirement"].items() if k != "hardware_profile"}
    if not any(v is not None for v in targets.values()):
        lines.append("- No deployment targets (maximum size, speed or error) were set for this run, so none "
                     "were checked.")
    else:
        for r in ok:
            req = _requirement_plain(r)
            if req:
                lines.append(f"- {r.plain_name} {req}.")
    failed = [r for r in rep.rows if r.status != "ok"]
    if failed:
        lines.append(f"- {len(failed)} run(s) failed: {', '.join(r.plain_name for r in failed)}. The error "
                     "messages are under *Anomalies and failures* below.")
    lines.append("")

    if rep.plain_intro:
        lines += ["## How this stage works", "", rep.plain_intro, ""]
    return lines


def _layer_cell(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 100 else f"{v:,.0f}"
    return _fmt(v)


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
    headers = ["Row", "Status"] + [label(m) for m in metrics] + ["Requirement met"]
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
