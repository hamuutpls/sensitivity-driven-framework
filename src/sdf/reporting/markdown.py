"""report.md output for StageReporter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sdf.reporting.metrics import METRICS
from sdf.utils.cache import atomic_write_text

if TYPE_CHECKING:
    from pathlib import Path

    from sdf.reporting.reporter import StageReporter


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


def write_report(rep: "StageReporter", path: "Path") -> None:
    lines = [f"# Stage {rep.stage} — {rep.title}", "", f"Started {rep.started}. Output: `{rep.dir}`.", ""]

    lines += ["## What was run", ""]
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

    atomic_write_text(path, "\n".join(lines) + "\n")
