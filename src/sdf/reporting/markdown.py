"""report.md output for StageReporter.

The report opens with a plain-language part for readers with no AI background (a short summary with one small
verdict table per technique, charts, a glossary, what the stage does, the findings, and the big results table at the
back; every section opens with a "What this shows / What it means" box) and keeps the full technical detail below it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sdf.reporting.metrics import METRICS, MetricSpec, is_backend_ppl, is_number, label
from sdf.utils.cache import atomic_write_text

if TYPE_CHECKING:
    from pathlib import Path

    from sdf.reporting.charts import Chart
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


_HEADS = {"better": "beat the standard method on every measure compared",
          "worse": "lost to the standard method on every measure compared",
          "mixed": "is a trade-off against the standard method: better on some measures, worse on others",
          "even": "showed no clear difference from the standard method on the measures compared"}
_SHORT = {"better": "Better", "worse": "Worse", "mixed": "Trade-off", "even": "No clear difference"}


def _judge(rep: "StageReporter", fw: "ComparisonRow") -> tuple[str, list[str], list[str], list[str]] | None:
    """(better | worse | mixed | even, plain names it wins on, loses on, one-off costs) of a framework row against
    its standard row; None without one."""
    orig = rep.find_original(fw)
    if orig is None:
        return None
    wins, losses, cost = rep.verdict(fw)
    plain = lambda m: METRICS[m].plain or METRICS[m].label  # noqa: E731 - verdict() only judges known metrics
    kind = "mixed" if wins and losses else "better" if wins else "worse" if losses else "even"
    one_off = [f"{plain(m)} ({_value(fw.metrics[m] - orig.metrics[m], METRICS.get(m))} more, paid once)"
               for m in cost]
    return kind, [plain(m) for m in wins], [plain(m) for m in losses], one_off


def _plain_verdicts(rep: "StageReporter") -> list[str]:
    """One plain sentence per method: did the framework beat the standard method, and on what."""
    out = []
    for fw in (r for r in rep.rows if r.variant == "framework" and r.status == "ok"):
        j = _judge(rep, fw)
        if j is None:
            out.append(f"There is no working standard-method result for {fw.method}, "
                       "so the framework could not be compared against it.")
            continue
        kind, wins, losses, one_off = j
        detail = []
        if wins:
            detail.append("better on " + ", ".join(wins))
        if losses:
            detail.append("worse on " + ", ".join(losses))
        text = f"**{fw.plain_name}** {_HEADS[kind]}" + (" (" + "; ".join(detail) + ")." if detail else ".")
        if one_off:
            text += " It costs more to prepare: " + "; ".join(one_off) + "."
        out.append(text)
    return out


def _box(shows: str, means: str) -> list[str]:
    """The "What this shows / What it means" box every section opens with."""
    return [f"> **What this shows:** {shows}", ">", f"> **What it means:** {means}", ""]


def _wxay(r: "ComparisonRow") -> str:
    w, a = r.metrics.get("avg_bits_per_weight"), r.metrics.get("avg_activation_bits")
    return f"W{w if is_number(w) else 16:.3g}A{a if is_number(a) else 16:.3g}"


def _size(rep: "StageReporter", r: "ComparisonRow | None") -> str:
    """A row's bits (Stage 1: WxAy) and memory, for the summary."""
    if r is None:
        return "–"
    memory = "predicted_kv_memory_gb" if rep.stage == 3 else "predicted_weight_memory_gb"
    parts = [_wxay(r)] if rep.stage == 1 else []
    if is_number(r.metrics.get(memory)):
        parts.append(f"{r.metrics[memory]:.3g} GB")
    return ", ".join(parts) or "–"


def _summary(rep: "StageReporter") -> list[str]:
    """Per technique, a small table: each Our method version's size against the standard one's, and the verdict."""
    groups: dict[str, list[list[str]]] = {}
    counts = dict.fromkeys(_SHORT, 0)
    for fw in (r for r in rep.rows if r.variant == "framework" and r.status == "ok"):
        j = _judge(rep, fw)
        if j is None:
            continue
        kind, wins, losses, _ = j
        counts[kind] += 1
        t = rep.technique(fw)
        groups.setdefault(t, []).append([fw.plain_name.removeprefix(t + ": "),
                                         f"{_size(rep, fw)} vs {_size(rep, rep.find_original(fw))}",
                                         _SHORT[kind], ", ".join(wins) or "–", ", ".join(losses) or "–"])
    if not groups:
        return []
    n = sum(counts.values())
    backends = [m for m in rep.main_metrics if is_backend_ppl(m)]
    lines = _box(
        "For each technique and each version of our method, whether it beat the standard method on the measures "
        "that decide it (accuracy measured in Hugging Face, memory, speed). "
        + ("Backend perplexity (" + ", ".join(METRICS[m].label.split(" perplexity")[0]
                                              for m in backends if m.endswith("_ppl_val") and m in METRICS)
           + ") is not part of the verdict: those programs re-round the saved file, so it is only shown in the "
             "Results table. " if backends else "")
        + "A larger size is shown but never counted against our method.",
        f"Our method beat the standard method in {counts['better']} of {n} comparisons, lost in {counts['worse']}, "
        f"was a trade-off in {counts['mixed']} and showed no clear difference in {counts['even']}.")
    head = ["Version", "Bits (WxAy) and memory, ours vs standard" if rep.stage == 1 else "Memory, ours vs standard",
            "Result", "Better on", "Worse on"]
    if max(map(len, groups.values())) <= 3:  # a version or two per technique: one table, a line per technique
        lines += [_table(["Technique"] + head, [[t] + row for t, table in groups.items() for row in table]), ""]
    else:
        for t, table in groups.items():
            lines += [f"**{t}**", "", _table(head, table), ""]
    if rep.stage == 1:
        lines += ["WxAy: average bits per weight over the whole model (the input and output word tables stay at 16 "
                  "bits, so a 4-bit model reads about W5.7) and average bits per number passed between layers; A16 "
                  "means those numbers are not rounded.", ""]
    return lines


def requirement_cell(req: dict[str, Any]) -> str:
    """Table cell for a row's requirement check."""
    if not req.get("targets_set", True):
        return "no targets set"
    met = req.get("met")
    return "n/a (not all targets measurable yet)" if met is None else "yes" if met else "no"


def _requirement_plain(r: "ComparisonRow") -> str | None:
    met = r.requirement.get("met")
    if met is True:
        return "meets every deployment target that was set"
    if met is False:
        return "misses a deployment target (" + ", ".join(r.requirement.get("shortfall", {})) + ")"
    return None


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _plain_part(rep: "StageReporter", charts: list["Chart"]) -> list[str]:
    """Plain-language part: a short summary, charts, key terms, then the long findings and the big results table at
    the back (owner's rule: big tables go at the back)."""
    from sdf.reporting.charts import condition
    from sdf.reporting.reporter import VARIANT_PLAIN

    ok = [r for r in rep.rows if r.status == "ok"]

    # 1. summary: the stage's own paragraph when it writes one (Stage 0), else per technique a small verdict table
    lines = ["## Summary", ""]
    verdicts = [] if rep.plain_summary else _summary(rep)
    lines += [rep.plain_summary, ""] if rep.plain_summary else verdicts
    if not rep.plain_summary and not verdicts:
        lines += [("No comparison could be made; see the failures below." if any(r.status != "ok" for r in rep.rows)
                   else "This report has no standard-versus-our-method comparison; the results are in the tables "
                        "below."), ""]

    # 2. charts
    if charts:
        lines += ["## Charts", ""]
        for c in charts:
            lines += [f"![{c.path.stem.replace('_', ' ')}]({c.path.name})", ""] + _box(c.shows, c.means)
        if not any(c.path.stem.endswith("speed") for c in charts):
            lines += ["There is no speed chart: every version here is simulated (kept in the uncompressed format), or "
                      "has no speed measurement, so its speed would not be that of a real compressed model.", ""]

    # 3. key terms: the versions compared, the measures, then stage-specific words
    lines += ["## Key terms", ""] + _box("The words this report uses, in everyday language.",
                                         "Read these first if a name in the tables is unfamiliar.")
    seen = set()
    for r in rep.rows:  # one entry per version (Standard method, Our method...), not per pruning level
        if condition(r) in seen or r.info.get("key_term") is False:  # rows explained by a glossary entry
            continue
        seen.add(condition(r))
        lines.append(f"- **{condition(r)}**: {_cap(r.info.get('plain_desc') or VARIANT_PLAIN[r.variant][1])}")
    metrics = [m for m in rep.main_metrics if m in METRICS and any(m in r.metrics for r in ok)]
    lines += [f"- **{term}**: {text}" for term, text in rep.glossary.items()]
    lines.append("")

    model = original_model_lines(rep.original_model)
    lines += model[:2] + _box("The uncompressed model every version starts from.",
                              "Every size and accuracy below is measured against this reference.") + model[2:]
    if rep.plain_intro:
        lines += ["## How this stage works", ""] + _box("What this stage changes in the model, and how the standard "
                                                        "method and our method differ.",
                                                        "The comparisons above only differ in that change.")
        lines += [rep.plain_intro, ""]

    # 4. per-layer table, when the stage provides one
    if rep.plain_layer_columns and rep.per_layer:
        heading, intro, cols = rep.plain_layer_columns
        lines += [f"## {heading}", ""] + _box(intro, "Each column is explained under the table.")
        lines += [_table([c[1] for c in cols],
                         [[_layer_cell(row.get(c[0])) for c in cols] for row in rep.per_layer]), ""]
        lines += _column_notes([(c[1], c[2]) for c in cols if len(c) > 2])

    # 5. findings, one per version
    lines += ["## Findings", ""] + _box("Every comparison in a sentence, with the reasons and caveats the stage noted.",
                                        "The summary at the top is these sentences in short.")
    lines += [f"- {v}" for v in _plain_verdicts(rep)]
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

    # 6. every version in one table, units in the headers
    lines += ["## Results", ""] + _box(
        "Every version's numbers, measured under exactly the same conditions (same data, same computer, same "
        "settings), so differences come from the method alone. A dash means the measure is not available at this "
        "stage.",
        "The summary and findings above are read from this table."
        + (" A backend's perplexity column is that program's re-rounded file and is not part of the verdict."
           if any(is_backend_ppl(m) for m in metrics) else ""))
    headers = ["Version"] + [_headline(METRICS[m], m) for m in metrics]
    table = [[r.plain_name] + [_value(r.metrics.get(m), None) if m in r.metrics else "–" for m in metrics]
             for r in ok]
    lines += [_table(headers, table), ""]
    lines += _column_notes([(headers[0], "Which version of the model the row describes (see Key terms above).")]
                           + [(_cap(METRICS[m].plain or METRICS[m].label), METRICS[m].meaning) for m in metrics])
    return lines


def _model_value(key: str, v: Any) -> str:
    if key == "num_parameters":
        return f"{v:,} ({v / 1e9:.2f} billion)" if v >= 1e8 else f"{v:,}"
    if key == "fp16_size_gb":
        return f"{v:.2f} GB"
    if key == "bits_per_parameter":
        return f"{v} bits"
    if key == "max_context":
        return f"{v:,} tokens"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return _fmt(v)


def original_model_lines(info: dict[str, Any]) -> list[str]:
    """"Original model" section: what the uncompressed model looks like, so every number has a reference
    point. Used by every report writer."""
    from sdf.utils.model_info import MODEL_FACTS

    lines = ["## Original model", ""]
    if not info:
        return lines + ["The original model's parameters were not recorded for this run.", ""]
    lines += ["Every version in this report starts from this model, unchanged as published.", ""]
    rows = [[MODEL_FACTS.get(k, (k, ""))[0], _model_value(k, v), MODEL_FACTS.get(k, ("", ""))[1]]
            for k, v in info.items()]
    return lines + [_table(["What", "Value", "What it means"], rows), ""]


def _column_notes(notes: list[tuple[str, str]]) -> list[str]:
    """"What each column means" list under a table, so every column is explained where it is read."""
    notes = [(h, m) for h, m in notes if m]
    if not notes:
        return []
    return ["**What each column means**", ""] + [f"- **{h}**: {m}" for h, m in notes] + [""]


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

    if rep.original_model:
        lines += ["", "Original model (from its config):", ""]
        lines += [f"- **{k}**: {_fmt(v)}" for k, v in rep.original_model.items()]

    lines += ["", "## Environment", ""]
    lines += [f"- **{k}**: {_fmt(v)}" for k, v in rep.environment.items()]

    lines += ["", "## Results", ""]
    metrics = [m for m in rep.main_metrics if any(m in r.metrics for r in rep.rows)]
    headers = ["Row", "Status"] + [label(m) for m in metrics] + ["Requirement met"]
    rows = []
    for r in rep.rows:
        rows.append([f"`{r.key}`", r.status] + [r.metrics.get(m) for m in metrics] + [requirement_cell(r.requirement)])
    lines += [_table(headers, rows), ""]
    lines += _column_notes(
        [("Row", "Method and version, as `method/variant`: `fp16` is the uncompressed model, `original` the method "
                 "used the standard way, `framework` the method guided by Stage 0."),
         ("Status", "`ok` if the row finished, `failed` if it raised an error (the error is under Anomalies).")]
        + [(label(m), _technical_note(m)) for m in metrics]
        + [("Requirement met", "Whether the row meets every deployment target that was set; `no targets set` "
                               "when the run set none, `n/a` when a target cannot be measured at this stage.")])
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


def _headline(spec: MetricSpec, metric: str = "") -> str:
    """Plain name with "(unit, lower is better)"; a backend's perplexity is labelled as its re-rounded file."""
    bits = [spec.unit] if spec.unit else []
    if spec.better:
        bits.append(f"{spec.better} is better")
    if is_backend_ppl(metric):
        bits.append(f"{spec.label.split(' perplexity')[0]}'s re-rounded file, not in the verdict")
    return _cap(spec.plain or spec.label) + (f" ({', '.join(bits)})" if bits else "")


def _technical_note(m: str) -> str:
    spec = METRICS.get(m)
    return "" if spec is None else _headline(spec) + "."


def write_report(rep: "StageReporter", path: "Path", charts: list["Chart"] = ()) -> None:
    lines = [f"# Stage {rep.stage}: {rep.title}", "", f"Run started {rep.started}.", ""]
    lines += _plain_part(rep, list(charts))
    lines += _technical_part(rep)
    atomic_write_text(path, "\n".join(lines) + "\n")
