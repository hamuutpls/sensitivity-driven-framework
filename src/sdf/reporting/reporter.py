"""StageReporter: the one reporting module every stage uses.

Each stage adds comparison rows (method x variant), per-layer details and raw measurements; the reporter
computes deltas, checks the DeploymentRequirement, and writes, under <run_dir>/stage_<N>/:

    report.md                  plain-language report
    stage_<N>_comparison.xlsx  Summary / Config / Per-layer / Raw / Charts
    results.json               the same data, machine-readable

results.json is rewritten after every row, so a disconnect loses at most the method in progress.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from sdf.reporting.metrics import METRICS, is_better
from sdf.requirements import DeploymentRequirement
from sdf.utils.cache import atomic_write_text
from sdf.utils.logging import get_logger

log = get_logger(__name__)

# fp16: uncompressed baseline. original: the method used the standard way, no Stage 0 guidance.
# framework: Stage 0 plan + the method.
VARIANTS = ("fp16", "original", "framework")

# Plain-language names and explanations of the three variants, for readers with no AI background.
VARIANT_PLAIN = {
    "fp16": ("Uncompressed model", "the model exactly as published, with nothing removed or simplified. It is "
             "the reference point: the best accuracy we can hope for, and the most memory."),
    "original": ("Standard method", "the compression technique applied the usual way, treating every part of "
                 "the model the same."),
    "framework": ("Sensitivity-guided framework", "our approach: first measure which parts of the model are "
                  "fragile, then compress the robust parts hard and leave the fragile parts mostly intact."),
}

# Terms every report uses. Stages add their own with `glossary`.
BASE_GLOSSARY = {
    "Language model": "A program that has learned from large amounts of text to predict the next word. "
                      "Chat assistants are built on these. The one used here is TinyLlama, a small open model "
                      "with about 1.1 billion numbers inside it.",
    "Compression": "Making the model smaller and faster by storing its numbers with less detail or removing "
                   "some of them, ideally without making its answers worse.",
    "Layer": "The model is a stack of similar building blocks called layers; text passes through them one "
             "after another (TinyLlama has 22).",
    "Bits": "Computers store numbers as strings of 0s and 1s (bits). More bits per number keeps more detail "
            "but takes more space. The uncompressed model uses 16 bits per number.",
    "FP16": "\"16-bit floating point\", the standard precise format the model is published in.",
}


@dataclass
class ComparisonRow:
    method: str
    variant: str
    status: str = "running"  # running | ok | failed
    metrics: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)  # free-form settings for this row
    error: str | None = None
    requirement: dict[str, Any] = field(default_factory=dict)
    deltas: dict[str, dict[str, float | None]] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.method}/{self.variant}"


class StageReporter:
    def __init__(
        self,
        stage: int,
        run_dir: str | Path,
        title: str,
        config: dict[str, Any],
        environment: dict[str, Any],
        conditions: dict[str, Any],
        requirement: DeploymentRequirement,
        main_metrics: list[str],
    ):
        self.stage = stage
        self.title = title
        self.dir = Path(run_dir) / f"stage_{stage}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.environment = environment
        self.conditions = conditions  # the identical conditions every row shares (calibration, seed, eval data...)
        self.requirement = requirement
        self.main_metrics = main_metrics
        self.rows: list[ComparisonRow] = []
        self.per_layer: list[dict[str, Any]] = []
        self.raw: list[dict[str, Any]] = []
        self.findings: list[str] = []  # stage-specific findings; the framework-vs-original ones are generated
        self._auto_findings: list[str] = []
        self.anomalies: list[str] = []
        self.next_steps: list[str] = []
        self.sections: list[tuple[str, str]] = []  # extra technical (heading, markdown) sections
        # Plain-language parts of report.md, for readers with no AI background. Stages fill these in.
        self.plain_intro: str = ""  # what this stage does and why, in everyday words
        self.plain_why: list[str] = []  # why the framework won or lost, in everyday words
        self.glossary: dict[str, str] = dict(BASE_GLOSSARY)
        self.started = time.strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------ collecting

    @contextmanager
    def method(self, method: str, variant: str, **info: Any) -> Iterator[ComparisonRow]:
        """Run one (method, variant) row. An exception marks the row failed and is recorded, not raised,
        so one broken method does not take down the stage."""
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        row = ComparisonRow(method=method, variant=variant, info=dict(info))
        self.rows.append(row)
        log.info("[stage %d] %s: start", self.stage, row.key)
        try:
            yield row
        except Exception as exc:  # noqa: BLE001 - deliberately broad: record and continue
            row.status = "failed"
            row.error = f"{type(exc).__name__}: {exc}"
            row.info["traceback"] = traceback.format_exc()
            self.anomalies.append(f"{row.key} failed: {row.error}")
            log.exception("[stage %d] %s failed", self.stage, row.key)
        else:
            row.status = "ok"
            self._sanity_check(row)
            log.info("[stage %d] %s: done", self.stage, row.key)
        row.requirement = self.requirement.check(row.metrics).to_dict()
        self.flush()

    def add_raw(self, method: str, variant: str, records: list[dict[str, Any]]) -> None:
        self.raw.extend({"method": method, "variant": variant, **r} for r in records)

    def _sanity_check(self, row: ComparisonRow) -> None:
        for name, value in row.metrics.items():
            if isinstance(value, float) and not math.isfinite(value):
                self.anomalies.append(f"{row.key}: {name} is {value}")

    # ------------------------------------------------------------------ analysis

    def _find(self, method: str | None, variant: str) -> ComparisonRow | None:
        for r in self.rows:
            if r.variant == variant and r.status == "ok" and (method is None or r.method == method):
                return r
        return None

    def compute_deltas(self) -> None:
        fp16 = self._find(None, "fp16")
        for row in self.rows:
            row.deltas = {}
            if row.status != "ok":
                continue
            original = self._find(row.method, "original") if row.variant == "framework" else None
            for name, value in row.metrics.items():
                if not _num(value):
                    continue
                d: dict[str, float | None] = {}
                if fp16 is not None and row is not fp16:
                    d.update(_delta(value, fp16.metrics.get(name), "vs_fp16"))
                if original is not None:
                    d.update(_delta(value, original.metrics.get(name), "vs_original"))
                if d:
                    row.deltas[name] = d

    def comparison_summary(self) -> list[str]:
        """Plain-language framework-vs-original findings, one line per method."""
        lines = []
        for fw in (r for r in self.rows if r.variant == "framework" and r.status == "ok"):
            wins, losses = [], []
            for name, d in fw.deltas.items():
                abs_d = d.get("vs_original_abs")
                if abs_d is None:
                    continue
                better = is_better(name, abs_d)
                if better is None:
                    continue
                pct = d.get("vs_original_pct")
                desc = f"{_label(name)} ({_fmt_signed(abs_d)}{'' if pct is None else f', {pct:+.1f}%'})"
                (wins if better else losses).append(desc)
            if not wins and not losses:
                lines.append(f"**{fw.method}**: no original-method row to compare against.")
                continue
            verdict = "beats" if wins and not losses else "trades off against" if wins else "loses to"
            text = f"**{fw.method}**: the framework {verdict} the original method."
            if wins:
                text += " Better on " + "; ".join(wins) + "."
            if losses:
                text += " Worse on " + "; ".join(losses) + "."
            lines.append(text)
        for r in self.rows:
            if r.status == "ok" and r.requirement.get("met") is False:
                gaps = ", ".join(f"{k} over by {v:.4g}" for k, v in r.requirement["shortfall"].items())
                lines.append(f"{r.key} misses the deployment requirement: {gaps}.")
        return lines

    # ------------------------------------------------------------------ output

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "title": self.title,
            "started": self.started,
            "config": self.config,
            "conditions": self.conditions,
            "environment": self.environment,
            "requirement": asdict(self.requirement),
            "rows": [asdict(r) for r in self.rows],
            "per_layer": self.per_layer,
            "raw": self.raw,
            "findings": self.all_findings,
            "anomalies": self.anomalies,
            "next_steps": self.next_steps,
            "sections": [{"heading": h, "body": b} for h, b in self.sections],
            "plain": {"intro": self.plain_intro, "why": self.plain_why, "glossary": self.glossary},
        }

    @property
    def all_findings(self) -> list[str]:
        return self._auto_findings + self.findings

    @property
    def json_path(self) -> Path:
        return self.dir / "results.json"

    @property
    def xlsx_path(self) -> Path:
        return self.dir / f"stage_{self.stage}_comparison.xlsx"

    @property
    def report_path(self) -> Path:
        return self.dir / "report.md"

    def flush(self) -> None:
        """Rewrite results.json with everything collected so far (incremental save)."""
        self.compute_deltas()
        atomic_write_text(self.json_path, json.dumps(self.to_dict(), indent=2, default=_json_default))

    def finalize(self) -> dict[str, Path]:
        from sdf.reporting.excel import write_workbook
        from sdf.reporting.markdown import write_report

        self.compute_deltas()
        self._auto_findings = self.comparison_summary()
        self.flush()
        write_workbook(self, self.xlsx_path)
        write_report(self, self.report_path)
        log.info("[stage %d] wrote %s", self.stage, self.dir)
        return {"report": self.report_path, "xlsx": self.xlsx_path, "json": self.json_path}


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _delta(value: float, ref: Any, prefix: str) -> dict[str, float | None]:
    if not _num(ref):
        return {}
    return {f"{prefix}_abs": value - ref, f"{prefix}_pct": (value - ref) / ref * 100 if ref != 0 else None}


def _label(name: str) -> str:
    spec = METRICS.get(name)
    return spec.label if spec else name


def _fmt_signed(x: float) -> str:
    return f"{x:+.4g}"


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)
