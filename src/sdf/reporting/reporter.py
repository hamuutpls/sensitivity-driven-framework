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
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator

from sdf.reporting.metrics import NOT_IN_VERDICT, ONE_OFF_COSTS, SIZE_METRICS, is_better, is_number, label
from sdf.requirements import DeploymentRequirement
from sdf.utils.cache import atomic_write_text
from sdf.utils.logging import get_logger

log = get_logger(__name__)

# fp16: uncompressed baseline. original: the method used the standard way, no Stage 0 guidance.
# framework: Stage 0 plan + the method.
VARIANTS = ("fp16", "original", "framework")

# Plain-language names and explanations of the three variants, for readers with no AI background.
VARIANT_PLAIN = {
    "fp16": ("Original model (uncompressed)", "the model exactly as published, with nothing removed or simplified. It is "
             "the reference point: the best accuracy we can hope for, and the most memory."),
    "original": ("Standard method", "the compression technique applied the usual way, treating every part of "
                 "the model the same."),
    "framework": ("Our method", "our approach: first measure which parts of the model are "
                  "fragile, then compress the robust parts hard and leave the fragile parts mostly intact."),
}

W8A8 = "_w8a8"  # sdf.stages.methods.W8A8: a Stage 1 standard row with every layer at W8A8


def standard_name(stage: int, prose: bool = False) -> str:
    """The standard row's name: "Standard quantization" in Stage 1 and for Stage 0's unlabelled (weight plan) row,
    "Standard method" elsewhere (pruning and KV cache are not quantization). `prose`: the lower-case words the
    report text uses, where Stage 0 says "standard method" (its text also covers activation and KV rows)."""
    quant = stage == 1 or (stage == 0 and not prose)
    name = "Standard quantization" if quant else "Standard method"
    return name.lower() if prose else name

# Terms every report uses. Stages add their own with `glossary`.
BASE_GLOSSARY = {
    "Language model": "A program that has learned from large amounts of text to predict the next word. "
                      "Chat assistants are built on these. The one used here is described under "
                      "\"Original model\".",
    "Compression": "Making the model smaller and faster by storing its numbers with less detail or removing "
                   "some of them, ideally without making its answers worse.",
    "Layer": "The model is a stack of similar building blocks called layers; text passes through them one "
             "after another.",
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
    # framework rows: main metrics won / lost / one-off costs against the original row (StageReporter.verdict), for
    # the sheets
    verdict: dict[str, list[str]] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.method}/{self.variant}"

    @property
    def compare_to(self) -> str:
        """Method whose "original" row this row is compared against (itself unless info["compare_to"])."""
        return self.info.get("compare_to", self.method)

    @property
    def plain_name(self) -> str:
        return self.info.get("label") or VARIANT_PLAIN[self.variant][0]


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
        original_model: dict[str, Any] | None = None,
        subdir: str | None = None,
    ):
        self.stage = stage
        self.title = title
        self.dir = Path(run_dir) / (subdir or f"stage_{stage}")  # subdir: a side study next to stage_<N>/
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.environment = environment
        self.conditions = conditions  # the identical conditions every row shares (calibration, seed, eval data...)
        self.requirement = requirement
        self.main_metrics = main_metrics
        # The uncompressed model's parameters (sdf.utils.model_info.describe_model), shown in every report.
        self.original_model: dict[str, Any] = dict(original_model or {})
        self.rows: list[ComparisonRow] = []
        self.per_layer: list[dict[str, Any]] = []
        self.raw: list[dict[str, Any]] = []
        self._auto_findings: list[str] = []
        self.anomalies: list[str] = []
        self.next_steps: list[str] = []
        self.sections: list[tuple[str, str]] = []  # extra technical (heading, markdown) sections
        # Plain-language parts of report.md, for readers with no AI background. Stages fill these in.
        self.plain_summary: str = ""  # one paragraph: what was done, what came out, what it means
        self.plain_intro: str = ""  # what this stage does and why, in everyday words
        self.plain_why: list[str] = []  # findings: why the framework won or lost, caveats, in everyday words
        # Optional plain per-layer table: (heading, intro sentence, [(per_layer key, column header, meaning), ...]);
        # the meaning is listed under the table.
        self.plain_layer_columns: tuple[str, str, list[tuple[str, ...]]] | None = None
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
        self._name_standard(row)
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

    def _name_standard(self, row: ComparisonRow) -> None:
        if row.variant == "original" and self.stage in (0, 1):
            row.info.setdefault("label", standard_name(self.stage)
                                + (", every layer W8A8" if row.method.endswith(W8A8) else ""))

    def technique(self, row: ComparisonRow) -> str:
        """The technique a row belongs to, in words: a pruning level row's label before ":", else its (or its
        standard row's) description before ", standard settings" / ", Stage 0 ", else the method name."""
        if row.variant == "fp16":
            return VARIANT_PLAIN["fp16"][0]
        for r in (row, self.find_original(row)):
            if r is None:
                continue
            if r.info.get("kind_label") and ": " in r.info.get("label", ""):
                return r.info["label"].split(": ")[0]
            desc = r.info.get("description", "")
            for cut in (", standard settings", ", Stage 0 "):
                if cut in desc:
                    return desc.split(cut)[0].removesuffix(", every layer W8A8")
        return row.compare_to

    def add_raw(self, method: str, variant: str, records: list[dict[str, Any]]) -> None:
        self.raw.extend({"method": method, "variant": variant, **r} for r in records)

    def _sanity_check(self, row: ComparisonRow) -> None:
        for name, value in row.metrics.items():
            if isinstance(value, float) and not math.isfinite(value):
                self.anomalies.append(f"{row.key}: {name} is {value}")

    # ------------------------------------------------------------------ analysis

    def find_original(self, row: ComparisonRow) -> ComparisonRow | None:
        return self._find(row.compare_to, "original")

    def _find(self, method: str | None, variant: str) -> ComparisonRow | None:
        for r in self.rows:
            if r.variant == variant and r.status == "ok" and (method is None or r.method == method):
                return r
        return None

    def compute_deltas(self) -> None:
        fp16 = self._find(None, "fp16")
        for row in self.rows:
            row.deltas, row.verdict = {}, {}
            if row.status != "ok":
                continue
            original = self._find(row.compare_to, "original") if row.variant == "framework" else None
            if original is not None:
                wins, losses, one_off = self.verdict(row)
                row.verdict = {"wins": wins, "losses": losses, "one_off": one_off}
            for name, value in row.metrics.items():
                if not is_number(value):
                    continue
                d: dict[str, float | None] = {}
                if fp16 is not None and row is not fp16:
                    d.update(_delta(value, fp16.metrics.get(name), "vs_fp16"))
                if original is not None:
                    d.update(_delta(value, original.metrics.get(name), "vs_original"))
                if d:
                    row.deltas[name] = d

    def verdict(self, fw: ComparisonRow) -> tuple[list[str], list[str], list[str]] | None:
        """Main metrics on which framework row `fw` beats, loses to, or (one-off costs) costs more than its
        original row; None when there is no original row. The one judgement behind both the technical findings
        and the plain summary. Perplexity counts once (see below); a backend's perplexity is not counted
        (NOT_IN_VERDICT). A difference within the larger of the two rows' std (a repeated timing's `<name>_std`) is
        a tie; HF timings of two rows with different `latency_mode` are not compared. See judge() for stored size."""
        orig = self.find_original(fw)
        if orig is None:
            return None
        wins, losses, one_off = [], [], []
        for m in (m for m in self.main_metrics if m not in NOT_IN_VERDICT):
            if judged := self.judge(fw, orig, m):
                {"win": wins, "loss": losses, "one_off": one_off}[judged].append(m)
        # validation and held-out perplexity are one measure on two halves of the data: count it once, and only
        # when both halves agree (a disagreement is a tie)
        if "ppl_val" in wins + losses and "ppl_heldout" in wins + losses:
            agree = ("ppl_val" in wins) == ("ppl_heldout" in wins)
            wins, losses = ([m for m in ms if m != "ppl_heldout" and (agree or m != "ppl_val")]
                            for ms in (wins, losses))
        return wins, losses, one_off

    def judge(self, fw: ComparisonRow, orig: ComparisonRow, m: str) -> str | None:
        """One metric of `fw` against its original row `orig`: "win", "loss", "one_off" (a one-off cost that is
        higher) or None (missing, not comparable, or inside the measurement noise). Perplexity's two halves are
        each judged here; verdict() counts them once."""
        a, b = fw.metrics.get(m), orig.metrics.get(m)
        if not (is_number(a) and is_number(b)):
            return None
        # eager timing is ~10x slower than CUDA graph replay: rows timed differently are not compared. A row
        # saved before the mode was recorded has none; it was most likely timed with the default CUDA graphs,
        # so it is compared with a CUDA graph row (or another row without a mode) but not with an eager one
        modes = (fw.metrics.get("latency_mode"), orig.metrics.get("latency_mode"))
        if m.startswith(("prefill_", "decode_")) and modes[0] != modes[1] and (None not in modes or "eager" in modes):
            return None
        better = is_better(m, a - b)
        if better is None:
            return None
        spread = [x for x in (fw.metrics.get(m.removesuffix("_mean") + "_std"),
                              orig.metrics.get(m.removesuffix("_mean") + "_std")) if is_number(x)]
        if spread and abs(a - b) <= max(spread):
            return None  # inside the measurement noise
        if m in ONE_OFF_COSTS:
            return None if better else "one_off"
        if m in SIZE_METRICS and not better:
            return None  # a larger stored size is never a loss (owner's rule); a smaller one is still a win
        return "win" if better else "loss"

    def comparison_summary(self) -> list[str]:
        """Framework-vs-original findings, one line per method, with the size of each difference."""
        lines = []
        for fw in (r for r in self.rows if r.variant == "framework" and r.status == "ok"):
            v = self.verdict(fw)
            if v is None:
                lines.append(f"**{fw.method}**: no original-method row to compare against.")
                continue
            if not any("vs_original_abs" in fw.deltas.get(m, {}) for m in self.main_metrics if m not in NOT_IN_VERDICT):
                lines.append(f"**{fw.method}**: no measure could be compared with the original method.")
                continue
            wins, losses, one_off = ([self._delta_text(fw, m) for m in ms] for ms in v)
            verdict = ("beats" if wins and not losses else "trades off against" if wins and losses
                       else "loses to" if losses else "ties with")
            text = f"**{fw.method}**: the framework {verdict} the original method."
            if wins:
                text += " Better on " + "; ".join(wins) + "."
            if losses:
                text += " Worse on " + "; ".join(losses) + "."
            if one_off:
                text += " One-off cost: " + "; ".join(one_off) + "."
            lines.append(text)
        for r in self.rows:
            if r.status == "ok" and r.requirement.get("met") is False:
                gaps = ", ".join(f"{k} over by {v:.4g}" for k, v in r.requirement["shortfall"].items())
                lines.append(f"{r.key} misses the deployment requirement: {gaps}.")
        return lines

    @staticmethod
    def _delta_text(row: ComparisonRow, m: str) -> str:
        d = row.deltas.get(m, {})
        abs_d, pct = d.get("vs_original_abs"), d.get("vs_original_pct")
        return label(m) + ("" if abs_d is None else f" ({abs_d:+.4g}{'' if pct is None else f', {pct:+.1f}%'})")

    # ------------------------------------------------------------------ output

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "title": self.title,
            "main_metrics": self.main_metrics,
            "started": self.started,
            "config": self.config,
            "conditions": self.conditions,
            "original_model": self.original_model,
            "environment": self.environment,
            "requirement": asdict(self.requirement),
            "rows": [asdict(r) for r in self.rows],
            "per_layer": self.per_layer,
            "raw": self.raw,
            "findings": self.all_findings,
            "anomalies": self.anomalies,
            "next_steps": self.next_steps,
            "sections": [{"heading": h, "body": b} for h, b in self.sections],
            "plain": {"summary": self.plain_summary, "intro": self.plain_intro, "findings": self.plain_why,
                      "glossary": self.glossary},
        }

    @property
    def all_findings(self) -> list[str]:
        return self._auto_findings

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

    def analyse(self) -> None:
        """Deltas and findings from the rows as they are."""
        self.compute_deltas()
        self._auto_findings = self.comparison_summary()

    def finalize(self) -> dict[str, Path]:
        self.analyse()
        self.flush()
        return {**self.write_outputs(), "json": self.json_path}

    def write_outputs(self) -> dict[str, Path]:
        """report.md, the workbook and the charts from the rows as they are; results.json is left alone."""
        from sdf.reporting.charts import write_charts
        from sdf.reporting.excel import write_workbook
        from sdf.reporting.markdown import write_report

        write_workbook(self, self.xlsx_path)
        write_report(self, self.report_path)  # first without charts, so a chart error never costs the report
        try:
            charts = write_charts(self)
        except Exception:  # noqa: BLE001 - charts are optional; never abort a stage (or a search point) for them
            log.warning("[stage %d] charts failed, the report has none", self.stage, exc_info=True)
            charts = []
        if charts:
            write_report(self, self.report_path, charts)
        log.info("[stage %d] wrote %s", self.stage, self.dir)
        return {"report": self.report_path, "xlsx": self.xlsx_path, **{c.path.stem: c.path for c in charts}}

    @classmethod
    def load(cls, json_path: str | Path, out_dir: str | Path | None = None) -> "StageReporter":
        """A reporter holding a saved results.json, writing to `out_dir` (default: the file's folder). main_metrics
        is empty when the file predates saving them (sdf.reporting.rebuild fills it in)."""
        d = json.loads(Path(json_path).read_text(encoding="utf-8"))
        folder = Path(out_dir or Path(json_path).parent)
        rep = cls(d["stage"], folder.parent, d.get("title", ""), d.get("config", {}), d.get("environment", {}),
                  d.get("conditions", {}), DeploymentRequirement(**_known(DeploymentRequirement, d.get("requirement"))),
                  d.get("main_metrics", []), d.get("original_model"), subdir=folder.name)
        rep.started = d.get("started", "")
        rep.rows = [ComparisonRow(**_known(ComparisonRow, r)) for r in d["rows"]]
        for r in rep.rows:
            rep._name_standard(r)
        rep.per_layer, rep.raw = d.get("per_layer", []), d.get("raw", [])
        rep.anomalies, rep.next_steps = d.get("anomalies", []), d.get("next_steps", [])
        rep.sections = [(s["heading"], s["body"]) for s in d.get("sections", [])]
        p = d.get("plain", {})
        rep.plain_summary, rep.plain_intro = p.get("summary", ""), p.get("intro", "")
        rep.plain_why, rep.glossary = p.get("findings", []), p.get("glossary", dict(BASE_GLOSSARY))
        return rep


def _known(cls: type, d: dict[str, Any] | None) -> dict[str, Any]:
    """The keys of `d` that are fields of dataclass `cls` (a file from an older or newer version may have others)."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in (d or {}).items() if k in names}


def _delta(value: float, ref: Any, prefix: str) -> dict[str, float | None]:
    if not is_number(ref):
        return {}
    return {f"{prefix}_abs": value - ref, f"{prefix}_pct": (value - ref) / abs(ref) * 100 if ref != 0 else None}


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)
