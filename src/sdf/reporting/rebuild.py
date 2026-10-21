"""Rebuild the Stage 1-3 report.md, workbook and charts from saved results.json files, without running any model:

    python -m sdf.reporting.rebuild <folder> [--out <dir>]

<folder> is a stage folder or a whole run: every Stage 1-3 results.json under it is rebuilt. results.json itself is
never rewritten; with --out, the outputs and a copy of results.json go to the same relative folders under <dir>.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from sdf.reporting.reporter import VARIANT_PLAIN, StageReporter
from sdf.utils.logging import get_logger

log = get_logger(__name__)


def rebuild(folder: str | Path, out: str | Path | None = None) -> list[Path]:
    """Rebuild every Stage 1-3 results.json under `folder`; returns the report.md paths written."""
    folder = Path(folder)
    written = []
    for path in sorted(folder.rglob("results.json")):
        if json.loads(path.read_text(encoding="utf-8")).get("stage") not in (1, 2, 3):
            continue  # Stage 0 and side studies set report parts that results.json does not keep
        dest = Path(out) / path.parent.relative_to(folder) if out else None
        rep = StageReporter.load(path, dest)
        if not rep.main_metrics:
            rep.main_metrics = _main_metrics(rep)
        _relabel(rep)
        if dest is not None:
            shutil.copyfile(path, dest / "results.json")
        rep.compute_deltas()
        rep._auto_findings = rep.comparison_summary()
        written.append(rep.write_outputs()["report"])
        log.info("rebuilt %s", written[-1])
    return written


def _main_metrics(rep: StageReporter) -> list[str]:
    """The main metrics the runner gave this stage (results.json saves them since 2026-10-21)."""
    from sdf.eval import llamacpp, trtllm_backend, vllm_backend
    from sdf.stages.runner import MAIN_METRICS

    present = {k for r in rep.rows for k in r.metrics}
    chained = (["predicted_weight_memory_gb", "decoder_weights_removed"]
               if rep.stage == 3 and any("built_on" in r.info for r in rep.rows) else [])
    return MAIN_METRICS[rep.stage] + chained + [m for mod in (llamacpp, vllm_backend, trtllm_backend)
                                                for m in mod.MAIN_METRICS if m in present]


def _relabel(rep: StageReporter) -> None:
    """Row names earlier runs saved, as the runner writes them now."""
    from sdf.stages.runner import _PLAN_ROWS

    _, label, desc = _PLAN_ROWS["prune_same_size"]
    old = {"Our method, same amount removed as the standard method": label,
           "our method removing exactly as many numbers as the standard method, but taking them from the least "
           "sensitive layers instead of evenly, so the two can be compared fairly, size for size.": desc}
    if rep.stage != 1:  # pruning and KV cache are not quantization
        old["Standard quantization"] = VARIANT_PLAIN["original"][0]
    for r in rep.rows:
        for k in ("label", "kind_label", "plain_desc"):
            if isinstance(r.info.get(k), str):
                for a, b in old.items():
                    r.info[k] = r.info[k].replace(a, b)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m sdf.reporting.rebuild",
                                 description="Rebuild Stage 1-3 reports from their saved results.json files.")
    ap.add_argument("folder", help="a stage folder or a whole run folder")
    ap.add_argument("--out", help="write here instead of next to each results.json")
    args = ap.parse_args(argv)
    for p in rebuild(args.folder, args.out):
        print(p)


if __name__ == "__main__":
    main()
