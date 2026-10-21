"""Rebuild the Stage 1-3 report.md, workbook and charts from saved results.json files, without running any model:

    python -m sdf.reporting.rebuild <folder> [--out <dir>]

<folder> is a stage folder or a whole run: every Stage 1-3 results.json under it is rebuilt. When <folder> is a run
folder (it holds stage_<N>/results.json), master_report.md and all_stages_comparison.xlsx are refreshed too, from the
rebuilt stages, so they show the same verdicts. results.json itself is never rewritten; with --out, the outputs and a
copy of results.json go to the same relative folders under <dir>.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from sdf.reporting.master import load_stages, write_master
from sdf.reporting.reporter import VARIANT_PLAIN, StageReporter
from sdf.utils.logging import get_logger

log = get_logger(__name__)


def rebuild(folder: str | Path, out: str | Path | None = None) -> list[Path]:
    """Rebuild every Stage 1-3 results.json under `folder`, then the master files when `folder` is a run folder;
    returns the paths written (each report.md, then master_report.md)."""
    folder = Path(folder).resolve()
    out = Path(out).resolve() if out else folder
    written, rebuilt = [], {}
    for path in sorted(folder.rglob("results.json")):
        if json.loads(path.read_text(encoding="utf-8")).get("stage") not in (1, 2, 3):
            continue  # Stage 0 and side studies set report parts that results.json does not keep
        dest = out / path.parent.relative_to(folder)
        rep = StageReporter.load(path, dest)
        if not rep.main_metrics:
            rep.main_metrics = _main_metrics(rep)
        _relabel(rep)
        if dest != path.parent:
            shutil.copyfile(path, dest / "results.json")
        rep.analyse()
        written.append(rep.write_outputs()["report"])
        rebuilt[path.parent] = rep.to_dict()
        log.info("rebuilt %s", written[-1])
    if saved := load_stages(folder):  # a run folder: the master files, the rebuilt stages in place of the saved ones
        stages = [rebuilt.get(folder / f"stage_{st['stage']}", st) for st in saved]
        written.append(write_master(out, stages)["master_report"])
    elif rebuilt:
        print("master_report.md was not refreshed: rebuild the run folder to refresh it")
    return written


def _main_metrics(rep: StageReporter) -> list[str]:
    """The main metrics the runner gave this stage (results.json saves them since 2026-10-21)."""
    from sdf.eval import llamacpp, trtllm_backend, vllm_backend
    from sdf.stages.runner import stage_main_metrics

    present = {k for r in rep.rows for k in r.metrics}
    return stage_main_metrics(rep.stage, rep.stage == 3 and any("built_on" in r.info for r in rep.rows),
                              [m for m in (llamacpp, vllm_backend, trtllm_backend)
                               if any(k in present for k in m.MAIN_METRICS)])


# ponytail: migration for results saved before 2026-10-21; delete when those runs are no longer rebuilt
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
