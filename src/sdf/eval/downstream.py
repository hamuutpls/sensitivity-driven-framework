"""Downstream tasks: multiple-choice accuracy through lm-evaluation-harness (`pip install lm-eval`).

Off unless `eval.downstream_tasks` lists tasks (e.g. ["arc_easy", "hellaswag", "piqa", "winogrande"]); the task
suite is still an open choice (SyRS open issue #1). The model is evaluated in place, so whatever a method changed
(rounded weights, activation or cache hooks) is what the tasks see.
"""

from __future__ import annotations

import statistics
from typing import Any

from torch import nn

from sdf.reporting.metrics import METRICS, MetricSpec
from sdf.utils.logging import get_logger

log = get_logger(__name__)


def task_metric(task: str) -> str:
    """Metric name for one task's accuracy, registered so reports know higher is better."""
    name = f"acc_{task}"
    METRICS.setdefault(name, MetricSpec(
        f"Accuracy: {task}", "", "higher", f"share of {task} questions answered correctly",
        f"A multiple-choice test ({task}): the share of questions the model answers correctly, from 0 to 1. "
        "Higher is better."))
    return name


def task_limits(tasks: list[str], limit: int | dict[str, int] | None) -> dict[int | None, list[str]]:
    """The tasks grouped by their question limit: one number for every task, or {task: number} (a task not named
    runs all its questions; paper review, 2026-11-22: Mohammad chose 500 HellaSwag questions per search try)."""
    if not isinstance(limit, dict):
        return {limit: list(tasks)}
    if unknown := sorted(set(limit) - set(tasks)):
        raise ValueError(f"question limits for tasks {unknown} that are not run ({list(tasks)})")
    groups: dict[int | None, list[str]] = {}
    for t in tasks:
        groups.setdefault(limit.get(t), []).append(t)
    return groups


def downstream_accuracy(model: nn.Module, tokenizer: Any, tasks: list[str], limit: int | dict[str, int] | None,
                        batch_size: int, seed: int) -> dict[str, float]:
    """Accuracy per task (normalised accuracy where the task reports it, as is usual for HellaSwag, ARC and PIQA)
    and their mean. `limit`: questions per task, for every task or per task ({task: number})."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    res = {}
    for n, group in task_limits(tasks, limit).items():
        res.update(simple_evaluate(model=lm, tasks=group, limit=n, random_seed=seed, numpy_random_seed=seed,
                                   torch_random_seed=seed, fewshot_random_seed=seed)["results"])
    out = {}
    for task in tasks:
        r = res[task]
        out[task_metric(task)] = float(r.get("acc_norm,none", r["acc,none"]))
    out["downstream_acc_mean"] = statistics.fmean(out.values())
    log.info("downstream: %s", ", ".join(f"{k}={v:.3f}" for k, v in out.items()))
    return out
