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


def downstream_accuracy(model: nn.Module, tokenizer: Any, tasks: list[str], limit: int | None, batch_size: int,
                        seed: int) -> dict[str, float]:
    """Accuracy per task (normalised accuracy where the task reports it, as is usual for HellaSwag, ARC and PIQA)
    and their mean."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    res = simple_evaluate(model=lm, tasks=list(tasks), limit=limit, random_seed=seed, numpy_random_seed=seed,
                          torch_random_seed=seed, fewshot_random_seed=seed)["results"]
    out = {}
    for task in tasks:
        r = res[task]
        out[task_metric(task)] = float(r.get("acc_norm,none", r["acc,none"]))
    out["downstream_acc_mean"] = statistics.fmean(out.values())
    log.info("downstream: %s", ", ".join(f"{k}={v:.3f}" for k, v in out.items()))
    return out
