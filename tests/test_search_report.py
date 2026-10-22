"""The settings search's report, written from synthetic tries (no model is built)."""

import json
import random
import re
from pathlib import Path

import openpyxl
import pytest

from sdf.cli import search_report_main
from sdf.reporting.search import _tokens_per_s, best_so_far
from sdf.search import FAILED, _sample, pareto, technique_space

OBJ = ["ppl_val", "vllm_decode_ms_per_token"]


@pytest.fixture
def run_dir(small_cfg):
    cfg = small_cfg.with_overrides({"requirement.target_memory_gb": 1.1, "search.objectives": OBJ,
                                    "search.stage1_methods": ["rtn_with_rtn_act", "gptq_with_smoothquant"],
                                    "search.stage3_methods": ["rtn_kv", "h2o"]})
    run = Path(small_cfg.run.output_root) / "20261021-colab-a100-search"
    (run / "search").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(cfg.to_dict()))
    space, rng, trials = technique_space(cfg), random.Random(0), []
    for name in cfg.search.searchers:
        for n in range(1, 13):
            size = rng.uniform(0.7, 1.4)
            over, failed = size > 1.1, n == 5
            rough = name == "mfbo" and n % 3
            ok = not over and not failed
            metrics = {} if over or failed else {
                "ppl_val": rng.uniform(8, 14), "ppl_heldout": rng.uniform(8, 14), "predicted_weight_memory_gb": size,
                "vllm_decode_ms_per_token": rng.uniform(4, 9), "vllm_ppl_val": rng.uniform(8, 14),
                "avg_bits_per_weight": 4.5, "avg_activation_bits": 6.0}
            trials.append({
                "searcher": name, "n": n, "point": _sample(space, rng), "windows": 2 if rough else 6,
                "cost": 0.0 if over else (1 / 3 if rough else 1.0), "built": not over, "over_limit": over,
                "full": not rough, "ok": ok, "met": ok, "values": [metrics[o] for o in OBJ] if ok else [FAILED] * 2,
                "violation": 0.0 if ok else 1.0, "metrics": metrics, "dir": "x", "time_s": 60.0, "cached": False,
                "error": "plan over the limit" if over else ("CUDA out of memory" if failed else None)})
    (run / "search" / "trials.json").write_text(json.dumps(trials))
    return run, trials


def test_the_report_is_rebuilt_from_saved_tries(run_dir, capsys):
    run, trials = run_dir
    search_report_main([str(run), "--tests", "417 passed, 3 skipped at 5f440dc"])
    assert "search_report:" in capsys.readouterr().out
    d = run / "search"
    wb = openpyxl.load_workbook(d / "search_results.xlsx")
    assert wb.sheetnames == ["Trials", "Pareto", "Searcher comparison", "Best configs"]
    assert wb["Trials"].max_row == len(trials) + 1
    assert wb["Pareto"].max_row == len(pareto(trials)) + 1
    text = (d / "search_report.md").read_text()
    assert "417 passed, 3 skipped at 5f440dc" in text
    heads = re.findall(r"^## (.+)$", text, re.M)
    assert heads[:4] == ["Summary", "1. Best trade-off settings", "2. Search methods compared",
                         "3. Best setting for each goal"]
    assert heads[-2:] == ["Appendix A: the other settings of each best trade-off option",
                          "Appendix B: search methods, every column"]
    assert text.count("> **What this shows.**") == text.count("> **What it means.**") == 4
    for png in re.findall(r"!\[[^]]*\]\(([^)]+)\)", text):  # every chart the report shows exists
        assert (d / png).exists(), png
    assert {"pareto_ppl_vs_predicted_weight_memory_gb.png", "pareto_ppl_vs_vllm_decode_ms_per_token.png",
            "best_so_far.png", "hypervolume.png"} <= {p.name for p in d.glob("*.png")}
    # the most accurate trade-off setting's speed is also given in tokens per second, next to its time
    top = min(pareto(trials), key=lambda t: t["values"][0])
    ms = top["metrics"]["vllm_decode_ms_per_token"]
    row = next(r for r in text.splitlines() if r.startswith("| 1 |"))
    assert f"{ms:.4g} | {1000 / ms:.4g}" in row and "| W4.5A6 |" in row
    assert "Same speed, tokens per second, higher is better" in text
    assert "predicted weight memory at most 1.1 GB" in text


def test_the_report_says_when_no_test_result_was_recorded_or_nothing_met_the_limit(run_dir):
    run, trials = run_dir
    for t in trials:
        t["met"] = False
    (run / "search" / "trials.json").write_text(json.dumps(trials))
    search_report_main([str(run)])
    text = (run / "search" / "search_report.md").read_text()
    assert "not recorded for this report" in text and "No setting met the limit" in text


def test_best_so_far_keeps_the_lowest_full_try_under_the_limit():
    t = lambda c, v, full=True, met=True: {"searcher": "a", "cost": c, "full": full, "met": met, "values": [v]}  # noqa: E731
    trials = [t(1, 9.0, met=False), t(1 / 3, 5.0, full=False), t(1, 8.0), t(1, 8.5), t(0, FAILED, met=False), t(1, 7.0)]
    assert best_so_far(trials, "a") == [(pytest.approx(7 / 3), 8.0), (pytest.approx(10 / 3), 8.0),
                                         (pytest.approx(10 / 3), 8.0), (pytest.approx(13 / 3), 7.0)]
    assert best_so_far(trials, "b") == []


def test_tokens_per_second_only_for_a_real_time():
    assert _tokens_per_s(4.0) == 250.0
    assert _tokens_per_s(None) is None and _tokens_per_s(0) is None and _tokens_per_s(FAILED) is None
