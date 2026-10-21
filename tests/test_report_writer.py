"""The stage report writer against the owner's report rules: row names, verdict, summary layout, charts, rebuild."""

import json
import sys

import pytest

from sdf.eval import vllm_backend  # noqa: F401 - registers the vLLM metrics
from sdf.reporting.charts import write_charts
from sdf.reporting.markdown import _plain_verdicts
from sdf.reporting.metrics import is_backend_ppl, is_size
from sdf.reporting.rebuild import main as rebuild_main
from sdf.reporting.reporter import StageReporter
from sdf.requirements import DeploymentRequirement

METRICS = ["ppl_val", "predicted_weight_memory_gb", "decode_ms_per_token_mean", "vllm_ppl_val"]


def _stage(tmp_path, stage, metrics=METRICS):
    return StageReporter(stage, tmp_path, "t", {}, {}, {}, DeploymentRequirement(), list(metrics))


def _pruning(tmp_path, levels=((0.3, 12.0, 11.0), (0.5, 30.0, 14.0))):
    """A Stage 2 report shaped like the runner's: levels (share, standard ppl, our ppl) of one technique, labelled
    as _label_level does."""
    rep = _stage(tmp_path, 2)
    with rep.method("baseline", "fp16") as r:
        r.metrics.update(ppl_val=10.0, predicted_weight_memory_gb=2.0, decoder_weights_removed=0.0,
                         decode_tokens_per_s=100.0)
    for level, ppl_std, ppl_ours in levels:
        tag = f"wanda_{round(level * 100):03d}"
        for variant, ppl, removed, gb in (("original", ppl_std, level, 2 - 2 * level),
                                          ("framework", ppl_ours, level / 2, 2 - level)):
            with rep.method(tag, variant, description="Unstructured pruning (Wanda), standard settings: x",
                            simulated=variant == "original") as r:
                r.info["kind_label"] = r.plain_name
                r.info["label"] = f"Unstructured pruning (Wanda): {r.plain_name}, level {level:.0%}"
                r.metrics.update(ppl_val=ppl, predicted_weight_memory_gb=gb, decoder_weights_removed=removed,
                                 decode_ms_per_token_mean=10.0, decode_tokens_per_s=100.0, vllm_ppl_val=ppl + 5)
    return rep


def test_metric_kinds():
    assert is_backend_ppl("vllm_ppl_val") and is_backend_ppl("llamacpp_ppl_heldout")
    assert not is_backend_ppl("ppl_val") and not is_backend_ppl("ppl_heldout")
    assert is_size("predicted_weight_memory_gb") and is_size("avg_bits_per_weight") and is_size("vllm_size_gb")
    assert not is_size("ppl_val") and not is_size("decode_ms_per_token_mean")


def test_standard_row_is_quantization_only_in_stage_1(tmp_path):
    for stage, name in ((1, "Standard quantization"), (2, "Standard method"), (3, "Standard method")):
        rep = _stage(tmp_path / str(stage), stage)
        with rep.method("m", "original") as r:
            r.metrics["ppl_val"] = 10.0
        assert rep.rows[0].plain_name == name
    rep = _pruning(tmp_path / "p")
    assert rep.rows[1].info["label"] == "Unstructured pruning (Wanda): Standard method, level 30%"
    md = rep.finalize()["report"].read_text(encoding="utf-8")
    assert "quantization" not in md.lower()
    terms = md.split("## Key terms")[1].split("\n## ")[0]
    assert "- **Standard method**: The compression technique applied the usual way" in terms
    assert terms.count("- **Our method**") == 1  # one entry per version, not per level


def test_verdict_ignores_backend_perplexity_and_never_counts_a_larger_size(tmp_path):
    rep = _stage(tmp_path, 1)
    with rep.method("m", "original") as r:
        r.metrics.update(ppl_val=10.0, predicted_weight_memory_gb=1.0, vllm_ppl_val=10.0)
    with rep.method("m", "framework") as r:
        r.metrics.update(ppl_val=9.0, predicted_weight_memory_gb=1.5, vllm_ppl_val=12.0)
    assert rep.verdict(rep.rows[1]) == (["ppl_val"], [], [])  # bigger, and worse in vLLM: neither is a loss
    assert "beat the standard method" in _plain_verdicts(rep)[0]
    rep.rows[1].metrics["predicted_weight_memory_gb"] = 0.5
    assert rep.verdict(rep.rows[1]) == (["ppl_val", "predicted_weight_memory_gb"], [], [])  # smaller still wins


def test_summary_is_a_small_table_per_technique_and_big_tables_go_at_the_back(tmp_path):
    rep = _pruning(tmp_path)
    md = rep.finalize()["report"].read_text(encoding="utf-8")
    plain = md.split("# Technical details")[0]
    summary = plain.split("## Summary")[1].split("\n## ")[0]
    assert "> **What this shows:**" in summary and "> **What it means:** Our method beat the standard method in 2 " \
                                                    "of 2 comparisons" in summary
    assert "vLLM) is not part of the verdict" in summary and "never counted against our method" in summary
    # one or two versions per technique: one table, a line per version
    assert "| Unstructured pruning (Wanda) | Our method, level 30% | 1.7 GB vs 1.4 GB | Better | prediction error" \
        in summary
    assert plain.index("## Summary") < plain.index("## Charts") < plain.index("## Key terms") \
        < plain.index("## Findings") < plain.index("## Results")
    for section in plain.split("\n## ")[1:]:
        assert "> **What this shows:**" in section and "> **What it means:**" in section, section[:40]
    assert "re-rounded file, not in the verdict" in plain.split("## Results")[1]
    # more versions per technique: a small table under each technique's name
    md = _pruning(tmp_path / "sweep", [(x / 10, 20.0, 15.0) for x in range(1, 5)]).finalize()["report"].read_text()
    assert "**Unstructured pruning (Wanda)**\n\n| Version | Memory, ours vs standard |" in md


def test_charts_are_written_and_embedded_with_captions(tmp_path):
    pytest.importorskip("matplotlib")
    outputs = _pruning(tmp_path).finalize()
    names = sorted(p.name for k, p in outputs.items() if k.startswith("chart_"))
    # the speed chart leaves out the simulated standard rows but keeps our method's
    assert names == ["chart_perplexity_vs_memory.png", "chart_perplexity_vs_share_removed.png",
                     "chart_perplexity_vs_speed.png"]
    assert all((tmp_path / "stage_2" / n).stat().st_size > 0 for n in names)
    md = outputs["report"].read_text(encoding="utf-8")
    assert "![chart perplexity vs share removed](chart_perplexity_vs_share_removed.png)" in md
    assert "Lower is better" in md and "does not start at 0" in md
    assert "logarithmic" not in md  # perplexity spans 10 to 30: a linear axis
    rep = _stage(tmp_path / "s1", 1)
    with rep.method("gptq", "original", description="GPTQ, standard settings: x") as r:
        r.metrics.update(ppl_val=10.0)
    with rep.method("gptq", "framework", description="GPTQ, Stage 0 plan") as r:
        r.metrics.update(ppl_val=9.0)
    assert [c.path.name for c in write_charts(rep)] == ["chart_perplexity.png"]  # Stage 1: a dot per version


def test_no_matplotlib_means_no_charts_and_no_crash(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    outputs = _pruning(tmp_path).finalize()
    assert not [k for k in outputs if k.startswith("chart_")]
    assert "## Charts" not in outputs["report"].read_text(encoding="utf-8")


def test_rebuild_from_an_older_results_json(tmp_path):
    rep = _pruning(tmp_path / "run")
    rep.finalize()
    path = rep.json_path
    data = json.loads(path.read_text())
    del data["main_metrics"]  # saved before main_metrics was
    for r in data["rows"]:  # and with the old names
        r["info"] = {k: v.replace("Standard method", "Standard quantization") if isinstance(v, str) else v
                     for k, v in r["info"].items()}
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    (rep.dir / "report.md").unlink()

    rebuild_main([str(tmp_path / "run"), "--out", str(tmp_path / "out")])
    md = (tmp_path / "out" / "stage_2" / "report.md").read_text(encoding="utf-8")
    assert "Standard method, level 30%" in md and "Standard quantization" not in md
    assert (tmp_path / "out" / "stage_2" / "stage_2_comparison.xlsx").exists()
    assert (tmp_path / "out" / "stage_2" / "results.json").read_bytes() == before
    assert not (rep.dir / "report.md").exists()  # --out leaves the run alone

    rebuild_main([str(tmp_path / "run")])  # in place: the report comes back, results.json is not rewritten
    assert path.read_bytes() == before
    assert (rep.dir / "report.md").read_text(encoding="utf-8") == md
