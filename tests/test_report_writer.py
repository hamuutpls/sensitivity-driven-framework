"""The stage report writer against the owner's report rules: row names, verdict, summary layout, charts, rebuild."""

import json
import sys

import pytest

from sdf.eval import vllm_backend  # noqa: F401 - registers the vLLM metrics
from sdf.reporting.charts import write_charts
from sdf.reporting.markdown import _plain_verdicts
from sdf.reporting.metrics import SIZE_METRICS, backend, is_backend_ppl
from sdf.reporting.rebuild import main as rebuild_main
from sdf.reporting.reporter import StageReporter
from sdf.requirements import DeploymentRequirement

METRICS = ["ppl_val", "predicted_weight_memory_gb", "decode_ms_per_token_mean", "vllm_ppl_val"]


def _stage(tmp_path, stage, metrics=METRICS):
    return StageReporter(stage, tmp_path, "t", {}, {}, {}, DeploymentRequirement(), list(metrics))


def _pruning(tmp_path, levels=((0.3, 12.0, 11.0), (0.5, 40.0, 14.0))):
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
    assert backend("trtllm_engine_gb") == "TensorRT-LLM" and backend("predicted_weight_memory_gb") is None
    assert {"predicted_weight_memory_gb", "avg_bits_per_weight", "vllm_size_gb", "trtllm_engine_gb"} <= SIZE_METRICS
    # running memory is judged like any measure
    assert not {"peak_memory_gb", "predicted_kv_memory_gb", "avg_kv_bits", "avg_activation_bits", "vllm_weight_memory_gb",
                "vllm_kv_cache_reserved_gb", "vllm_peak_memory_gb", "trtllm_gpu_memory_loaded_gb"} & SIZE_METRICS


def test_standard_row_is_quantization_only_in_stages_0_and_1(tmp_path):
    for stage, name in ((0, "Standard quantization"), (1, "Standard quantization"), (2, "Standard method"),
                        (3, "Standard method")):
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


def test_stage_1_w8a8_row_is_its_own_version_of_the_technique(tmp_path):
    rep = _stage(tmp_path, 1)
    for method, variant, desc in (("gptq_with_quarot", "original", "GPTQ + QuaRot, standard settings: x"),
                                  ("gptq_with_quarot_w8a8", "original",
                                   "GPTQ + QuaRot, every layer W8A8, standard settings: x"),
                                  ("gptq_with_quarot", "framework", "GPTQ + QuaRot, Stage 0 weights")):
        with rep.method(method, variant, description=desc) as r:
            r.metrics["ppl_val"] = 10.0
    assert [r.plain_name for r in rep.rows] == ["Standard quantization", "Standard quantization, every layer W8A8",
                                                "Our method"]
    assert {rep.technique(r) for r in rep.rows} == {"GPTQ + QuaRot"}


def test_verdict_ignores_backend_perplexity_and_never_counts_a_larger_size(tmp_path):
    rep = _stage(tmp_path, 1)
    with rep.method("m", "original") as r:
        r.metrics.update(ppl_val=10.0, predicted_weight_memory_gb=1.0, vllm_ppl_val=10.0)
    with rep.method("m", "framework") as r:
        r.metrics.update(ppl_val=9.0, predicted_weight_memory_gb=1.5, vllm_ppl_val=12.0)
    assert rep.verdict(rep.rows[1]) == (["ppl_val"], [], [])  # bigger, and worse in vLLM: neither is a loss
    assert "beat the standard quantization" in _plain_verdicts(rep)[0]
    rep.rows[1].metrics["predicted_weight_memory_gb"] = 0.5
    assert rep.verdict(rep.rows[1]) == (["ppl_val", "predicted_weight_memory_gb"], [], [])  # smaller still wins
    rep.rows[1].metrics["predicted_weight_memory_gb"] = 0.996
    assert rep.verdict(rep.rows[1]) == (["ppl_val", "predicted_weight_memory_gb"], [], [])  # no tolerance band


def test_verdict_judges_running_memory_and_simulated_rows_like_any_row(tmp_path):
    rep = _stage(tmp_path, 3, ["ppl_val", "predicted_kv_memory_gb", "peak_memory_gb", "decode_ms_per_token_mean"])
    with rep.method("m", "original", simulated=True) as r:
        r.metrics.update(ppl_val=10.0, predicted_kv_memory_gb=0.10, peak_memory_gb=1.0, decode_ms_per_token_mean=5.0)
    with rep.method("m", "framework", simulated=True) as r:
        r.metrics.update(ppl_val=9.0, predicted_kv_memory_gb=0.12, peak_memory_gb=1.2, decode_ms_per_token_mean=4.0)
    assert rep.verdict(rep.rows[1]) == (["ppl_val", "decode_ms_per_token_mean"],
                                        ["predicted_kv_memory_gb", "peak_memory_gb"], [])


def test_summary_is_a_small_table_per_technique_and_big_tables_go_at_the_back(tmp_path):
    rep = _pruning(tmp_path)
    md = rep.finalize()["report"].read_text(encoding="utf-8")
    plain = md.split("# Technical details")[0]
    summary = plain.split("## Summary")[1].split("\n## ")[0]
    assert "> **What this shows:**" in summary and "> **What it means:** Our method beat the standard method in 2 " \
                                                    "of 2 comparisons" in summary
    assert "vLLM) is not part of the verdict" in summary and "never counted against our method" in summary
    assert "most often" not in summary and "badly damaged" not in summary
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
    assert "**Unstructured pruning (Wanda)**\n\n| Version | Memory, ours vs standard method |" in md


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
    assert "logarithmic" not in md  # perplexity spans 10 to 40: a linear axis
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
    assert not (rep.dir / "report.md").exists() and path.read_bytes() == before  # the run is left alone


def test_rebuild_writes_a_new_folder_and_never_overwrites(tmp_path, capsys):
    rep = _pruning(tmp_path / "run")
    rep.finalize()
    old = rep.report_path.read_bytes()
    rebuild_main([str(tmp_path / "run")])  # default: a new folder next to the run
    (new,) = [p for p in tmp_path.iterdir() if p.name.startswith("run-rebuilt-")]
    assert (new / "stage_2" / "report.md").exists() and (new / "master_report.md").exists()
    assert rep.report_path.read_bytes() == old
    for out in (tmp_path / "run", new, tmp_path):  # the input itself, a used folder, a folder holding the input
        with pytest.raises(SystemExit):
            rebuild_main([str(tmp_path / "run"), "--out", str(out)])
        assert "never overwrites" in capsys.readouterr().err
    inside = tmp_path / "run" / "rebuilt"
    rebuild_main([str(tmp_path / "run"), "--out", str(inside)])  # inside the input: allowed, and skipped later
    assert not (inside / "rebuilt").exists()
    rebuild_main([str(tmp_path / "run"), "--out", str(tmp_path / "again")])
    assert not (tmp_path / "again" / "rebuilt").exists()  # the earlier rebuild folder was not rebuilt again
    rebuild_main([str(new), "--out", str(tmp_path / "of_a_rebuild")])  # a rebuild folder itself is rebuilt
    assert (tmp_path / "of_a_rebuild" / "stage_2" / "report.md").exists()
    (tmp_path / "a_file").write_text("x")
    for out in ("a_file", "a_file/sub"):
        with pytest.raises(SystemExit):
            rebuild_main([str(tmp_path / "run"), "--out", str(tmp_path / out)])
        assert "never overwrites" in capsys.readouterr().err


def test_rebuild_fails_cleanly_on_a_file_that_is_not_a_results_object(tmp_path, capsys):
    bad = [b"[1, 2]", b"{not json", b"\xff\xfe", b'{"stage": 2}', b'{"stage": 2, "rows": [1]}',
           b'{"stage": 2, "rows": [{"variant": "fp16"}]}', b'{"stage": 2, "rows": [{"method": "m", "variant": "fp16", '
           b'"metrics": [1]}]}', b'{"stage": 2, "rows": [], "requirement": 5}']
    for i, data in enumerate(bad):
        (tmp_path / str(i)).mkdir()
        (tmp_path / str(i) / "results.json").write_bytes(data)
        with pytest.raises(SystemExit):
            rebuild_main([str(tmp_path / str(i)), "--out", str(tmp_path / f"out{i}")])
        assert str(tmp_path / str(i) / "results.json") in capsys.readouterr().err, data
        assert not (tmp_path / f"out{i}").exists()  # nothing is written


def test_stage_0_standard_label():
    from sdf.stage0.run import standard_label

    assert standard_label(0.0) == "Standard quantization"
    assert standard_label(0.2) == "Standard quantization and pruning"


def _stage1(tmp_path, techniques=("GPTQ",), metrics=METRICS):
    rep = _stage(tmp_path, 1, metrics)
    with rep.method("baseline", "fp16") as r:
        r.metrics.update(ppl_val=10.0, vllm_ppl_val=10.0)
    for t in techniques:
        with rep.method(t, "original", description=f"{t}, standard settings: x", simulated=True) as r:
            r.metrics.update(ppl_val=12.0, predicted_weight_memory_gb=0.7, avg_bits_per_weight=5.65)
        with rep.method(t, "framework", description=f"{t}, Stage 0 sensitivity plan", simulated=True) as r:
            r.metrics.update(ppl_val=11.0, predicted_weight_memory_gb=0.8, avg_bits_per_weight=7.4)
    return rep


def test_stage_1_report_names_techniques_and_says_standard_quantization(tmp_path):
    rep = _stage1(tmp_path, ("GPTQ", "OmniQuant (LWC)"))
    with rep.method("awq", "original", description="AWQ, standard settings: x"):
        raise RuntimeError("no kernel")
    md = rep.finalize()["report"].read_text(encoding="utf-8")
    plain = md.split("# Technical details")[0]
    assert "whether it beat the standard quantization" in plain and "ours vs standard quantization" in plain
    assert "- **GPTQ: Our method** beat the standard quantization on every measure compared" in plain
    assert "- 1 run(s) failed: AWQ: Standard quantization." in plain
    assert "| GPTQ: Our method |" in plain.split("## Results")[1]  # the Results table names the technique
    assert "how the standard quantization and our method differ" not in plain  # no intro set here
    terms = plain.split("## Key terms")[1].split("\n## ")[0]
    assert "- **Standard quantization**: Every layer at 4-bit weights and 8-bit activations (W4A8)" in terms
    assert "- **Our method**: Our method, following the Stage 0 plan" in terms
    for t in ("Hugging Face", "Simulated", "Perplexity", "LWC"):
        assert f"- **{t}**:" in terms, t
    assert "- **vLLM**" not in terms and "Re-rounded file" not in terms  # only the uncompressed row has vLLM
    assert "vLLM" not in plain.split("## Results")[1].split("**What each column means**")[0]  # no empty column
    assert "Backend perplexity" not in plain


def test_a_chart_error_does_not_stop_the_report(tmp_path, monkeypatch):
    from sdf.reporting import charts

    monkeypatch.setattr(charts, "write_charts", lambda rep: 1 / 0)
    outputs = _pruning(tmp_path).finalize()
    assert outputs["report"].exists() and outputs["xlsx"].exists()
    assert "## Summary" in outputs["report"].read_text(encoding="utf-8")


def test_more_than_seven_techniques_split_each_chart_and_never_repeat_a_colour(tmp_path):
    pytest.importorskip("matplotlib")
    from sdf.reporting.charts import PALETTE

    assert "#F0E442" not in PALETTE and len(set(PALETTE)) == len(PALETTE) == 7  # yellow is hard to see
    charts = write_charts(_stage1(tmp_path, [f"T{i}" for i in range(13)]))
    assert [c.path.name for c in charts] == ["chart_perplexity_part1.png", "chart_perplexity_vs_memory_part1.png",
                                             "chart_perplexity_part2.png", "chart_perplexity_vs_memory_part2.png"]
    assert "Part 1 of 2: 7 of the 13 techniques" in charts[0].shows
    assert "Part 2 of 2: 6 of the 13 techniques" in charts[2].shows


def test_a_part_chart_counts_only_its_own_techniques(tmp_path):
    pytest.importorskip("matplotlib")
    rep = _stage1(tmp_path, [f"T{i}" for i in range(9)])
    for r in rep.rows:  # every row has a speed, but only T0's rows are real (_stage1 marks rows simulated)
        r.metrics["decode_tokens_per_s"] = 100.0
        r.info["simulated"] = rep.technique(r) != "T0"
    speed = [c for c in write_charts(rep) if "vs_speed" in c.path.name]
    assert [c.path.name for c in speed] == ["chart_perplexity_vs_speed_part1.png"]
    assert "Part 1 of 2: 1 of the 9 techniques" in speed[0].shows  # the simulated techniques are not counted


def test_rebuild_paths_stage_1_and_master(tmp_path, monkeypatch):
    rep = _stage1(tmp_path / "run")
    rep.finalize()
    before = rep.json_path.read_bytes()
    monkeypatch.chdir(rep.dir)
    rebuild_main([".", "--out", "../rebuilt_stage"])  # relative paths from inside the stage folder
    md = (tmp_path / "run" / "rebuilt_stage" / "report.md").read_text(encoding="utf-8")
    assert "Standard quantization" in md and "Standard method" not in md
    assert not (tmp_path / "run" / "rebuilt_stage" / "master_report.md").exists()  # a stage folder: no master
    rebuild_main([str(tmp_path / "run"), "--out", str(tmp_path / "out")])  # a run folder: the master too
    master = (tmp_path / "out" / "master_report.md").read_text(encoding="utf-8")
    assert "beats the original method" in master and (tmp_path / "out" / "all_stages_comparison.xlsx").exists()
    assert not (tmp_path / "out" / "rebuilt_stage").exists()  # an earlier rebuild inside the run is skipped
    assert rep.json_path.read_bytes() == before


def test_load_tolerates_missing_and_unknown_keys(tmp_path):
    path = tmp_path / "stage_2" / "results.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"stage": 2, "requirement": {"target_ppl": 11.0, "new_target": 1},
                                "rows": [{"method": "m", "variant": "fp16", "status": "ok", "metrics": {"ppl_val": 10.0},
                                          "added_later": True}]}))
    rep = StageReporter.load(path)
    assert rep.requirement.target_ppl == 11.0 and rep.rows[0].metrics == {"ppl_val": 10.0} and rep.sections == []
