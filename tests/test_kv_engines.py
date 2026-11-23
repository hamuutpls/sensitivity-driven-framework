"""The engines' own KV-cache settings (eval.vllm_kv_cache_dtype, eval.llamacpp_kv_cache_type) and the script that
measures them on the final models (scripts/kv_engines_20261123.py), 2026-11-23."""

import copy
import csv
import importlib.util
import sys
import types
from dataclasses import asdict
from pathlib import Path

import pytest

from sdf.config import KV_ENGINE_FIELDS, EvalConfig, kv_engine_key
from sdf.eval import llamacpp
from conftest import fake_texts


def test_kv_settings_are_checked_and_stay_out_of_keys_at_their_defaults():
    assert kv_engine_key(EvalConfig()) == {}
    ev = EvalConfig(vllm_kv_cache_dtype="fp8", llamacpp_kv_cache_type="q4_0")
    assert kv_engine_key(ev) == {"vllm_kv_cache_dtype": "fp8", "llamacpp_kv_cache_type": "q4_0"}
    assert kv_engine_key(ev, "vllm") == {"vllm_kv_cache_dtype": "fp8"}
    with pytest.raises(ValueError, match="llamacpp_kv_cache_type"):
        EvalConfig(llamacpp_kv_cache_type="q3_k")
    with pytest.raises(ValueError, match="vllm_kv_cache_dtype"):
        EvalConfig(vllm_kv_cache_dtype="int4")


def test_point_key_is_unchanged_at_the_defaults(small_cfg):
    from sdf.run import start_run
    from sdf.search import point_key, technique_space

    ctx = start_run(small_cfg)
    point = technique_space(small_cfg).make({})
    key = point_key(ctx, small_cfg, point, [])
    old = {k: v for k, v in asdict(small_cfg.eval).items() if k not in KV_ENGINE_FIELDS}
    assert key["eval"] == old  # the keys of every earlier try
    fp8 = small_cfg.with_overrides({"eval.vllm_kv_cache_dtype": "fp8"})
    assert point_key(ctx, fp8, point, [])["eval"] == {**old, "vllm_kv_cache_dtype": "fp8"}


def test_llamacpp_passes_the_cache_type_and_reads_the_cache_size(monkeypatch, tmp_path):
    assert llamacpp.kv_args(EvalConfig()) == []
    assert llamacpp.kv_args(EvalConfig(llamacpp_kv_cache_type="q4_0")) == ["-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on"]
    line = ("llama_kv_cache: size =   11.00 MiB (  2048 cells,  22 layers,  4/4 seqs), K (q4_0):    5.50 MiB, "
            "V (q4_0):    5.50 MiB")
    assert llamacpp.kv_cache_gb("x\n" + line + "\n") == pytest.approx(11 * 2**20 / 1e9)
    assert llamacpp.kv_cache_gb("no such line") is None
    seen = []
    rows = '[{"n_prompt": 8, "n_gen": 0, "avg_ts": 1.0, "stddev_ts": 0.1}, ' \
           '{"n_prompt": 0, "n_gen": 4, "avg_ts": 2.0, "stddev_ts": 0.2}]'
    monkeypatch.setattr(llamacpp, "_run", lambda cmd: seen.append([str(c) for c in cmd]) or rows)
    monkeypatch.setattr(llamacpp, "_exe", lambda cfg, name: name)
    llamacpp._bench(EvalConfig(llamacpp_kv_cache_type="q8_0"), "m.gguf")
    assert seen[0][seen[0].index("-ctk") + 1] == "q8_0" and seen[0][seen[0].index("-ctv") + 1] == "q8_0"
    monkeypatch.setattr(llamacpp, "_run", lambda cmd: seen.append([str(c) for c in cmd])
                        or "Final estimate: PPL = 9.5\n" + line)
    ppl, gb = llamacpp._perplexity(EvalConfig(llamacpp_kv_cache_type="q4_0"), "m.gguf", "text", 2,
                                   Path(tmp_path) / "t.txt")
    assert (ppl, gb) == (9.5, pytest.approx(11 * 2**20 / 1e9)) and "-ctk" in seen[-1] and "-fa" in seen[-1]


def test_fp16_key_is_unchanged_at_the_defaults(small_cfg):
    """The FP16 and stage-row cache keys hold no KV-cache setting: the engine keys (lc_key, the serving keys)
    carry it, so earlier entries stay valid and a compressed-cache run still measures the engines again."""
    import torch

    from sdf.run import start_run
    from sdf.stage0.run import fp16_key

    key = fp16_key(start_run(small_cfg), torch.device("cpu"))
    assert not set(key["eval"]) & set(KV_ENGINE_FIELDS)
    fp8 = small_cfg.with_overrides({"eval.vllm_kv_cache_dtype": "fp8", "eval.llamacpp_kv_cache_type": "q4_0"})
    assert fp16_key(start_run(fp8), torch.device("cpu")) == key


@pytest.mark.parametrize("dtype", ["auto", "fp8"])
def test_vllm_worker_gives_vllm_the_cache_dtype(monkeypatch, dtype):
    """The worker on a stand-in vLLM: kv_cache_dtype reaches LLM() only when it is not the default."""
    seen = {}

    class Stop(Exception):
        pass

    def fake_llm(**kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=fake_llm, SamplingParams=None))
    from sdf.eval import vllm_worker

    args = {"model_dir": "m", "seq_len": 16, "prompt_len": 8, "decode_tokens": 4, "val": [[1] * 16],
            "gpu_memory_utilization": 0.5, "kv_cache_dtype": dtype}
    with pytest.raises(Stop):
        vllm_worker.main(args)
    assert seen.get("kv_cache_dtype") == (None if dtype == "auto" else "fp8")
    assert "enforce_eager" not in seen  # a weight-only row keeps CUDA graphs


def test_vllm_backend_sends_the_cache_dtype_to_the_worker(monkeypatch, tiny_llama, tokenizer, tmp_path):
    import torch

    from sdf.eval import vllm_backend

    seen = {}
    monkeypatch.setattr(vllm_backend, "run_worker", lambda py, script, args, d, err: seen.update(args) or {})
    tokenizer.save_pretrained = lambda path: None
    w = torch.zeros(2, 16, dtype=torch.long)
    vllm_backend.measure(tiny_llama, tokenizer, None, w, w, EvalConfig(vllm_kv_cache_dtype="fp8"), 16, 128)
    assert seen["kv_cache_dtype"] == "fp8"


def test_kv_engines_script_measures_each_model_with_both_caches(small_cfg, tiny_llama, tokenizer, monkeypatch):
    """scripts/kv_engines_20261123.py end to end on the toy model, with stand-in engines that answer by cache
    setting: every compressed row gets its own engine numbers (no cache entry of the 16-bit row is reused)."""
    from sdf.eval import vllm_backend

    root = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("kv_engines_20261123", root / "kv_engines_20261123.py")
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    monkeypatch.setattr(K, "_cfg", lambda run, run_id, **over: small_cfg.with_overrides(
        {"run.run_id": run_id, "eval.vllm_python": "stand-in", **over}))
    monkeypatch.setenv("LLAMACPP_DIR", "stand-in")
    monkeypatch.setenv("LLAMACPP_CONVERT", "stand-in.py")
    monkeypatch.setattr(K, "FINALS", {"toy": {"sensitive_threshold": 0.5, "gptq_groupsize": 32,
                                              "calib_dataset": "c4", "calib_samples": 16, "stage1_method": "rtn",
                                              "stage2_method": "unstructured_prune", "stage3_method": "snapkv"}})
    monkeypatch.setattr(K, "PRUNE", {"toy": 0.1})
    monkeypatch.setattr(vllm_backend, "measure", lambda model, tok, plan, val, held, cfg, *a, **k: {
        "vllm_kv_cache_tokens": 2.0 if cfg.vllm_kv_cache_dtype == "fp8" else 1.0})
    monkeypatch.setattr(llamacpp, "measure", lambda model, tok, plan, val, held, cfg, *a: {
        "llamacpp_kv_cache_gb": 0.25 if cfg.llamacpp_kv_cache_type == "q4_0" else 1.0})
    out = K.run(inject=dict(model_factory=lambda: copy.deepcopy(tiny_llama), tokenizer=tokenizer,
                            text_loader=fake_texts))
    rows = list(csv.DictReader((out / "kv_engines.csv").open(encoding="utf-8")))
    assert [(r["model"], r["kv_cache"]) for r in rows] == [
        (m, kv) for m in ("baseline_fp16", "toy") for kv in ("16-bit KV cache", "compressed KV cache")]
    assert not [r for r in rows if r.get("error")], rows
    for r in rows:
        compressed = r["kv_cache"] == "compressed KV cache"
        assert float(r["vllm_kv_cache_tokens"]) == (2.0 if compressed else 1.0)
        assert float(r["llamacpp_kv_cache_gb"]) == (0.25 if compressed else 1.0)
    assert rows[0]["ppl_val"] == rows[1]["ppl_val"]  # same model: only the engines' cache differs
    assert rows[0]["stage1"] == "no_quant" and rows[2]["stage1"] == "rtn" and float(rows[2]["prune_share"]) == 0.1
    assert (out / "summary.md").exists()


def test_kv_engines_script_refuses_without_both_engines(small_cfg, monkeypatch):
    root = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("kv_engines_20261123", root / "kv_engines_20261123.py")
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    monkeypatch.setattr(K, "_cfg", lambda run, run_id, **over: small_cfg.with_overrides(
        {"run.run_id": run_id, "eval.vllm_python": "stand-in", **over}))
    monkeypatch.delenv("LLAMACPP_DIR", raising=False)
    with pytest.raises(SystemExit, match="both engines"):
        K.run()
