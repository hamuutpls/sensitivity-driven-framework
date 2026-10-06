"""Stage 4 backend: llama.cpp, measured with its ready-made programs (github.com/ggml-org/llama.cpp releases).

For each row the compressed model (FP16 storage, already rounded by the Stage 1 method) is written as an HF
checkpoint, converted to a GGUF file (convert_hf_to_gguf.py), stored by llama-quantize with each layer's planned
bits (one llama.cpp type per bit width, GGUF_TYPES), and measured: file size, perplexity (llama-perplexity, on
the same validation / held-out text as HF), GPU memory, and prompt / writing speed (llama-bench, which runs a
warmup and repeats). The FP16 row is the converted file without llama-quantize.

What differs from the HF measurement, and is said in the report:
- llama.cpp stores numbers in its own formats (blocks of 32 inside groups of 256), so it rounds the method's
  weights once more; gptq_groupsize does not apply.
- llama.cpp has no sparse format: pruned weights are stored as zeros and save no space.
- llama-perplexity scores only the second half of each window (the first half is context), so its perplexity
  is lower than HF's on the same text; compare rows within one backend.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sdf.stage0.planner import CompressionPlan
from sdf.utils.logging import get_logger

log = get_logger(__name__)

# llama.cpp storage type per planned bit width (k-quants: 256-number groups with 6-8-bit scales).
GGUF_TYPES = {16: "f16", 8: "q8_0", 6: "q6_k", 5: "q5_k", 4: "q4_k", 3: "q3_k", 2: "q2_k"}


def tensor_type_args(plan: CompressionPlan) -> list[str]:
    """llama-quantize --tensor-type arguments: one regex per storage type over the layers ("blk.<i>.") with it."""
    by_type: dict[str, list[int]] = {}
    for lp in plan.layers:
        if lp.bit_width not in GGUF_TYPES:
            raise ValueError(f"llama.cpp has no {lp.bit_width}-bit type (has {sorted(GGUF_TYPES)})")
        by_type.setdefault(GGUF_TYPES[lp.bit_width], []).append(lp.layer)
    out = []
    for t, layers in by_type.items():
        out += ["--tensor-type", rf"blk\.({'|'.join(map(str, layers))})\.={t}"]
    return out


def parse_perplexity(log_text: str) -> tuple[float, float]:
    """(ppl, GPU buffers in GB) from llama-perplexity's output."""
    m = re.search(r"Final estimate: PPL = ([\d.]+)", log_text)
    if not m:
        raise RuntimeError("llama-perplexity printed no final estimate:\n" + log_text[-2000:])
    mib = sum(float(x) for x in re.findall(r"CUDA\d+ [\w ]*?buffer size =\s*([\d.]+) MiB", log_text))
    return float(m.group(1)), mib * 2 ** 20 / 1e9


def parse_bench(results: list[dict[str, Any]]) -> dict[str, float]:
    """Speed metrics from llama-bench -o json (a prompt-only test and a writing-only test)."""
    out = {}
    for r in results:
        ts, sd = r["avg_ts"], r.get("stddev_ts", 0.0)
        if r["n_gen"] == 0:  # prompt reading: n_prompt tokens at ts tok/s
            ms = r["n_prompt"] * 1e3 / ts
            out |= {"prefill_tokens_per_s": ts, "prefill_ms_mean": ms, "prefill_ms_std": ms * sd / ts}
        else:  # writing: one token at a time
            ms = 1e3 / ts
            out |= {"decode_tokens_per_s": ts, "decode_ms_per_token_mean": ms, "decode_ms_per_token_std": ms * sd / ts}
    return out


class LlamaCpp:
    name = "llama_cpp"
    label = "llama.cpp"
    notes = [  # plain-language caveats for the report
        "llama.cpp stores numbers in its own compressed formats, so it rounds each method's numbers once more; "
        "the file size, memory and speed here are real, not predicted.",
        "llama.cpp cannot skip removed (pruned) numbers: it stores them as zeros, so pruning saves no space here.",
        "llama.cpp's perplexity tool scores only the second half of each piece of text (the first half is context), "
        "so its perplexities are lower than in Stage 1. Compare rows within this report, not with Stage 1.",
    ]

    def __init__(self, bin_dir: str | Path, convert_script: str | Path, eval_cfg, work_dir: str | Path,
                 gpu_layers: int = 99):
        self.bin_dir, self.convert = Path(bin_dir), Path(convert_script)
        self.eval, self.gpu_layers = eval_cfg, gpu_layers
        self.work = Path(work_dir)
        if not self.convert.is_file():
            raise FileNotFoundError(f"{self.convert} not found; set LLAMA_CPP_CONVERT to llama.cpp's "
                                    "convert_hf_to_gguf.py")
        self.version = self._run("llama-perplexity", "--version").strip().splitlines()[-1]
        self.texts: dict[str, Path] = {}

    @property
    def key(self) -> dict[str, Any]:
        """Cache-key part: what makes a llama.cpp measurement."""
        return {"backend": self.name, "version": self.version, "types": GGUF_TYPES, "gpu_layers": self.gpu_layers,
                "bench": [self.eval.latency_prompt_len, self.eval.latency_decode_tokens, self.eval.latency_repeats]}

    def _exe(self, name: str) -> str:
        exe = shutil.which(name, path=str(self.bin_dir))
        if exe is None:
            raise FileNotFoundError(f"{name} not in {self.bin_dir}; set LLAMA_CPP_DIR to the unpacked llama.cpp release")
        return exe

    def _run(self, name: str, *args: Any, stdout_only: bool = False) -> str:
        cmd = [sys.executable, str(self.convert)] if name == "convert" else [self._exe(name)]
        cmd += [str(a) for a in args]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode:
            raise RuntimeError(f"{' '.join(cmd)} failed ({p.returncode}):\n{(p.stdout + p.stderr)[-3000:]}")
        return p.stdout if stdout_only else p.stdout + p.stderr

    def _write_texts(self, windows: tuple[torch.Tensor, torch.Tensor], tokenizer) -> None:
        """The HF evaluation windows as text, so llama.cpp reads the same validation / held-out text."""
        self.work.mkdir(parents=True, exist_ok=True)
        for name, w in zip(("val", "heldout"), windows):
            path = self.work / f"{name}.txt"
            path.write_text("\n".join(tokenizer.decode(x.tolist()) for x in w), encoding="utf-8")
            self.texts[name] = path

    def measure(self, model: nn.Module, plan: CompressionPlan | None, windows, tokenizer
                ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Every llama.cpp metric for one model; plan None = FP16 file. Returns (metrics, raw records)."""
        if not self.texts:
            self._write_texts(windows, tokenizer)
        with tempfile.TemporaryDirectory(dir=self.work) as tmp:
            tmp = Path(tmp)
            model.save_pretrained(tmp / "hf")
            tokenizer.save_pretrained(tmp / "hf")
            gguf = tmp / "model-f16.gguf"
            self._run("convert", tmp / "hf", "--outtype", "f16", "--outfile", gguf)
            shutil.rmtree(tmp / "hf")
            if plan is not None:
                q = tmp / "model.gguf"
                # --pure: exactly the planned type per layer (no llama.cpp mixtures); embeddings and output stay FP16
                self._run("llama-quantize", "--pure", "--output-tensor-type", "f16", "--token-embedding-type", "f16",
                          *tensor_type_args(plan), gguf, q, GGUF_TYPES[plan.layers[0].bit_width])
                gguf.unlink()
                gguf = q
            metrics: dict[str, Any] = {"model_size_gb": gguf.stat().st_size / 1e9}
            raw = []
            ngl = ("-ngl", self.gpu_layers)
            for half, metric in (("val", "ppl_val"), ("heldout", "ppl_heldout")):
                ppl, gpu_gb = parse_perplexity(self._run("llama-perplexity", "-m", gguf, "-f", self.texts[half],
                                                         "-c", self.eval.seq_len, *ngl))
                if not math.isfinite(ppl):
                    raise FloatingPointError(f"llama.cpp perplexity is not finite ({ppl})")
                metrics[metric] = ppl
                metrics["peak_memory_gb"] = max(metrics.get("peak_memory_gb", 0.0), gpu_gb)
                raw.append({"measurement": metric, "value": ppl})
            bench = json.loads(self._run("llama-bench", "-m", gguf, "-p", self.eval.latency_prompt_len,
                                         "-n", self.eval.latency_decode_tokens, "-r", self.eval.latency_repeats,
                                         *ngl, "-o", "json", stdout_only=True))
            metrics |= parse_bench(bench)
            raw += [{"measurement": "llama_bench", **{k: r.get(k) for k in ("n_prompt", "n_gen", "avg_ts", "stddev_ts")}}
                    for r in bench]
        log.info("llama.cpp: ppl_val=%.3f size=%.3fGB decode=%.1f tok/s", metrics["ppl_val"], metrics["model_size_gb"],
                 metrics.get("decode_tokens_per_s", float("nan")))
        return metrics, raw

