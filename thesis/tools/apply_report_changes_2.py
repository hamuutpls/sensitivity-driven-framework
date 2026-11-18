"""Second set of anchored changes to thesis-single-file.tex (three measured engines; dangling literature-review sentence).

    python apply_report_changes_2.py <thesis-single-file.tex> <output.tex> [more files to patch in place ...]

Each old text must occur exactly once in the single file, otherwise nothing is written.
"""
import sys
from pathlib import Path

PAIRS = [
    ("lit review sentence", " Finally, Section~\\ref{sec:lr-conclusion} concludes the survey by summarizing its scope, main findings, and the principal challenges that motivate continued research.", ""),
    ("RQ4", "carry over to real file size, memory and decode latency in llama.cpp, vLLM and TensorRT-LLM?", "carry over to real file size, memory and decode latency in llama.cpp and vLLM?"),
    ("O4", "Export every compressed model to llama.cpp, vLLM and TensorRT-LLM and measure", "Export the compressed models to llama.cpp and vLLM and measure"),
    ("scope", "Deployment metrics are measured in four engines: Hugging Face Transformers \\cite{wolf2020transformers}, which evaluates the quality of the simulated compression, and llama.cpp \\cite{ggml2025llamacpp}, vLLM \\cite{kwon2023vllm} and TensorRT-LLM \\cite{nvidia2025tensorrtllm}, which give real file size, memory and speed after re-rounding the weights onto their own grids.",
     "Deployment metrics are measured in three engines: Hugging Face Transformers \\cite{wolf2020transformers}, which evaluates the quality of the simulated compression, and llama.cpp \\cite{ggml2025llamacpp} and vLLM \\cite{kwon2023vllm}, which give real file size, memory and speed after re-rounding the weights onto their own grids. The framework can also export a model to TensorRT-LLM \\cite{nvidia2025tensorrtllm}, but no TensorRT-LLM measurement is reported in this thesis."),
    ("ch4 overview", "Stage~4 evaluates the resulting model in four inference engines.", "Stage~4 evaluates the resulting model in the inference engines."),
    ("ch4 stage 4", "and it exports the model to llama.cpp \\cite{ggml2025llamacpp}, vLLM \\cite{kwon2023vllm} and TensorRT-LLM \\cite{nvidia2025tensorrtllm} to measure real file size, memory and speed.",
     "and it exports the model to llama.cpp \\cite{ggml2025llamacpp} and vLLM \\cite{kwon2023vllm} to measure real file size, memory and speed. An export to TensorRT-LLM \\cite{nvidia2025tensorrtllm} is also implemented, but no TensorRT-LLM measurement is reported in this thesis."),
    ("ch4 engines table", "TensorRT-LLM & The same GPTQ-format checkpoint is built into a TensorRT engine, with one bit width or with a per-layer mixed-precision configuration. \\\\",
     "TensorRT-LLM (implemented, not measured in this thesis) & The same GPTQ-format checkpoint is built into a TensorRT engine, with one bit width or with a per-layer mixed-precision configuration. \\\\"),
    ("ch4 search caption", "After the last try the final model is evaluated in all four engines and on the accuracy tasks.}", "After the last try the final model is evaluated in the engines and on the accuracy tasks.}"),
    ("ch4 final model", "It is then evaluated in all four engines and on the four accuracy tasks.", "It is then evaluated in the engines that are installed on the machine of the run (Hugging Face and vLLM for the searches of Chapter~5) and on the four accuracy tasks."),
    ("ch4 setup table", "TensorRT-LLM & version 1.2, in its own Python environment (Linux only) \\\\", "TensorRT-LLM & version 1.2, in its own Python environment (Linux only); not measured in this thesis \\\\"),
    ("ch4 setup text", "Because vLLM and TensorRT-LLM run on Linux only, every measurement that involves them, including all search runs, was made on the A100.", "Because vLLM runs on Linux only, every measurement that involves it, including all search runs, was made on the A100."),
    ("ch4 summary", "Stage~4 measures every model in Hugging Face Transformers and, after export, in llama.cpp, vLLM and TensorRT-LLM.", "Stage~4 measures the models in Hugging Face Transformers and, after export, in llama.cpp and vLLM."),
    ("ch6 summary", "The six objectives of Chapter~1 were carried out on TinyLlama, with the exception that no TensorRT-LLM measurement is reported.", "The six objectives of Chapter~1 were carried out on TinyLlama."),
]
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
raw = src.read_bytes(); crlf = b"\r\n" in raw
s = raw.decode("utf-8").replace("\r\n", "\n")
for name, old, new in PAIRS:
    if s.count(old) != 1:
        sys.exit(f"STOP: '{name}' found {s.count(old)} times (need exactly 1); nothing written")
    s = s.replace(old, new); print("done", name)
dst.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
for extra in sys.argv[3:]:
    p = Path(extra); t = p.read_text(encoding="utf-8"); n = 0
    for name, old, new in PAIRS[1:]:
        if t.count(old) == 1: t = t.replace(old, new); n += 1
    p.write_text(t, encoding="utf-8", newline="\n"); print(extra, n, "changes")
