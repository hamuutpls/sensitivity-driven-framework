"""Seventh set of anchored changes to thesis-single-file.tex: KV-cache compression named as its own
technique next to quantization and pruning (aims and objectives, Section 2.5, executive summary,
RQ3 answer, relation to the gaps, conclusion 4).

    python apply_report_changes_7.py <thesis-single-file.tex> <output.tex>

Each old text must occur exactly once, otherwise nothing is written.
"""
import sys
from pathlib import Path

CHANGES = [
    ("aim",
     r"improves the quality, size and latency trade-off of post-training compression for a decoder-only LLM, and to characterise",
     r"improves the quality, size and latency trade-off of three post-training compression techniques, quantization, pruning and KV-cache compression, for a decoder-only LLM, and to characterise"),
    ("RQ2",
     r"improve the quality, size and latency trade-off of published weight, activation and KV-cache compression techniques compared with their usual uniform settings?",
     r"improve the quality, size and latency trade-off of published techniques of each of the three kinds, quantization of weights and activations, pruning, and KV-cache compression, compared with their usual uniform settings?"),
    ("RQ3",
     r"\item \textbf{RQ3.} When is quantization-only, pruning-only or a combined pipeline preferable,",
     r"\item \textbf{RQ3.} What does each of the three techniques, quantization, pruning and KV-cache compression, contribute to quality, size and latency, and when is quantization-only, pruning-only or a combined pipeline preferable,"),
    ("O2",
     r"Implement or adapt published quantization, pruning, low-rank and KV-cache techniques so that each can be run with and without the plan on identical code.",
     r"Implement or adapt published techniques of three kinds, each as its own stage: quantization of weights and activations, pruning (including low-rank factorisation), and KV-cache compression, so that each can be run with and without the plan on identical code."),
    ("O6",
     r"derive guidance on precision levels, pipeline choice, pruning, engine behaviour and adaptive planning for practitioners.",
     r"derive guidance on precision levels, pipeline choice, pruning, KV-cache compression, engine behaviour and adaptive planning for practitioners."),
    ("section 2.5 opening",
     r"This section explains quantization and pruning, adds low-rank factorisation, KV-cache compression and mixed precision, and defines the terms used later.",
     r"This section explains the three techniques used in this thesis, quantization, pruning and KV-cache compression, adds low-rank factorisation and mixed precision, and defines the terms used later."),
    ("executive summary: what it means",
     r"pruning is not worth its quality cost unless the engine can use sparse weights; and a model should be chosen",
     r"pruning is not worth its quality cost unless the engine can use sparse weights; a KV-cache technique lowers the cache memory but was the step that cost task accuracy; and a model should be chosen"),
    ("RQ3 heading",
     r"\textbf{RQ3: quantization only, pruning only, or both?}",
     r"\textbf{RQ3: what does each technique contribute, and which pipeline is preferable?}"),
    ("RQ3 answer",
     r"A KV-cache technique reduces cache memory and does not change the exported model.",
     r"The KV-cache step lowers the predicted cache memory, does not change the exported model, and is the step that lowers task accuracy (Section~\ref{sec:ch5-checks})."),
    ("relation to the gaps",
     r"quantize, and prune only if the engine can use it (gaps two and three).",
     r"quantize, prune only if the engine can use it, and add a KV-cache technique for memory only after checking task accuracy (gaps two and three)."),
    ("conclusion 4",
     r"Pruned models keep the FP16 file size and the FP16 latency in llama.cpp and vLLM, because the removed weights are stored as zeros.",
     r"Pruned models keep the FP16 file size and the FP16 latency in llama.cpp and vLLM, because the removed weights are stored as zeros. The KV-cache techniques of the pipeline could not be exported; the engines' own compressed caches held twice as many tokens in vLLM and were 3.6 times smaller in llama.cpp, at a cost in decode speed."),
]


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    t = src.read_bytes().decode("utf-8")
    for name, old, new in CHANGES:
        n = t.count(old)
        if n != 1:
            sys.exit(f"STOP: '{name}' found {n} times, nothing written")
        t = t.replace(old, new)
        print("done", name)
    dst.write_bytes(t.encode("utf-8"))


if __name__ == "__main__":
    main()
