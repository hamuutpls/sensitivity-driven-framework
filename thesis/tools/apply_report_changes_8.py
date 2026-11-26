"""Eighth set of anchored changes to thesis-single-file.tex: revised future work, without the seed.

    python apply_report_changes_8.py <thesis-single-file.tex> <output.tex>

Each old text must occur exactly once, otherwise nothing is written.
"""
import sys
from pathlib import Path

CHANGES = [
    ("future work: other models",
     r"\item \textbf{Other models and seeds.} Repeating the experiments on larger models of the Llama family and with several seeds would show whether the findings transfer and how large the variation between runs is.",
     r"\item \textbf{Other models.} Repeating the experiments on larger models of the Llama family and on models of other families would show whether the findings transfer."),
    ("future work: KV cache in the engines",
     r"\item \textbf{KV cache in the engines.} Implementing the KV-cache techniques inside an engine, and testing them with longer contexts and larger batches where the cache is a larger share of the memory, would allow their effect on memory and speed to be measured.",
     r"\item \textbf{KV cache in the engines.} The compressed caches measured in vLLM and llama.cpp are the engines' own methods, at a batch of one and with short texts. Implementing the KV-cache techniques of the pipeline inside an engine, and testing them with longer contexts and larger batches where the cache is a larger share of the memory, would allow their effect on memory and speed to be measured."),
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
