"""Ninth set of anchored changes to thesis-single-file.tex: no mention of the seed.

    python apply_report_changes_9.py <thesis-single-file.tex> <output.tex>

Each old text must occur exactly once, otherwise nothing is written.
"""
import sys
from pathlib import Path

CHANGES = [
    ("executive summary limits",
     r"The results are for one model and one seed, most techniques",
     r"The results are for one model, most techniques"),
    ("RQ2",
     r"Under identical calibration data, seed, evaluation data, hardware and software,",
     r"Under identical calibration data, evaluation data, hardware and software,"),
    ("O3",
     r"with the same calibration data, seed, evaluation data, hardware and software throughout.",
     r"with the same calibration data, evaluation data, hardware and software throughout."),
    ("scope",
     r"Each configuration is run with one seed, so the variation between runs with different seeds is not measured. Quality is measured",
     r"Each configuration is run once, so the variation between repeated runs is not measured. Quality is measured"),
    ("stage 0 cache",
     r"reused by every later run that has the same model, data and seed.",
     r"reused by every later run that has the same model and data."),
    ("calibration data",
     r"of the training split of WikiText-2 with a fixed seed.",
     r"of the training split of WikiText-2."),
    ("comparison protocol",
     r"the same calibration windows, the same seed, the same evaluation windows,",
     r"the same calibration windows, the same evaluation windows,"),
    ("experimental setup",
     r"The seed fixes the random number generators of Python, NumPy and PyTorch, and the deterministic mode of PyTorch is switched on where a deterministic implementation exists. Each configuration is run with this one seed (Section~\ref{sec:ch1-scope}).",
     r"The random number generators of Python, NumPy and PyTorch are fixed, and the deterministic mode of PyTorch is switched on where a deterministic implementation exists. Each configuration is run once (Section~\ref{sec:ch1-scope})."),
    ("experiments",
     r"All of them use TinyLlama, the seed 0 and the data of Section~\ref{sec:ch4-data}.",
     r"All of them use TinyLlama and the data of Section~\ref{sec:ch4-data}."),
    ("optimisers",
     r"on the same search space and seed and with the same budget",
     r"on the same search space and with the same budget"),
    ("limitations",
     r"The results hold for one model, TinyLlama, and one seed; the variation between seeds was not measured, so small differences between rows should not be over-read.",
     r"The results hold for one model, TinyLlama, and each configuration was run once, so small differences between rows should not be over-read."),
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
