"""Sixth set of anchored changes to thesis-single-file.tex: the two equations of the paper that the
thesis did not have (the search problem and the constraint value given to the optimiser).

    python apply_report_changes_6.py <thesis-single-file.tex> <output.tex>

Each old text must occur exactly once, otherwise nothing is written.
"""
import sys
from pathlib import Path

CHANGES = [
    ("search problem",
     "The objectives of each search run are given with its results in Chapter~5. The held-out perplexity cannot be selected as an objective or as a limit.\n",
     "The objectives of each search run are given with its results in Chapter~5. The held-out perplexity cannot be selected as an objective or as a limit.\n\n"
     "Written as a formula, let $x = (t_1, t_2, t_3, \\theta)$ be a point of the search space $\\mathcal{X}$ (Section~\\ref{sec:ch4-space}), with $t_i$ the technique of Stage~$i$ and $\\theta$ the numeric settings, and let $P(x)$ be the validation perplexity, $L(x)$ the vLLM decode latency in ms per token and $S(x)$ the predicted weight memory in GB of the model built from $x$. For the requirement of this thesis the search solves\n"
     "\\begin{equation}\n"
     "  \\min_{x \\in \\mathcal{X}} \\; \\bigl(f_1(x), \\ldots, f_k(x)\\bigr) \\quad \\text{subject to} \\quad P(x) \\le P_{\\max}, \\;\\; L(x) \\le L_{\\max},\n"
     "  \\label{eq:ch4-search}\n"
     "\\end{equation}\n"
     "where the objectives $f_1, \\ldots, f_k$ ($k \\le 3$) are taken from $P$, $L$ and $S$ in a stated order, $P_{\\max} = 11.5$ and $L_{\\max} = 2.5$~ms per token.\n"),
    ("constraint value",
     "the requirement is passed to the sampler as a constraint whose value is the sum of the shortfalls of the point, which is zero for a point that meets every limit.",
     "the requirement is passed to the sampler as a constraint whose value is the sum of the shortfalls of the point. For the two limits of Equation~\\ref{eq:ch4-search} it is\n"
     "\\begin{equation}\n"
     "  c(x) = \\max\\bigl(0, P(x) - P_{\\max}\\bigr) + \\max\\bigl(0, L(x) - L_{\\max}\\bigr),\n"
     "  \\label{eq:ch4-constraint}\n"
     "\\end{equation}\n"
     "which is zero for a point that meets every limit."),
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
