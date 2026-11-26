"""Puts Chapters 5 and 6, the abstract and the executive summary into an existing thesis-single-file.tex.

    python apply_report_changes.py <thesis-single-file.tex> <output.tex>

Everything else in the file is left exactly as it is. Each change is anchored on a text that must occur exactly once;
the script stops without writing anything if a required anchor is missing or if the file already has Chapter 5.
"""
import sys
from pathlib import Path

T = Path(__file__).resolve().parent.parent
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
raw = src.read_bytes()
crlf = b"\r\n" in raw
s = raw.decode("utf-8").replace("\r\n", "\n")
read = lambda p: (T / p).read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n") + "\n"
log = []


def change(name, anchor, new, required=True):
    global s
    n = s.count(anchor)
    if n != 1:
        if required:
            sys.exit(f"STOP: anchor for '{name}' found {n} times (need exactly 1); nothing written")
        log.append(f"SKIPPED {name}: anchor found {n} times")
        return
    s = s.replace(anchor, new)
    log.append(f"done {name}")


if "\\chapter{Results and Discussion}" in s:
    sys.exit("STOP: the file already has Chapter 5; nothing written")

change("abstract", "Add Abstract here\n", read("FrontMatter/abstract.tex"))
change("executive summary", "\\end{abstract}\n",
       "\\end{abstract}\n\n\\cleardoublepage\n\n" + read("FrontMatter/executive-summary.tex"))
change("chapters 5 and 6", "Chapter~5 reports the results obtained with this method.\n",
       "Chapter~5 reports the results obtained with this method.\n\\clearpage\n" + read("Chapters/Chapter5v2.tex")
       + "\\clearpage\n" + read("Chapters/Chapter6v2.tex"))
# the two references to parts of Chapter 4 that were deleted from the single file
if "\\label{eq:ch4-cost}" not in s:
    change("reference to the deleted memory equation",
           "The predicted weight memory of the plan (Equation~\\ref{eq:ch4-cost}) is checked first;",
           "The weight memory that the plan would need is estimated first;", required=False)
if "\\label{tab:ch4-stage0-settings}" not in s:
    change("reference to the deleted settings table",
           "and the settings of Table~\\ref{tab:ch4-stage0-settings}.",
           "and the Stage~0 settings given in Section~\\ref{sec:ch4-stage0}.", required=False)
dst.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
print("\n".join(log))
