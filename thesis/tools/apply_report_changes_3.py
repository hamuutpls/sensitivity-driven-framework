"""Third set of anchored changes to thesis-single-file.tex: the presentation-style graphs of Section 5.3.

    python apply_report_changes_3.py <thesis-single-file.tex> <output.tex>

Each old text must occur exactly once, otherwise nothing is written.
"""
import sys
from pathlib import Path

PAIRS = [('quantization figure', '\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.95\\textwidth]{Figures/ch5-quantization.pdf}\n\\caption{Hugging Face validation perplexity of weight-only quantization (lower is better). The dashed line is the FP16 model.}\n\\label{fig:ch5-quant}\n\\end{figure}\n', '\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.9\\textwidth]{Figures/ch5-quant-dumbbell.pdf}\n\\caption{Weight-only quantization: Hugging Face validation perplexity of each technique on the Standard plan (circle) and with Our method (square). Further left is better. The dashed line is the FP16 model.}\n\\label{fig:ch5-quant}\n\\end{figure}\n\n\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.9\\textwidth]{Figures/ch5-quant-gain.pdf}\n\\caption{How much lower the validation perplexity of each weight technique is with Our method than on the Standard plan, in percent of the Standard value (a longer bar is a larger gain).}\n\\label{fig:ch5-quant-gain}\n\\end{figure}\n'), ('gain sentence', 'where the range is 10.27 to 11.70.', 'where the range is 10.27 to 11.70. Figure~\\ref{fig:ch5-quant-gain} gives the same gain in percent: 10.4 percent for PB-LLM, 5.1 percent for round-to-nearest and 0.6 percent for SpQR.'), ('size figure', 'for 0.24~GB of extra file size and about 15 percent of the llama.cpp speed.', 'for 0.24~GB of extra file size and about 15 percent of the llama.cpp speed. Figure~\\ref{fig:ch5-quant-size} shows this trade of quality against size for every technique.\n\n\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.9\\textwidth]{Figures/ch5-quant-size.pdf}\n\\caption{Quality against size for weight-only quantization. Each line joins one technique on the Standard plan (circle, 0.81~GB) to the same technique with Our method (square, 1.05~GB). Lower is better for perplexity and smaller is better for size. The dashed line is the FP16 model.}\n\\label{fig:ch5-quant-size}\n\\end{figure}'), ('activation figure', 'Table~\\ref{tab:ch5-quant-wa} reports the combined techniques.', 'Table~\\ref{tab:ch5-quant-wa} and Figure~\\ref{fig:ch5-quant-wa} report the combined techniques.\n\n\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.9\\textwidth]{Figures/ch5-quant-wa-dumbbell.pdf}\n\\caption{Weights and activations: Hugging Face validation perplexity of each combined technique on the Standard plan (circle) and with Our method (square). Further left is better. The dashed line is the FP16 model.}\n\\label{fig:ch5-quant-wa}\n\\end{figure}\n')]
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
raw = src.read_bytes(); crlf = b"\r\n" in raw
s = raw.decode("utf-8").replace("\r\n", "\n")
for name, old, new in PAIRS:
    if s.count(old) != 1:
        sys.exit(f"STOP: '{name}' found {s.count(old)} times (need exactly 1); nothing written")
    s = s.replace(old, new); print("done", name)
dst.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
