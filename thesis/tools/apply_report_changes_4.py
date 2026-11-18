"""Fourth set of anchored changes to thesis-single-file.tex: the paper changes of 18 Nov 2026
(round-to-nearest model as main result, comparison with published TinyLlama work,
search budget rule, two clarifying sentences).

    python apply_report_changes_4.py <thesis-single-file.tex> <output.tex>

Each old text must occur exactly once, otherwise nothing is written.
"""
import sys
from pathlib import Path

RTN = "round-to-nearest on the weights only (threshold 0.800, groups of 64, 128 windows of C4), no pruning and SnapKV"
OMNI = "OmniQuant on the weights only (threshold 0.857, per-channel groups, 32 windows of C4), unstructured pruning with a ratio of 0.082 and InfiniGen"

PUBLISHED = r"""\section{Comparison with Published Work on TinyLlama}
\label{sec:ch5-published}

The final models are compared with published studies that compress the same model, TinyLlama, and only results measured in the same engine are compared. Because the studies use different evaluation settings, the comparison is made on the change against each study's own uncompressed model. Table~\ref{tab:ch5-published} covers Hugging Face, where two pruning studies report perplexity and accuracy \cite{ilin2026qkwanda, ilin2025thanos}. The two final models found under both limits raise the Hugging Face perplexity by 9.2 and 13.0 percent while the vLLM file becomes 59 and 62 percent smaller; pruning half of the weights raises it by 38.6 to 44.4 percent, and pruning 30 percent by 6.9 percent. The pruning studies lose less accuracy than the two final models, on a different set of tasks, and their uncompressed perplexity (7.77 and 7.97 against 10.18) shows that the evaluation settings differ.

\begin{table}[htbp]
\centering
\small
\caption{Hugging Face results on TinyLlama: change against each study's own uncompressed model. Values for other studies are calculated from the numbers reported in the cited papers (perplexity on WikiText-2). The perplexity of this work is the Hugging Face value on the validation set. Lower is better for the change in perplexity; a smaller loss of accuracy is better.}
\label{tab:ch5-published}
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}p{2.2cm}>{\raggedright\arraybackslash}p{1.7cm}>{\raggedright\arraybackslash}p{1.6cm}}
\hline
\textbf{Study} & \textbf{Methods used} & \textbf{Uncompressed perplexity} & \textbf{Perplexity} & \textbf{Mean accuracy} \\
\hline
This work: round-to-nearest, weights only (threshold 0.800, group 64); no pruning; SnapKV & Quantization, pruning, KV cache & 10.18 & $+9.2\%$ & $-21\%$ \\
This work: OmniQuant, weights only (threshold 0.857, per channel); pruning 0.082; InfiniGen & Quantization, pruning, KV cache & 10.18 & $+13.0\%$ & $-11\%$ \\
QK-Wanda \cite{ilin2026qkwanda}, 30\% pruned & Unstructured pruning & 7.77 & $+6.9\%$ & unchanged \\
QK-Wanda \cite{ilin2026qkwanda}, 50\% pruned & Unstructured pruning & 7.77 & $+44.4\%$ & $-2.7\%$ \\
Thanos \cite{ilin2025thanos}, 50\% pruned & Unstructured pruning & 7.97 & $+38.6\%$ & $-7.3\%$ \\
\hline
\end{tabularx}
\end{table}

For the other engines there is no result on both sides. In llama.cpp, Das et al.\ prune 7 percent of TinyLlama, recover with LoRA and quantize to 4-bit GGUF; the FP16 file becomes 4.9 percent smaller and the SQuAD F1 score falls by 52 percent \cite{das2026largemodels}; the final models of this thesis were not measured in llama.cpp. On a phone processor, MobileQuant quantizes weights and activations to 8 bits with 4.7 percent higher perplexity and 15 percent lower decode time than with 16-bit activations \cite{tan2024mobilequant}. EdgeProfiler estimates with an analytical model that 4-bit TinyLlama is 71 percent smaller and 2.45 times faster \cite{pinnock2025edgeprofiler}, and Belhaouari and Kraidia report 45 percent compression and a 1.68 times speedup from pruning with retraining \cite{belhaouari2025smartpruning}. No published study was found that measures a compressed TinyLlama in vLLM, or that combines quantization, pruning and KV-cache compression on it.

"""

PAIRS = [
 ('abstract',
  "The search produced two final models that are about 60 percent smaller and 20 percent faster than the uncompressed model in vLLM: OmniQuant weight-only quantization with light unstructured pruning and InfiniGen (0.838~GB, 1.98~ms per token), and round-to-nearest weight-only quantization with SnapKV (0.898~GB, 2.00~ms per token), against 2.200~GB and 2.54~ms per token. Their mean accuracy on four tasks is 11 and 21 percent lower,",
  "The main result is a final model that uses round-to-nearest weight-only quantization (threshold 0.800, groups of 64), no pruning and SnapKV: in vLLM it has 0.898~GB and 2.00~ms per token against 2.200~GB and 2.54~ms per token for the uncompressed model, which is 59 percent smaller and 21 percent faster, and its vLLM perplexity of 11.10 against 10.18 is within both limits. A second final model, OmniQuant weight-only quantization (threshold 0.857) with unstructured pruning at a ratio of 0.082 and InfiniGen (0.838~GB, 1.98~ms per token), is within the perplexity limit in Hugging Face (11.50) but not in vLLM (13.00). Their mean accuracy on four tasks is 21 and 11 percent lower,"),
 ('executive summary',
  "The search gave two final models. The first uses OmniQuant on the weights only (threshold 0.857), unstructured pruning with a ratio of 0.082 and InfiniGen: 0.838~GB and 1.98~ms per token. The second uses round-to-nearest on the weights only (threshold 0.800), no pruning and SnapKV: 0.898~GB and 2.00~ms per token. The uncompressed model has 2.200~GB and 2.54~ms per token. Both final models are about 60 percent smaller and about 20 percent faster, and their mean accuracy on four tasks is 11 and 21 percent lower.",
  "The search gave two final models. The main result uses round-to-nearest on the weights only (threshold 0.800, groups of 64), no pruning and SnapKV: 0.898~GB and 2.00~ms per token, against 2.200~GB and 2.54~ms per token for the uncompressed model. It is 59 percent smaller and 21 percent faster, and it is within both limits in vLLM (perplexity 11.10 against 10.18). The second uses OmniQuant on the weights only (threshold 0.857), unstructured pruning with a ratio of 0.082 and InfiniGen: 0.838~GB and 1.98~ms per token, with a perplexity of 11.50 in Hugging Face and 13.00 in vLLM. Their mean accuracy on four tasks is 21 and 11 percent lower.\n  \\item Compared with published work on TinyLlama in Hugging Face, the two final models raise perplexity by 9.2 and 13.0 percent, against 38.6 to 44.4 percent for pruning half of the weights and 6.9 percent for pruning 30 percent; the pruning studies lose less accuracy, on a different set of tasks."),
 ('resource constraints',
  "The scope is deliberately bounded as follows.",
  "The scope is deliberately bounded as follows.\n\nResource constraints in this thesis mean that the deployment needs a small model and a short time per token, which the search receives as limits and objectives; the GPUs of the experiments serve only as test machines."),
 ('perplexity is not accuracy',
  "The experiments in this thesis use the default score.",
  "The experiments in this thesis use the default score. Perplexity decides where the model is compressed; it is not a guarantee of task accuracy, which is measured separately on the final models."),
 ('budget rule',
  "The search results in Chapter~5 come from the Bayesian optimiser.",
  "The budget is counted in full evaluations, that is, models measured on all validation windows, and a try on a fraction of the windows uses that fraction of one evaluation. For the same budget, successive halving therefore builds more models than the Bayesian optimiser: it first measures random points on one ninth of the windows, then the best third of them on one third of the windows, and then the best third of those on all windows. The search results in Chapter~5 come from the Bayesian optimiser."),
 ('budget in chapter 5',
  "These two searches were also run with successive halving on the same search space and seed,",
  "These two searches were also run with successive halving on the same search space and seed and with the same budget of 30 full evaluations (Section~\\ref{sec:ch4-optimiser}),"),
 ('published work section',
  "\\section{Discussion}\n\\label{sec:ch5-discussion}",
  PUBLISHED + "\\section{Discussion}\n\\label{sec:ch5-discussion}"),
 ('RQ5 main result',
  "The final model of the weight-only search is within both limits in vLLM as well.",
  "The final model of the weight-only search is within both limits in vLLM as well, and it is the main result of the search: round-to-nearest on the weights only (threshold 0.800, groups of 64), no pruning and SnapKV, 59 percent smaller and 21 percent faster than FP16 in vLLM."),
 ('conclusion 6',
  "The first uses " + OMNI + ": 0.838~GB, 1.98~ms per token, perplexity 11.50 in Hugging Face and 13.00 in vLLM, mean accuracy 0.556. The second uses " + RTN + ": 0.898~GB, 2.00~ms per token, perplexity 11.12 in Hugging Face and 11.10 in vLLM, mean accuracy 0.495.",
  "The main result uses " + RTN + ": 0.898~GB, 2.00~ms per token, perplexity 11.12 in Hugging Face and 11.10 in vLLM, mean accuracy 0.495; it is within both limits in vLLM. The second uses " + OMNI + ": 0.838~GB, 1.98~ms per token, perplexity 11.50 in Hugging Face and 13.00 in vLLM, mean accuracy 0.556, so it exceeds the perplexity limit in vLLM."),
 ('conclusion published',
  "When the perplexity limit was checked in vLLM, two searches of 40 tries each found no model within both limits.",
  "When the perplexity limit was checked in vLLM, two searches of 40 tries each found no model within both limits. Against published work on TinyLlama in Hugging Face, the two final models raise perplexity by 9.2 and 13.0 percent, less than pruning half of the weights (38.6 to 44.4 percent), and lose more accuracy than the pruning studies, which use a different set of tasks (Section~\\ref{sec:ch5-published})."),
]
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
raw = src.read_bytes(); crlf = b"\r\n" in raw
s = raw.decode("utf-8").replace("\r\n", "\n")
for name, old, new in PAIRS:
    if s.count(old) != 1:
        sys.exit(f"STOP: '{name}' found {s.count(old)} times (need exactly 1); nothing written")
for name, old, new in PAIRS:
    if s.count(old) != 1:
        sys.exit(f"STOP: '{name}' found {s.count(old)} times after earlier changes; nothing written")
    s = s.replace(old, new); print("done", name)
dst.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
