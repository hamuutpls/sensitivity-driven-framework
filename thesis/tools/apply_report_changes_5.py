"""Fifth set of anchored changes to thesis-single-file.tex: the paper changes up to 26 Nov 2026
(checks of the score, the per-layer plan and the stages; corrected statements on the
round-to-nearest perplexity and on the cause of the accuracy loss; no GPU card size).

    python apply_report_changes_5.py <thesis-single-file.tex> <output.tex>

Each old text must occur exactly once, otherwise nothing is written.
"""
import sys
from pathlib import Path

CHECKS = r"""\section{Checks of the Score, the Plan and the Stages}
\label{sec:ch5-checks}

Three further experiments test the parts of the framework one at a time: whether the sensitivity score predicts the damage of compressing a layer, whether the layers chosen by the score are a better choice than other layers at the same size, and what each stage contributes to the two final models found under both limits. A fourth measures a compressed KV cache in the engines. They were run on the A100; the latencies of the stage ablation come from a separate session, so they are compared only with each other.

\subsection{Does the Score Predict Compression Damage?}

Each of the 22 layers was changed alone in four ways: its weights were rounded to 8 bits and to 4 bits (round-to-nearest, groups of 128), and 25 percent of its weights were pruned by magnitude and by Wanda. The rise in perplexity was compared with the rise when the layer is bypassed, by Spearman rank correlation. On the calibration text, which the score uses, the correlation with bypassing is 0.04 for 8 bits, 0.18 for 4 bits, 0.15 for magnitude pruning and 0.41 for Wanda; on the validation set it is 0.46, 0.51, 0.54 and 0.42. Rounding one layer to 8 bits changes the perplexity by less than 0.002 in every layer, and the largest rise from 4 bits in one layer is 0.13. Layer~0, by far the most sensitive to bypassing, is hardly affected by 4-bit rounding alone (0.004 on the validation set). The score of Equation~\ref{eq:ch4-score} is therefore only a weak to moderate predictor of the damage of compressing a single layer.

\subsection{Per-layer Plan at Equal Size and at Equal Perplexity}

Table~\ref{tab:ch5-equal} compares bit plans of the same size for weight-only quantization alone, for four of the weight techniques. With 8 bits in 11 layers and 4 bits in the rest, the plan chosen by the score has a lower perplexity than the inverse plan and than each of five random plans, for all four techniques. With 8 bits in 5 layers it is better than the inverse plan for all four techniques and better than every random plan for round-to-nearest and OmniQuant; for GPTQ and SqueezeLLM one of the five random plans is better (10.46 against 10.49 and 10.47 against 10.49). So the plan is a better choice of layers than the alternatives at the same size. The plan is not a gain over 4 bits in every layer at equal size: giving 8 bits to the five most sensitive layers and 3 bits to the rest has about the size of uniform 4 bits but a much higher perplexity (12.55 against 10.97 for round-to-nearest), and every plan with 8 bits in some layers and 4 bits in the rest is larger than uniform 4 bits (0.84~GB or more against 0.78~GB). The plan trades size for quality.

\begin{table}[htbp]
\centering
\footnotesize
\caption{Weight-only quantization in Hugging Face with bit plans of equal size: validation perplexity (FP16: 10.18; lower is better). Plan: 8 bits in the most sensitive layers; inverse: 8 bits in the least sensitive layers; random: range of five random choices of layers. Sizes are the predicted weight memory for round-to-nearest, GPTQ and OmniQuant; for SqueezeLLM they are 0.79, 0.84, 0.94 and 1.12~GB.}
\label{tab:ch5-equal}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{1.7cm}>{\raggedright\arraybackslash}p{1.1cm}>{\raggedright\arraybackslash}p{1.5cm}>{\raggedright\arraybackslash}p{0.75cm}>{\raggedright\arraybackslash}p{0.9cm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{0.75cm}>{\raggedright\arraybackslash}p{0.9cm}>{\raggedright\arraybackslash}X}
\hline
 & \textbf{4 bits in every layer} & \textbf{8 bits in 5 layers, 3 bits in the rest} & \multicolumn{3}{>{\raggedright\arraybackslash}p{4.2cm}}{\textbf{8 bits in 5 layers, 4 bits in the rest (0.89~GB)}} & \multicolumn{3}{>{\raggedright\arraybackslash}p{4.2cm}}{\textbf{8 bits in 11 layers, 4 bits in the rest (1.02~GB)}} \\
\textbf{Technique} & (0.78~GB) & (0.79~GB) & Plan & Inverse & Random & Plan & Inverse & Random \\
\hline
Round-to-nearest & 10.97 & 12.55 & 10.61 & 10.84 & \mbox{10.68--10.80} & 10.41 & 10.71 & \mbox{10.49--10.63} \\
GPTQ & 10.63 & 11.66 & 10.49 & 10.58 & \mbox{10.46--10.57} & 10.32 & 10.51 & \mbox{10.36--10.41} \\
OmniQuant & 10.89 & 12.37 & 10.56 & 10.79 & \mbox{10.62--10.75} & 10.39 & 10.66 & \mbox{10.45--10.58} \\
SqueezeLLM & 10.60 & 11.29 & 10.49 & 10.52 & \mbox{10.47--10.51} & 10.33 & 10.43 & \mbox{10.36--10.45} \\
\hline
\end{tabularx}
\end{table}

Table~\ref{tab:ch5-equalppl} turns the comparison around: for a target perplexity, it gives the smallest plan of each kind, among the plans of Table~\ref{tab:ch5-equal}, that reaches the target. The targets are the perplexities of the two score plans. To reach the perplexity of the score plan with 8 bits in 5 layers (0.89~GB), a random plan needs 11 layers at 8 bits (1.02~GB) for round-to-nearest and OmniQuant, and no inverse plan reaches it; for GPTQ and SqueezeLLM one random plan of the same size reaches it, and for SqueezeLLM the inverse plan needs 11 layers (1.12~GB). No inverse or random plan reaches the perplexity of the score plan with 8 bits in 11 layers, for any of the four techniques. At equal perplexity the plan chosen by the score is therefore as small as or smaller than the other choices of layers.

\begin{table}[htbp]
\centering
\footnotesize
\caption{Weight-only quantization in Hugging Face at equal perplexity: predicted weight memory in GB of the smallest plan of each kind that reaches the target validation perplexity (smaller is better). The targets are the perplexities of the score plans with 8 bits in 5 and in 11 layers. Inverse and random plans were built with 5 and with 11 layers at 8 bits; none: no such plan reaches the target.}
\label{tab:ch5-equalppl}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}
\hline
 & \multicolumn{4}{>{\raggedright\arraybackslash}p{5.2cm}}{\textbf{Target: score plan, 5 layers at 8 bits}} & \multicolumn{4}{>{\raggedright\arraybackslash}p{5.2cm}}{\textbf{Target: score plan, 11 layers at 8 bits}} \\
\textbf{Technique} & Target & Plan & Inverse & Random & Target & Plan & Inverse & Random \\
\hline
Round-to-nearest & 10.61 & 0.89 & none & 1.02 & 10.41 & 1.02 & none & none \\
GPTQ & 10.49 & 0.89 & none & 0.89 & 10.32 & 1.02 & none & none \\
OmniQuant & 10.56 & 0.89 & none & 1.02 & 10.39 & 1.02 & none & none \\
SqueezeLLM & 10.49 & 0.94 & 1.12 & 0.94 & 10.33 & 1.12 & none & none \\
\hline
\end{tabularx}
\end{table}

\subsection{Contribution of Each Stage}

Table~\ref{tab:ch5-ablation} repeats the two final models found under both limits with stages left out. The first uses round-to-nearest on the weights only (threshold 0.800, groups of 64) and SnapKV; the second uses OmniQuant on the weights only (threshold 0.857, per-channel groups) and InfiniGen. In the rows with pruning, both are pruned with unstructured pruning at a ratio of 0.082 in the layers below their threshold, which removes 5.6 percent of the weights.

\begin{table}[htbp]
\centering
\scriptsize
\caption{Stage ablation with the settings of the two final models found under both limits. Q: quantization; P: unstructured pruning at a ratio of 0.082; KV: KV-cache technique. Perplexity on the validation set; accuracy measured in Hugging Face. vLLM was measured on the model before the KV-cache step, so rows that differ only in KV share their vLLM values. Lower is better except for accuracy.}
\label{tab:ch5-ablation}
\renewcommand{\arraystretch}{1.1}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{1.1cm}>{\raggedright\arraybackslash}p{1.5cm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}
\hline
\textbf{Row} & \textbf{Stages} & HF perplexity & ARC-Easy & PIQA & Hella\-Swag & Wino\-Grande & Mean accuracy & vLLM file (GB) & Peak memory (GB) & Decode (ms per token) \\
\hline
\multicolumn{11}{l}{\textbf{Round-to-nearest (threshold 0.800), SnapKV}} \\
Baseline & none & 10.18 & 0.548 & 0.745 & 0.604 & 0.603 & 0.625 & 2.200 & 3.77 & 2.80 \\
E1 & Q & 10.44 & 0.548 & 0.739 & 0.598 & 0.609 & 0.624 & 0.898 & 2.45 & 1.98 \\
E2 & P & 10.18 & 0.548 & 0.746 & 0.606 & 0.605 & 0.626 & 2.200 & 3.77 & 2.82 \\
E3 & KV & 10.82 & 0.332 & 0.632 & 0.521 & 0.504 & 0.497 & 2.200 & 3.77 & 2.80 \\
E4 & Q + P & 10.44 & 0.548 & 0.739 & 0.598 & 0.609 & 0.624 & 0.898 & 2.45 & 2.03 \\
E5 & Q + KV & 11.11 & 0.325 & 0.630 & 0.516 & 0.510 & 0.495 & 0.898 & 2.45 & 1.98 \\
E6 & Q + P + KV & 11.11 & 0.325 & 0.630 & 0.516 & 0.510 & 0.495 & 0.898 & 2.45 & 2.03 \\
\hline
\multicolumn{11}{l}{\textbf{OmniQuant (threshold 0.857), InfiniGen}} \\
Baseline & none & 10.18 & 0.548 & 0.745 & 0.604 & 0.603 & 0.625 & 2.200 & 3.77 & 2.81 \\
E1 & Q & 11.25 & 0.529 & 0.742 & 0.586 & 0.594 & 0.613 & 0.838 & 2.40 & 2.03 \\
E2 & P & 10.18 & 0.548 & 0.745 & 0.606 & 0.605 & 0.626 & 2.200 & 3.77 & 2.81 \\
E3 & KV & 10.36 & 0.464 & 0.694 & 0.584 & 0.505 & 0.562 & 2.200 & 3.77 & 2.81 \\
E4 & Q + P & 11.25 & 0.529 & 0.742 & 0.586 & 0.594 & 0.613 & 0.838 & 2.40 & 2.09 \\
E5 & Q + KV & 11.47 & 0.432 & 0.689 & 0.565 & 0.526 & 0.553 & 0.838 & 2.40 & 2.03 \\
E6 & Q + P + KV & 11.47 & 0.432 & 0.689 & 0.565 & 0.526 & 0.553 & 0.838 & 2.40 & 2.09 \\
\hline
\end{tabularx}
\end{table}

Quantization gives all of the gain in size, memory and speed: for round-to-nearest the vLLM file falls from 2.200 to 0.898~GB, the peak memory from 3.77 to 2.45~GB and the decode latency from 2.80 to 1.98~ms per token, for a Hugging Face perplexity of 10.44 and an unchanged mean accuracy (0.624 against 0.625). Pruning alone at this ratio leaves perplexity, file size and latency unchanged, and after 4-bit rounding it changes nothing at all: rows E4 and E6 equal rows E1 and E5 in every quality number. The KV-cache step causes the loss of accuracy: SnapKV alone raises the perplexity to 10.82 and lowers the mean accuracy from 0.625 to 0.497, and InfiniGen alone gives 10.36 and 0.562. ARC-Easy loses the most: SnapKV alone lowers it from 0.548 to 0.332 and the other three tasks by 0.08 to 0.11, and InfiniGen alone lowers it to 0.464. The loss of 11 to 21 percent in Table~\ref{tab:ch5-finals} therefore comes from the KV-cache technique and not from quantization. Because that step does not run in vLLM, the model that vLLM serves is row E1.

The ablation also separates the two effects in the comparison of the engines. The Hugging Face values of Table~\ref{tab:ch5-finals} include the KV-cache step and the vLLM values do not. For the quantized model alone (row E1) the perplexity is 10.44 in Hugging Face and 11.10 in vLLM for round-to-nearest, and 11.25 and 12.96 for OmniQuant: the export raises it in both cases. The agreement of 11.12 and 11.10 for the round-to-nearest final model is therefore a coincidence of two different effects.

The ratio of 0.082 is the value chosen by the search, and it is too small to show what pruning does; larger ratios are covered by Section~\ref{sec:ch5-prune}. In the searches, 14 tries combined OmniQuant on the weights only with unstructured pruning at ratios of 0.20 to 0.29 and thresholds of 0.857 to 0.891. Their vLLM perplexity is 13.34 to 13.88, against 12.96 for OmniQuant alone in row E1, with files of 0.815 to 0.838~GB and 1.99 to 2.14~ms per token. These tries also differ from row E1 in the threshold and in the other settings, so the difference cannot be assigned to pruning alone.

\subsection{KV-cache Compression in the Engines}

To measure a compressed cache in a real engine, the KV-cache technique of the pipeline was replaced by the method that each engine provides: an 8-bit cache in vLLM and a 4-bit cache in llama.cpp. The two final models, built without their KV-cache step, were measured with the 16-bit and with the compressed cache. In vLLM the 8-bit cache holds twice as many tokens in the same 1~GiB (95{,}312 against 47{,}648). For the round-to-nearest model the perplexity rises from 11.10 to 11.14, the decode latency from 1.97 to 2.50~ms per token and the peak memory from 2.45 to 3.17~GB; for the OmniQuant model the values are 12.96 to 13.02, 2.01 to 2.48~ms and 2.40 to 3.12~GB. In llama.cpp the 4-bit cache is 3.6 times smaller (0.013 against 0.046~GB); the perplexity of the round-to-nearest model rises from 8.40 to 8.53 and its decode speed falls from 507 to 463 tokens per second. At a batch of one and short texts a compressed cache saves little memory and costs time; its gain, more text in the same memory, was not measured with longer contexts or larger batches.

"""

CHANGES = [
    ("abstract",
     "Their mean accuracy on four tasks is 21 and 11 percent lower, which shows that a perplexity limit alone does not protect accuracy.",
     "Their mean accuracy on four tasks is 21 and 11 percent lower. A stage ablation shows that this loss comes from the KV-cache step, which runs only in Hugging Face, and not from quantization: with quantization alone the mean accuracy is 0.624 and 0.613 against 0.625."),
    ("executive summary: accuracy",
     "with a perplexity of 11.50 in Hugging Face and 13.00 in vLLM. Their mean accuracy on four tasks is 21 and 11 percent lower.",
     "with a perplexity of 11.50 in Hugging Face and 13.00 in vLLM. Their mean accuracy on four tasks is 21 and 11 percent lower. A stage ablation shows that this loss comes from the KV-cache step and not from quantization: with quantization alone the mean accuracy is 0.624 and 0.613 against 0.625."),
    ("executive summary: equal size",
     "was worse for eight of twelve techniques.\n",
     "was worse for eight of twelve techniques.\n  \\item For weight-only quantization at equal size, the layers chosen by the score gave a lower perplexity than the least sensitive layers in all eight comparisons and than every random choice of layers in six of eight. A plan with 8 and 3 bits that has the size of 4 bits in every layer was much worse (12.55 against 10.97 for round-to-nearest).\n"),
    ("scope: card size", "NVIDIA A100 80~GB GPU on Google Colab", "NVIDIA A100 GPU on Google Colab"),
    ("setup table: card size", "NVIDIA A100 80~GB on Google Colab", "NVIDIA A100 on Google Colab"),
    ("chapter 5 introduction",
     "reports the sequential path and the search. Section~\\ref{sec:ch5-discussion}",
     "reports the sequential path and the search. Section~\\ref{sec:ch5-checks} tests the score, the per-layer plan and the contribution of each stage, and Section~\\ref{sec:ch5-published} compares the final models with published work. Section~\\ref{sec:ch5-discussion}"),
    ("experiments table",
     "each try is one model built through Stages~1, 2 and 3 & A100 & Hugging Face, vLLM \\\\\n",
     "each try is one model built through Stages~1, 2 and 3 & A100 & Hugging Face, vLLM \\\\\nChecks (Section~\\ref{sec:ch5-checks}) & Each layer changed alone; bit plans of equal size for 4 weight techniques; stage ablation of the two final models found under both limits; the compressed caches of the engines & A100 & Hugging Face, llama.cpp, vLLM \\\\\n"),
    ("weight-only search paragraph",
     "Its perplexity is almost the same in both engines, 11.12 in Hugging Face and 11.10 in vLLM, so this model is within both limits also when the perplexity is judged in vLLM.",
     "Its perplexity is 11.12 in Hugging Face and 11.10 in vLLM, so this model is within both limits also when the perplexity is judged in vLLM. The two values are close by coincidence: the Hugging Face value includes the KV-cache step and the vLLM value does not (Section~\\ref{sec:ch5-checks})."),
    ("checks section", "\\section{Comparison with Published Work on TinyLlama}", CHECKS + "\\section{Comparison with Published Work on TinyLlama}"),
    ("RQ1",
     "A layer that can be bypassed with little damage cannot, on this evidence, absorb a much larger share of the pruning.",
     "A layer that can be bypassed with little damage cannot, on this evidence, absorb a much larger share of the pruning. Measured layer by layer, the score is only a weak to moderate predictor of the damage of compressing a single layer (Spearman correlations of 0.04 to 0.54, Section~\\ref{sec:ch5-checks})."),
    ("RQ2",
     "between the uniform 4-bit and the uniform 8-bit model, and not a gain at equal size.",
     "between the uniform 4-bit and the uniform 8-bit model, and not a gain at equal size. At equal size the layers chosen by the score are a better choice than the least sensitive or random layers, and at equal perplexity the plan is as small as or smaller than these choices; a plan with 8 and 3 bits at the size of the uniform 4-bit model is much worse (Section~\\ref{sec:ch5-checks})."),
    ("RQ4",
     "while the round-to-nearest final model stays at 11.1 and one try",
     "while the round-to-nearest final model has 11.12 and 11.10, an agreement that is a coincidence of two effects (without the KV-cache step it moves from 10.44 to 11.10, Section~\\ref{sec:ch5-checks}), and one try"),
    ("RQ5",
     "and every final model lost at least 11 percent. A requirement",
     "and every final model lost at least 11 percent. The stage ablation shows that this loss comes from the KV-cache step and not from quantization (Section~\\ref{sec:ch5-checks}). A requirement"),
    ("limitations: equal size",
     "only the same-size pruning plan is a comparison at equal size, and no equal-size comparison was made for quantization.",
     "the comparisons at equal size are the same-size pruning plan and, for four weight techniques in Hugging Face, the plans of Table~\\ref{tab:ch5-equal}."),
    ("limitations: accuracy",
     "Accuracy was measured for the final models only.",
     "Accuracy was measured for the final models and for the stage ablation only."),
    ("conclusion 3",
     "The KV-cache plan gives a cache 3 to 4 times smaller at a higher perplexity for five of six techniques.\n",
     "The KV-cache plan gives a cache 3 to 4 times smaller at a higher perplexity for five of six techniques. For weight-only quantization at equal size, the layers chosen by the score gave a lower perplexity than the least sensitive layers in all eight comparisons and than every random choice of layers in six of eight, but a plan with 8 and 3 bits at the size of 4 bits in every layer was much worse.\n"),
    ("conclusion 7",
     "and the final model with the lowest perplexity lost the most.",
     "and the final model with the lowest perplexity lost the most. The stage ablation shows that the loss comes from the KV-cache step: with quantization alone the mean accuracy is 0.624 for round-to-nearest and 0.613 for OmniQuant against 0.625 for FP16, while SnapKV alone lowers it to 0.497 and InfiniGen alone to 0.562."),
    ("future work: equal size",
     "A quantization plan that has the same file size as the uniform 4-bit model, for example with layers below 4 bits, would show whether the plan helps at equal size, which the pruning experiment could test and the quantization experiment could not.",
     "The comparison of bit plans at equal size was made for four weight techniques in Hugging Face. Repeating it in the engines, for the other techniques and with other ways of choosing the layers below 4 bits would show how far it holds."),
    ("scope: export statement",
     "(for one compressed model in this work, 11.50 in Hugging Face and 13.0 in vLLM), so the search judges perplexity in vLLM.",
     "(for one quantized model in this work, 11.25 in Hugging Face and 12.96 in vLLM), so the last two searches judge perplexity in vLLM."),
    ("scope: host PC", "and a desktop PC with an NVIDIA RTX~5070~Ti 16~GB GPU", "and the host PC with an NVIDIA RTX~5070~Ti 16~GB GPU"),
    ("setup table: host PC", "NVIDIA RTX~5070~Ti 16~GB in a desktop PC", "NVIDIA RTX~5070~Ti 16~GB in the host PC"),
    ("executive summary: export",
     "The perplexity of a model can change when it is exported: 11.50 in Hugging Face and 13.00 in vLLM for the same model.",
     "The perplexity of a model can change when it is exported: for the same quantized model, without the KV-cache step, 11.25 in Hugging Face and 12.96 in vLLM with OmniQuant, and 10.44 and 11.10 with round-to-nearest."),
    ("RQ4: OmniQuant",
     "the OmniQuant final model moves from 11.50 to 13.00 and the GPTQ final model",
     "the OmniQuant final model moves from 11.50 to 13.00 (from 11.25 to 12.96 without its KV-cache step) and the GPTQ final model"),
    ("conclusion 5",
     "The export can change the perplexity (11.50 in Hugging Face and 13.00 in vLLM for the same OmniQuant model)",
     "The export can change the perplexity (for the OmniQuant model without its KV-cache step, 11.25 in Hugging Face and 12.96 in vLLM; for round-to-nearest, 10.44 and 11.10)"),
]


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    t = src.read_bytes().decode("utf-8")
    crlf = "\r\n" in t
    if crlf:
        t = t.replace("\r\n", "\n")
    for name, old, new in CHANGES:
        n = t.count(old)
        if n != 1:
            sys.exit(f"STOP: '{name}' found {n} times, nothing written")
        t = t.replace(old, new)
        print("done", name)
    if crlf:
        t = t.replace("\n", "\r\n")
    dst.write_bytes(t.encode("utf-8"))


if __name__ == "__main__":
    main()
