"""Tenth set of anchored changes to thesis-single-file.tex: the final model described as one model
with the note that SnapKV runs only in Hugging Face; the setting in which the eviction techniques
are scored; corrected numbers; three caption fragments removed from Chapter 3; one engine removed.

    python apply_report_changes_10.py <thesis-single-file.tex> <output.tex>

Each old text must occur exactly once, otherwise nothing is written.
"""
import sys
from pathlib import Path

KV_ONLY_HF = r" SnapKV runs only in Hugging Face, so these vLLM numbers are of the model without it."

CHANGES = [
    # ---- engines -------------------------------------------------------------------------------
    ("exec summary limits",
     r"The results are for one model, most techniques are simplified re-implementations, and no TensorRT-LLM measurement is reported.",
     r"The results are for one model, and most techniques are simplified re-implementations."),
    ("problem statement gap four",
     r"such as Hugging Face Transformers, llama.cpp, vLLM and TensorRT-LLM. Fifth,",
     r"such as Hugging Face Transformers, llama.cpp and vLLM. Fifth,"),
    ("scope engines",
     r" The framework can also export a model to TensorRT-LLM \cite{nvidia2025tensorrtllm}, but no TensorRT-LLM measurement is reported in this thesis.",
     r""),
    ("section 2.6 opening",
     r"Four inference engines are used in this thesis.",
     r"Three inference engines are used in this thesis."),
    ("section 2.6 vLLM",
     r"TensorRT-LLM is NVIDIA's inference library, which optimises LLM inference with specialised kernels, an efficient runtime and a Python framework for customisation \cite{nvidia2025tensorrtllm}; it builds a TensorRT engine for the model. Both vLLM and TensorRT-LLM re-round the exported weights onto their own quantization grids, so the per-layer plan survives export only to the extent that the engine's format can represent different bit widths in different layers. Measuring the same model in all four engines",
     r"vLLM re-rounds the exported weights onto its own quantization grids, so the per-layer plan survives export only to the extent that the engine's format can represent different bit widths in different layers. Measuring the same model in all three engines"),
    ("section 2.9",
     r"so this thesis measures every model in Hugging Face Transformers, llama.cpp, vLLM and TensorRT-LLM. Finally,",
     r"so this thesis measures every model in Hugging Face Transformers, llama.cpp and vLLM. Finally,"),
    ("stage 4 opening",
     r" An export to TensorRT-LLM \cite{nvidia2025tensorrtllm} is also implemented, but no TensorRT-LLM measurement is reported in this thesis.",
     r""),
    ("engine table row",
     "TensorRT-LLM (implemented, not measured in this thesis) & The same GPTQ-format checkpoint is built into a TensorRT engine, with one bit width or with a per-layer mixed-precision configuration. \\\\\n",
     ""),
    ("worker processes",
     r"vLLM and TensorRT-LLM run in Python environments of their own, separate from the framework, and are started by the framework as worker processes. In both, the latency is measured",
     r"vLLM runs in a Python environment of its own, separate from the framework, and is started by the framework as a worker process. There the latency is measured"),
    ("tries",
     r"llama.cpp and TensorRT-LLM are not run during the tries.",
     r"llama.cpp is not run during the tries."),
    ("setup table row",
     "TensorRT-LLM & version 1.2, in its own Python environment (Linux only); not measured in this thesis \\\\\n",
     ""),
    ("experiments table note",
     r" No TensorRT-LLM measurement is reported in this chapter; the measured engines are Hugging Face Transformers, llama.cpp and vLLM.",
     r" The measured engines are Hugging Face Transformers, llama.cpp and vLLM."),
    ("limitations engine",
     r" No TensorRT-LLM measurement is reported. In the searches,",
     r" In the searches,"),
    ("future work engines",
     r"\item \textbf{Engines and sparse formats.} Measurements in TensorRT-LLM, and in an engine that stores pruned weights in a sparse format, would complete the comparison across engines and give pruning a fair test of size and speed.",
     r"\item \textbf{Sparse formats.} Measurements in an engine that stores pruned weights in a sparse format would give pruning a fair test of size and speed."),

    # ---- the final model as one model -----------------------------------------------------------
    ("abstract: one model, 11 layers",
     r"Giving 8 bits to the sensitive layers lowers the perplexity of all eleven weight-quantization techniques",
     r"Giving 8 bits to the 11 most sensitive layers lowers the perplexity of all eleven weight-quantization techniques"),
    ("abstract: SnapKV only in HF",
     r"and its vLLM perplexity of 11.10 against 10.18 is within both limits. A second final model,",
     r"and its vLLM perplexity of 11.10 against 10.18 is within both limits." + KV_ONLY_HF + r" A second final model,"),
    ("abstract: KV setting",
     r"A stage ablation shows that this loss comes from the KV-cache step, which runs only in Hugging Face, and not from quantization: with quantization alone the mean accuracy is 0.624 and 0.613 against 0.625.",
     r"A stage ablation shows that this loss comes from the KV-cache step, which runs only in Hugging Face, and not from quantization: with quantization alone the mean accuracy is 0.624 and 0.613 against 0.625. In this step tokens are dropped while the scored text is still being read, which is harsher than the published SnapKV, so the loss holds for this setting."),
    ("exec summary: 11 layers",
     r"\item Giving 8 bits to the sensitive layers and 4 bits to the rest lowered",
     r"\item Giving 8 bits to the 11 most sensitive layers and 4 bits to the rest lowered"),
    ("exec summary: one model",
     r"and it is within both limits in vLLM (perplexity 11.10 against 10.18). The second uses",
     r"and it is within both limits in vLLM (perplexity 11.10 against 10.18)." + KV_ONLY_HF + r" The second uses"),
    ("exec summary: KV setting",
     r"A stage ablation shows that this loss comes from the KV-cache step and not from quantization: with quantization alone the mean accuracy is 0.624 and 0.613 against 0.625.",
     r"A stage ablation shows that this loss comes from the KV-cache step and not from quantization: with quantization alone the mean accuracy is 0.624 and 0.613 against 0.625. In this step tokens are dropped while the scored text is still being read, which is harsher than the published SnapKV, so the loss holds for this setting."),
    ("exec summary: what it means",
     r"a KV-cache technique lowers the cache memory but was the step that cost task accuracy;",
     r"a KV-cache technique lowers the cache memory but, in the harsh setting used here, was the step that cost task accuracy;"),
    ("section 5.6.1 weight-only search",
     r"and it decodes in 2.00~ms per token (21 percent faster). Its perplexity is 11.12 in Hugging Face",
     r"and it decodes in 2.00~ms per token (21 percent faster). SnapKV runs only in Hugging Face, so the vLLM file, latency and perplexity are of the model without it, and the Hugging Face perplexity and the accuracy are of the model with it. Its perplexity is 11.12 in Hugging Face"),
    ("finals table caption",
     r"\caption{The final models, measured in vLLM on the A100, and their accuracy. Perplexity is given on the validation set and on the held-out set.",
     r"\caption{The final models, measured in vLLM on the A100, and their accuracy. The KV-cache technique of a column runs only in Hugging Face: the vLLM rows are of the model without it, and the Hugging Face perplexity and the accuracy are of the model with it. Perplexity is given on the validation set and on the held-out set."),
    ("RQ5: one model and KV setting",
     r"no pruning and SnapKV, 59 percent smaller and 21 percent faster than FP16 in vLLM. With the perplexity judged in vLLM,",
     r"no pruning and SnapKV, 59 percent smaller and 21 percent faster than FP16 in vLLM." + KV_ONLY_HF + r" With the perplexity judged in vLLM,"),
    ("RQ5: KV setting",
     r"The stage ablation shows that this loss comes from the KV-cache step and not from quantization (Section~\ref{sec:ch5-checks}). A requirement",
     r"The stage ablation shows that this loss comes from the KV-cache step and not from quantization (Section~\ref{sec:ch5-checks}). That step drops tokens while the scored text is still being read, a harsher setting than the published one (Section~\ref{sec:ch4-stage3}), so the size of the loss holds for this setting. A requirement"),
    ("conclusion 6: one model",
     r"mean accuracy 0.495; it is within both limits in vLLM. The second uses",
     r"mean accuracy 0.495; it is within both limits in vLLM. SnapKV runs only in Hugging Face, so the vLLM numbers are of the model without it, and the Hugging Face perplexity and the accuracy are of the model with it. The second uses"),

    # ---- the setting in which the eviction techniques are scored --------------------------------
    ("section 4.5 protocol",
     r"keys and values are rounded in place, and evicted tokens are masked out of the attention. The cache memory",
     r"keys and values are rounded in place, and evicted tokens are masked out of the attention. For the eviction techniques this is done while the scored text is being read: every position of the text is treated as the end of a prompt and may attend only to a cache of fixed size, which holds the kept share of the tokens of the text. In the published SnapKV the positions are selected once, from an observation window at the end of the prompt (Section~\ref{sec:ch2-kvcompress}), and the reduced cache is then used for the generation \cite{li2024snapkv}. The setting used here is therefore harsher, and the perplexity and accuracy of the eviction techniques in Chapter~5 are results for this setting and not for the published techniques. The cache memory"),
    ("table 4.3 SnapKV row",
     r"in a cache of the same fixed size as H2O. \\" + "\n" + r"InfiniGen",
     r"in a cache of the same fixed size as H2O. The selection is made at every position of the scored text, not once at the end of the prompt as in the paper. \\" + "\n" + r"InfiniGen"),
    ("section 5.7.3 KV setting",
     r"therefore comes from the KV-cache technique and not from quantization. Because that step does not run in vLLM,",
     r"therefore comes from the KV-cache technique and not from quantization. The loss is measured in the setting of Section~\ref{sec:ch4-stage3}, in which tokens are dropped while the scored text is still being read and not only after the prompt as in the published SnapKV; it is a result for this harsher setting. Because that step does not run in vLLM,"),
    ("RQ3 answer",
     r"and is the step that lowers task accuracy (Section~\ref{sec:ch5-checks}).",
     r"and, in the harsh setting in which it is scored here (Section~\ref{sec:ch4-stage3}), is the step that lowers task accuracy (Section~\ref{sec:ch5-checks})."),
    ("conclusion 7",
     r"while SnapKV alone lowers it to 0.497 and InfiniGen alone to 0.562.",
     r"while SnapKV alone lowers it to 0.497 and InfiniGen alone to 0.562. In the KV-cache step tokens are dropped while the scored text is still being read, and not only after the prompt as in the published SnapKV, so this loss is measured under a harsher setting than the published one."),
    ("limitations KV setting",
     r"and the KV-cache techniques could not be exported. Latency was measured",
     r"and the KV-cache techniques could not be exported. The eviction techniques drop tokens while the scored text is still being read, a harsher setting than the published one, so their perplexity and accuracy are results for this setting. Latency was measured"),

    # ---- numbers --------------------------------------------------------------------------------
    ("standard deviation",
     r"repeated 10 times, and the mean and the standard deviation are reported.",
     r"repeated 10 times, and the mean is reported."),
    ("activation latency range",
     r"decodes in about 35 to 42~ms per token.",
     r"decodes in 33 to 43~ms per token."),
    ("llama.cpp speed range",
     r"and 1.7 to 2.0 times the FP16 tokens per second in llama.cpp",
     r"and 1.6 to 2.0 times the FP16 tokens per second in llama.cpp"),
    ("table 5.7 row, 2.0 ms search",
     r"HF & 30 & 1 & In effect the final model of the first search (same technique and calibration, perplexity 11.50) \\",
     r"HF & 30 & 1 & In effect the final model of the first search (same technique and calibration, perplexity 11.50); 1.96~ms in the try and 2.04~ms when built again as the final model \\"),
    ("2.0 ms search text",
     r"and lowering the latency limit to 2.0~ms left one, which is in effect the final model of the first search.",
     r"and lowering the latency limit to 2.0~ms left one, which is in effect the final model of the first search. That try decoded in 1.96~ms per token in the search and in 2.04~ms when it was built again as the final model, so on the second measurement it is above the 2.0~ms limit; a difference of a few percent in latency is within the variation between measurements."),
    ("published table, methods used",
     r"no pruning; SnapKV & Quantization, pruning, KV cache & 10.18",
     r"no pruning; SnapKV & Quantization, KV cache & 10.18"),
    ("ablation rebuild note",
     r"which removes 5.6 percent of the weights.",
     r"which removes 5.6 percent of the weights. The models are built again for this table, so a value can differ in the last digit from Table~\ref{tab:ch5-finals} (11.11 against 11.12 for the first model)."),

    # ---- same-size pruning ----------------------------------------------------------------------
    ("exec summary same-size",
     r"a pruning plan that removes the same total as uniform pruning was worse in 19 of 20 comparisons,",
     r"a pruning plan that removes the same total as uniform pruning was worse at all 13 levels where it could remove the same total,"),
    ("table 5.5 caption",
     r"\caption{Hugging Face validation perplexity after pruning (lower is better), measured on the A100. FP16: 10.18.}",
     r"\caption{Hugging Face validation perplexity after pruning (lower is better), measured on the A100. FP16: 10.18. $^{*}$: the plan has reached the most it can remove (75 percent of every layer that is not guarded), so it removes less than Standard and the row is not a comparison at equal size.}"),
    ("table 5.5 structured 60",
     r" & 60 & 111 & 25.84 & 2{,}343 \\",
     r" & 60 & 111 & 25.84 & 2{,}343$^{*}$ \\"),
    ("table 5.5 structured 70",
     r" & 70 & 1{,}187 & 67.86 & 2{,}343 \\",
     r" & 70 & 1{,}187 & 67.86 & 2{,}343$^{*}$ \\"),
    ("section 5.4 same-size text",
     r"The same-size plan gives the comparison at equal size, and it is worse than Standard in 19 of the 20 comparisons: at 30 percent unstructured pruning 11.60 against 10.66.",
     r"The same-size plan gives the comparison at equal size. It can remove the same total as Standard only while the layers that are not guarded can hold it: up to 70 percent for unstructured pruning and for the low-rank factorisation, and up to 50 percent for structured pruning, where at most 75 percent of a layer is removed. At these 13 levels it is worse than Standard every time: at 30 percent unstructured pruning 11.60 against 10.66. At the seven higher levels the plan removes less than Standard and its perplexity stops changing (2{,}343 for structured pruning at 60, 70 and 80 percent), so those rows are not comparisons at equal size."),
    ("RQ1 same-size",
     r"was worse than uniform pruning in 19 of 20 comparisons. A layer",
     r"was worse than uniform pruning at all 13 levels where it could remove the same total. A layer"),
    ("conclusion 2 same-size",
     r"was worse than uniform pruning in 19 of 20 comparisons. The KV-cache plan",
     r"was worse than uniform pruning at all 13 levels where it could remove the same total. The KV-cache plan"),

    # ---- caption fragments in Chapter 3 ---------------------------------------------------------
    ("chapter 3 fragment 1",
     r" illation techniques for model-size reduction. ssion and memory-aware inference techniques.",
     r""),
    ("chapter 3 fragment 2",
     r"according to the specific deployment requirement. on of the five compression technique families.",
     r"according to the specific deployment requirement."),
    ("chapter 3 fragment 3",
     r" TABLE VII: Approaches to estimating weight importance and computational cost, and typical use.",
     r""),
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
