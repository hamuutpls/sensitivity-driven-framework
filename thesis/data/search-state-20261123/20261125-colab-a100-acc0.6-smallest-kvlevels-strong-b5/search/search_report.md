# Settings search

## What this shows

The framework tried many settings and built each one as a chained model: the Stage 0 plan (which layers are sensitive), then Stage 1 (weights and activations stored with fewer bits, following the threshold plan: weights 8 bits in sensitive layers and 4 in the rest, activations from the activation plan, [4, 8] bits averaging 6.0; models from runs on the joint plan are not counted), then Stage 2 (pruning: removing numbers), then Stage 3 (compressing the KV cache, the model's memory of earlier words). A setting is the technique of each stage plus the numbers that tune them (protection threshold, pruning share, calibration text and so on). Only settings that meet the limit count (vLLM decode time per token at most 2.5; Downstream accuracy (mean) at least 0.6). Among those the search looks for the lowest predicted weight memory. Speed comes from vLLM, which runs the model up to Stage 2 (it cannot run the Stage 3 techniques); unstructured pruning keeps the removed numbers as zeros, so it does not make vLLM faster. Stage 0's refinement was off: every plan is the one-shot joint plan. 1 search method each had the same budget of 5 full tries, where a full try is one model measured on all the evaluation windows. MFBO's rough tries use fewer windows and count by that share, so it builds more models for the same budget (see Models built below). A setting whose plan is already over the limit is not built and does not count, nor does one that failed before building or a repeat of the method's own try that came from the cache. The methods: MOBO (multi-objective Bayesian optimisation).

## What it means

0 setting(s) are on the best trade-off line: none of them can get better on one goal without getting worse on another, so each is a fair choice depending on what matters more. They are listed below, best predicted weight memory first (lower is better on every goal). Held-out perplexity was measured but never used to choose, so it shows how well the choice holds on unseen text.

## Best trade-off settings

| # | Stage 1 quantization | Stage 2 pruning | Pruning share | Stage 3 technique | Predicted weight memory (lower is better) | vLLM perplexity of the re-rounded file (validation half) (lower is better) | Downstream accuracy (mean) (higher is better) | Perplexity (held-out half) (lower is better) |
|---|---|---|---|---|---|---|---|---|

The other settings of each option:

| # | Protection threshold | Calibration text | Calibration passages | Group size | SmoothQuant strength | QuaRot key bits |
|---|---|---|---|---|---|---|

## KV-cache levels

The levels were searched in this order, each with the same budget; the next level was searched only when no try of the one before met the limits.

| Level | Settings | Tries | Met the limits |
|---|---|---|---|
| strong | average KV bits None; attention kept 0.95 | 5 | 0 |

## Search methods compared

| Search method | Tries | Models built | Of those, on all evaluation windows | Cost, in full tries | Time, minutes | Share of tries under the limit (higher is better) | Best predicted weight memory (lower is better) | Cost until it found that best (lower is better) | Hypervolume (higher is better) | Share of the best trade-off line it found (higher is better) |
|---|---|---|---|---|---|---|---|---|---|---|
| MOBO (multi-objective Bayesian optimisation) | 5 | 0 | 0 | 5 | 0 | 0 |  |  | 0 | 0 |

A full try measures on every evaluation window (a slice of the test text); MFBO's rough tries use fewer windows and count by that share, though each still builds the whole model, so its time column is the fairer cost. Hypervolume measures how much of the space of good results (low predicted weight memory) the method's settings under the limit cover; larger is better.

## Conditions

- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0; seed 0; evaluation windows: all of 512 tokens.
- Techniques allowed: Stage 1 Round-to-nearest + round-to-nearest activations, Round-to-nearest + SmoothQuant, GPTQ + round-to-nearest activations, GPTQ + SmoothQuant, AWQ + round-to-nearest activations, AWQ + SmoothQuant, OmniQuant (LWC) + round-to-nearest activations, OmniQuant (LWC) + SmoothQuant, SqueezeLLM + round-to-nearest activations, SqueezeLLM + SmoothQuant, SpQR + round-to-nearest activations, SpQR + SmoothQuant, EfficientQAT (Block-AP) + round-to-nearest activations, EfficientQAT (Block-AP) + SmoothQuant, QuIP + round-to-nearest activations, QuIP + SmoothQuant, QuIP# + round-to-nearest activations, QuIP# + SmoothQuant, PB-LLM + round-to-nearest activations, PB-LLM + SmoothQuant, bitsandbytes (NF4 / int8) + round-to-nearest activations, bitsandbytes (NF4 / int8) + SmoothQuant, Round-to-nearest, GPTQ, AWQ, OmniQuant (LWC), EfficientQAT (Block-AP); Stage 2 Unstructured pruning (Wanda), Structured pruning (feed-forward channels); Stage 3 Round-to-nearest KV cache, QuaRot KV, KVQuant, H2O, SnapKV, InfiniGen.
- Tries over the limit, so not built: 0 of 5.
- Tries whose model an earlier run already built, so not built again (its earlier result is used): 5 of 5. Tries that were the same model as an earlier try of this run (they differ only in a setting their techniques do not read), so not built either: 0. Every combination tried so far is listed in /content/thesis_compression/cache/search_tried.csv.
- Tries that failed (could not be built or measured): 0 of 5; the reasons are in the Trials sheet of search_results.xlsx.

## Extra Stage 1 techniques

**What this shows:** SpQR, QuIP and QuIP# (the techniques asked for) when no search method built them: one chained try each, the best setting under the limit with only its Stage 1 technique changed. **What it means:** how each would have done in the best model; they are not part of the comparison or the trade-off line, but one is the final model (Stage 4 above) when it is the best under the limit on predicted weight memory.

| Stage 1 technique | Our method WxAy (average bits) | Under the limit | Predicted weight memory (lower is better) | vLLM perplexity of the re-rounded file (validation half) (lower is better) | Downstream accuracy (mean) (higher is better) | Perplexity (held-out half) (lower is better) | Why it failed |
|---|---|---|---|---|---|---|---|
| SpQR + round-to-nearest activations |  | no | 0.9261 | 12.27 | 0.4871 | 13.18 |  |
| QuIP + round-to-nearest activations |  | no | 0.8614 | 12.52 | 0.4889 | 13.54 |  |
