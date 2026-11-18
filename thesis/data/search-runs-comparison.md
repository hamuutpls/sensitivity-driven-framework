# All search runs: comparison for the Results and Discussion section

Model: TinyLlama-1.1B-Chat-v1.0, seed 0, Colab A100-SXM4-80GB for every run. Perplexity (ppl) is WikiText-2 test, split into a validation half (used by the search) and a held-out half (never used to choose). HF = Hugging Face measurement; vLLM = the model exported and served in vLLM (Stage 4). Size for HF is the predicted weight memory; size for vLLM is the real file size.

Sources: run notes and search reports on D:\Thesis\results\colab\<run id>\, and /mnt/project-files/chained-search/*.md. Gathered 2026-10-31; runs 9 to 11 added 2026-11-01, runs 12 to 14 added 2026-11-01, runs 15 to 18 added 2026-11-03 (from config.json, trials.json and the report in each run's out\ folder on D:, read by the PC session).

## Reference: uncompressed TinyLlama (FP16)

| Size | ppl val / held-out | vLLM decode | vLLM peak memory | Accuracy (mean of 4 tasks) |
|---|---|---|---|---|
| 2.200 GB | 10.18 / 10.62 | 2.54 ms/token | 3.766 GB | 0.625 |

## Table 1. Every run at a glance

| # | Run | Code (tests on the run machine: passed/skipped/failed) | Goal (limits) | Search | Tries | Met the limits | How it ended |
|---|---|---|---|---|---|---|---|
| 1 | run1 (20261025-colab-a100-chained-search) | 9cd4ea2 (full suite 423/3/0 at 1cbc67d; search tests 19/0/0) | size <= 1.0 GB and ppl <= 11.0 | MOBO + MFBO, 30 each | MOBO 17 built (stopped at 600-try cap); MFBO 16.9 of 30 budget | 0 | Colab machine lost during MFBO |
| 2 | b40 (20261026-...-chained-search-b40) | d40d1c0 (426/3/0) | size <= 1.0 GB and ppl <= 11.0 | MOBO + MFBO, 40 each | MOBO 131 tries, 7 built (rest skipped as over 1 GB) | 0 | Stopped: goal changed |
| 3 | ppl11 (20261026-...-smallest-ppl11) | b77e868 (430/3/0) | smallest model, ppl <= 11.0 | MOBO + MFBO, 30 each | MOBO 19 | 1 | Stopped: limit raised to 11.5 |
| 4 | ppl11.5 (20261026-...-smallest-ppl11.5) | b77e868 (430/3/0) | smallest model, ppl <= 11.5 | MOBO + MFBO, 30 each | MOBO 30 | 4 | Stopped at MFBO #1 to relaunch on a bug fix |
| 5 | r2 (20261026-...-smallest-ppl11.5-r2) | 78f9416 (432/3/0) | smallest model, ppl <= 11.5 | MOBO + MFBO, 30 each | MOBO 30 (reused from run 4); MFBO 130 | MOBO 4, MFBO 10 | Finished, with Stage 4 and accuracy |
| 6 | thr (20261027-...-smallest-ppl11.5-thr) | cb99c65 (436/3/0) | smallest model, ppl <= 11.5; layer bits 8 or 4 by threshold | MOBO + MFBO, 30 each | MOBO 30; MFBO 3 | 2 | Colab machine lost during MFBO |
| 7 | 3bit (20261027-...-smallest-ppl11.5-3bit) | 2e4e0c8 (437/3/0) | smallest model, ppl <= 11.5; 3-bit layers allowed | MOBO + MFBO, 30 each | MOBO 30; MFBO 130 | MOBO 8, MFBO 0 | Finished, with Stage 4 and accuracy |
| 8 | lat2.5 (20261029-colab-a100-ppl11.5-lat2.5) | 5675e83 (453/4/0, plus bitsandbytes tests 6/0/0) | ppl <= 11.5 and vLLM decode <= 2.5 ms/token; then lowest ppl | MOBO only, 30 | 30 (+1 extra QuIP try) | 2 | Finished, with Stage 4 and accuracy |
| 9 | weightonly (20261031-colab-a100-weightonly) | 5675e83 (454/3/0); accuracy redone at c808d0b (= main af9dde7) after a SnapKV fix | ppl <= 11.5 and vLLM decode <= 2.5 ms/token; weight-only Stage 1 only | MOBO only, 30 | 30 | 17 | Finished, with Stage 4 and accuracy |
| 10 | ppl11-lat2.5 (20261031-colab-a100-ppl11-lat2.5) | 5675e83 (454/3/0) | ppl <= 11.0 and vLLM decode <= 2.5 ms/token; then lowest ppl | MOBO only, 30 | 30 | 0 | Finished, no final model |
| 11 | ppl11.5-lat2.0 (20261031-colab-a100-ppl11.5-lat2.0) | 5675e83 (454/3/0) | ppl <= 11.5 and vLLM decode <= 2.0 ms/token; then lowest ppl | MOBO only, 30 | 30 | 1 | Finished, with Stage 4 and accuracy |
| 12 | structonly (20261031-colab-a100-structonly) | af9dde7 (476/3/0) | ppl <= 11.5 and vLLM decode <= 2.5 ms/token; structured pruning only in Stage 2 | MOBO only, 30 | 30 | 0 | Finished, no final model (follow-up structonly-b40 cancelled by Mohammad) |
| 13 | smallest (20261031-colab-a100-smallest) | 01baa92 (478/3/0) | ppl <= 11.5 and vLLM decode <= 2.5 ms/token; then smallest size | MOBO only, 30 | 30 | 0 | Finished, no final model (follow-up smallest-b40 cancelled by Mohammad) |
| 14 | b40 (20261031-colab-a100-b40) | 01baa92 (478/3/0) | ppl <= 11.5 and vLLM decode <= 2.5 ms/token; then lowest ppl | MOBO only, 40 | 40 | 0 | Finished, no final model; last run (queue cancelled) |
| 15 | lowppl-small-r2 (20261101-colab-a100-lowppl-small-r2) | a178012 (481/3/0) | vLLM ppl <= 11.5 (judged on the vLLM file, not HF) and vLLM decode <= 2.5 ms/token | MOBO only, 30 | 30 (27 built; tries 1-3 reused from 20261101-colab-a100-lowppl-small, whose Colab machine was lost after 3 tries) | 0 | Finished; no pass or fail (outcome = best tries vs FP16 below) |
| 16 | lowlat-small-r2 (20261101-colab-a100-lowlat-small-r2) | a178012 (481/3/0) | vLLM ppl <= 11.5 and vLLM decode <= 2.5 ms/token; then fastest, then smallest | MOBO only, 30 | 30 (tries 1-10 reused from run 15) | 0 | Finished; no pass or fail (outcome = best tries vs FP16 below) |
| 17 | lowppl-small-r2-b40 (20261101-colab-a100-lowppl-small-r2-b40) | a178012 (481/3/0) | run 15's goal with 10 more tries | MOBO only, 40 | 40 (tries 1-30 reused, 31-40 built) | 0 | Finished; no pass or fail (outcome = best tries vs FP16 below) |
| 18 | lowlat-small-r2-b40 (20261101-colab-a100-lowlat-small-r2-b40) | a178012 (481/3/0) | run 16's goal with 10 more tries | MOBO only, 40 | 40 (tries 1-30 reused, 31-40 built) | 0 | Finished; no pass or fail (outcome = best tries vs FP16 below) |

## Table 1b. Search settings of each run

Same in every run: TinyLlama-1.1B-Chat-v1.0, seed 0, all WikiText-2 test windows of 512 tokens (333 validation, 333 held-out), Stage 0 refinement rounds 0, accuracy tasks ARC-Easy, PIQA, HellaSwag and WinoGrande on the final model, Stage 4 backend vLLM only (llama.cpp and TensorRT-LLM were not installed on Colab), extra Stage 1 tries SpQR, QuIP and QuIP# (with RTN activations) on the best setting (not in run 9).

Settings the search could choose (same ranges in every run unless the table says otherwise): calibration text wikitext2, c4 or pile10k; calibration passages 16, 32, 64 or 128; group size 32, 64, 128 or per-channel; SmoothQuant strength 0 to 1; QuaRot key bits 2, 3, 4 or 8. Techniques: Stage 1 = 11 weight methods (RTN, GPTQ, AWQ, OmniQuant, SqueezeLLM, SpQR, EfficientQAT, QuIP, QuIP#, PB-LLM, bitsandbytes), each with RTN activations or SmoothQuant; Stage 2 = unstructured (Wanda) or structured pruning; Stage 3 = RTN KV, QuaRot KV, KVQuant, H2O, SnapKV or InfiniGen.

| # | Run | Searchers and budget | Limits | Objectives (made as small as possible, in order) | Protection threshold range | Pruning share range | Layer bits plan | 3-bit layers | Weight-only Stage 1 options | vLLM measured on | Earlier tries reused |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | run1 | MOBO + MFBO, 30 each | size <= 1.0 GB, ppl <= 11.0 | ppl val, vLLM decode | 0.1 to 0.9 | 0 to 0.6 | joint weight + activation plan | no | no | every try | cache of the earlier stage-search run |
| 2 | b40 | MOBO + MFBO, 40 each | size <= 1.0 GB, ppl <= 11.0 | ppl val, predicted size | 0.1 to 0.9 | 0 to 0.6 | joint plan | no | no | final model only | no (fresh) |
| 3 | ppl11 | MOBO + MFBO, 30 each | ppl <= 11.0, no size limit | predicted size, ppl val | 0.1 to 0.9 | 0 to 0.6 | joint plan | no | no | final model only | yes, b40's cache |
| 4 | ppl11.5 | MOBO + MFBO, 30 each | ppl <= 11.5 | predicted size, ppl val | 0.1 to 0.9 | 0 to 0.6 | joint plan | no | no | final model only | yes (tries 1-10 from run 3) |
| 5 | r2 | MOBO + MFBO, 30 each | ppl <= 11.5 | predicted size, ppl val | 0.1 to 0.9 | 0 to 0.6 | joint plan | no | no | final model only | yes (31 of 160 tries, all MOBO from run 4) |
| 6 | thr | MOBO + MFBO, 30 each | ppl <= 11.5 | predicted size, ppl val | 0.8 to 0.9 | 0 to 0.3 | threshold plan: 8 bits above the threshold, 4 below | no | no | final model only | no (fresh) |
| 7 | 3bit | MOBO + MFBO, 30 each | ppl <= 11.5 | predicted size, ppl val | 0.8 to 0.9 | 0 to 0.3 | threshold plan (8 / 4 bits) | yes, 3-bit threshold 0 to 0.8 | no | final model only | no (fresh) |
| 8 | lat2.5 | MOBO only, 30 | ppl <= 11.5, vLLM decode <= 2.5 ms/token | ppl val, vLLM decode, predicted size | 0.8 to 0.9 | 0 to 0.3 | threshold plan (8 / 4 bits) | no | yes (RTN, GPTQ, AWQ, OmniQuant, EfficientQAT with 16-bit activations) | every try | no (fresh) |
| 9 | weightonly | MOBO only, 30 | ppl <= 11.5, vLLM decode <= 2.5 ms/token | ppl val, vLLM decode, predicted size | 0.8 to 0.9 | 0 to 0.3 | threshold plan (8 / 4 bits) | no | only these (RTN, GPTQ, AWQ, OmniQuant, EfficientQAT with 16-bit activations); no WxAy options | every try | no (fresh) |
| 10 | ppl11-lat2.5 | MOBO only, 30 | ppl <= 11.0, vLLM decode <= 2.5 ms/token | ppl val, vLLM decode, predicted size | 0.8 to 0.9 | 0 to 0.3 | threshold plan (8 / 4 bits) | no | yes (RTN, GPTQ, AWQ, OmniQuant, EfficientQAT with 16-bit activations) | every try | no (fresh) |
| 11 | ppl11.5-lat2.0 | MOBO only, 30 | ppl <= 11.5, vLLM decode <= 2.0 ms/token | ppl val, vLLM decode, predicted size | 0.8 to 0.9 | 0 to 0.3 | threshold plan (8 / 4 bits) | no | yes (RTN, GPTQ, AWQ, OmniQuant, EfficientQAT with 16-bit activations) | every try | no (fresh) |
| 12 | structonly | MOBO only, 30 | ppl <= 11.5, vLLM decode <= 2.5 ms/token | ppl val, vLLM decode, predicted size | 0.8 to 0.9 | 0 to 0.3 | threshold plan (8 / 4 bits) | no | yes (RTN, GPTQ, AWQ, OmniQuant, EfficientQAT with 16-bit activations) | every try | no (fresh); Stage 2 structured pruning only |
| 13 | smallest | MOBO only, 30 | ppl <= 11.5, vLLM decode <= 2.5 ms/token | predicted size, ppl val, vLLM decode | 0.8 to 0.9 | 0 to 0.3 | threshold plan (8 / 4 bits) | no | yes (RTN, GPTQ, AWQ, OmniQuant, EfficientQAT with 16-bit activations) | every try | no (fresh) |
| 14 | b40 | MOBO only, 40 | ppl <= 11.5, vLLM decode <= 2.5 ms/token | ppl val, vLLM decode, predicted size | 0.8 to 0.9 | 0 to 0.3 | threshold plan (8 / 4 bits) | no | yes (RTN, GPTQ, AWQ, OmniQuant, EfficientQAT with 16-bit activations) | every try | no (fresh) |
| 15 | lowppl-small-r2 | MOBO only, 30 | vLLM ppl <= 11.5, vLLM decode <= 2.5 ms/token | vLLM ppl val, predicted size | 0.8 to 0.9 | 0 to 0.3 | threshold plan (8 / 4 bits) | no | yes (RTN, GPTQ, AWQ, OmniQuant, EfficientQAT with 16-bit activations), plus all WxAy options | every try | tries 1-3 from the lost first attempt; extra tries: only QuIP ran (HF ppl 13.28, not met) |
| 16 | lowlat-small-r2 | MOBO only, 30 | vLLM ppl <= 11.5, vLLM decode <= 2.5 ms/token | vLLM decode, predicted size | 0.8 to 0.9 | 0 to 0.3 | threshold plan (8 / 4 bits) | no | yes (RTN, GPTQ, AWQ, OmniQuant, EfficientQAT with 16-bit activations), plus all WxAy options | every try | tries 1-10 from run 15; otherwise same settings as run 15 |
| 17 | lowppl-small-r2-b40 | MOBO only, 40 | vLLM ppl <= 11.5, vLLM decode <= 2.5 ms/token | vLLM ppl val, predicted size | 0.8 to 0.9 | 0 to 0.3 | threshold plan (8 / 4 bits) | no | yes (RTN, GPTQ, AWQ, OmniQuant, EfficientQAT with 16-bit activations), plus all WxAy options | every try | tries 1-30 from earlier runs (cache seeded from run 16); otherwise same settings as run 15 |
| 18 | lowlat-small-r2-b40 | MOBO only, 40 | vLLM ppl <= 11.5, vLLM decode <= 2.5 ms/token | vLLM decode, predicted size | 0.8 to 0.9 | 0 to 0.3 | threshold plan (8 / 4 bits) | no | yes (RTN, GPTQ, AWQ, OmniQuant, EfficientQAT with 16-bit activations), plus all WxAy options | every try | tries 1-30 from earlier runs (cache seeded from run 17); otherwise same settings as run 16 |

Sources: each run's script and run notes on D:, and main.py / src/sdf/search_space.py at each run's commit.

## Table 2. Best model of each run, with all its settings (HF measurements)

For runs that met the limits this is the chosen model; for runs 1 and 2, which met nothing, it is the closest try. "Not used" means the chosen techniques do not read that setting (for example SmoothQuant strength with RTN activations, or QuaRot key bits with InfiniGen). b40's threshold and pruning share come from the same point (a3bfcddfcc, identical scores) rebuilt as MOBO #6 of run 3, in that run's search_tried.csv; lat2.5's values are from its search_tried-final.csv.

| # | Run | Try | Quantization (weights + activations) | Protection threshold | Group size | Calibration | SmoothQuant strength | 3-bit threshold | Pruning | Pruning share | KV cache | QuaRot key bits | Size (GB) | ppl val / held-out | Met? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | run1 | MFBO #56 | bitsandbytes + SmoothQuant | 0.528 | not used (64) | c4 x 128 | 0.851 | – | Unstructured (Wanda) | 0.359 | InfiniGen | not used | 0.939 | 11.44 / 11.88 | No (ppl over 11.0) |
| 2 | b40 | MOBO #6 | EfficientQAT + RTN activations | 0.791 | 128 | pile10k x 32 | not used | – | Structured | 0.071 | SnapKV | not used | 0.858 | 13.36 / 13.67 | No |
| 3 | ppl11 | MOBO #11 | RTN + RTN activations | 0.100 | 32 | wikitext2 x 64 | not used | – | Unstructured (Wanda) | 0.205 | InfiniGen | not used | 1.271 | 10.92 / 11.33 | Yes |
| 4 | ppl11.5 | MOBO #15 | GPTQ + RTN activations | 0.780 | 128 | wikitext2 x 16 | not used | – | Unstructured (Wanda) | 0.122 | InfiniGen | not used | 0.924 | 11.39 / 11.85 | Yes |
| 5 | r2 (final) | run 3's MOBO #15 | GPTQ + RTN activations | 0.759 | 128 | wikitext2 x 64 | not used | – | Unstructured (Wanda) | 0 | InfiniGen | not used | 0.909 | 11.23 / 11.62 | Yes |
| 5b | r2 extra try | QuIP# on row 5's settings | QuIP# + RTN activations | 0.759 | 128 | wikitext2 x 64 | not used | – | Unstructured (Wanda) | 0 | InfiniGen | not used | 0.879 | 11.02 / 11.44 | Yes (extras could not be final in that code) |
| 6 | thr | MOBO #13 | GPTQ + RTN activations | 0.833 | 32 | wikitext2 x 64 | not used | – | Unstructured (Wanda) | 0.300 | InfiniGen | not used | 0.909 | 11.43 / 11.86 | Yes |
| 7 | 3bit (final) | MOBO #20 | SqueezeLLM + SmoothQuant | 0.876 | 32 | pile10k x 128 | 0.638 | 0 (no 3-bit layers) | Unstructured (Wanda) | 0.262 | InfiniGen | not used | 0.868 | 11.44 / 11.87 | Yes |
| 8 | lat2.5 (final) | MOBO #27 | OmniQuant, weights only (activations 16-bit) | 0.857 | per-channel | c4 x 32 | not used | – (off) | Unstructured (Wanda) | 0.082 | InfiniGen | not used | 0.899 | 11.50 / 11.91 | Yes |
| 9 | weightonly (final) | MOBO #29 | RTN, weights only (activations 16-bit) | 0.800 | 64 | c4 x 128 | not used | – (off) | Unstructured (Wanda) | 0 | SnapKV (kept share 0.259) | not used | 0.917 | 11.12 / 11.52 (redo; first run 11.20 / 11.60) | Yes |
| 10 | ppl11-lat2.5 (closest) | MOBO #27 | OmniQuant, weights only (activations 16-bit) | 0.834 | per-channel | c4 x 64 | not used | – (off) | Unstructured (Wanda) | 0.030 | InfiniGen | not used | 0.919 | 11.50 / 11.90 | No (ppl over 11.0; vLLM 2.0 ms/token, vLLM ppl 13.0) |
| 11 | ppl11.5-lat2.0 (final) | MOBO #25 | OmniQuant, weights only (activations 16-bit) | 0.851 | per-channel | c4 x 32 | not used | – (off) | Unstructured (Wanda) | 0.031 | InfiniGen | not used | 0.919 | 11.50 / 11.91 | Yes |
| 12 | structonly (closest) | MOBO #30 | OmniQuant, weights only (activations 16-bit) | 0.810 | per-channel | pile10k x 128 | not used | – (off) | Structured | 0 | InfiniGen (kept share 0.259) | not used | 0.836 | 11.52 / 11.95 | No (ppl over 11.5; vLLM 1.99 ms/token, vLLM ppl 13.0) |
| 13 | smallest (closest fast) | MOBO #21 | OmniQuant, weights only (activations 16-bit) | 0.881 | per-channel | c4 x 64 | not used | – (off) | Unstructured (Wanda) | 0.126 | InfiniGen (kept share 0.259) | not used | 0.861 | 11.75 / 12.17 | No (ppl over 11.5; vLLM 2.00 ms/token, vLLM ppl 13.21) |
| 13b | smallest (lowest ppl) | MOBO #1 | GPTQ + RTN activations | 0.855 | 64 | wikitext2 x 128 | not used | – (off) | Unstructured (Wanda) | 0.215 | InfiniGen (kept share 0.259) | not used | 0.899 | 11.26 / 11.69 | No (vLLM 42.6 ms/token, vLLM ppl 25.22) |
| 14 | b40 (closest fast) | MOBO #25 | OmniQuant, weights only (activations 16-bit) | 0.875 | per-channel | c4 x 64 | not used | – (off) | Unstructured (Wanda) | 0.074 | InfiniGen (kept share 0.259) | not used | 0.880 | 11.75 / 12.16 | No (ppl over 11.5; vLLM 2.14 ms/token, vLLM ppl 13.21) |
| 15 | lowppl-small-r2 (closest fast) | MOBO #29 | OmniQuant, weights only (activations 16-bit) | 0.800 | per-channel | c4 x 64 | not used | – (off) | Unstructured (Wanda) | 0.083 | RTN KV (4 bits average) | not used | 0.921 | 12.68 / 12.98 | No (vLLM ppl 12.97 / 13.36 over 11.5; vLLM 2.09 ms/token, file 0.860 GB) |
| 15b | lowppl-small-r2 (lowest vLLM ppl) | MOBO #2 (from the lost first attempt) | RTN + SmoothQuant | 0.810 | 32 | c4 x 16 | 0.821 | – (off) | Unstructured (Wanda) | 0.063 | H2O (kept share 0.259) | not used | 1.020 | 12.06 / 12.44 | No (vLLM ppl 10.98 / 11.44 meets 11.5, but vLLM 42.0 ms/token; file 0.896 GB) |
| 16 | lowlat-small-r2 (fastest) | MOBO #27 | OmniQuant, weights only (activations 16-bit) | 0.800 | per-channel | c4 x 32 | not used | – (off) | Unstructured (Wanda) | 0 | InfiniGen (kept share 0.259) | not used | 0.858 | 11.42 / 11.81 | No (vLLM ppl 12.98 / 13.37 over 11.5; vLLM 1.99 ms/token, file 0.860 GB) |
| 16b | lowlat-small-r2 (lowest vLLM ppl) | MOBO #26 | OmniQuant, weights only (activations 16-bit) | 0.800 | per-channel | c4 x 128 | not used | – (off) | Unstructured (Wanda) | 0 | H2O (kept share 0.259) | not used | 0.858 | 12.67 / 12.98 | No (vLLM ppl 12.97 / 13.35 over 11.5; vLLM 2.10 ms/token, file 0.860 GB) |
| 17 | lowppl-small-r2-b40 (lowest vLLM ppl of new tries) | MOBO #31 | OmniQuant, weights only (activations 16-bit) | 0.800 | per-channel | c4 x 128 | not used | – (off) | Unstructured (Wanda) | 0.087 | SnapKV (kept share 0.259) | not used | 0.919 | 12.03 / 12.41 | No (vLLM ppl 12.97 / 13.36 over 11.5; vLLM 2.02 ms/token, file 0.860 GB) |
| 17b | lowppl-small-r2-b40 (fastest of new tries) | MOBO #35 | OmniQuant, weights only (activations 16-bit) | 0.800 | per-channel | c4 x 64 | not used | – (off) | Unstructured (Wanda) | 0.170 | InfiniGen (kept share 0.259) | not used | 0.888 | 11.43 / 11.82 | No (vLLM ppl 13.00 / 13.39 over 11.5; vLLM 1.97 ms/token, file 0.860 GB) |
| 18 | lowlat-small-r2-b40 (fastest of new tries) | MOBO #32 | OmniQuant, weights only (activations 16-bit) | 0.847 | per-channel | c4 x 128 | not used | – (off) | Unstructured (Wanda) | 0.015 | SnapKV (kept share 0.259) | not used | 0.924 | 12.14 / 12.54 | No (vLLM ppl 13.00 / 13.41 over 11.5; vLLM 1.98 ms/token, file 0.838 GB) |
| 18b | lowlat-small-r2-b40 (lowest vLLM ppl of new tries) | MOBO #34 | OmniQuant, weights only (activations 16-bit) | 0.800 | per-channel | c4 x 128 | not used | – (off) | Unstructured (Wanda) | 0.042 | QuaRot KV (4 bits) | 8 | 0.936 | 12.82 / 13.13 | No (vLLM ppl 12.97 / 13.36 over 11.5; vLLM 2.06 ms/token, file 0.860 GB) |

## Table 3. Final models on vLLM (Stage 4) and accuracy

Only runs 5, 7, 8, 9 and 11 reached Stage 4. vLLM runs the model up to pruning (it has no InfiniGen), and its export re-rounds the weights, so its perplexity differs from HF.

| | FP16 | r2 final (run 5) | 3bit final (run 7) | lat2.5 final (run 8) | weightonly final (run 9) | lat2.0 final (run 11) |
|---|---|---|---|---|---|---|
| Settings | none | GPTQ + RTN activations, thr 0.759, gs 128, wikitext2 x 64; Wanda 0; InfiniGen | SqueezeLLM + SmoothQuant 0.638, thr 0.876, gs 32, pile10k x 128; Wanda 0.262; InfiniGen | OmniQuant weights only, thr 0.857, per-channel, c4 x 32; Wanda 0.082; InfiniGen | RTN weights only, thr 0.800, gs 64, c4 x 128; Wanda 0; SnapKV | OmniQuant weights only, thr 0.851, per-channel, c4 x 32; Wanda 0.031; InfiniGen |
| vLLM file size (GB) | 2.200 | 0.894 | 0.873 | 0.838 | 0.898 | 0.838 |
| vLLM ppl val / held-out | 10.18 / 10.62 | 12.09 / 12.52 | 11.46 / 11.89 | 13.00 / 13.41 | 11.10 / 11.54 | 13.0 / 13.41 |
| vLLM decode (ms/token) | 2.54 | 34.42 | 42.45 | **1.98** | 2.00 | 2.04 (1.96 in the search try) |
| Decode vs FP16 | – | 13.5x slower | 16.7x slower | 22% faster | 21% faster | 20% faster |
| vLLM prefill (ms) | 9.16 | 35.0 | 43.71 | – | 9.73 | 9.15 |
| vLLM peak memory (GB) | 3.766 | 2.458 | 2.442 | 2.397 | 2.448 | 2.397 |
| ARC-Easy | 0.548 | 0.402 | 0.415 | 0.443 | 0.325 | 0.443 |
| PIQA | 0.745 | 0.676 | 0.676 | 0.691 | 0.630 | 0.691 |
| HellaSwag | 0.604 | 0.551 | 0.555 | 0.566 | 0.516 | 0.566 |
| WinoGrande | 0.603 | 0.521 | 0.509 | 0.523 | 0.510 | 0.523 |
| Accuracy, mean | 0.625 | 0.538 (-14.0%) | 0.539 (-13.8%) | 0.556 (-11.1%) | 0.495 (-20.8%) | 0.556 (-11.1%) |

Runs 5 and 7 keep activations at low bits (WxAy), which vLLM runs without its fast kernels (eager mode), so they are slow. Runs 8, 9 and 11 are weight-only, which vLLM runs with CUDA graphs. Run 9's accuracy is from a redo after a bug fix: SnapKV broke on short prompts in the first run (mean 0.374, near chance); vLLM numbers are unaffected because vLLM runs no KV step. Run 11's final is in effect the same model as run 8's (same method and calibration, almost the same scores; threshold 0.851 vs 0.857, pruning 0.031 vs 0.082).

## Table 4. MOBO vs MFBO (runs that ran both to the end)

| Run | Searcher (same space and budget, seed 0) | Tries | Models built | Build time | Met the limit | Best size under the limit |
|---|---|---|---|---|---|---|
| r2 | MOBO | 30 | 30 | 245 min | 4 (13%) | 0.924 GB (11.39) |
| r2 | MFBO | 130 | 129 | 495 min | 10 (8%) | 0.939 GB (11.44) |
| 3bit | MOBO | 30 | 30 | about 3 h 10 min | 8 | 0.868 GB (11.42) |
| 3bit | MFBO | 130 | 130 | about 8 h 20 min | 0 | none (closest 0.916 GB at 11.66) |

## Main points for the discussion

1. Changing the goal mattered more than adding tries: under "size <= 1 GB and ppl <= 11" (runs 1 and 2) nothing qualified; under "smallest model with ppl <= 11.5" every later run found qualifying models.
2. The smallest model kept getting smaller across runs: 0.924 GB (run 4) to 0.909 GB (runs 5 and 6) to 0.868 GB (run 7), while ppl stayed under 11.5.
3. Small size did not mean fast: the WxAy finals of runs 5 and 7 were 13 to 17 times slower than FP16 in vLLM. Adding a speed limit (run 8) moved the search to weight-only quantization and gave the first final model faster than FP16 (1.98 vs 2.54 ms/token), at 0.84 GB and the best accuracy of the finals (run 11 later matched it with in effect the same model).
4. Accuracy dropped 11 to 14% in every final model with InfiniGen, and 21% in run 9's SnapKV model, even though run 9 has the lowest ppl of the weight-only finals. With OmniQuant, GPTQ or SqueezeLLM, ppl on vLLM is higher than on HF because the export re-rounds the weights; run 9's plain RTN weights export almost unchanged (vLLM 11.10 vs HF 11.12).
5. MOBO beat MFBO: in both full comparisons it found qualifying models in fewer builds and less time.
6. Every qualifying model in Table 2 uses unstructured (Wanda) pruning, and all but run 9's use InfiniGen for the KV cache (run 9 chose SnapKV).
7. Tightening either limit after run 8 gave nothing better: with ppl <= 11.0 (run 10) no try qualified (closest 11.50 at 2.0 ms/token); with 2.0 ms/token (run 11) one try qualified, in effect run 8's model.
8. With structured pruning only (run 12) no try qualified: the closest kept a pruning share of 0 (nothing removed) and missed by 0.02 ppl (11.52). In that run the same model was sometimes rebuilt under a different threshold (tries #11, #14 and #17); this was fixed after the run (PR #94).
9. Putting size first (run 13) found no qualifying model either: the low-ppl tries were WxAy and 17 times slower than FP16 in vLLM (42.6 ms/token), and the fast weight-only tries stayed at ppl 11.75 or above.
10. Ten more tries (run 14, budget 40) did not help: no try qualified, and the closest fast try was again at ppl 11.75. Its lowest-ppl try (#1) is the same point as run 13's #1: HF ppl matched to every digit, while vLLM ppl moved only in the third decimal (25.2245 vs 25.2216) between the two builds.
11. Judging ppl on the vLLM file instead of HF (run 15) did not help either: the only tries under vLLM ppl 11.5 were WxAy models at about 42 ms/token, while the fast weight-only tries stayed at vLLM ppl 12.97 or above. For WxAy models the vLLM ppl can be lower than HF (try #2: 10.98 vs 12.06), the opposite of the weight-only models.
12. Putting speed first under the vLLM ppl limit (run 16) found no qualifying model: the fastest try (1.99 ms/token) had HF ppl 11.42, under 11.5, but vLLM ppl 12.98. Its two best tries used the same OmniQuant settings with nothing pruned and differed mainly in the KV method (InfiniGen vs H2O). That changed HF ppl by 1.25 but vLLM ppl by only 0.01, because vLLM runs no KV step. So the vLLM ppl limit is set by the weight export, not by pruning or KV choices.
13. Ten more tries on runs 15 and 16 (runs 17 and 18) changed nothing. All 20 new tries were OmniQuant weight-only, at 1.97-2.09 ms/token, and their vLLM ppl stayed between 12.97 and 13.31 (14.63 for one structured-pruning try). The vLLM ppl floor of about 12.97 seen since run 15 did not move, whatever pruning or KV method was chosen.
