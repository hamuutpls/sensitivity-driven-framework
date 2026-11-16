# Stage 0 summary (source: stage_0 folder of 20261101-pc-5070ti-stage014, code 01baa92, PC RTX 5070 Ti; plans byte-identical in Colab stage024/034 at a178012)
- Baseline calibration ppl 14.1186 (WikiText-2, 64 x 512 tokens, seed 0, float32 profile).
- Layer-removal profile: 96.0 s on PC (121 s on Colab A100), peak 4.60 GB. Activation profile 188.3 s, peak 4.66 GB. KV plan build 540.5 s. Joint plan 1685 s.
- Same-size prune plan % removed per layer: 0,0,0,31,47,42,39,0,44,50,61,58,53,55,33,22,19,36,25,28,17,0.
- Predicted weights: FP16 2.200 GB (16 bits); Standard uniform 4-bit group 128: 0.777 GB, 5.65 avg bits; Our quant plan (11 of 22 layers 8-bit): 1.019 GB, 7.41 avg bits; bits + 30% pruning of unprotected layers: 1.003 GB, 7.29 avg bits (sparsity 0.1321). Floor (embeddings, head, norms; 131,164,160 params at 16 bits) 0.262 GB.
- Pruning predictions at 16-bit: Standard 30% every layer 1.740 GB (26% removed); pruning plan 1.970 GB (13%); same-size plan 1.712 GB (26%).
- KV at 2048 tokens, batch 1: FP16 0.04614 GB; Standard 4-bit 0.01298 GB (rise 0.1085); Standard eviction 16-bit 20% kept 9.2 MB; Our plan (bits + eviction) 0.003232 GB, 3.825 avg bits, 0.2592 kept, attention on kept 0.9672; plan bits only 0.01285 GB, 3.955 bits.
- Activations: Standard 8-bit predicted rise 0.0109; plan 6.00 bits 0.5244; copied from weight plan 6.00 bits 0.6454.
- FP16 (HF, PC): ppl 10.18 / 10.62; prefill 5.869 ms; decode 4.324 ms/token; peak GPU 2.412 GB.

# Stage 1 run facts (source: stage_1/results.json of 20261101-pc-5070ti-stage014, 01baa92, RTX 5070 Ti; 87 rows, all ok)
- FP16: peak GPU 2.412 GB; prefill 5.869 ms (std 0.222); decode 4.324 ms/token (std 0.075); llama.cpp FP16 2.201 GB, 323.7 tok/s, ppl val 8.1204.
- Predicted weight memory / avg bits: Standard 0.777 GB / 5.65; Our 1.019 GB / 7.41 (rtn, gptq, awq, omniquant, efficientqat, quip, pbllm). squeezellm 0.786/5.71 vs 1.122/8.16; spqr 0.861/6.26 vs 1.103/8.02; quipsharp 0.747/5.43 vs 0.989/7.19; bitsandbytes 0.807/5.87 vs 1.020/7.42. W8A8 rows 1.261 GB / 9.17.
- llama.cpp file: Standard 0.808 GB, Our 1.050 GB, W8A8 1.293 GB.
- Stage 3 run (20261102-colab-a100-stage034, a178012, torch 2.11.0+cu130, transformers 5.21.0): 13 rows all ok; see stage034-kv-predicted.csv. rtn_kv Our row == h2o Our row by design.
- Unit tests PC 9 Nov at a178012: 481 passed, 3 skipped (compressed_tensors not installed).

# Final models of earlier searches (source: run folders on D:\Thesis\results\colab)
- 20261029-colab-a100-ppl11.5-lat2.5 final (try 27): OmniQuant weights only + Wanda 0.0818 + InfiniGen; threshold 0.8567; c4 x 32; avg bits 6.54. Accuracy final/FP16: arc_easy 0.4428/0.5476, piqa 0.691/0.7454, hellaswag 0.5657/0.6044, winogrande 0.5225/0.603, mean 0.5555/0.6251 (-11.13%). Protected layers not recorded for this model (plans in that threshold range on disk have 4 layers at 8 bits: 0, 2, 7, 21; inference only).
- 20261031-colab-a100-weightonly final (try 29): RTN weights only + Wanda 0 + SnapKV; threshold 0.8; c4 x 128; group 64; 5 of 22 layers at 8 bits (0,1,2,7,21). Accuracy with fixed SnapKV code (c808d0b): arc_easy 0.3249, piqa 0.6300, hellaswag 0.5163, winogrande 0.5099, mean 0.4953 vs FP16 0.5476/0.7454/0.6044/0.6030 mean 0.6251. HF ppl val 11.11, vLLM ppl val 11.10. (Before the fix: mean 0.3741.)
- Final two searches (40 tries each, a178012): no try within both limits (vLLM ppl <= 11.5 and vLLM decode <= 2.5). 30 of 40 tries weight-only (decode 1.97-2.14 ms, vLLM ppl 12.97-14.63, file 0.815-0.860 GB); 10 tries with rounded activations (33.3-43.1 ms; lowest vLLM ppl 10.98).
