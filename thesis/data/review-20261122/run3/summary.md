# Run 3: ablation of the three stages

Commit 49f7fdb63d1a9ed0bf61b0e19b07ea5902878341; model TinyLlama/TinyLlama-1.1B-Chat-v1.0; seed 0; written 2026-11-22 18:48:13 UTC. Colab A100-SXM4-40GB (speed numbers are from the 40 GB card); archive md5 b1597bc56b149ef7cca18b0920cb3654 (D:\Thesis\results\colab\20261122-review-run3-ablation); wall clock about 3 h 01 min.

Each final model's settings with stages left out: no quantization (16-bit weights), pruning share 0, no KV method. Pruning rows use unstructured pruning (Wanda) at 0.082 on the layers below each model's own threshold (RTN model 0.800, OmniQuant model 0.857). vLLM runs the model before the KV step (it cannot run it), so vLLM was measured once, on Baseline, E1 and E4, and those numbers are repeated for E3, E5 and E6 (column vllm_from).

Note: E4 equals E1 and E6 equals E5 in every quality number, to all digits, in both models. Pruning alone (E2) does change the model. Likely cause (checked on random weights, not on TinyLlama): 4-bit rounding with an integer zero point already sets more than 8.2% of every row to exactly zero (13.8% on average, at least 10.8% per row, for normal weights at group 128), and Wanda removes the weights with the smallest score, which are those zeros. So pruning 8.2% after 4-bit rounding removes nothing new, while the predicted size still counts its bitmask.

| model | row | ppl val | ppl held-out | acc mean | vLLM size GB | vLLM decode ms/token | vLLM ppl val | predicted KV GB |
|---|---|---|---|---|---|---|---|---|
| rtn_snapkv | Baseline | 10.183 | 10.620 | 0.625 | 2.200 | 2.803 | 10.183 | 0.0461 |
| rtn_snapkv | E1 | 10.438 | 10.868 | 0.624 | 0.898 | 1.980 | 11.101 | 0.0461 |
| rtn_snapkv | E2 | 10.183 | 10.614 | 0.626 | 2.200 | 2.817 | 10.184 | 0.0461 |
| rtn_snapkv | E3 | 10.824 | 11.225 | 0.497 | 2.200 | 2.803 | 10.183 | 0.0034 |
| rtn_snapkv | E4 | 10.438 | 10.868 | 0.624 | 0.898 | 2.031 | 11.101 | 0.0461 |
| rtn_snapkv | E5 | 11.115 | 11.517 | 0.495 | 0.898 | 1.980 | 11.101 | 0.0034 |
| rtn_snapkv | E6 | 11.115 | 11.517 | 0.495 | 0.898 | 2.031 | 11.101 | 0.0034 |
| omniquant_infinigen | Baseline | 10.183 | 10.620 | 0.625 | 2.200 | 2.807 | 10.183 | 0.0461 |
| omniquant_infinigen | E1 | 11.248 | 11.667 | 0.613 | 0.838 | 2.034 | 12.958 | 0.0461 |
| omniquant_infinigen | E2 | 10.184 | 10.615 | 0.626 | 2.200 | 2.814 | 10.184 | 0.0461 |
| omniquant_infinigen | E3 | 10.356 | 10.770 | 0.562 | 2.200 | 2.807 | 10.183 | 0.0034 |
| omniquant_infinigen | E4 | 11.248 | 11.667 | 0.613 | 0.838 | 2.086 | 12.958 | 0.0461 |
| omniquant_infinigen | E5 | 11.466 | 11.879 | 0.553 | 0.838 | 2.034 | 12.958 | 0.0034 |
| omniquant_infinigen | E6 | 11.466 | 11.879 | 0.553 | 0.838 | 2.086 | 12.958 | 0.0034 |
