# Run 1: does the layer-skipping score predict compression damage?

Commit 49f7fdb63d1a9ed0bf61b0e19b07ea5902878341; model TinyLlama/TinyLlama-1.1B-Chat-v1.0; seed 0; written 2026-11-22 13:04:16 UTC. Colab A100-SXM4-40GB; archive md5 4660cd4e373b44b0195a4e112946e81c (D:\Thesis\results\colab\20261122-review-run1-layer-damage).

22 decoder layers; each changed alone. Calibration text: wikitext2 train, 64 batches of 512 tokens (the text the score uses); model in float32 (as Stage 0 scores layers); validation: 333 windows of 512 tokens. Rounding: round-to-nearest, group 128, integer zero point. Pruning: 25% of each Linear's weights per output row, by magnitude and by Wanda (calibration inputs). Each column is the perplexity rise over the unchanged model (base: 14.12 calibration, 10.18 validation).
Wall clock 3264 s.

| text | a | b | Spearman | layers |
|---|---|---|---|---|
| calib | skip | w8 | 0.040 | 22 |
| calib | skip | w4 | 0.180 | 22 |
| calib | skip | prune25_magnitude | 0.153 | 22 |
| calib | skip | prune25_wanda | 0.411 | 22 |
| calib | w8 | w4 | -0.168 | 22 |
| calib | w8 | prune25_magnitude | 0.160 | 22 |
| calib | w8 | prune25_wanda | -0.068 | 22 |
| calib | w4 | prune25_magnitude | 0.604 | 22 |
| calib | w4 | prune25_wanda | 0.223 | 22 |
| calib | prune25_magnitude | prune25_wanda | 0.517 | 22 |
| val | skip | w8 | 0.460 | 22 |
| val | skip | w4 | 0.509 | 22 |
| val | skip | prune25_magnitude | 0.539 | 22 |
| val | skip | prune25_wanda | 0.422 | 22 |
| val | w8 | w4 | 0.256 | 22 |
| val | w8 | prune25_magnitude | 0.290 | 22 |
| val | w8 | prune25_wanda | 0.319 | 22 |
| val | w4 | prune25_magnitude | 0.730 | 22 |
| val | w4 | prune25_wanda | 0.609 | 22 |
| val | prune25_magnitude | prune25_wanda | 0.761 | 22 |
