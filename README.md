# Sensitivity-Driven Adaptive LLM Compression

A search-driven compression pipeline for TinyLlama-1.1B-Chat. Three searchers (MOBO, MFBO and NSGA-III) explore
a shared 5-D space (sensitivity threshold, robust-layer pruning ratio, group size, migration strength and cache
bit-width). Each trial runs:

| Stage | What it does | Status |
|---|---|---|
| 0 | Per-layer sensitivity profiling, then a compression plan | implemented |
| 1 | Weight compression (quantisation + pruning) | todo |
| 2 | Activation compression (outlier-aware) | todo |
| 3 | KV-cache compression | todo |
| 4 | Deployment-aware evaluation (memory, latency, accuracy) | todo |
| Search | MOBO / MFBO / NSGA-III, Pareto pooling, searcher benchmark | todo |

## Layout

```
src/sdf/
  config.py        search space, candidate config, deployment targets, calibration settings
  trial_log.py     JSONL trial log shared by all searchers
  calibration.py   WikiText-2 / C4 / pile-10k calibration windows
  stage0/
    sensitivity.py per-layer gradient x weight sensitivity profile
    planner.py     compression plan + Stage 0 report
  cli.py           `sdf-stage0` command
```

## Stage 0

For each decoder layer *l* the score is the gradient × weight saliency summed over calibration batches:

    s_l = Σ_batches Σ_{w ∈ layer l} |∂L/∂w · w|

The scores are min-max normalised to [0, 1]. A layer with a score at or above the sensitivity threshold is
**protected** (8-bit, no pruning). Every other layer is **compressed** (4-bit, pruned at the robust-layer
pruning ratio).

The profile depends only on the model and calibration data, so it is computed once and saved. Each trial then
only re-plans with its own threshold and pruning ratio.

## Usage

```bash
pip install -e ".[dev]"
pytest

# profile + plan (downloads the model and WikiText-2 from the Hugging Face Hub)
sdf-stage0 --threshold 0.5 --pruning-ratio 0.3 --out runs/stage0

# re-plan from a saved profile without re-profiling
sdf-stage0 --profile runs/stage0/sensitivity_profile.json --threshold 0.7 --pruning-ratio 0.4 --out runs/t1
```

Outputs are `sensitivity_profile.json`, `compression_plan.json` and `stage0_report.json`.

Profiling TinyLlama in float32 needs roughly 9 GB for the weights plus gradients (only decoder-layer weights get
gradients). Use `--dtype bfloat16` on a GPU to halve that.
