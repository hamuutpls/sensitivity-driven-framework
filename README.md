# Sensitivity-Driven Adaptive LLM Compression

Research code for an MSc thesis. A sensitivity-driven adaptive precision allocation framework for LLM
compression, applied to TinyLlama-1.1B-Chat.

| Stage | What it does | Status |
|---|---|---|
| 0 | Per-layer sensitivity profiling, then a bit-width / pruning plan | implemented |
| 1 | Weight compression (GPTQ, AWQ, structured / unstructured / low-rank pruning) | todo |
| 2 | Activation compression (SmoothQuant, QuaRot, RPTQ, SpinQuant) | todo |
| 3 | KV-cache compression (QuaRot KV, KVQuant, H2O, SnapKV, InfiniGen) | todo |
| 4 | Evaluation across backends (HF Transformers, llama.cpp, vLLM, TensorRT-LLM) | todo |
| Search | MOBO (Optuna), MFBO (successive halving), NSGA-III (pymoo), benchmarked against each other | todo |

Every stage compares, under identical conditions, three variants of each method: the **FP16** baseline, the
**original** method used on its own, and the **framework** version (Stage 0 plan + the method).

## Layout

```
configs/tinyllama.yaml   every setting, in one file
src/sdf/
  config.py              FrameworkConfig: the single config object (YAML + --set overrides)
  search_space.py        the shared search space; adding a parameter is one line
  requirements.py        DeploymentRequirement and its met / shortfall check
  run.py                 run directory, config snapshot, logging, seeding, cache
  data.py                calibration windows; WikiText-2 test split into validation / held-out halves
  eval/metrics.py        perplexity (both halves), model size, peak memory, prefill / decode latency
  reporting/             StageReporter: report.md, stage_<N>_comparison.xlsx, results.json for every stage
  stage0/                sensitivity profile, planner + memory prediction, end-to-end Stage 0 run
  trial_log.py           per-searcher JSONL trial log
  utils/                 logging, seeding, environment capture, artifact cache
```

## Stage 0

For each decoder layer *l* the score is the gradient × weight saliency summed over calibration batches:

    s_l = Σ_batches Σ_{w ∈ layer l} |∂L/∂w · w|

The scores are min-max normalised to [0, 1]. A layer at or above `sensitive_threshold` is **protected**
(8-bit, no pruning). Every other layer is **compressed** (4-bit, pruned at `prune_ratio_aggressive`).

Stage 0 compares:

- **fp16**: the uncompressed model, measured once and cached (perplexity on both halves, size, peak memory,
  prefill and decode latency).
- **original**: a uniform allocation (every layer 4-bit, no pruning), i.e. what a method uses with no guidance.
- **framework**: the sensitivity plan.

For the two plans it reports predicted weight memory, average bits per weight, sparsity and *sensitivity
exposure*, which measures how much compression lands on sensitive layers (lower is better). Accuracy and
latency of the plans are measured once Stage 1 applies them.

The sensitivity profile depends only on the model and calibration settings, so it is cached and reused by
every trial. Each trial then only re-plans.

## Usage

```bash
pip install -e ".[dev]"
pytest

# Colab: outputs go to Drive (see run.output_root in the config)
sdf-stage0 --config configs/tinyllama.yaml
sdf-stage0 --config configs/tinyllama.yaml --set hyperparams.sensitive_threshold=0.7 --set run.run_id=my-run
```

Outputs are written to `<output_root>/<run_id>/stage_0/`:

- `report.md`
- `stage_0_comparison.xlsx` (Summary, Config, Per-layer, Raw, Charts)
- `results.json`
- `compression_plan.json`
- `sensitivity_profile.json`

`results.json` is rewritten after every row. Reusing a `run_id` resumes into the same folder.

Profiling TinyLlama in float32 needs about 9 GB for weights plus gradients (only decoder-layer weights get
gradients). On a GPU, `--set stage0.profile_dtype=bfloat16` halves that.
