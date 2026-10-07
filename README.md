# Sensitivity-Driven Adaptive LLM Compression

Research code for an MSc thesis. A sensitivity-driven adaptive precision allocation framework for LLM
compression, applied to TinyLlama-1.1B-Chat.

| Stage | What it does | Status |
|---|---|---|
| 0 | Per-layer sensitivity profiling, then a bit-width / pruning plan | implemented |
| 1 | Quantization only, nothing removed: weights (RTN, GPTQ, AWQ) and activations (RTN, SmoothQuant, QuaRot, RPTQ, SpinQuant) | weights done (simulated); activations: RTN baseline |
| 2 | Pruning only, nothing rounded (Wanda, structured, low-rank) | done (simulated) |
| 3 | KV-cache compression (QuaRot KV, KVQuant, H2O, SnapKV, InfiniGen) | todo |
| 4 | Evaluation across backends (HF Transformers, llama.cpp, vLLM, TensorRT-LLM) | todo |
| Search | MOBO (Optuna), MFBO (successive halving), NSGA-III (pymoo), benchmarked against each other | todo |

Paths: 0 > 1 > 4, 0 > 2 > 4 (prunes the FP16 model), 0 > 3 > 4, and 0 > 1 > 2 > 4 (a Stage 2 method on a model
Stage 1 has quantized; `stage2_after` in `main.py`).

Every stage compares, under identical conditions, three variants of each method: the **FP16** baseline, the
**original** method used on its own, and the **framework** version (Stage 0 plan + the method).

## Quick start

Open `main.py`, change the settings at the top (every tunable parameter is there, with its allowed values),
then run:

```bash
pip install -e .
python main.py
```

`MODE = "single"` runs Stage 0 once; `MODE = "sweep"` tries every combination in `SWEEP` and writes one
comparison report; `MODE = "prune_sweep"` really prunes the model at every level in `PRUNE_SWEEP_RATIOS`
(10% to 100% by default) and measures the error, standard method vs framework, plus the fair test at the same size and share removed with the pruning placed by sensitivity (`stage_0_prune_sweep/`). The command-line tools below do the same with `--set` overrides.

### On Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamuutpls/sensitivity-driven-framework/blob/main/notebooks/colab_run.ipynb)

`notebooks/colab_run.ipynb` pulls `main`, installs the package, runs the tests and then `main.py` on the Colab
GPU, with results and caches in `MyDrive/thesis_compression/` so they survive the session.

## Layout

```
main.py                  every tunable parameter, edit and run
configs/tinyllama.yaml   local output paths; every default lives in config.py
src/sdf/
  config.py              FrameworkConfig: the single config object (YAML + --set overrides)
  search_space.py        the search space (Stage 0 parameters for now); adding a parameter is one line
  requirements.py        DeploymentRequirement and its met / shortfall check
  run.py                 run directory, config snapshot, logging, seeding, cache
  data.py                calibration windows; WikiText-2 test split into validation / held-out halves
  eval/metrics.py        perplexity (both halves), model size, peak memory, prefill / decode latency
  reporting/             StageReporter: report.md, stage_<N>_comparison.xlsx, results.json for every stage
  stage0/                sensitivity profile, planner + memory prediction, end-to-end Stage 0 run
  utils/                 logging, seeding, environment capture, artifact cache
docs/diagrams/           class and sequence diagrams, per stage and for the whole project
docs/specs/              system requirements (ISO/IEC/IEEE 29148) and subsystem design (IEEE 1016)
```

## Stage 0

For each decoder layer *l* the default score (`stage0.score: layer_removal`) is the rise in calibration
perplexity when that layer is skipped (its output = its input). On TinyLlama the older gradient × weight
estimate ranked layer 0 as the least sensitive layer, although skipping it raises perplexity from 14 to about
1,190.

The scores are rank-normalised to [0, 1]: a layer's position in the sorted order, divided by (n − 1). A layer
at or above `sensitive_threshold` is **protected** (8-bit, no pruning). Every other layer is **compressed**
(4-bit, pruned at `prune_ratio_aggressive`). A threshold *t* therefore protects about the top (1 − *t*) share
of layers.

`stage0.score` switches the measurement: `layer_quant` (perplexity rise when only that layer is compressed) or
`grad_x_weight` (Σ_batches Σ_{w ∈ layer l} |∂L/∂w · w|, one backward pass, a first-order estimate), or one of four
more gradient scores: `fisher` (mean_b ½ Σ_w (g·w)², empirical Fisher), `taylor_ema` (moving average over batches
of |Σ_w g·w|, as in LLM-Pruner), `hessian` (½ Σ_w max(diag(H), 0)·w², Hutchinson probes with a finite-difference
Hessian-vector product, `hessian_probes` per batch, as in Optimal Brain Damage) and `movement` (Σ_w |Σ_steps g·w| over a short SGD
fine-tune, as in movement pruning). `MODE = "compare_scores"` in `main.py` runs all seven and checks the
estimates against the two measured ones.

Rank is used instead of min-max because one outlier layer skews min-max. On TinyLlama, layer 0 scores 2705
against 6596–8647 for the other layers, so min-max put layers 1–21 between 0.66 and 1.0 and threshold 0.5
protected 21 of 22. Min-max is still available as `stage0.normalization: minmax`.

Stage 0 compares:

- **fp16**: the uncompressed model, measured once and cached (perplexity on both halves, size, peak memory,
  prefill and decode latency).
- **original**: a uniform allocation (every layer 4-bit, no pruning), i.e. what a method uses with no guidance.
- **framework**: the sensitivity plan.
- **framework, budget plan**: the sensitivity plan limited to the uniform plan's predicted memory (it protects
  as many of the most sensitive layers as fit). This is the fair quality comparison, since the threshold plan
  is usually bigger than uniform.
- **benchmark, nothing removed**: the budget plan without pruning (robust layers at 3 bits). A Stage 0
  comparison only; later stages do not receive it.

For the two plans it reports predicted weight memory, average bits per weight, sparsity and *sensitivity
exposure*, which measures how much compression lands on sensitive layers (lower is better). Accuracy and
latency of the plans are measured once Stage 1 applies them.

**KV cache plan** (`stage0/kv_cache.py`, section 4 of `main.py`). Per layer, Stage 0 also chooses key bits
and value bits separately (each layer's keys, then values, are rounded alone to 2/4/8 bits and the perplexity
rise is measured; a greedy allocation spends the uniform 4-bit average where damage is largest) and a token
budget (the fewest past tokens that still receive 95% of the layer's attention, H2O / SnapKV style). It
compares uniform 4-bit, the plan, and the plan without eviction on predicted cache memory at
`KV_CONTEXT_LEN` tokens, average bits, share of tokens kept, predicted perplexity rise and attention kept.
Stage 3 applies the plan.

The sensitivity profile depends only on the model and calibration settings, so it is cached and reused by
every trial. Each trial then only re-plans.

### Sweep

`sdf-stage0-sweep` tries several values of every setting and writes one comparison report
(`stage_0_sweep/report.md`, `sweep.xlsx`, `results.json`). By default it tries every calibration text and
sample count, every group size, and 5 evenly spaced thresholds and prune ratios. Each calibration setting is
profiled once (cached); the rest only re-plans.

```bash
sdf-stage0-sweep --config configs/tinyllama.yaml                          # full sweep, needs the GPU
sdf-stage0-sweep --grid sensitive_threshold=0.4,0.5,0.6 --grid calib_samples=64
sdf-stage0-sweep --profile results/<run>/stage_0/sensitivity_profile.json # planning only, no GPU
```

## Usage

```bash
pip install -e ".[dev]"
pytest

# outputs go to ./thesis_compression/ in the folder you run from (see run.output_root)
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
