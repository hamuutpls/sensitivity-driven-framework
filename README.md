# Sensitivity-Driven Adaptive LLM Compression

Research code for an MSc thesis. A sensitivity-driven adaptive precision allocation framework for LLM
compression, applied to TinyLlama-1.1B-Chat.

| Stage | What it does | Status |
|---|---|---|
| 0 | Per-layer sensitivity profiling, then a bit-width / pruning plan | implemented |
| 1 | Quantization only, nothing removed: weights (RTN, GPTQ, AWQ, OmniQuant, SqueezeLLM, SpQR, EfficientQAT, QuIP, QuIP#, PB-LLM, bitsandbytes and others) and activations (RTN, SmoothQuant, QuaRot, RPTQ, SpinQuant), on the joint weight x activation plan | done (simulated) |
| 2 | Pruning only, nothing rounded (Wanda, structured, low-rank) | done (simulated) |
| 3 | KV-cache compression (RTN KV, QuaRot KV, KVQuant, H2O, SnapKV, InfiniGen) | done (simulated on HF) |
| 4 | Evaluation across backends (HF Transformers, llama.cpp, vLLM, TensorRT-LLM), plus lm-eval task accuracy | done (vLLM and TensorRT-LLM on Linux/Colab) |
| Search | MODE "search": each try builds one chained model 0 > 1 > 2 > 3 and measures it; the current run (2026-10-29) keeps validation perplexity at most 11.5 and vLLM decode below 2.5 ms per token (FP16: 2.53) and makes size, perplexity and vLLM latency as small as possible, with MOBO (Optuna) only; MFBO (successive halving) and NSGA-III (pymoo) also work. The smallest final model under the limits is then tested on the backends (Stage 4) and for accuracy next to TinyLlama. See `docs/diagrams/search.md` and [Reproduce the 2026-10-29 search run](#reproduce-the-2026-10-29-search-run). | done |

Paths: 0 > 1 > 4, 0 > 2 > 4 (prunes the FP16 model), 0 > 3 > 4, and 0 > 1 > 2 > 4 (a Stage 2 method on a model
Stage 1 has quantized; `stage2_after` in `main.py`).

Every stage compares, under identical conditions, three variants of each method: the **FP16** baseline, the
**original** method used on its own, and the **framework** version (Stage 0 plan + the method).

## Requirements

- Python 3.10 or newer (3.11 on the PC; Colab's own Python on Colab), and a CUDA GPU for real runs (the tests also
  run on CPU). PyTorch >= 2.1 and Transformers >= 4.56 come with the package.
- From a fresh clone:

  ```bash
  git clone https://github.com/hamuutpls/sensitivity-driven-framework
  cd sensitivity-driven-framework
  python -m venv .venv
  pip install -e ".[dev,search,downstream,figures]" pytest-xdist
  ```

  (activate `.venv` first). `dev`: pytest; `search`: Optuna (MOBO) and pymoo (NSGA-III, hypervolume), needed by
  MODE "search" and its tests; `downstream`: lm-eval, for `DOWNSTREAM_TASKS` and `SEARCH_ACCURACY_TASKS`;
  `figures`: matplotlib; `pytest-xdist`: `-n 4` below. Install the CUDA build of PyTorch for your GPU first if
  pip's default does not match it.
- vLLM (Linux only: a Colab A100 or a Linux PC), in its own virtual environment so its pinned PyTorch never touches
  the framework's. The code is tested with vLLM 0.31:

  ```bash
  python -m venv /content/vllm-venv
  /content/vllm-venv/bin/pip install "vllm==0.31.*"
  ```

  Then set `VLLM_PYTHON = "/content/vllm-venv/bin/python"` in `main.py`. The search run below needs it: every try
  runs vLLM.
- Optional: llama.cpp (`eval.llamacpp_dir`: the folder with `llama-quantize`, `llama-bench`, `llama-perplexity`;
  `eval.llamacpp_convert`: its `convert_hf_to_gguf.py`) and TensorRT-LLM (`eval.trtllm_python`, Linux only, its own
  venv). Left at None, they are not measured.

## Tests

Run the full suite before every run and every report, on the machine that does the run (GPU visible), and record
the passed and skipped counts and the commit:

```bash
python -m pytest -q tests
```

On a CPU-only machine set `OMP_NUM_THREADS=1` and run in parallel with `-n 4` (pytest-xdist), else the workers'
threads fight each other:

```bash
OMP_NUM_THREADS=1 python -m pytest -q -n 4 tests
```

The tests use a tiny 4-layer Llama and made-up text, so they need no download. Tests for vLLM, TensorRT-LLM and
lm-eval are skipped where those are not installed.

## Quick start

Open `main.py`, change the settings at the top (every tunable parameter is there, with its allowed values),
then run:

```bash
python main.py
```

What to change at the top of `main.py`:

| Section | Settings |
|---|---|
| 1. What to run | `MODE`, `MODEL`, `OUTPUT_ROOT`, `CACHE_DIR`, `RUN_ID` (None: timestamp + config hash; reuse a name to resume), `SEED` |
| 2. Search-space parameters | `SENSITIVE_THRESHOLD`, `PRUNE_RATIO_AGGRESSIVE`, `CALIB_DATASET`, `CALIB_SAMPLES`, `GPTQ_GROUPSIZE` (the values a single run uses) |
| 3-4b. Stage 0 | sensitivity score, bits, KV-cache plan, activation plan |
| 5-6. Sweeps | `SWEEP`, `PRUNE_SWEEP_RATIOS`, `PRUNING_LEVELS`, `THRESHOLD_SWEEP`, `GUARD_SWEEP` |
| 7. Stages 1-3 | `STAGE0_DIR`, `STAGE1_METHODS`, `STAGE2_METHODS`, `STAGE2_AFTER`, `STAGE3_METHODS` |
| 8. Search | limits `TARGET_PPL`, `TARGET_MEMORY_GB`, `TARGET_LATENCY_MS` (checked on `SEARCH_LATENCY_METRIC`); `SEARCHERS`, `SEARCH_BUDGET`, `SEARCH_OBJECTIVES`, `VLLM_PYTHON`; the techniques each stage may pick (`SEARCH_STAGE1`, `SEARCH_STAGE2`, `SEARCH_STAGE3`, `SEARCH_EXTRA_STAGE1`); `SEARCH_JOINT_PLAN`, `SEARCH_RANGES`, `SEARCH_LOW_THRESHOLD_MAX` |

The modes:

- `MODE = "single"`: Stage 0 once.
- `"sweep"`: every combination in `SWEEP`, one comparison report.
- `"compare_scores"`: every sensitivity score, compared on how they rank layers.
- `"prune_sweep"`: really prunes the model at every level in `PRUNE_SWEEP_RATIOS` (10% to 100% by default) and
  measures the error, standard method vs framework, plus the fair test at the same size and share removed with the
  pruning placed by sensitivity (`stage_0_prune_sweep/`).
- `"threshold_sweep"`: builds and measures the plan at every threshold and guard size in section 6.
- `"stages"`: Stages 1-3 with the methods in section 7 on a Stage 0 plan (run first unless `STAGE0_DIR` is set),
  then the master report.
- `"chain"`: Stages 0 > 1 > 2 > 3 > 4 in a row, every row on a trade-off line going into the next stage.
- `"search"`: the settings search of section 8 (below).
- `"stage_search"`: the search folded into the stages: Stage 1 and Stage 2 each search their own settings, Stage 3
  tries every KV technique, then the best pairs are built as chained models.

The command-line tools below do the same as `"single"` and `"sweep"` with `--set` overrides.

## Reproduce the 2026-10-29 search run

MODE "search" with MOBO only, on a Colab A100 (vLLM needs Linux). `main.py` holds these settings except `MODE`,
`SEARCH_BUDGET`, `VLLM_PYTHON`, `RUN_ID` and the folders, which step 4 of the Colab notebook sets:

| Setting | Value |
|---|---|
| `MODE` | `"search"` |
| `SEARCHERS` | `["mobo"]` |
| Limits | `TARGET_PPL = 11.5` (validation half), `TARGET_LATENCY_MS = 2.5` on `SEARCH_LATENCY_METRIC = "vllm_decode_ms_per_token"`, `TARGET_MEMORY_GB = None` |
| `SEARCH_OBJECTIVES` | `["predicted_weight_memory_gb", "ppl_val", "vllm_decode_ms_per_token"]` (size first: the final model is the smallest under the limits) |
| `SEARCH_STAGE1` | every WxAy row (`<weights>_with_rtn_act` and `<weights>_with_smoothquant` for rtn, gptq, awq, omniquant, squeezellm, spqr, efficientqat, quip, quipsharp, pbllm, bitsandbytes; kept to show they make the model slower on vLLM) plus the weight-only techniques `rtn`, `gptq`, `awq`, `omniquant`, `efficientqat` |
| `SEARCH_STAGE2`, `SEARCH_STAGE3` | `unstructured_prune`, `structured_prune`; every KV technique |
| `SEARCH_EXTRA_STAGE1` | `spqr_with_rtn_act`, `quip_with_rtn_act`, `quipsharp_with_rtn_act` |
| `SEARCH_JOINT_PLAN`, `SEARCH_RANGES`, `SEARCH_LOW_THRESHOLD_MAX` | `False`; threshold [0.8, 0.9] and prune ratio [0.0, 0.3]; `0` (vLLM 0.31 has no 3-bit GPTQ kernel) |
| `SEARCH_BUDGET`, `SEED` | `30` (main.py's default is 100), `0` |
| `VLLM_PYTHON` | `"/content/vllm-venv/bin/python"` |
| `RUN_ID` | `"20261029-colab-a100-ppl11.5-lat2.5"` |
| Cache | fresh: an empty `CACHE_DIR` (a new Colab session has one), so no earlier run's tries (`search_tried.csv`) are reused |

Why the weight-only techniques: vLLM runs WxAy rows on Humming kernels in eager mode (`src/sdf/eval/vllm_worker.py`),
35-42 ms per token, while weight-only rows go through the GPTQ export with CUDA graphs at about 2 ms per token
(2026-10-07: GPTQ 1.98, RTN 2.06, AWQ 2.06). Only weight techniques on a uniform per-group grid are listed, which the
export re-rounds faithfully. With a vLLM objective, or the latency limit on a vLLM metric, every try runs vLLM
(`sdf.search.tries_need_vllm`); otherwise tries run no backend, and vLLM, llama.cpp and TensorRT-LLM measure only the
final model. A try whose vLLM measurement fails does not meet the latency limit.

Colab notebook (`notebooks/colab_run.ipynb`), step 4:

```python
main.MODE = "search"
main.SEARCH_BUDGET = 30
main.VLLM_PYTHON = "/content/vllm-venv/bin/python"
main.RUN_ID = "20261029-colab-a100-ppl11.5-lat2.5"
main.OUTPUT_ROOT = f"{ROOT}/results"
main.CACHE_DIR = f"{ROOT}/cache"  # empty in a new Colab session
```

Outputs go to `<OUTPUT_ROOT>/<RUN_ID>/search/`: `points/<hash>/` (each try's Stage 0 and Stage 3 reports),
`trials.json`, `search_report.md`, `search_results.xlsx` (Trials, Pareto, Searcher comparison, Best configs) and
the plots; every try is also logged in `<CACHE_DIR>/search_tried.csv`.

To run it on a model other than TinyLlama, see [docs/other-models.md](docs/other-models.md).

### On Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamuutpls/sensitivity-driven-framework/blob/main/notebooks/colab_run.ipynb)

`notebooks/colab_run.ipynb` pulls `main`, installs the package, runs the tests and then `main.py` on the Colab
GPU, with results and caches in `/content/thesis_compression/` (no Google Drive, so nothing to sign in to). Its last
cell packs the results into `/content/thesis_results.tgz`, which the host PC session copies to `D:\Thesis`; on a local
runtime on the PC the results are written to `D:\Thesis\thesis_compression\` directly.

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
- **standard quantization** (`original`): a uniform allocation, every layer at one bit length (4-bit, no pruning),
  i.e. what a method uses with no guidance.
- **our method** (`framework`): the sensitivity plan, which mixes bit lengths per layer.

For both it reports predicted weight memory, average bits per weight, sparsity and *sensitivity
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

## Changes

Every change, with the reason and Mohammad's words where he asked for it, is in [CHANGELOG.md](CHANGELOG.md),
newest first.
