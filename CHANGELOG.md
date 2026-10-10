# Changelog

## 2026-10-10: joint plan refined with every layer rounded; SpinQuant on GPU

- The per-layer joint plan scored worse than the separate plans on Colab A100. Stage 0 now refines it in context
  (`stage0.joint_refine_rounds`, default 2): starting from the better fully rounded plan of the per-layer joint
  plan and the separate plans, each layer is re-measured at every pair with every other layer rounded, the plan is
  re-made, and a new plan is kept only if the fully rounded model's calibration perplexity drops. The row info
  records each start's and each round's perplexity.
- SpinQuant's Hadamard rotation is now built on the model's device (it stayed on the CPU, so GPTQ + SpinQuant
  failed on GPU; PR #58).

## 2026-10-10: joint weight x activation plan (Stage 0e)

- Stage 0 measures every decoder layer with its weights and its Linear inputs rounded together, at every pair in
  `joint_w_options` x `joint_a_options` (4, 8, 16 each), and picks one pair per layer with the smallest summed rise
  under both budgets at once (exact dynamic programming; average weight bits = the Stage 1 weight plan's,
  average activation bits = `act_avg_bits`, unless set). 16 bits is an option, so no layer is protected by a fixed
  rule. Files: `joint_weight_plan.json`, `joint_activation_plan.json`, `joint_profile.json`; Stage 0 row
  `joint_weights_activations` reports the pairs, the predicted rise of the joint and of the separate plans, and
  how far rounding both differs from the sum of rounding each alone.
- Stage 1 joint methods get a second framework row, `<name>_joint`, that follows the joint plan.

## 2026-10-10: weights and activations quantized together, per layer

- Stage 1 joint methods `<weight method>_with_<activation method>` quantize the weights at the Stage 0 weight plan's
  bits and the Linear inputs at the Stage 0 activation plan's bits, layer by layer, on one model (`JointPlan`), so
  layers get W8A8, W8A4, W4A8 or W4A4. Standard row: W4A8 on every layer; `..._w8a8` adds a W8A8-everywhere
  standard row. The activation method rewrites the model first (smoothing, rotation; nothing for `rtn_act`), then
  the weight method quantizes the rewritten weights. Row descriptions count the layers per combination.
- `scripts/colab_stage1_joint.py` runs the working weight techniques with `rtn_act`, plus SmoothQuant + RTN, QuaRot +
  GPTQ and SpinQuant + GPTQ, on Colab.

## 2026-10-10: diagrams redrawn from the code

- `docs/diagrams/`: 4 class and 4 sequence diagrams of the current code (Mermaid `.mmd` + `.svg` + `.png`),
  indexed in its README; joint `<weights>_with_<activations>` methods (JointPlan) drawn in. Replaces the stale per-stage pages (stage0-4.md); `search.md` (planned) stays.

## 2026-10-09: every Stage 1 technique from the forks

- New Stage 1 weight methods (simplified plain-torch ports of the forks, one commit and file cited above each):
  `omniquant`, `squeezellm`, `spqr`, `efficientqat`, `aqlm`, `quip`, `quipsharp`, `pbllm`, `billm`, `bitsandbytes`;
  activation methods `smoothquant`, `quarot`, `spinquant`, `rptq`. `qtip` and `abqllm` are registered as failed rows
  with the reason. All follow the plan's per-layer bits and refuse a plan that removes anything.
- BiLLM has no bit-width knob: its standard row runs, its plan-following row fails with that reason
  (`Method.fixed_bits`). Rows carry a `note` (simplified port, nominal size) and the metric `zero_weight_share`.
- `scripts/colab_stage1.py` runs Stage 0 + Stage 1 on Colab with no Drive and prints every number.

## 2026-10-08: the budget plan is removed; standard quantization against our method

- Removed: the Stage 0 budget plan and its nothing-removed benchmark (rows `allocation_same_size`,
  `allocation_same_size_no_prune`, `planner.budget_matched_plan`, `stage0.no_prune_compressed_bits`, the files
  `compression_plan_budget_matched.json` and `quant_plan_budget_matched.json`, the budget columns and sections of
  the Stage 0 report and handoff, the budget parts of the sweeps) and the Stage 1 plan key `quant_same_size`.
- The comparison is two-way: standard quantization (every weight at one bit length) against our method (bit lengths
  mixed per layer, the Stage 0 plan). Reports label the two rows "Standard method" and "Our method". The uniform
  6-bit and 8-bit activation rows stay as the standard side for activations. Stage 2 keeps its same-size pruning plan.
- Cached results on disk are untouched; budget rows in older run folders are just no longer read.

## 2026-10-08: TensorRT-LLM backend; vLLM backend fixes from the first real run (Colab A100)

- `eval.trtllm_python` (the Python with TensorRT-LLM; Linux only) adds TensorRT-LLM rows: the same GPTQ export,
  plus TensorRT-LLM's own `hf_quant_config.json` (W4A16_GPTQ / W8A16_GPTQ, MIXED_PRECISION for mixed plans; it does
  not read a GPTQ `quantization_config`), built into a TensorRT engine by `eval/trtllm_worker.py`: engine size,
  GPU memory loaded/after the run (read from the device; TensorRT does not allocate through torch), build time,
  prefill and decode time, perplexity of both halves from the context logits.
- vLLM: the export is symmetric (vLLM 0.31 rejects sym=False GPTQ), rounds against the FP16 scale it stores, the
  worker gets its venv's bin/ on PATH (ninja), and reserves a fixed 1 GB KV cache so peak memory measures the model.
- A plan a serving backend cannot load (vLLM has no 3-bit GPTQ kernel) keeps its HF/llama.cpp row and records the
  backend's own reason in the row info (`vllm_error`, `trtllm_error`).

## 2026-10-07: vLLM backend for Stage 1 weight rows

- `eval.vllm_python` (the Python with vLLM; Linux only, so run inside WSL2 or Colab) turns it on. Each weight row is
  saved as a GPTQ-format checkpoint at the plan's per-layer bits (`eval/gptq_export.py`: 2/3/4/8-bit packing, vLLM
  `dynamic` per-layer overrides) and measured by `eval/vllm_worker.py`: file size, weight memory, peak memory,
  prefill and decode time (warm-up and repeats as in HF), perplexity of both halves.
- Weights are re-rounded onto grids of their own when saved (as in llama.cpp), so `vllm_ppl_*` is comparable between
  rows, not with HF. Cache keys are unchanged when the backend is off.

## 2026-10-07: Stage 1 is quantization only, Stage 2 is pruning only

Mohammad's split: Stage 1 = quantization of weights and activations (path 0 > 1 > 4), Stage 2 = pruning
(0 > 2 > 4 on the FP16 model, 0 > 1 > 2 > 4 after a Stage 1 method), Stage 3 = KV cache (0 > 3 > 4).

- Stage 1 methods: `rtn`, `gptq`, `awq` (weights) and `rtn_act`, `smoothquant`, `quarot`, `rptq`, `spinquant`
  (activations, formerly Stage 2). They refuse a plan that removes anything (`require_bits_only`). GPTQ's fixed-mask
  (SparseGPT) path and RTN's magnitude pruning are gone; `rtn` no longer prunes.
- Stage 2 methods: `unstructured_prune` (Wanda), `structured_prune`, `low_rank`, in `stages/pruning.py`. They round
  nothing, except low-rank, whose factors are stored at the plan's bits. Run alone the weights stay FP16; run as
  `<pruning>_after_<quantization>` (config `stages.stage2_after`, e.g. `["gptq"]`) the Stage 1 method runs first.
- Stage 0 now also writes `quant_plan.json` (main plan, bits only), `quant_plan_budget_matched.json` (the
  nothing-removed budget plan, robust layers at 3 bits), `prune_plan.json` (the main plan's ratios, bits at 16) and
  `prune_plan_same_size.json` (as many numbers removed as the standard method, placed by sensitivity). The runner
  reads these instead of `compression_plan*.json`, so Stage 0 must be rerun.
- Standard versions: Stage 1 = uniform 4-bit, nothing removed; Stage 2 = every layer pruned at
  `prune_ratio_aggressive`, nothing rounded (4-bit uniform first when after Stage 1).
- llama.cpp rows are made for Stage 1 weight methods only. Cache versions bumped for every changed method.
- All earlier combined prune-plus-4-bit Stage 1 numbers (Wanda, structured, low-rank rows) belong to the series path
  and are stale; Stage 1 numbers for RTN, GPTQ and AWQ on the main and budget plans change (no pruning).

## 2026-10-07: Stage 1/4 audit fixes

- Stage 1-3 cache keys now include the settings a plan does not carry (`weight_zero_point`, `baseline_bits`,
  `act_group_size`, `kv_group_size`, `kv_module_names`) when they differ from the defaults. Before, changing one
  reused stale rows. Default-settings entries stay valid.
- Structured pruning and low-rank rows predict their size without the 1-bit-per-weight mask (whole channels or
  factors are removed). Only `predicted_weight_memory_gb` and `avg_bits_per_weight` of those rows change; no
  perplexity does. Cached rows are not stale (the prediction is recomputed every run).
- Wanda ranks weights with float32 scores (FP16 scores overflowed to inf when a channel's summed squares passed
  65504 squared). `unstructured_prune` is version 2, so its cached rows are measured again.
- llama-bench JSON is parsed up to its end, so text after it on stderr cannot break it.

## 2026-10-06: CUDA-graph latency works with transformers 5.18

- Timing passes an explicit all-ones attention mask. Without one, transformers 5.18 reads a GPU value to decide
  whether to skip the causal mask, which breaks CUDA graph capture; latency then fell back to eager (~40 ms/token
  on Colab) and each failed capture left 32 MiB allocated, so peak memory grew row by row.
- Every capture reuses one warm-up stream. A new stream per capture kept a new 32 MiB cuBLAS workspace alive, so peak
  memory grew 64 MiB per measured row.

## 2026-10-06: Latency timed with deterministic algorithms off

- The run's deterministic mode (`run.deterministic`) made HF decode 21% slower (4.12 -> 4.99 ms/token on the host
  PC) through slower kernels. `measure_latency` now turns it off while timing and restores it after; perplexity and
  everything else stay deterministic. The FP16 cache key has `latency: 2`, so cached rows are re-measured.

## 2026-10-06: Stage 4 llama.cpp backend (Stage 1 rows)

- With `eval.llamacpp_dir` and `eval.llamacpp_convert` set, each Stage 1 row is also saved as a GGUF file with every
  decoder layer at its planned bits (8 -> Q8_0, 4 -> Q4_K, ...; embeddings and output head F16, `--pure`), then
  measured: file size, llama-bench prefill/decode tok/s, llama-perplexity on both halves. llama.cpp re-rounds on its
  own grid and scores only the second half of each window, so its perplexity is compared between rows only.
- The saved model keeps its SentencePiece `tokenizer.model` (transformers 5 omits it; without it the GGUF had flat
  token scores and FP16 perplexity 17.8 instead of 8.1).
- Pruned weights are stored as zeros (GGUF has no sparse format), so pruning saves no size here.

## 2026-10-06: Latency measures the GPU, not Python

- Eager HF decode of a 1B model is mostly Python and kernel-launch overhead (32.8 ms/token on the host PC), so it
  said nothing about compression. `eval.latency_mode = "cuda_graph"` (default) runs prefill and each decode step
  from a fixed-size KV cache and replays them as CUDA graphs. If capture fails (e.g. a method's hook syncs with
  the CPU) it times eagerly; each raw latency record has `mode`. `"eager"` times the same loop without
  graphs. The setting is in the FP16 cache key, so cached baselines are re-measured.

## 2026-10-06: Stage 1 pruning and low-rank methods

- `unstructured_prune` (Wanda: |w| x input norm per output row), `structured_prune` (whole feed-forward channels,
  found by shape, lowest input norm x output-column norm first) and `low_rank` (activation-aware SVD, rank keeping
  1 - planned share of each layer's numbers). Each then rounds to the planned bits.
- Their standard version removes `prune_ratio_aggressive` from every layer (Method.prunes), since removing nothing
  would just be RTN. All six Stage 1 methods are the default.
- The plan's predicted size assumes a bitmask for removed numbers; structured and low-rank need none, so their
  real size is slightly smaller than predicted (Stage 4 measures it).

## 2026-10-06: Stage 1 GPTQ and AWQ

- `gptq` and `awq` (src/sdf/stages/weights.py), plain torch, layer by layer with inputs from the already-compressed
  layers before. Both use the round-to-nearest grid, follow each layer's planned bits, and keep the plan's pruned
  weights at 0 (GPTQ feeds their error back, as SparseGPT does). AWQ has no weight-clipping search yet.
- Default Stage 1 methods: rtn, gptq, awq.

## 2026-10-05: "Nothing removed" is a benchmark, not a plan for later stages

- The budget-size plan with nothing removed stays in the Stage 0 rows, sweeps and report, labelled
  "Benchmark: budget size, nothing removed". It is no longer saved as a plan file, returned in `Stage0Result`,
  listed in handoff.md or run as a Stage 1 framework row. Handoff and Stage 1 call the 0.768 GB plan the budget plan.

## 2026-10-05: Realistic size for pruned weights; comparable fragile-parts score; code version in reports

- Pruning is unstructured, so deleted weights only save space in a sparse file. `stage0.sparse_storage`:
  "bitmask" (default, +1 bit per weight of a partly pruned layer), "dense" (no saving), "free" (old ideal).
  The fair pruning test now pairs versions by share removed (a bitmask makes its plan slightly smaller).
- The fragile-parts score (exposure) of a non-removal plan is scored against the layer-removal ranks: against
  its own ranks every measure protecting k layers scores the same (Hessian 0.58 = removal 0.58 by construction).
- `environment` in every results.json records `git_commit` and `git_dirty`.

## 2026-10-05: Weight rounding keeps 0 on the grid (integer zero point)

- `round_to_nearest` offset its grid by the group minimum (a float), so 0 was usually not one of the 16 values.
  Magnitude pruning keeps pruned weights at exactly 0, so pruned layers got a 17th value for free: the standard
  method scored 10.87 with 10% removed against 11.32 with nothing removed. Weights now use an integer zero point
  (`stage0.weight_zero_point = "int"`, GPTQ/AWQ format); `"float"` reproduces the old numbers. Activations and the
  KV cache keep the float grid (unchanged). Cache keys of pruning/threshold measurements and the one-layer
  compression profile include the grid, and the RTN method version is 2, so nothing old is reused by mistake.

## 2026-10-05: "Budget plan" instead of "same size"

- The two plans that protect as many layers as fit in the standard method's memory were called "same size", but
  they come out 1-1.4% smaller (0.768 / 0.766 GB against 0.777 GB). Report text, labels and figures 3 and 6 now
  call them the budget plan and the budget plan with nothing removed. Internal keys (`allocation_same_size`, ...)
  are unchanged, so caches and saved plans still load. The fair pruning test keeps "same size": there the sizes match.

## 2026-10-04: Protection threshold sweep

- `MODE = "threshold_sweep"` (src/sdf/stage0/threshold_sweep.py): builds the Stage 0 plan at every threshold in
  `THRESHOLD_SWEEP` (0.1 to 0.9) and guard size in `GUARD_SWEEP` (0, 3, 5, 8), applies it to the real weights and
  measures perplexity, next to the standard method and the same-size plans. The report marks the best trade-offs
  (no other version both smaller and more accurate). Measurements are cached per plan and shared with
  `prune_sweep` (`prune_sweep.make_evaluator`).

## 2026-10-04: Report text for every sensitivity score

- Stage 0's report explained only `grad_x_weight`, `layer_removal` and `layer_quant`, so a single run with
  `fisher`, `taylor_ema`, `hessian` or `movement` crashed with a `KeyError` after the profile was computed. All
  seven scores now have a plain-language description, and a test checks every score in `SCORES` has one.

## 2026-10-04: Downstream tasks and the master report

- `eval/downstream.py`: multiple-choice accuracy through lm-evaluation-harness (`pip install -e .[downstream]`),
  set with `DOWNSTREAM_TASKS` / `DOWNSTREAM_LIMIT` in `main.py` (off by default). Stages 1-3 measure it on every
  row; the FP16 result is cached on its own key, and the FP16 perplexity cache key no longer includes the
  downstream settings, so existing cache entries stay valid.
- `reporting/master.py`: `all_stages_comparison.xlsx` and `master_report.md` from every `stage_<N>/results.json`,
  written at the end of `MODE = "stages"` (or any time with `write_master(run_dir)`).

## 2026-10-04: Groundwork for Stages 1-3

- **`MODE = "stages"`** (`src/sdf/stages/`): one runner for Stages 1-3. It loads the Stage 0 plans
  (`STAGE0_DIR`, or runs Stage 0 first), then for each method builds the uncompressed row (Stage 0's cache entry),
  the standard method on the uniform plan (cached by config) and the framework on each Stage 0 plan, every one from
  a fresh FP16 model, and reports to `stage_<N>/` through `StageReporter`.
- `stages/methods.py` lists every method in the spec. Round-to-nearest baselines work now (`rtn`, `rtn_act`,
  `rtn_kv`); the rest are failed rows saying "not implemented yet" until they are written.
- Search space: `smoothquant_alpha` (0-1, default 0.5) and `quarot_k_bits` (2/3/4/8, default 4). The Stage 0 sweep's
  default grid only covers the Stage 0/1 parameters.
- `docs/stage-methods-feasibility.md`: which library each method needs and whether it runs on the Windows PC and
  Colab.

## 2026-10-04: Fair pruning test

- `prune_sweep` adds the fair comparison (`stage0.prune_sweep_same_size`, on by default; rows `same_NNN`): every
  layer at `uniform_bits` and the same number of weights removed as the standard method, so the same predicted
  size and share removed, but placed by sensitivity (`planner.same_size_pruning_plan`: robust layers lose more,
  fragile ones less, guarded ones nothing). Its own report section, "Fair test: same size, same share removed".
  Where the guarded layers can't absorb the pruning (high levels) the plan stays bigger and the report says so.

## 2026-10-04: Run on Google Colab

- `notebooks/colab_run.ipynb`: mounts Drive, clones or pulls `main`, installs, runs the tests, then runs `main.py`
  with `MODE` and the output / cache folders set in the notebook (`MyDrive/thesis_compression/results` and
  `cache`). No code changes were needed: every path was already a setting.

## 2026-10-03: Hessian score fixed

- `hessian` came out negative for some TinyLlama layers (one probe per passage is mostly noise). It now uses
  `stage0.hessian_probes` (default 8) probes per passage, averages the curvature estimate per weight over all
  passages and probes, and drops the negative part before weighting by w². Scores are never negative. Costs
  1 + `hessian_probes` backward passes per passage two CPU copies of the layer weights (originals and the
  curvature sum) and one extra gradient on the device.

## 2026-10-03: Four more sensitivity scores

- `stage0.score` adds `fisher`, `taylor_ema`, `hessian` and `movement` (`profile_sensitivity(method=...)`; settings
  `taylor_ema_beta` 0.9, `movement_lr` 1e-4, `hessian_eps` 1e-3). Hessian uses Hutchinson's diagonal estimate with
  a finite-difference Hessian-vector product, so it needs no second-order graph; movement runs one plain SGD step
  per calibration passage. Both restore the weights from a CPU copy and keep per-weight state on the CPU, so GPU
  memory is about one backward pass. Layer removal stays the default.
- `compare_scores` runs all seven, survives a failing score, and opens its report with how well each score
  matches the two measured ones (agreement, top 5 vs the never-pruned set, time, peak memory).

## 2026-10-03: Pruning levels, measured

- **`MODE = "prune_sweep"`** (`stage0/prune_sweep.py`): prunes the real weights at every share in
  `PRUNE_SWEEP_RATIOS` (default 10% to 100% in 10% steps, `stage0.prune_sweep_ratios`) and measures perplexity on
  both test halves, standard method (every layer 4-bit, pruned) vs framework (Stage 0 plan, protected and guarded
  layers not pruned). Magnitude pruning per output row plus round-to-nearest at the plan's bits
  (`PRUNE_SWEEP_QUANTIZE = False` for pruning only), simulated on FP16 weights; each level is cached. Outputs in
  `stage_0_prune_sweep/` with a "Pruning curve" table. Before, pruning was only predicted, at the one fixed
  ratio (0.3).
- Pruning ratio 1.0 is now allowed in plans (the layer keeps no weights and passes its input on).
- `predict_cost` counts scale factors only for kept weights, consistent with its "pruned weights are free"
  assumption; plans that prune are predicted slightly smaller (about 0.5% at 30% pruning on TinyLlama).
- `load_fp16` shared by Stage 0 and the new study; `StageReporter(subdir=...)` for side studies.

## 2026-10-01: Annotated figures

- `docs/figures/`: six annotated figures (SVG + PNG) for the thesis: pipeline overview, how Stage 0 builds its
  plans, the weight, activation and KV cache plans with the final run's numbers, and what Stage 0 hands to each
  later stage. Drawn by `docs/figures/make_figures.py` (needs matplotlib, not a package dependency).

## 2026-10-01: Stage 0 finalised: measured activation plan, hand-off report, honest verdicts

- **Activation plan is measured** (`ACT_PLAN = "measured"`, default): each layer's Linear inputs are rounded to
  4 and 8 bits on their own (64 passages) and the perplexity rise recorded; an average of 6 bits per layer goes
  where the rise is largest. The weight-derived plan stays as `ACT_PLAN = "from_weights"` and as the comparison
  row `activations_from_weights/framework`. New metric `predicted_act_ppl_rise`, `activation_profile.json`,
  per-layer column "Activation damage at fewest bits". `allocate_bits` is now shared by the KV and activation
  plans.
- **KV cache measured on 64 passages by default** (`KV_CALIB_SAMPLES`): the 2026-09-30 rerun showed 16 passages
  were too noisy for the bit choice. The KV bit plan keeps the standard method's 4-bit average (same size); the
  hand-off report says when the bit choice predicts no gain.
- **`handoff.md`**: what Stage 0 hands to Stages 1-4 and the search, layer by layer, with predicted costs, the
  Original model section and every column explained. Linked from report.md and returned in `outputs`.
- **Report fixes:** build time no longer turns every framework row into a "trade-off" (stated as a one-off
  cost instead); with no deployment target set, rows say "no targets set" instead of "yes"; `kv_budget_gb` is
  checked against the predicted KV memory; the FP16 row is called "Original model (uncompressed)".
- Specs v0.4 (S0-15, S0-17, REP-09; SDD §5.9, §5.10) and the Stage 0 diagram updated.

## 2026-09-30: Stage 0 hands everything later stages need

- **Pruning guard.** `stage0.guard_top_k` (default 5, `GUARD_TOP_K` in main.py): the layers whose removal raises
  perplexity most are never pruned in any framework plan (threshold, same-size, same-size without pruning),
  whatever score picks the bits. On TinyLlama these are 0, 2, 7, 21, 1. When `stage0.score` is not
  `layer_removal`, a removal profile is measured and cached for the guard. Before, the critical layers were only
  safe because the default threshold happened to protect them; a higher threshold, the size-matched plans or
  the gradient score could prune layer 0, which alone takes perplexity from 14 to ~1190. The sweep guards when
  its profiles are layer-removal ones.
- **Activation plan for Stage 2** (`stage0/activation.py`): protected and guarded layers keep
  `act_protected_bits` (8) activations, the rest `act_compressed_bits` (4); original = `act_uniform_bits` (8).
  Derived from the weight plan, not measured; Stage 2 should validate it. New rows `activations/original` and
  `activations/framework` (`avg_activation_bits`), `activation_plan.json`, report section and per-layer columns
  "Never pruned" and "Activation bits".
- **Hand-off.** `Stage0Result` also returns the same-size plans, the activation plan and both KV plans;
  `CompressionPlan.load`, `KVPlan.load` and `ActivationPlan.load` read the saved JSON back.
- Specs v0.3 (S0-14 to S0-16, SDD §5.4, §5.9, §7.3) and the Stage 0 class diagram updated.

## 2026-09-30: Every report table explains its columns

- `report.md` lists "What each column means" under the plain results table, the per-layer table and the
  technical results table. Metric columns use the metric registry's plain explanation (moved there from Key
  terms); per-layer columns take an optional third element, the meaning, in `plain_layer_columns`.
- Stage 0 explains all 12 per-layer columns; the raw-score explanation follows the sensitivity score used.

## 2026-09-30: Every report describes the original model

- New `utils/model_info.py`: `describe_model(config, name, num_parameters)` reads the original model's
  parameters from its Hugging Face config (architecture, parameter count, layers, hidden size, feed-forward
  size, attention heads, key/value heads, numbers per head, vocabulary, maximum context, published number
  format, bits per number, size at 16 bits, shared word tables). Fields a model's config lacks are left out,
  so it works for any model.
- `StageReporter(original_model=...)`: an "Original model" section near the top of `report.md` (value and a
  plain-language meaning for each), `original_model.*` rows in the Config sheet, and `original_model` in
  `results.json`. The sweep and score-comparison reports show the same section.
- Stage 0 reads only the config file when everything comes from cache; the parameter count then comes from
  the sensitivity profile. The glossary no longer hardcodes TinyLlama's size.

## 2026-09-30: Specs describe layer removal and the KV cache plan

- `docs/specs/` version 0.2: S0-01 names layer removal as the default sensitivity score, with one-layer
  compression and gradient x weight selectable (SDD §5.3); new S0-10 to S0-13 and SDD §5.8 for the Stage 0 KV
  cache plan; MET-06, S3-02 and SDD §8 follow that plan; traceability updated.

## 2026-09-29: System and subsystem specifications

- **`docs/specs/`**: a System Requirements Specification following ISO/IEC/IEEE 29148:2018 (every requirement
  from the thesis spec with an ID, priority, status and verification method) and a Subsystem Design
  Description following IEEE 1016-2009 (shared core, Stages 0 to 4 and the search layer, each described from
  the same viewpoints, with a traceability table from requirement to design, code and test). Stage 0 and the
  shared core are described from the code; the rest is marked planned, consistent with `docs/diagrams/`.

## 2026-09-30: Diagrams describe layer removal and the KV cache plan

- `docs/diagrams/stage0.md`: layer removal is the default sensitivity score (grad x weight and one-layer
  compression still drawn as options); new KV cache classes (`KVProfile`, `KVPlan`, `KVLayerPlan`, `KVCost`,
  `kv_cache` module) and the three KV cache rows in the sequence diagram. `stage3.md` now reads the Stage 0 KV
  cache plan.

## 2026-09-29: Class and sequence diagrams

- **`docs/diagrams/`**: a class diagram and a sequence diagram for the project as a whole and for each stage
  and the search layer, in Mermaid so GitHub draws them. Stage 0 and the shared parts are drawn from the
  code; Stages 1 to 4 and the search are drawn from the design and labelled `<<planned>>`.

## 2026-09-30: Stage 0 plans the KV cache

- New `stage0/kv_cache.py`, on by default (`KV_CACHE` in main.py, section 4). Per decoder layer it plans:
  1. key bits and 2. value bits, chosen separately: each layer's keys (then values) alone are rounded to each
     of `KV_BITS_OPTIONS` and the calibration perplexity rise is measured. A greedy allocation spends the
     same average bits as the uniform `KV_UNIFORM_BITS` cache where the measured damage is largest. Keys are
     rounded per channel and values per token (KIVI / KVQuant), keys before RoPE.
  3. a token budget (H2O / SnapKV style eviction): the fewest of `KV_KEEP_RATIOS` whose top tokens still
     receive `KV_ATTENTION_COVERAGE` of the layer's attention (oracle top-k, measured with eager attention).
  4. predicted cache memory at `KV_CONTEXT_LEN` tokens x `KV_BATCH_SIZE`, with scale overhead per group.
- Report rows `kv_cache/original` (uniform bits, no eviction), `kv_cache/framework` (bits + budget) and
  `kv_cache_bits_only/framework` (bits alone, to separate the two effects); the FP16 row gets the 16-bit cache
  size. New metrics, per-layer columns, a "KV cache plan" section and plain-language text. Plans and the KV
  profile are saved as JSON; the profile is cached.
- Key/value layers are found by name (`KV_MODULE_NAMES`), so any model with separate key/value projections
  works; fused-QKV models need their own names.

## 2026-09-30: Default sensitivity score is layer removal

- `stage0.score` defaults to `layer_removal`, Mohammad's choice: a layer's sensitivity is the rise in
  calibration perplexity when it is skipped. `layer_quant` and `grad_x_weight` stay selectable.

## 2026-09-29: Default sensitivity score is now single-layer compression

- `stage0.score` defaults to `layer_quant`. On TinyLlama (WikiText-2, 64 passages) gradient × weight ranked
  layer 0 least sensitive, so the plan compressed and pruned it, yet skipping layer 0 raises perplexity from
  14.1 to about 1,190. Its ranking also did not agree with the measured one-layer compression damage (rank
  agreement −0.01). `layer_quant` measures the damage protection prevents directly. Old runs are unaffected:
  the score is part of the profile cache key.

## 2026-09-29: Layer-removal and one-layer-compression sensitivity scores

- **`stage0.score`** (`SENSITIVITY_SCORE` in main.py) chooses how sensitivity is measured:
  - `grad_x_weight` (default): unchanged.
  - `layer_removal`: skip one decoder layer at a time and measure the rise in perplexity on the
    calibration text.
  - `layer_quant`: compress only that layer (round-to-nearest at `compressed_bits` and the candidate's
    group size) and measure the rise in perplexity. This is closest to what the plan does.

  The ablation scores need one forward pass over the calibration text per layer. They work on any decoder
  stack through hooks and temporary weight rounding, and the model is restored afterwards. The score is
  part of the profile cache key.
- **`MODE = "compare_scores"`** in main.py profiles all three ways on the same text and writes
  `stage_0_scores/report.md`, `scores.xlsx` and `results.json`: rank agreement between the ways, the layers
  each would protect, and a per-layer table.
- The report's "How this stage works" section describes whichever score was used.

## 2026-09-29: Report explains the sensitivity-driven method step by step

- The plain part's background section is now "How this stage works". It explains in four numbered steps
  how the score is measured (weight size times gradient, summed per layer), how it is ranked onto a 0-1
  scale, what the threshold does, and how bits and pruning follow, including the two same-size versions.

## 2026-09-29: Local output folders

- Results and the cache now go to `thesis_compression/results` and `thesis_compression/cache` under the
  folder you run from, in `main.py` and `configs/tinyllama.yaml`, instead of Google Drive paths. The
  folder is git-ignored. The hardware profile in the config is now `RTX-5070-Ti`.

## 2026-09-29: Tunable parameters in main.py

- **`main.py`** lists every tunable parameter at the top of one file: the run settings, the five
  search-space parameters (with their allowed values in comments), the fixed Stage 0 settings and the sweep
  values. Edit it and run `python main.py`. `MODE` picks a single run or a sweep. Values outside the search
  space are rejected before anything runs. Settings not listed keep their defaults from `config.py`.

## 2026-09-29: Stage 0 sweep

- **`sdf-stage0-sweep`** tries several values of every search-space setting and writes one plain-language
  report plus `sweep.xlsx` (every plan) and `results.json`. It covers the threshold, the prune ratio, the
  group size, and the calibration text and amount. For each combination it shows the threshold plan, the
  same-size plan and the no-removal plan against the standard method, how much the calibration setting
  changes the layer ranking (rank agreement), and which plans are both smaller and safer than the standard
  method.
- A calibration setting that fails (for example, a dataset that can't be downloaded) is recorded in the
  report instead of stopping the sweep.
- `--profile` plans from saved sensitivity profiles, so the planning part runs without a GPU.
- Profile loading moved into `stage0.run.load_profile`, shared by the single run and the sweep.

## 2026-09-28: Report gaps from the first local run

- **Every core metric in both report tables.** Held-out perplexity, peak GPU memory and per-token decode
  latency were only in results.json; they are now columns in the plain and technical tables, along with
  sparsity.
- **Pruning caveat for the same-size plan.** On TinyLlama the same-size plan protects 5 layers and prunes
  the other 17 by 30%, while uniform prunes nothing, so the size match is bought with pruning. Sensitivity
  exposure scores 4 bits with 30% pruned as 2.8 effective bits, which likely understates the damage. The
  report now says so in both parts.
- **Same-size plan without pruning** (`allocation_same_size_no_prune`). Robust layers drop to
  `stage0.no_prune_compressed_bits` (3) instead of being pruned, and as many top layers as fit are
  protected at 8 bits within the uniform plan's memory. This isolates the effect of the sensitivity
  guidance itself. Its plan is saved as `compression_plan_budget_matched_no_prune.json`.
- **Per-layer table** now shows the share removed for the same-size plan and the bits per layer of the
  no-pruning plan.
- **Prefill latency** is a column in both report tables, next to decode latency.
- **Unused budget explained.** The same-size plans keep the protected set to the top-k layers by
  sensitivity, so packing stops at the first layer that doesn't fit (0.011 GB left on TinyLlama). The
  report now states the leftover and what the next layer would cost. Every TinyLlama decoder layer is the
  same size, so skipping ahead would not fit another layer either.
- **Unit test for the no-pruning plan**: it fits the uniform budget, removes nothing, keeps robust layers
  at 3 bits and protects the 4 most sensitive layers on TinyLlama shapes.

## 2026-09-28: Leaner code after the ponytail audit

About 220 lines were removed with no change to what a run produces:

- **One copy of every default.** The defaults live only in `config.py`. `configs/tinyllama.yaml` now holds
  only the Colab paths and the hardware profile, so it shows at a glance what a run changes.
- **Candidates are plain dicts.** The `Candidate` class, the unused sampling helpers and `stage0.score`
  (a setting with one allowed value) are gone.
- **Sensitivity is normalised once**, when planning. The saved profile holds only the raw scores, and the
  planner functions take the normalised score list.
- **Smaller helpers.** The cache has one `get_or_compute` method, the number and label helpers exist once,
  `load_texts(sources, dataset, split)` replaces the loader factory, and the package `__init__` files no
  longer re-export names (import from the modules directly).

## 2026-09-28: Report layout matches the hand-written version

- **New plain-part order.** The plain part of report.md now follows the layout Mohammad liked:
  1. a one-paragraph summary;
  2. key terms, covering the versions compared, every measure and stage-specific words;
  3. one results table with every version, units and "lower is better" in the headers;
  4. a sensitivity table for every layer;
  5. findings.

  The Stage 0 findings cover the same-size plan, the 16-bit size floor and outlier layers. The background
  explanation comes after the findings, and the technical detail follows unchanged.

## 2026-09-28: Fair same-size comparison, size floor, outlier flag

- **Same-size framework plan.** The threshold plan (0.947 GB predicted on TinyLlama) is about 22% bigger
  than uniform 4-bit (0.777 GB), so comparing its quality against uniform isn't fair. Stage 0 now adds a
  second framework row, `allocation_same_size/framework`, compared against the same uniform plan. It
  protects as many of the most sensitive layers as fit in the uniform plan's predicted memory, which works
  out to 5 of 22 layers at 0.775 GB. Its plan is saved as `compression_plan_budget_matched.json`.
- **Size floor reported.** Embeddings, the LM head and norms stay at 16 bits in every plan. The report now
  states this floor (0.26 GB on TinyLlama) in both the plain and the technical part.
- **Outlier layers flagged.** A layer whose raw sensitivity is far from the rest is flagged in the report
  and in the Per-layer sheet, together with what the plan does to it. Outliers are found by a robust
  z-score (median / MAD, above 3.5), which catches TinyLlama's layer 0. They are flagged, not
  auto-protected.

## 2026-09-28: Plain-language reports

- **Two-part report.md.** Every report now opens with a part written for readers with no AI background:
  - the short version (did the framework win);
  - what the stage does;
  - what was compared;
  - what each number means, with the three versions side by side and a "which is better/worse" reading;
  - why the framework won or lost;
  - a glossary.

  The technical tables follow unchanged under "Technical details".
- **Built into the shared reporter.** This lives in `StageReporter`, so every stage gets it. Each metric in
  the registry carries a plain name and explanation, and a stage supplies its own intro, "why" and glossary
  terms.

## 2026-09-28: Fixes from the first real run (RTX 5070 Ti)

- **Degenerate plan fixed.** Sensitivity scores are now rank-normalised by default
  (`stage0.normalization: rank`). With min-max, TinyLlama's layer 0 (raw score 2705 against 6596–8647 for
  the other layers) squeezed layers 1–21 into 0.66–1.0, so threshold 0.5 protected 21 of 22 layers. That
  plan averaged 8.97 bits against 5.65 for uniform. With rank, `sensitive_threshold` *t* protects about the
  top (1 − *t*) share of layers, whatever the score distribution. Normalisation is applied when planning,
  so cached profiles are reused.
- **Windows encoding fixed.** Every text file is now written and read as UTF-8: reports, JSON, config and
  `run.log`. Before this, Windows wrote cp1252, which garbled "—" and "Δ" in report.md.

## 2026-09-28: Dataset IDs moved to the config

- **Fix.** WikiText-2 now loads from `Salesforce/wikitext`. Current `datasets` versions no longer resolve
  the bare `wikitext` id, so Stage 0 failed before profiling.
- **Config.** All Hub dataset ids and splits now live in the config (`data.sources`), so a moved dataset is
  a config change rather than a code change.

## 2026-09-28: Stage 0 aligned with the thesis spec

- **Config.** Added a single `FrameworkConfig` (`configs/tinyllama.yaml` plus `--set key=value` overrides).
  Stage logic no longer hardcodes anything: bit-widths, the uniform baseline, eval settings and output
  paths all come from the config.
- **Search space.** It now uses the spec's parameter names and is defined in one place
  (`search_space.py`), so adding a parameter is one line. The scope is Stage 0 only, so it holds
  `sensitive_threshold`, `prune_ratio_aggressive`, `calib_dataset` and `calib_samples`, plus
  `gptq_groupsize`, which Stage 0 uses to predict memory. The Stage 2 and 3 parameters are added with
  those stages.
- **Reporting.** Added `StageReporter`, the shared reporting module. Every stage writes `report.md`,
  `stage_<N>_comparison.xlsx` and `results.json` with one schema. The reporter:
  - computes deltas against FP16 and against the original method;
  - highlights framework wins and losses with conditional formatting;
  - checks the DeploymentRequirement for each row and records the shortfall;
  - records a failed method as a failed row instead of stopping the stage.
- **FP16 baseline.** Added measurement of perplexity on the validation and held-out halves of the
  WikiText-2 test split, model size, peak GPU memory, and prefill and decode latency (warmup runs, repeated
  timings, mean and std). It is computed once per model, eval settings and hardware, then cached.
- **Stage 0 comparison.** Stage 0 now compares FP16, a uniform allocation (the original method) and the
  sensitivity plan. It reports predicted weight memory, bits per weight, sparsity, sensitivity exposure,
  build cost and per-layer allocations.
- **Profile caching.** The sensitivity profile is cached by model and calibration settings. Before this, it
  would have been recomputed for every trial, even though the searched Stage 0 thresholds don't affect it.
- **Seeding and logging.** Seeding covers python, numpy, torch and cuda, with a deterministic-algorithms
  flag, and the seed is logged. Logging is timestamped and goes to stderr and to `<run_dir>/run.log`.
- **Crash-safe output.** Writes are atomic and `results.json` is rewritten after every row, so a Colab
  disconnect keeps completed work.
- **Trial log removed.** It belongs to the search layer, which is out of scope for now.

## 2026-09-28: Initial Stage 0

- Project skeleton, gradient × weight sensitivity profile and threshold planner.
