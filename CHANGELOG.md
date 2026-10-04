# Changelog

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
