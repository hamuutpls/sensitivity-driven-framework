# Changelog

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
