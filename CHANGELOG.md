# Changelog

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
