"""Run Stage 0 of the sensitivity-driven compression framework.

Edit the settings below, then run:  python main.py
Every tunable parameter is here. Anything not listed keeps its default from src/sdf/config.py.
"""

# =====================================================================================================
# 1. What to run
# =====================================================================================================
MODE = "single"  # "single": one Stage 0 run with the settings in section 2
#                  "sweep":  try every combination of the values in section 4 and write one comparison report
#                  "compare_scores": measure sensitivity all three ways (section 3) and compare how they rank layers

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_ROOT = "thesis_compression/results"  # local folder, relative to where you run; results/<run_id>/stage_0/
CACHE_DIR = "thesis_compression/cache"  # sensitivity profiles and the FP16 baseline are reused from here
RUN_ID = None  # None -> timestamp + config hash; reuse a name to resume a run
SEED = 0
MEASURE_FP16 = True  # measure the uncompressed model's perplexity, memory and latency (cached after the first run)

# =====================================================================================================
# 2. Search-space parameters (the ones a search tunes). Allowed values are in src/sdf/search_space.py.
# =====================================================================================================
SENSITIVE_THRESHOLD = 0.5  # 0.1-0.9: layers at or above this sensitivity (0-1 rank scale) are protected
PRUNE_RATIO_AGGRESSIVE = 0.3  # 0.0-0.6: share of numbers removed from each unprotected layer
CALIB_DATASET = "wikitext2"  # wikitext2 | c4 | pile10k: text used to measure sensitivity
CALIB_SAMPLES = 64  # 16 | 32 | 64 | 128: number of calibration passages
GPTQ_GROUPSIZE = 128  # 32 | 64 | 128 | -1 (one scale per row): numbers sharing one scale factor

# =====================================================================================================
# 3. Stage 0 fixed settings (not searched)
# =====================================================================================================
SENSITIVITY_SCORE = "layer_removal"  # how a layer's sensitivity is measured:
#   "grad_x_weight": size of each number x its gradient, summed per layer (one pass, fast estimate)
#   "layer_removal": perplexity rise when the layer is skipped entirely
#   "layer_quant":   perplexity rise when only that layer is compressed to COMPRESSED_BITS
NORMALIZATION = "rank"  # rank | minmax: how raw sensitivity scores are put on the 0-1 scale
PROTECTED_BITS = 8  # bits per number in protected layers
COMPRESSED_BITS = 4  # bits per number in unprotected layers
NO_PRUNE_COMPRESSED_BITS = 3  # bits for unprotected layers in the same-size plan that removes nothing
UNIFORM_BITS = 4  # the standard method: every layer at this many bits ...
UNIFORM_PRUNE_RATIO = 0.0  # ... with this share removed
CALIB_SEQ_LEN = 512  # tokens per calibration passage
EVAL_SEQ_LEN = 512  # tokens per perplexity window
EVAL_MAX_WINDOWS = None  # cap on perplexity windows (None = the whole WikiText-2 test split)

# =====================================================================================================
# 4. Sweep values (MODE = "sweep"). Leave a list out, or set it to None, to use every allowed value
#    (5 evenly spaced values for the threshold and prune ratio).
# =====================================================================================================
SWEEP = {
    "sensitive_threshold": [0.1, 0.3, 0.5, 0.7, 0.9],
    "prune_ratio_aggressive": [0.0, 0.15, 0.3, 0.45, 0.6],
    "calib_dataset": ["wikitext2", "c4", "pile10k"],
    "calib_samples": [16, 32, 64, 128],
    "gptq_groupsize": [32, 64, 128, -1],
}


# =====================================================================================================
def build_config():
    from sdf.config import FrameworkConfig

    return FrameworkConfig().with_overrides({
        "model.name": MODEL,
        "run.output_root": OUTPUT_ROOT, "run.cache_dir": CACHE_DIR, "run.run_id": RUN_ID, "run.seed": SEED,
        "hyperparams.sensitive_threshold": SENSITIVE_THRESHOLD,
        "hyperparams.prune_ratio_aggressive": PRUNE_RATIO_AGGRESSIVE,
        "hyperparams.calib_dataset": CALIB_DATASET,
        "hyperparams.calib_samples": CALIB_SAMPLES,
        "hyperparams.gptq_groupsize": GPTQ_GROUPSIZE,
        "stage0.score": SENSITIVITY_SCORE,
        "stage0.normalization": NORMALIZATION,
        "stage0.protected_bits": PROTECTED_BITS,
        "stage0.compressed_bits": COMPRESSED_BITS,
        "stage0.no_prune_compressed_bits": NO_PRUNE_COMPRESSED_BITS,
        "stage0.uniform_bits": UNIFORM_BITS,
        "stage0.uniform_prune_ratio": UNIFORM_PRUNE_RATIO,
        "calibration.seq_len": CALIB_SEQ_LEN,
        "eval.seq_len": EVAL_SEQ_LEN,
        "eval.max_windows": EVAL_MAX_WINDOWS,
    })


def main():
    from sdf.run import start_run
    from sdf.search_space import SEARCH_SPACE

    cfg = build_config()
    candidate = SEARCH_SPACE.make(cfg.hyperparams)  # rejects values outside the allowed ranges
    if MODE == "single":
        from sdf.stage0.run import run_stage0

        outputs = run_stage0(start_run(cfg), candidate, measure_fp16=MEASURE_FP16).outputs
    elif MODE == "sweep":
        from sdf.stage0.sweep import run_sweep

        grid = {k: v for k, v in SWEEP.items() if v}
        for name, values in grid.items():
            for v in values:
                SEARCH_SPACE.make({**candidate, name: v})
        outputs = run_sweep(start_run(cfg), grid)
    elif MODE == "compare_scores":
        from sdf.stage0.compare import compare_scores

        outputs = compare_scores(start_run(cfg), candidate)
    else:
        raise SystemExit(f"MODE must be 'single', 'sweep' or 'compare_scores', not {MODE!r}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
