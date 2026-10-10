"""Run Stage 0 of the sensitivity-driven compression framework.

Edit the settings below, then run:  python main.py
Every tunable parameter is here. Anything not listed keeps its default from src/sdf/config.py.
"""

# =====================================================================================================
# 1. What to run
# =====================================================================================================
MODE = "single"  # "single": one Stage 0 run with the settings in section 2
#                  "sweep":  try every combination of the values in section 5 and write one comparison report
#                  "compare_scores": measure sensitivity every way (section 3) and compare how they rank layers
#                  "prune_sweep": really prune at every level in section 6 and measure the error (standard vs framework, and the fair same-size test)
#                  "threshold_sweep": build and measure the plan at every protection threshold and guard size in section 6
#                  "stages": Stages 1-3 with the methods in section 7, on top of a Stage 0 plan

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
#   "fisher":        0.5 x (gradient x number)^2 per layer, averaged over passages (second-order, Fisher estimate)
#   "taylor_ema":    |sum of gradient x number| per layer, as a moving average over passages
#   "hessian":       0.5 x curvature x number^2 per layer (curvature from 8 random probes per passage, negatives -> 0)
#   "movement":      how far each number is pushed away from zero over a short fine-tune, per layer
NORMALIZATION = "rank"  # rank | minmax: how raw sensitivity scores are put on the 0-1 scale
PROTECTED_BITS = 8  # bits per number in protected layers
COMPRESSED_BITS = 4  # bits per number in unprotected layers
UNIFORM_BITS = 4  # the standard method: every layer at this many bits ...
UNIFORM_PRUNE_RATIO = 0.0  # ... with this share removed
SPARSE_STORAGE = "bitmask"  # size of pruned weights: "bitmask" +1 bit/weight, "dense" no saving, "free" ideal (old)
WEIGHT_ZERO_POINT = "int"  # weight rounding grid: "int" keeps 0 representable (GPTQ/AWQ); "float" = before 2026-10-05
GUARD_TOP_K = 5  # never prune the layers whose removal hurts most, this many of them (0 = no guard)
CALIB_SEQ_LEN = 512  # tokens per calibration passage
EVAL_SEQ_LEN = 512  # tokens per perplexity window
EVAL_MAX_WINDOWS = None  # cap on perplexity windows (None = the whole WikiText-2 test split)
DOWNSTREAM_TASKS = []  # multiple-choice tests for Stages 1-3 (needs `pip install lm-eval`), e.g.
#                        ["arc_easy", "hellaswag", "piqa", "winogrande"]; [] = not measured
DOWNSTREAM_LIMIT = None  # questions per test (None = all; e.g. 500 for a quicker run)

# =====================================================================================================
# 4. KV cache plan (the model's short-term memory while writing). Stage 0 measures it and plans, per layer:
#    bits for keys, bits for values, and the share of past tokens kept. Stage 3 carries the plan out.
# =====================================================================================================
KV_CACHE = True  # False skips the KV cache measurement and plan
KV_BITS_OPTIONS = [2, 4, 8]  # bit widths tried for each layer's keys and values (ascending)
KV_UNIFORM_BITS = 4  # the standard method: every key and value at this many bits, every token kept
KV_AVG_BITS = None  # average bits the plan may spend; None = KV_UNIFORM_BITS (same size as the standard method)
KV_GROUP_SIZE = 64  # numbers sharing one scale factor
KV_CALIB_SAMPLES = 64  # passages for the KV measurement (one pass per layer x key/value x bit width);
#                        16 proved too noisy for the bit choice (2026-09-30 rerun)
KV_KEEP_RATIOS = [0.1, 0.2, 0.3, 0.5, 0.75]  # shares of past tokens a layer may keep
KV_ATTENTION_COVERAGE = 0.95  # a layer keeps the fewest tokens that still get this share of its attention
KV_CONTEXT_LEN = 2048  # text length (tokens) for the predicted cache memory
KV_BATCH_SIZE = 1  # texts held at once for the predicted cache memory
KV_MODULE_NAMES = ["k_proj", "v_proj"]  # names of the layers producing keys and values (Llama, Mistral, Qwen)

# =====================================================================================================
# 4b. Activation plan (the numbers passed between layers while the model runs). Stage 1 carries it out.
# =====================================================================================================
ACT_PLAN = "measured"  # "measured": test each layer's sensitivity to rounding its inputs (about 44 passes)
#                        "from_weights": no test; layers the weight plan protects or never prunes keep more bits
ACT_BITS_OPTIONS = [4, 8]  # bit widths a layer's activations may get (ascending)
ACT_AVG_BITS = 6.0  # average bits the plan may spend per layer
ACT_UNIFORM_BITS = 8  # the standard method: every layer's activations at this many bits
ACT_GROUP_SIZE = 128  # numbers sharing one scale factor
ACT_CALIB_SAMPLES = 64  # passages for the activation measurement

# =====================================================================================================
# 5. Sweep values (MODE = "sweep"). Leave a list out, or set it to None, to use every allowed value
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
# 6. Pruning levels (MODE = "prune_sweep"). Each level is applied to the real weights and measured.
# =====================================================================================================
PRUNE_SWEEP_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # shares removed from each pruned layer
PRUNE_SWEEP_QUANTIZE = True  # True: also store the rest at the plan's bits (4 / 8); False: pruning only
PRUNE_SWEEP_SAME_SIZE = True  # True: also the fair test, same size and share removed, pruning placed by sensitivity
THRESHOLD_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # MODE "threshold_sweep": protection thresholds to measure
GUARD_SWEEP = [0, 3, 5, 8]  # MODE "threshold_sweep": never-pruned layer counts to measure with each threshold

# =====================================================================================================
# 7. Stages 1-3 (MODE = "stages"). Each stage starts from the uncompressed model and follows the Stage 0 plan.
#    Methods: src/sdf/stages/methods.py. Not-yet-written methods show up as failed rows that say so.
# =====================================================================================================
STAGE0_DIR = None  # a finished run's stage_0 folder, e.g. "thesis_compression/results/<run_id>/stage_0";
#                    None = run Stage 0 first with the settings above, in the same run
STAGE1_METHODS = ["rtn", "gptq", "awq", "rtn_act"]  # quantization only, nothing removed. Weights: rtn | gptq | awq |
#   omniquant | squeezellm | spqr | efficientqat | aqlm | quip | quipsharp | pbllm | billm | bitsandbytes (| qtip | abqllm:
#   not ported, shown as failed rows). Activations: rtn_act | smoothquant | quarot | rptq | spinquant
STAGE2_METHODS = ["unstructured_prune", "structured_prune", "low_rank"]  # pruning only, nothing rounded (FP16 model)
STAGE2_AFTER = []  # Stage 1 weight methods to also run each Stage 2 method after (0 -> 1 -> 2 -> 4), e.g. ["gptq"]
STAGE3_METHODS = ["rtn_kv"]  # KV cache: rtn_kv | quarot_kv | kvquant | h2o | snapkv | infinigen
SMOOTHQUANT_ALPHA = 0.5  # 0.0-1.0: how much of the activation outliers SmoothQuant moves into the weights
QUAROT_K_BITS = 4  # 2 | 3 | 4 | 8: key bits for QuaRot KV (standard method)


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
        "hyperparams.smoothquant_alpha": SMOOTHQUANT_ALPHA,
        "hyperparams.quarot_k_bits": QUAROT_K_BITS,
        "stages.stage0_dir": STAGE0_DIR,
        "stages.stage1_methods": STAGE1_METHODS,
        "stages.stage2_methods": STAGE2_METHODS,
        "stages.stage2_after": STAGE2_AFTER,
        "stages.stage3_methods": STAGE3_METHODS,
        "stage0.score": SENSITIVITY_SCORE,
        "stage0.normalization": NORMALIZATION,
        "stage0.protected_bits": PROTECTED_BITS,
        "stage0.compressed_bits": COMPRESSED_BITS,
        "stage0.uniform_bits": UNIFORM_BITS,
        "stage0.uniform_prune_ratio": UNIFORM_PRUNE_RATIO,
        "stage0.weight_zero_point": WEIGHT_ZERO_POINT,
        "stage0.sparse_storage": SPARSE_STORAGE,
        "stage0.guard_top_k": GUARD_TOP_K,
        "stage0.act_plan": ACT_PLAN,
        "stage0.act_bits_options": ACT_BITS_OPTIONS,
        "stage0.act_avg_bits": ACT_AVG_BITS,
        "stage0.act_uniform_bits": ACT_UNIFORM_BITS,
        "stage0.act_group_size": ACT_GROUP_SIZE,
        "stage0.act_calib_samples": ACT_CALIB_SAMPLES,
        "stage0.kv_cache": KV_CACHE,
        "stage0.kv_bits_options": KV_BITS_OPTIONS,
        "stage0.kv_uniform_bits": KV_UNIFORM_BITS,
        "stage0.kv_avg_bits": KV_AVG_BITS,
        "stage0.kv_group_size": KV_GROUP_SIZE,
        "stage0.kv_calib_samples": KV_CALIB_SAMPLES,
        "stage0.kv_keep_ratios": KV_KEEP_RATIOS,
        "stage0.kv_attention_coverage": KV_ATTENTION_COVERAGE,
        "stage0.kv_context_len": KV_CONTEXT_LEN,
        "stage0.kv_batch_size": KV_BATCH_SIZE,
        "stage0.kv_module_names": KV_MODULE_NAMES,
        "stage0.prune_sweep_ratios": PRUNE_SWEEP_RATIOS,
        "stage0.prune_sweep_quantize": PRUNE_SWEEP_QUANTIZE,
        "stage0.prune_sweep_same_size": PRUNE_SWEEP_SAME_SIZE,
        "stage0.threshold_sweep": THRESHOLD_SWEEP,
        "stage0.guard_sweep": GUARD_SWEEP,
        "calibration.seq_len": CALIB_SEQ_LEN,
        "eval.seq_len": EVAL_SEQ_LEN,
        "eval.max_windows": EVAL_MAX_WINDOWS,
        "eval.downstream_tasks": DOWNSTREAM_TASKS,
        "eval.downstream_limit": DOWNSTREAM_LIMIT,
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
    elif MODE == "prune_sweep":
        from sdf.stage0.prune_sweep import run_prune_sweep

        outputs = run_prune_sweep(start_run(cfg), candidate)
    elif MODE == "threshold_sweep":
        from sdf.stage0.threshold_sweep import run_threshold_sweep

        outputs = run_threshold_sweep(start_run(cfg), candidate)
    elif MODE == "stages":
        from sdf.stage0.run import run_stage0
        from sdf.stages.runner import run_stages

        ctx = start_run(cfg)
        outputs = {}
        stage0_dir = cfg.stages.stage0_dir
        if stage0_dir is None:
            outputs = run_stage0(ctx, candidate, measure_fp16=MEASURE_FP16).outputs
            stage0_dir = outputs["report"].parent
        outputs.update(run_stages(ctx, candidate, stage0_dir, measure_fp16=MEASURE_FP16))
        from sdf.reporting.master import write_master

        outputs.update(write_master(ctx.run_dir))
    else:
        raise SystemExit(f"MODE must be 'single', 'sweep', 'compare_scores', 'prune_sweep', 'threshold_sweep' or 'stages', not {MODE!r}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
