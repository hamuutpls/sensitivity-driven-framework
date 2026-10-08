# Stage 0: sensitivity profiling and compression planning

Drawn from the code in `src/sdf/stage0/`. Back to the [overview](README.md).

## In plain words

Stage 0 finds out which layers of the model are fragile. It feeds the model some ordinary text and, for every
layer, measures how much worse the model gets when that layer is removed (skipped): the rise in perplexity is
the layer's **sensitivity score**. This **layer removal** score is the default; two other ways can be picked
with `SENSITIVITY_SCORE` in `main.py`: compressing only that layer (`layer_quant`), or a one-pass estimate from
the size of each number times its gradient (`grad_x_weight`). Layers scoring at or above a threshold are **protected** (kept at
8 bits, nothing removed); the rest are **compressed** (4 bits, and some of their numbers removed). The result is
a **compression plan**: one line per layer saying how many bits it gets and how much of it is removed.

Stage 0 does not compress anything yet. It predicts how much memory each plan would need and compares:

- the **uncompressed** model (FP16),
- the **standard method**: every layer treated the same (uniform 4 bits),
- the **framework**: the sensitivity plan, plus two versions squeezed into exactly the standard method's memory
  (one that removes numbers from robust layers, one that only lowers their bits), so the comparison is fair.

Stage 0 also plans the model's short-term memory while it writes, the **KV cache**. For every layer it measures
how much perplexity rises when only that layer's keys (then values) are rounded to 2, 4 or 8 bits, and gives
more bits where rounding hurts most, keeping the same average as a uniform 4-bit cache. It also finds, per
layer, the fewest earlier words that still get 95% of the layer's attention (a **token budget**), and predicts
the cache's memory at 2,048 tokens. The report compares the uniform 4-bit cache, the full plan, and the plan
with bits only (nothing forgotten).

Measuring the sensitivity is the slow part, so the scores are saved and reused by every later trial that uses
the same model and calibration text.

## Class diagram

```mermaid
classDiagram
    direction LR

    class Stage0Config {
        +str normalization
        +str profile_dtype
        +int protected_bits
        +int compressed_bits
        +int uniform_bits
        +float uniform_prune_ratio
        +int group_overhead_bits
        +int baseline_bits
        +str score
        +bool kv_cache
        +list~int~ kv_bits_options
        +int kv_uniform_bits
        +float kv_avg_bits
        +int kv_group_size
        +int kv_calib_samples
        +list~float~ kv_keep_ratios
        +float kv_attention_coverage
        +int kv_context_len
        +list~str~ kv_module_names
    }
    class SensitivityProfile {
        +list~float~ raw_scores
        +list~int~ layer_numel
        +list~int~ layer_rows
        +int other_numel
        +str method
        +dict cost
        +dict meta
        +num_layers() int
        +to_dict() dict
        +save(path)
        +from_dict(d) SensitivityProfile
        +load(path) SensitivityProfile
    }
    class LayerPlan {
        +int layer
        +int bit_width
        +float pruning_ratio
        +bool protected
        +float sensitivity
        +bool guarded
    }
    class CompressionPlan {
        +tuple~LayerPlan~ layers
        +str kind
        +float sensitive_threshold
        +float prune_ratio_aggressive
        +protected_layers() list
        +compressed_layers() list
        +guarded_layers() list
        +to_dict() dict
        +save(path)
        +load(path) CompressionPlan
    }
    class ActivationProfile {
        +list~int~ bits_options
        +list rise
        +dict cost
    }
    class ActivationLayerPlan {
        +int layer
        +int act_bits
        +bool protected
    }
    class ActivationPlan {
        +tuple~ActivationLayerPlan~ layers
        +str kind
        +avg_bits() float
        +save(path)
        +load(path) ActivationPlan
    }
    class activation {
        <<module>>
        +profile_activations(model, batches, bits_options, group_size) ActivationProfile
        +plan_activations(profile, avg_bits) ActivationPlan
        +activation_plan_from_weights(weight_plan, high, low) ActivationPlan
        +uniform_activation_plan(num_layers, bits) ActivationPlan
        +predicted_rise(plan, profile) float
    }
    class handoff {
        <<module>>
        +write_handoff(path, ...) Path
    }
    class PlanCost {
        +float weight_memory_gb
        +float avg_bits_per_weight
        +float sparsity
        +float sensitivity_exposure
        +tuple per_layer_mb
        +float fixed_memory_gb
    }
    class Stage0Result {
        +CompressionPlan plan
        +SensitivityProfile profile
        +dict outputs
        +ActivationPlan activation_plan
        +KVPlan kv_plan
        +KVPlan kv_plan_bits_only
    }
    class _ModelHandle {
        +tokenizer
        +model(dtype)
    }
    class KVProfile {
        +list~int~ bits_options
        +list~int~ key_dims
        +list~int~ value_dims
        +list key_rise
        +list value_rise
        +list~float~ keep_ratios
        +list coverage
        +dict cost
        +num_layers() int
        +save(path)
    }
    class KVLayerPlan {
        +int layer
        +int key_bits
        +int value_bits
        +float keep_ratio
    }
    class KVPlan {
        +tuple~KVLayerPlan~ layers
        +str kind
        +save(path)
        +load(path) KVPlan
    }
    class KVCost {
        +float memory_gb
        +float avg_bits
        +float kept_share
        +float ppl_rise
        +float attention_kept
        +tuple per_layer_mb
    }
    class kv_cache {
        <<module>>
        +profile_kv(model, batches, bits_options, group_size, keep_ratios, names) KVProfile
        +uniform_kv_plan(num_layers, bits) KVPlan
        +plan_kv(profile, avg_bits, coverage_target) KVPlan
        +predict_kv(plan, profile, context_len, batch_size, group_size, ...) KVCost
    }
    class sensitivity {
        <<module>>
        +profile_by_ablation(model, batches, method, device, bits, group_size, meta) SensitivityProfile
        +profile_sensitivity(model, batches, device, meta) SensitivityProfile
        +normalize(scores, method) list
        +outlier_layers(raw_scores, cutoff) list
        +find_decoder_layers(model) ModuleList
        +layer_shapes(model) tuple
    }
    class planner {
        <<module>>
        +guarded_layers(removal_scores, top_k) frozenset
        +plan_compression(scores, threshold, prune_ratio, protected_bits, compressed_bits, guarded) CompressionPlan
        +uniform_plan(scores, bits, prune_ratio) CompressionPlan
        +predict_cost(plan, profile, group_size, group_overhead_bits, baseline_bits) PlanCost
        +baseline_cost(profile, baseline_bits) PlanCost
    }
    class run {
        <<module>>
        +run_stage0(ctx, candidate, model, tokenizer, text_loader, measure_fp16) Stage0Result
        +load_profile(ctx, candidate, handle, text_loader, score) tuple
        +load_guard(ctx, candidate, handle, text_loader, profile) tuple
        +load_kv_profile(ctx, candidate, handle, text_loader) tuple
        +profile_key(ctx, candidate, score) dict
        +kv_profile_key(ctx, candidate) dict
        +fp16_key(ctx, device) dict
    }
    class StageReporter {
        +method(method, variant) ComparisonRow
        +add_raw(method, variant, records)
        +finalize() dict
    }
    class ArtifactCache {
        +get_or_compute(kind, key, compute) tuple
    }

    CompressionPlan *-- LayerPlan
    Stage0Result *-- CompressionPlan
    Stage0Result *-- SensitivityProfile
    Stage0Result *-- ActivationPlan
    Stage0Result *-- KVPlan
    ActivationPlan *-- ActivationLayerPlan
    activation ..> ActivationPlan : builds
    activation ..> ActivationProfile : builds
    run ..> handoff : writes handoff.md
    activation ..> CompressionPlan : reads protected and guarded layers
    run ..> activation
    sensitivity ..> SensitivityProfile : builds
    planner ..> CompressionPlan : builds
    planner ..> PlanCost : predicts
    planner ..> SensitivityProfile : reads sizes
    KVPlan *-- KVLayerPlan
    kv_cache ..> KVProfile : builds
    kv_cache ..> KVPlan : builds
    kv_cache ..> KVCost : predicts
    run ..> kv_cache : if kv_cache
    run ..> sensitivity
    run ..> planner
    run ..> _ModelHandle : loads model only on a cache miss
    run ..> ArtifactCache : profiles and FP16 metrics
    run ..> StageReporter : 5 weight rows + 3 KV rows
    run ..> Stage0Config
    run ..> Stage0Result : returns
```

`sensitivity`, `planner`, `kv_cache` and `run` are the modules `sdf.stage0.sensitivity`, `sdf.stage0.planner`,
`sdf.stage0.kv_cache` and `sdf.stage0.run`; they hold plain functions, not classes. `profile_by_ablation` gives
the `layer_removal` (default) and `layer_quant` scores; `profile_sensitivity` gives `grad_x_weight`. `StageReporter` and `ArtifactCache` are shared by
every stage (see the [overview](README.md)).

## Sequence diagram

`run_stage0(ctx, candidate)`, step by step. The candidate supplies `sensitive_threshold`,
`prune_ratio_aggressive`, `calib_dataset`, `calib_samples` and `gptq_groupsize` (used only to predict memory).

```mermaid
sequenceDiagram
    participant R as run_stage0
    participant C as ArtifactCache
    participant D as data
    participant M as _ModelHandle
    participant S as sensitivity
    participant P as planner
    participant K as kv_cache
    participant E as eval.measure_model
    participant Rep as StageReporter

    R->>C: get_or_compute("sensitivity_profile", profile_key)
    alt cache miss
        Note over C: runs run_stage0's compute_profile()
        C->>D: load_texts(calib_dataset, "train")
        C->>D: calibration_batches(texts, tokenizer, calib_samples, seq_len, seed)
        C->>M: model(profile_dtype)
        alt score = layer_removal (default) or layer_quant
            C->>S: profile_by_ablation(model, batches, score)
            S->>S: perplexity of the unchanged model
            loop each decoder layer
                S->>S: skip the layer (or round only its weights), perplexity, restore
            end
            Note over S: raw score = perplexity with the change minus without
        else score = grad_x_weight
            C->>S: profile_sensitivity(model, batches)
            loop each calibration batch
                S->>S: forward + backward, add sum of abs(grad x weight) per layer
            end
        end
        S-->>C: SensitivityProfile (raw scores, layer sizes, cost)
    end
    C-->>R: profile, was_cached
    R->>S: normalize(raw_scores, "rank")
    S-->>R: scores in 0..1, one per layer
    R->>Rep: new StageReporter(stage 0, config, environment, conditions)

    Note over R,Rep: row 1, baseline / fp16
    R->>P: baseline_cost(profile)
    opt measure_fp16
        R->>C: get_or_compute("fp16_baseline", fp16_key)
        alt cache miss
            C->>D: load_texts("wikitext2", "test"), eval_windows -> validation, held-out
            C->>E: measure_model(FP16 model, validation, held-out)
            E-->>C: ppl both halves, size, peak memory, latency + raw repeats
        end
        C-->>R: FP16 metrics
    end

    Note over R,Rep: row 2, allocation / original
    R->>P: uniform_plan(scores, 4 bits, prune 0)
    R->>P: predict_cost(uniform)

    Note over R,Rep: row 3, allocation / framework
    R->>P: plan_compression(scores, threshold, prune_ratio, 8, 4)
    loop each layer
        P->>P: score >= threshold ? protect at 8-bit : compress at 4-bit + prune
    end
    R->>P: predict_cost(plan)

    opt kv_cache (default on)
        Note over R,Rep: rows 4 to 6, KV cache
        R->>C: get_or_compute("kv_profile", kv_profile_key)
        alt cache miss
            C->>K: profile_kv(model, kv_calib_samples batches)
            loop each layer, keys then values, each bits option
                K->>K: round only that tensor, perplexity rise
            end
            K->>K: eager attention: share of attention on the top keep-ratio tokens
            K-->>C: KVProfile
        end
        R->>K: uniform_kv_plan(4 bits), predict_kv -> kv_cache / original
        R->>K: plan_kv(profile, avg bits, 95% coverage), predict_kv -> kv_cache / framework
        R->>K: plan_kv(profile, avg bits, no eviction), predict_kv -> kv_cache_bits_only / framework
        Note over K: bits: greedy, spend the uniform average where the rise is largest<br/>budget: fewest tokens keeping 95% of attention
    end

    Note over R,Rep: each row runs inside rep.method(...): an error marks<br/>that row failed and the rest continue, results.json is saved after each row
    R->>R: _add_stage0_details: per-layer table, outlier layers, plain-language text
    R->>R: save compression_plan*.json, sensitivity_profile.json, kv_profile.json, kv_cache_plan*.json
    R->>Rep: finalize()
    Rep->>Rep: deltas vs FP16 and vs original, requirement check
    Rep-->>R: report.md, stage_0_comparison.xlsx, results.json
```
