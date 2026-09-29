# Stage 0: sensitivity profiling and compression planning

Drawn from the code in `src/sdf/stage0/`. Back to the [overview](README.md).

## In plain words

Stage 0 finds out which layers of the model are fragile. It feeds the model some ordinary text and, for every
layer, estimates how much the model's predictions would suffer if that layer's numbers were changed. That
estimate is the layer's **sensitivity score**. Layers scoring at or above a threshold are **protected** (kept at
8 bits, nothing removed); the rest are **compressed** (4 bits, and some of their numbers removed). The result is
a **compression plan**: one line per layer saying how many bits it gets and how much of it is removed.

Stage 0 does not compress anything yet. It predicts how much memory each plan would need and compares:

- the **uncompressed** model (FP16),
- the **standard method**: every layer treated the same (uniform 4 bits),
- the **framework**: the sensitivity plan, plus two versions squeezed into exactly the standard method's memory
  (one that removes numbers from robust layers, one that only lowers their bits), so the comparison is fair.

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
        +int no_prune_compressed_bits
        +int uniform_bits
        +float uniform_prune_ratio
        +int group_overhead_bits
        +int baseline_bits
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
    }
    class CompressionPlan {
        +tuple~LayerPlan~ layers
        +str kind
        +float sensitive_threshold
        +float prune_ratio_aggressive
        +protected_layers() list
        +compressed_layers() list
        +to_dict() dict
        +save(path)
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
    }
    class _ModelHandle {
        +tokenizer
        +model(dtype)
    }
    class sensitivity {
        <<module>>
        +profile_sensitivity(model, batches, device, meta) SensitivityProfile
        +normalize(scores, method) list
        +outlier_layers(raw_scores, cutoff) list
        +find_decoder_layers(model) ModuleList
        +layer_shapes(model) tuple
    }
    class planner {
        <<module>>
        +plan_compression(scores, threshold, prune_ratio, protected_bits, compressed_bits) CompressionPlan
        +uniform_plan(scores, bits, prune_ratio) CompressionPlan
        +budget_matched_plan(scores, budget_gb, prune_ratio, protected_bits, compressed_bits, cost) CompressionPlan
        +predict_cost(plan, profile, group_size, group_overhead_bits, baseline_bits) PlanCost
        +baseline_cost(profile, baseline_bits) PlanCost
    }
    class run {
        <<module>>
        +run_stage0(ctx, candidate, model, tokenizer, text_loader, measure_fp16) Stage0Result
        +profile_key(ctx, candidate) dict
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
    sensitivity ..> SensitivityProfile : builds
    planner ..> CompressionPlan : builds
    planner ..> PlanCost : predicts
    planner ..> SensitivityProfile : reads sizes
    run ..> sensitivity
    run ..> planner
    run ..> _ModelHandle : loads model only on a cache miss
    run ..> ArtifactCache : profile and FP16 metrics
    run ..> StageReporter : 5 comparison rows
    run ..> Stage0Config
    run ..> Stage0Result : returns
```

`sensitivity`, `planner` and `run` are the modules `sdf.stage0.sensitivity`, `sdf.stage0.planner` and
`sdf.stage0.run`; they hold plain functions, not classes. `StageReporter` and `ArtifactCache` are shared by
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
    participant E as eval.measure_model
    participant Rep as StageReporter

    R->>C: get_or_compute("sensitivity_profile", profile_key)
    alt cache miss
        Note over C: runs run_stage0's compute_profile()
        C->>D: load_texts(calib_dataset, "train")
        C->>D: calibration_batches(texts, tokenizer, calib_samples, seq_len, seed)
        C->>M: model(profile_dtype)
        C->>S: profile_sensitivity(model, batches)
        loop each calibration batch
            S->>S: forward + backward, add sum of abs(grad x weight) per layer
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

    Note over R,Rep: rows 4 and 5, same size as the original
    R->>P: budget_matched_plan(scores, uniform size, prune_ratio, 8, 4)
    R->>P: budget_matched_plan(scores, uniform size, 0, 8, 3)
    loop k = 0, 1, 2 ... layers
        P->>P: protect the k most sensitive, stop when over budget
    end

    Note over R,Rep: each row runs inside rep.method(...): an error marks<br/>that row failed and the rest continue, results.json is saved after each row
    R->>R: _add_stage0_details: per-layer table, outlier layers, plain-language text
    R->>R: save compression_plan*.json, sensitivity_profile.json
    R->>Rep: finalize()
    Rep->>Rep: deltas vs FP16 and vs original, requirement check
    Rep-->>R: report.md, stage_0_comparison.xlsx, results.json
```
