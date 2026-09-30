# Stage 1: weight compression (planned)

Not written yet. Drawn from the design (v2_1 diagram, "Stage 1" page) and the thesis spec; names are proposals.
Back to the [overview](README.md).

## In plain words

A model is billions of stored numbers ("weights"). Stage 1 makes that store smaller in two ways: keeping each
number with fewer bits (**quantisation**, like rounding prices to the nearest dollar), and throwing away numbers
that barely matter (**pruning**). The standard methods treat every layer the same. The framework follows the
Stage 0 plan instead: protected layers keep 8 bits and are not pruned, the rest get 4 bits and are pruned.

For each method (GPTQ, AWQ, structured, unstructured and low-rank pruning) Stage 1 builds the standard version
and the framework version, really compresses the model, and measures accuracy, size, memory and speed against
the uncompressed model.

## Class diagram

```mermaid
classDiagram
    direction LR

    class WeightMethod {
        <<planned, Protocol>>
        +str name
        +str kind
        +apply(model, layer_plans, candidate, batches) Module
    }
    class GPTQ {
        <<planned>>
        +int groupsize
        +apply(...)
    }
    class AWQ {
        <<planned>>
        +int groupsize
        +apply(...)
    }
    class StructuredPrune {
        <<planned>>
        +apply(...)
    }
    class UnstructuredPrune {
        <<planned>>
        +apply(...)
    }
    class LowRank {
        <<planned>>
        +apply(...)
    }
    class Stage1Config {
        <<planned>>
        +list methods
        +dict original_defaults
    }
    class stage1_run {
        <<planned module>>
        +run_stage1(ctx, candidate, stage0) StageResult
    }
    class StageResult {
        <<planned>>
        +dict artifacts
        +dict outputs
    }
    class Stage0Result {
        +CompressionPlan plan
        +SensitivityProfile profile
    }
    class LayerPlan {
        +int layer
        +int bit_width
        +float pruning_ratio
        +bool protected
    }
    class StageReporter
    class ArtifactCache
    class measure_model {
        <<function>>
    }

    WeightMethod <|.. GPTQ
    WeightMethod <|.. AWQ
    WeightMethod <|.. StructuredPrune
    WeightMethod <|.. UnstructuredPrune
    WeightMethod <|.. LowRank
    stage1_run ..> WeightMethod : each method in Stage1Config.methods
    stage1_run ..> Stage0Result : reads plan
    WeightMethod ..> LayerPlan : bits and prune ratio per layer
    stage1_run ..> measure_model : ppl, size, memory, latency
    stage1_run ..> ArtifactCache : original-method results
    stage1_run ..> StageReporter : fp16 / original / framework rows
    stage1_run ..> StageResult : returns
    stage1_run ..> Stage1Config
```

`layer_plans` is `None` for the original method (uniform method defaults) and the Stage 0 plan's layers for the
framework. `gptq_groupsize` comes from the search space. `StageResult` is meant to be shared by Stages 1 to 3.

## Sequence diagram

```mermaid
sequenceDiagram
    participant R as run_stage1 (planned)
    participant C as ArtifactCache
    participant W as WeightMethod (planned)
    participant E as measure_model
    participant Rep as StageReporter

    Note over R,Rep: FP16 row reused from the cache (same key as Stage 0)
    loop each method: GPTQ, AWQ, structured, unstructured, low-rank
        R->>C: get_or_compute("stage1_original", method + defaults + calibration + seed)
        alt cache miss
            C->>W: apply(FP16 model, None, defaults, batches)
            W-->>C: compressed model
            C->>E: measure_model(model, validation, held-out)
            E-->>C: metrics
        end
        C-->>R: original-method metrics
        R->>Rep: row (method, original)

        R->>W: apply(FP16 model, plan.layers, candidate, batches)
        loop each layer plan
            alt protected
                W->>W: keep 8-bit, no pruning
            else compressed
                W->>W: prune to the planned ratio
                W->>W: quantise to 4-bit in groups of gptq_groupsize
            end
            opt GPTQ
                W->>W: Hessian of the layer inputs, round column by column, push the error to later columns
            end
        end
        W-->>R: compressed model + per-layer error, achieved sparsity
        R->>E: measure_model(model, validation, held-out)
        E-->>R: metrics
        R->>Rep: row (method, framework), per-layer table
    end
    R->>Rep: finalize()
    Rep-->>R: stage_1/report.md, stage_1_comparison.xlsx, results.json
```
