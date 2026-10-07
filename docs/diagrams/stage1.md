# Stage 1: quantization

Drawn from the code (`src/sdf/stages/`). Weights are implemented (RTN, GPTQ, AWQ); of the activation methods only
the RTN baseline is. Back to the [overview](README.md).

## In plain words

A model is billions of stored numbers ("weights") and, while it runs, computes new numbers at every step
("activations"). Stage 1 keeps both with fewer bits, like rounding prices to the nearest dollar, and **removes
nothing**. The standard methods treat every layer the same. The framework follows the Stage 0 plan: protected
layers keep more bits, the rest fewer. A plan handed to Stage 1 carries bits only; a plan that prunes is refused.

Path: Stage 0 > Stage 1 > Stage 4.

## Class diagram

```mermaid
classDiagram
    direction LR

    class Method {
        +str name
        +int stage
        +tuple plans
        +apply(MethodCall) ContextManager
    }
    class gptq_ {
        <<function>>
    }
    class awq_ {
        <<function>>
    }
    class rtn_weights {
        <<function>>
    }
    class rtn_activations {
        <<function>>
    }
    class require_bits_only {
        <<function>>
    }
    class run_stage {
        <<function>>
        +run_stage(ctx, 1, names, candidate, stage0_dir)
    }
    class Stage0Plans {
        +dict plans
        +load(dir)
    }
    class CompressionPlan {
        +LayerPlan[] layers
    }
    class ActivationPlan

    Method ..> gptq_ : gptq
    Method ..> awq_ : awq
    Method ..> rtn_weights : rtn
    Method ..> rtn_activations : rtn_act
    gptq_ ..> require_bits_only
    awq_ ..> require_bits_only
    run_stage ..> Method : each method
    run_stage ..> Stage0Plans : quant_plan.json, quant_plan_budget_matched.json, activation_plan.json
    Stage0Plans ..> CompressionPlan : bits only
    Stage0Plans ..> ActivationPlan
```

## Sequence diagram

```mermaid
sequenceDiagram
    participant R as run_stage (stage 1)
    participant C as ArtifactCache
    participant M as Method.apply
    participant E as measure_model
    participant Rep as StageReporter

    Note over R,Rep: FP16 row reused from the cache (same key as Stage 0)
    loop each method: rtn, gptq, awq, rtn_act
        R->>C: original row (uniform bits, nothing removed)
        alt cache miss
            C->>M: apply(fresh FP16 model, uniform plan)
            M-->>C: quantized model
            C->>E: measure_model
        end
        R->>Rep: row (method, original)
        loop each Stage 0 plan: quant, quant_same_size
            R->>M: apply(fresh FP16 model, plan)
            M->>M: round every layer to its planned bits (GPTQ: Hessian error feedback, AWQ: channel scaling)
            R->>E: measure_model
            R->>Rep: row (method, framework)
        end
    end
    R->>Rep: finalize()
```
