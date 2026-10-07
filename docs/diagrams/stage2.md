# Stage 2: pruning

Drawn from the code (`src/sdf/stages/pruning.py`, `methods.py`). Back to the [overview](README.md).

## In plain words

Stage 2 makes the model smaller by **removing** numbers that barely matter and rounds nothing: single weights
(**Wanda**), whole feed-forward channels (**structured pruning**), or the part of a weight table a smaller
**low-rank** copy can drop. The standard way removes the same share from every layer. The framework removes more
from the layers Stage 0 found robust and nothing from the fragile or never-pruned ones.

Two paths:

- Stage 0 > Stage 2 > Stage 4: prunes the uncompressed (FP16) model.
- Stage 0 > Stage 1 > Stage 2 > Stage 4: a Stage 1 method quantizes first, then the Stage 2 method prunes the
  quantized model (rows named `<pruning>_after_<quantization>`).

## Class diagram

```mermaid
classDiagram
    direction LR

    class Method {
        +str name
        +tuple plans
        +tuple quant_plans
        +apply(MethodCall)
    }
    class wanda_ {
        <<function>>
    }
    class structured_prune_ {
        <<function>>
    }
    class low_rank_ {
        <<function>>
    }
    class series {
        <<function>>
        +series(quant, prune) Method
    }
    class combine_plans {
        <<function>>
    }
    class run_stage {
        <<function>>
        +run_stage(ctx, 2, names, ...)
    }
    class Stage0Plans {
        +dict plans
    }

    Method ..> wanda_ : unstructured_prune
    Method ..> structured_prune_ : structured_prune
    Method ..> low_rank_ : low_rank
    series ..> Method : quant then prune
    run_stage ..> series : name contains _after_
    run_stage ..> combine_plans : quant_plan + prune_plan
    run_stage ..> Stage0Plans : prune_plan.json, prune_plan_same_size.json
```

## Sequence diagram

```mermaid
sequenceDiagram
    participant R as run_stage (stage 2)
    participant Q as Stage 1 method (series only)
    participant P as Stage 2 method
    participant E as measure_model
    participant Rep as StageReporter

    loop each method (alone, then after each stage2_after method)
        R->>P: original: every layer pruned at prune_ratio_aggressive
        opt series
            R->>Q: quantize fresh FP16 model with the bits of the plan
        end
        R->>P: prune with the plan's ratios (low-rank: factors at the plan's bits)
        R->>E: measure_model
        R->>Rep: row (method, original)
        loop each plan: prune, prune_same_size
            R->>P: same, with the Stage 0 plan
            R->>Rep: row (method, framework)
        end
    end
```
