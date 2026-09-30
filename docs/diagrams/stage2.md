# Stage 2: activation compression (planned)

Not written yet. Drawn from the design (v2_1 diagram, "Stage 2" page) and the thesis spec; names are proposals.
Back to the [overview](README.md).

## In plain words

Besides its stored numbers, a model computes new numbers at every step while it runs ("activations"). Storing
those with fewer bits too makes the model faster, but a few of them are huge **outliers** that get ruined by
rounding. The methods here tame the outliers first: **SmoothQuant** moves part of the difficulty into the stored
weights (how much is set by the "migration strength", `smoothquant_alpha`), while **QuaRot**, **RPTQ** and
**SpinQuant** mix the numbers so the outliers are spread evenly. None of this changes the model's output before
rounding.

The framework version keeps activations at higher precision in the layers Stage 0 protected. Stage 2 starts
from the uncompressed model and the Stage 0 plan, not from Stage 1's output, so each stage's effect is measured
on its own.

## Class diagram

```mermaid
classDiagram
    direction LR

    class ActivationMethod {
        <<planned, Protocol>>
        +str name
        +apply(model, layer_plans, candidate, batches) Module
    }
    class SmoothQuant {
        <<planned>>
        +float alpha
        +apply(...)
    }
    class QuaRot {
        <<planned>>
        +apply(...)
    }
    class RPTQ {
        <<planned>>
        +apply(...)
    }
    class SpinQuant {
        <<planned>>
        +apply(...)
    }
    class ChannelStats {
        <<planned>>
        +list max_abs_per_channel
        +list outlier_channels
        +collect(model, batches) ChannelStats
    }
    class Stage2Config {
        <<planned>>
        +list methods
        +int protected_act_bits
        +int compressed_act_bits
    }
    class stage2_run {
        <<planned module>>
        +run_stage2(ctx, candidate, stage0) StageResult
    }
    class Stage0Result
    class StageReporter
    class ArtifactCache
    class measure_model {
        <<function>>
    }

    ActivationMethod <|.. SmoothQuant
    ActivationMethod <|.. QuaRot
    ActivationMethod <|.. RPTQ
    ActivationMethod <|.. SpinQuant
    SmoothQuant ..> ChannelStats : per-channel scales
    RPTQ ..> ChannelStats : reorders channels
    stage2_run ..> ActivationMethod
    stage2_run ..> Stage0Result : reads plan
    stage2_run ..> measure_model
    stage2_run ..> ArtifactCache : original-method results
    stage2_run ..> StageReporter : fp16 / original / framework rows
    stage2_run ..> Stage2Config
```

`smoothquant_alpha` joins `SEARCH_SPACE` when this stage is written.

## Sequence diagram

```mermaid
sequenceDiagram
    participant R as run_stage2 (planned)
    participant A as ActivationMethod (planned)
    participant E as measure_model
    participant Rep as StageReporter

    loop each method: SmoothQuant, QuaRot, RPTQ, SpinQuant
        Note over R,Rep: original row: method defaults, same bits everywhere, cached like Stage 1
        R->>A: apply(FP16 model, plan.layers, candidate, batches)
        loop each calibration batch
            A->>A: record each channel's largest value
        end
        A->>A: mark the outlier channels
        alt smooth (SmoothQuant)
            loop each linear layer
                A->>A: scale per channel with smoothquant_alpha, move outliers into the weights
            end
        else rotate (QuaRot, SpinQuant)
            A->>A: build rotation (Hadamard or learned)
            loop each linear layer
                A->>A: fold norm scale into weights, rotate to spread outliers
            end
        end
        loop each linear layer
            alt protected in the plan
                A->>A: keep higher-precision activations
            else
                A->>A: quantise activations to the planned bits
            end
        end
        A-->>R: activation-quantised model + outlier ratio, activation error
        R->>E: measure_model(model, validation, held-out)
        E-->>R: metrics incl. added latency per token
        R->>Rep: row (method, framework)
    end
    R->>Rep: finalize()
```
