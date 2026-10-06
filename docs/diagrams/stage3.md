# Stage 3: KV-cache compression (planned)

Not written yet. Drawn from the design (v2_1 diagram, "Stage 3" page) and the thesis spec; names are proposals.
Back to the [overview](README.md).

## In plain words

While a model writes a reply, it keeps a short-term memory of everything said so far, called the **KV cache**.
For long conversations this memory can grow bigger than the model itself. Stage 3 shrinks it in two ways:
storing it with fewer bits (**QuaRot-KV**, **KVQuant**; the bit count is `quarot_k_bits`), or forgetting the
words the model pays least attention to (**H2O**, **SnapKV**, **InfiniGen**). It then checks whether the memory
fits the budget set in the deployment requirement.

The framework version follows Stage 0's KV cache plan: each layer's own key bits, value bits and token budget
(see [stage0.md](stage0.md)). Like Stage 2, it
starts from the uncompressed model and the Stage 0 plan.

## Class diagram

```mermaid
classDiagram
    direction LR

    class KVMethod {
        <<planned, Protocol>>
        +str name
        +str kind
        +wrap(model, layer_plans, candidate) Module
    }
    class QuaRotKV {
        <<planned>>
        +int k_bits
    }
    class KVQuant {
        <<planned>>
        +int bits
    }
    class H2O {
        <<planned>>
        +int keep_tokens
    }
    class SnapKV {
        <<planned>>
        +int keep_tokens
    }
    class InfiniGen {
        <<planned>>
    }
    class Stage3Config {
        <<planned>>
        +list methods
        +int context_len
    }
    class stage3_run {
        <<planned module>>
        +run_stage3(ctx, candidate, stage0) StageResult
        +kv_cache_gb(model, context_len) float
    }
    class DeploymentRequirement {
        +float kv_budget_gb
        +check(metrics) RequirementCheck
    }
    class Stage0Result
    class StageReporter
    class measure_model {
        <<function>>
    }

    KVMethod <|.. QuaRotKV : quantise
    KVMethod <|.. KVQuant : quantise
    KVMethod <|.. H2O : evict
    KVMethod <|.. SnapKV : evict
    KVMethod <|.. InfiniGen : evict
    stage3_run ..> KVMethod
    stage3_run ..> Stage0Result : reads the KV cache plan (kv_cache_plan.json)
    stage3_run ..> measure_model
    stage3_run ..> StageReporter : fp16 / original / framework rows
    StageReporter ..> DeploymentRequirement : kv_cache_gb vs kv_budget_gb
    stage3_run ..> Stage3Config
```

`quarot_k_bits` joins `SEARCH_SPACE` when this stage is written. `kv_cache_gb` is already in
`DeploymentRequirement`'s checks; its metric spec joins the registry when this stage first emits it.

## Sequence diagram

```mermaid
sequenceDiagram
    participant R as run_stage3 (planned)
    participant K as KVMethod (planned)
    participant M as model (generating)
    participant E as measure_model
    participant Rep as StageReporter

    loop each method: QuaRot-KV, KVQuant, H2O, SnapKV, InfiniGen
        Note over R,Rep: original row: method defaults, same cache bits in every layer
        R->>K: wrap(FP16 model, KV plan layers, candidate)
        K-->>R: model with a compressed cache
        R->>E: measure_model(model, validation, held-out)
        loop each generated token
            M->>M: compute its keys and values
            alt quantise (QuaRot-KV, KVQuant)
                M->>K: rotate to spread outliers, quantise to the layer's cache bits
            else evict (H2O, SnapKV, InfiniGen)
                M->>K: score tokens by attention received, keep recent and most-used
            end
            K-->>M: updated KV cache
        end
        E-->>R: metrics incl. latency
        R->>R: kv_cache_gb at context_len, tokens kept vs evicted
        R->>Rep: row (method, framework)
        Rep->>Rep: requirement check: fits the KV budget or records the overflow
    end
    R->>Rep: finalize()
```
