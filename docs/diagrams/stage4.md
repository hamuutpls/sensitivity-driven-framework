# Stage 4: evaluation across backends (planned)

Not written yet. Drawn from the design (v2_1 diagram, "Stage 4" page) and the thesis spec; names are proposals.
Back to the [overview](README.md).

Part of it exists: `eval.metrics.measure_model` already measures perplexity on both halves, model size, peak
memory and prefill / decode latency on HF Transformers, and every stage uses it.

## In plain words

A compressed model is only useful if it runs well on the software people actually use to serve models. Stage 4
loads each compressed model on four such programs (**HF Transformers**, **llama.cpp**, **vLLM**,
**TensorRT-LLM**) and measures, the same way every time: how accurate it is (prediction error on test text, plus
multiple-choice tests of common sense and reasoning), how much memory it needs, and how fast it reads a prompt
and writes a reply.

The test text is split in two. The search may look at the first half to pick settings; the second half is
never used for tuning and is the honest check. At the end of a search, Stage 4 also picks the final model: the
most accurate one that meets every deployment target, or the closest one with a note of how far it falls short.

## Class diagram

```mermaid
classDiagram
    direction LR

    class Backend {
        <<planned, Protocol>>
        +str name
        +export(model, out_dir) Path
        +load(path) Handle
        +perplexity(handle, windows) float
        +time_prefill_decode(handle, prompt_len, decode_tokens) dict
    }
    class HFBackend {
        <<partly exists: measure_model>>
    }
    class LlamaCppBackend {
        <<planned>>
    }
    class VLLMBackend {
        <<planned>>
    }
    class TensorRTLLMBackend {
        <<planned>>
    }
    class stage4_run {
        <<planned module>>
        +run_stage4(ctx, candidate, artifacts) StageResult
        +downstream_tasks(handle, tasks) dict
        +select_deployable(front, requirement) Trial
    }
    class EvalConfig {
        +str dataset
        +int seq_len
        +int max_windows
        +int latency_prompt_len
        +int latency_decode_tokens
        +int latency_warmup
        +int latency_repeats
    }
    class DeploymentRequirement {
        +check(metrics) RequirementCheck
    }
    class StageReporter
    class ArtifactCache

    Backend <|.. HFBackend
    Backend <|.. LlamaCppBackend
    Backend <|.. VLLMBackend
    Backend <|.. TensorRTLLMBackend
    stage4_run ..> Backend : every artifact on every backend
    stage4_run ..> EvalConfig : identical settings for every row
    stage4_run ..> DeploymentRequirement : met or shortfall
    stage4_run ..> ArtifactCache : FP16 and original results per backend
    stage4_run ..> StageReporter : fp16 / original / framework rows per backend
```

## Sequence diagram: evaluating one trial

```mermaid
sequenceDiagram
    participant R as run_stage4 (planned)
    participant B as Backend (planned)
    participant Rep as StageReporter

    loop each artifact from Stages 1 to 3 (fp16, original, framework)
        loop each backend: HF, llama.cpp, vLLM, TensorRT-LLM
            R->>B: export(model), load(path)
            B-->>R: handle, disk size, build time
            R->>B: perplexity(validation half)
            B-->>R: ppl_val (the search optimises this)
            R->>B: perplexity(held-out half)
            B-->>R: ppl_heldout (reported only)
            R->>B: downstream tasks: PIQA, WinoGrande, ARC
            R->>B: time_prefill_decode, warmup then repeats
            B-->>R: prefill / decode ms and tok/s (mean, std), peak memory, KV memory
            R->>Rep: row (method, variant, backend)
        end
    end
    R->>Rep: finalize()
    Rep-->>R: stage_4 report, objectives for the search
```

## Sequence diagram: choosing the deployable model after the search

```mermaid
sequenceDiagram
    participant R as select_deployable (planned)
    participant Q as DeploymentRequirement

    loop each config on the final Pareto front
        R->>Q: check(metrics)
        alt meets latency, memory, accuracy and KV targets
            Q-->>R: met
            R->>R: add to the shortlist
        else
            Q-->>R: shortfall per target
            R->>R: record the shortfall
        end
    end
    alt shortlist not empty
        R->>R: highest accuracy, ties broken by lowest memory
    else
        R->>R: smallest shortfall, reported with its gaps
    end
```
