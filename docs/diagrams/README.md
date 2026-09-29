# Diagrams

Class and sequence diagrams for the whole framework and for each stage. GitHub draws them from the Mermaid
text in these files.

| File | Covers | Drawn from |
|---|---|---|
| this file | the project as a whole | the code (Stage 0 and shared parts) + the design (the rest) |
| [stage0.md](stage0.md) | Stage 0: sensitivity profiling and planning | the code |
| [stage1.md](stage1.md) | Stage 1: weight compression | the design, **planned** |
| [stage2.md](stage2.md) | Stage 2: activation compression | the design, **planned** |
| [stage3.md](stage3.md) | Stage 3: KV-cache compression | the design, **planned** |
| [stage4.md](stage4.md) | Stage 4: evaluation across backends | the design, **planned** |
| [search.md](search.md) | the search layer (MOBO, MFBO, NSGA-III) | the design, **planned** |

Anything not written yet carries a `<<planned>>` label. Planned parts follow the flow of the v2_1 design diagram and the
thesis spec, and reuse the shared pieces that already exist (`FrameworkConfig`, `StageReporter`,
`ArtifactCache`, `measure_model`), so each new stage only adds its own methods. Names of planned classes and
functions are proposals and may change when the stage is written.

## How to read these diagrams

- A **class diagram** is a map of the parts of the program and how they connect. Each box is one kind of thing
  the program keeps track of (for example "a compression plan"), with the pieces of information it holds and the
  actions it can do. A line with a diamond means "is made of"; a dashed arrow means "uses".
- A **sequence diagram** is a timeline. Each column is a part of the program; arrows going down the page are
  the steps, in order, as one part asks another to do something and gets an answer back. `loop` boxes repeat,
  `alt` boxes show a choice, `opt` boxes are skipped when they don't apply.

## The project in plain words

The framework makes a language model smaller and faster while keeping its answers good. First it measures
which layers of the model are fragile (**Stage 0**). It then uses that measurement to decide how hard to
compress each layer, and applies three separate kinds of compression: to the model's stored numbers
(**Stage 1**), to the numbers it computes while running (**Stage 2**), and to its short-term memory of the
conversation (**Stage 3**). **Stage 4** measures the result on real software that runs models. Around all of
this, a **search** tries many settings and keeps the ones that give the best trade-off between accuracy, memory
and speed.

Every stage is judged the same way: the uncompressed model, the standard compression method on its own, and
the framework's sensitivity-guided version, all under identical conditions.

## Class diagram: the project as a whole

```mermaid
classDiagram
    direction LR

    class FrameworkConfig {
        +RunConfig run
        +ModelConfig model
        +DataConfig data
        +CalibrationConfig calibration
        +EvalConfig eval
        +Stage0Config stage0
        +DeploymentRequirement requirement
        +dict hyperparams
        +to_dict() dict
        +from_dict(d) FrameworkConfig
        +with_overrides(overrides) FrameworkConfig
    }
    class RunConfig {
        +str output_root
        +str run_id
        +str cache_dir
        +int seed
        +bool deterministic
    }
    class DeploymentRequirement {
        +float target_latency_ms
        +float target_memory_gb
        +float target_ppl
        +float kv_budget_gb
        +str hardware_profile
        +check(metrics) RequirementCheck
    }
    class RequirementCheck {
        +bool met
        +dict shortfall
        +dict checked
        +list unmeasured
    }
    class SearchSpace {
        +tuple params
        +make(overrides) dict
        +validate(values) dict
    }
    class Param {
        +str name
        +int stage
        +default
        +float low
        +float high
        +tuple choices
        +contains(value) bool
    }
    class RunContext {
        +FrameworkConfig cfg
        +str run_id
        +Path run_dir
        +ArtifactCache cache
    }
    class ArtifactCache {
        +Path root
        +path(kind, key) Path
        +get_or_compute(kind, key, compute) tuple
    }
    class StageReporter {
        +int stage
        +list rows
        +list per_layer
        +list raw
        +method(method, variant) ComparisonRow
        +add_raw(method, variant, records)
        +compute_deltas()
        +flush()
        +finalize() dict
    }
    class ComparisonRow {
        +str method
        +str variant
        +str status
        +dict metrics
        +dict deltas
        +dict requirement
        +str error
    }
    class Stage0Result {
        +CompressionPlan plan
        +SensitivityProfile profile
        +dict outputs
    }
    class CompressionPlan {
        +tuple layers
        +str kind
    }

    class Stage1Runner {
        <<planned>>
        +run_stage1(ctx, candidate, stage0) StageResult
    }
    class Stage2Runner {
        <<planned>>
        +run_stage2(ctx, candidate, stage0) StageResult
    }
    class Stage3Runner {
        <<planned>>
        +run_stage3(ctx, candidate, stage0) StageResult
    }
    class Stage4Evaluator {
        <<planned>>
        +evaluate(artifact, backend) dict
    }
    class Searcher {
        <<planned>>
        +ask() dict
        +tell(candidate, objectives)
    }
    class SearchRun {
        <<planned>>
        +run(searchers, budget) list~Trial~
        +pareto_front() list~Trial~
    }
    class MasterReport {
        <<planned>>
        +write(run_dir) all_stages_comparison.xlsx
    }

    FrameworkConfig *-- RunConfig
    FrameworkConfig *-- DeploymentRequirement
    DeploymentRequirement ..> RequirementCheck : returns
    SearchSpace *-- Param
    RunContext o-- FrameworkConfig
    RunContext *-- ArtifactCache
    StageReporter *-- ComparisonRow
    StageReporter ..> DeploymentRequirement : checks each row
    Stage0Result *-- CompressionPlan

    Stage1Runner ..> Stage0Result : reads plan
    Stage2Runner ..> Stage0Result : reads plan
    Stage3Runner ..> Stage0Result : reads plan
    Stage1Runner ..> StageReporter
    Stage2Runner ..> StageReporter
    Stage3Runner ..> StageReporter
    Stage4Evaluator ..> StageReporter
    SearchRun o-- Searcher
    SearchRun ..> SearchSpace : candidates
    SearchRun ..> Stage4Evaluator : objectives
    MasterReport ..> StageReporter : reads results.json
```

`FrameworkConfig` also holds `ModelConfig`, `DataConfig`, `CalibrationConfig`, `EvalConfig` and `Stage0Config`
(left out above for space; see [stage0.md](stage0.md)). Stages 1 to 4 will add their own `StageNConfig` block
to it, so there is still one config object.

## Sequence diagram: a Stage 0 run today

What happens when you type `sdf-stage0 --config configs/tinyllama.yaml`. Stage 0's inside is expanded in
[stage0.md](stage0.md).

```mermaid
sequenceDiagram
    actor User
    participant CLI as cli.stage0_main
    participant Cfg as config.load_config
    participant Space as SEARCH_SPACE
    participant Run as run.start_run
    participant S0 as stage0.run_stage0
    participant Rep as StageReporter

    User->>CLI: sdf-stage0 --config ... --set key=value
    CLI->>Cfg: load_config(path, overrides)
    Cfg-->>CLI: FrameworkConfig
    CLI->>Space: make(cfg.hyperparams)
    Space-->>CLI: candidate (validated dict)
    CLI->>Run: start_run(cfg)
    Note over Run: make results/run_id/, write config.json,<br/>start run.log, fix seeds, open cache
    Run-->>CLI: RunContext
    CLI->>S0: run_stage0(ctx, candidate)
    S0->>Rep: rows fp16 / original / framework
    Rep-->>S0: report.md, stage_0_comparison.xlsx, results.json
    S0-->>CLI: Stage0Result
    CLI-->>User: prints output paths
```

## Sequence diagram: the full run (planned)

The whole pipeline once the search exists. Each searcher runs its own loop over the same search space and trial
budget; their results are compared, not pooled into one search.

```mermaid
sequenceDiagram
    actor User
    participant SR as SearchRun (planned)
    participant Se as Searcher (planned)
    participant S0 as Stage 0
    participant S1 as Stage 1 (planned)
    participant S2 as Stage 2 (planned)
    participant S3 as Stage 3 (planned)
    participant S4 as Stage 4 (planned)
    participant MR as MasterReport (planned)

    User->>SR: start run (config, trial budget)
    loop each searcher: MOBO, MFBO, NSGA-III
        loop until the trial budget is used up
            SR->>Se: ask()
            Se-->>SR: candidate
            SR->>S0: run_stage0(candidate)
            S0-->>SR: compression plan (cached profile reused)
            par each stage starts from the same plan
                SR->>S1: run_stage1(candidate, plan)
                S1-->>SR: compressed weights
            and
                SR->>S2: run_stage2(candidate, plan)
                S2-->>SR: activation-quantised model
            and
                SR->>S3: run_stage3(candidate, plan)
                S3-->>SR: compressed KV cache
            end
            SR->>S4: evaluate(candidate artifacts)
            S4-->>SR: val ppl, memory, latency, build cost (+ held-out ppl, recorded only)
            SR->>SR: log trial, record requirement shortfall
            SR->>Se: tell(candidate, objectives)
        end
    end
    SR->>SR: Pareto front per searcher, hypervolume, searcher benchmark
    Note over SR: search_report.md, search_results.xlsx, Pareto plots
    loop each final Pareto candidate and best-per-objective config
        SR->>S0: full Stage 0 comparison
        SR->>S1: full Stage 1 comparison
        SR->>S2: full Stage 2 comparison
        SR->>S3: full Stage 3 comparison
        SR->>S4: full Stage 4 comparison
    end
    SR->>MR: write(run_dir)
    MR-->>User: all_stages_comparison.xlsx + master report
```
