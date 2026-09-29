# Search layer (planned)

Not written yet, except the search space itself (`src/sdf/search_space.py`). Drawn from the design (v2_1
diagram, "Search Loop" page) and the thesis spec; other names are proposals. Back to the [overview](README.md).

## In plain words

The framework has a handful of settings (how strict the "fragile layer" threshold is, how much to prune, how
many bits to keep, which calibration text to use, and so on). Each combination gives a different balance of
accuracy, memory, speed and build time, and no single one is best on all four. The search tries many
combinations and keeps the **Pareto front**: the ones that nothing else beats on every goal at once.

Three search strategies are compared, each with the same settings to choose from and the same number of tries:

- **MOBO** (Optuna) learns from past tries which regions look promising.
- **MFBO** (successive halving) tests many settings cheaply, then re-tests only the best third more carefully.
- **NSGA-III** (pymoo) evolves a population: it keeps good, varied settings and combines them into new ones.

Their results are kept separate and benchmarked: how fast each finds good settings and how much of the best
combined front it contributes.

## Class diagram

```mermaid
classDiagram
    direction LR

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
    }
    class Searcher {
        <<planned, Protocol>>
        +str name
        +ask() Proposal
        +tell(proposal, objectives)
    }
    class MOBOSearcher {
        <<planned>>
        +Study study
    }
    class MFBOSearcher {
        <<planned>>
        +list rungs
        +float keep_fraction
    }
    class NSGA3Searcher {
        <<planned>>
        +NSGA3 algorithm
        +ref_dirs
    }
    class Proposal {
        <<planned>>
        +dict candidate
        +float fidelity
    }
    class Trial {
        <<planned>>
        +str searcher
        +int number
        +dict candidate
        +float fidelity
        +dict objectives
        +float ppl_heldout
        +dict requirement
        +float wall_clock_s
    }
    class SearchRun {
        <<planned>>
        +int budget
        +run(searchers) list~Trial~
        +objective(candidate, fidelity) dict
        +pareto_front(trials) list~Trial~
        +hypervolume(trials) float
    }
    class SearchReport {
        <<planned>>
        +write(trials, out_dir)
    }
    class ArtifactCache

    SearchSpace *-- Param
    Searcher <|.. MOBOSearcher
    Searcher <|.. MFBOSearcher
    Searcher <|.. NSGA3Searcher
    Searcher ..> SearchSpace : proposes from
    Searcher ..> Proposal : returns
    SearchRun o-- Searcher
    SearchRun *-- Trial
    SearchRun ..> ArtifactCache : profiles and baselines reused across trials
    SearchReport ..> Trial
```

Objectives (all minimised): validation-half perplexity, memory, latency, build cost. Held-out perplexity is
stored on every `Trial` but never returned to a searcher. `SEARCH_SPACE` today holds `sensitive_threshold`,
`prune_ratio_aggressive`, `calib_dataset`, `calib_samples` and `gptq_groupsize`; `smoothquant_alpha` and
`quarot_k_bits` join it with Stages 2 and 3.

## Sequence diagram

```mermaid
sequenceDiagram
    actor User
    participant SR as SearchRun (planned)
    participant Se as Searcher (planned)
    participant P as Stages 0 to 4
    participant Rep as SearchReport (planned)

    User->>SR: run(MOBO, MFBO, NSGA-III), same budget and seed
    loop each searcher, separately
        loop until the trial budget is used up
            SR->>Se: ask()
            alt MOBO
                Se->>Se: split past trials into good and bad, sample near the good ones
            else MFBO
                Se->>Se: next config on the current rung, top third move to a pricier rung
            else NSGA-III
                Se->>Se: select by rank and diversity, crossover, mutate
            end
            Se-->>SR: candidate + fidelity
            SR->>P: objective(candidate, fidelity)
            P-->>SR: val ppl, memory, latency, build cost + held-out ppl
            SR->>SR: store Trial, record requirement shortfall
            SR->>Se: tell(candidate, objectives)
        end
    end
    SR->>SR: Pareto front per searcher and combined, hypervolume after every trial
    SR->>Rep: write(trials, results/run_id/search/)
    Rep-->>User: search_report.md, search_results.xlsx (Trials, Pareto, Searcher comparison, Best configs)
    Rep-->>User: Pareto plots (accuracy vs memory, accuracy vs latency), hypervolume over trials
    Note over SR,P: full per-stage comparisons then run only on the final Pareto candidates<br/>and the best config per objective (see the overview's full-run diagram)
```
