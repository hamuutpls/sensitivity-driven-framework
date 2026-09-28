"""Trial log: one JSON line per search trial. Each searcher writes its own log; they are compared, not pooled."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from sdf.search_space import SEARCH_SPACE, Candidate, SearchSpace


@dataclass(frozen=True)
class Objectives:
    """What the searchers optimise. The held-out perplexity is recorded on TrialRecord, never optimised."""

    ppl_val: float  # accuracy, as validation-half perplexity (lower is better)
    memory_gb: float
    latency_ms: float  # decode latency per token
    build_time_s: float


@dataclass
class TrialRecord:
    trial_id: int
    searcher: str  # "mobo" | "mfbo" | "nsga3"
    candidate: Candidate
    objectives: Objectives | None = None  # None while running or if the trial failed
    ppl_heldout: float | None = None
    requirement: dict[str, Any] = field(default_factory=dict)  # RequirementCheck.to_dict(): met + shortfall
    fidelity: str | None = None  # MFBO rung; None = full pipeline
    status: str = "ok"  # ok | failed
    error: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidate"] = self.candidate.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], space: SearchSpace = SEARCH_SPACE) -> "TrialRecord":
        d = dict(d)
        d["candidate"] = space.validate(d["candidate"])
        if d.get("objectives") is not None:
            d["objectives"] = Objectives(**d["objectives"])
        return cls(**d)


class TrialLog:
    """Append-only JSONL log; each append is flushed, so a disconnect loses at most the trial in progress."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: TrialRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record.to_dict(), default=str) + "\n")

    def __iter__(self) -> Iterator[TrialRecord]:
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    yield TrialRecord.from_dict(json.loads(line))

    def records(self, searcher: str | None = None) -> list[TrialRecord]:
        return [r for r in self if searcher is None or r.searcher == searcher]

    def __len__(self) -> int:
        return sum(1 for _ in self)
