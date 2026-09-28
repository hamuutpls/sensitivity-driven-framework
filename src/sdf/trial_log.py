"""Trial log: one JSON line per trial, shared by all searchers and pooled at the end."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from sdf.config import CandidateConfig


@dataclass(frozen=True)
class Objectives:
    """The three search objectives: memory down, latency down, accuracy up."""

    memory_gb: float
    latency_ms_per_token: float
    accuracy: float


@dataclass
class TrialRecord:
    trial_id: int
    searcher: str  # "mobo" | "mfbo" | "nsga3"
    config: CandidateConfig
    objectives: Objectives | None = None
    cost: dict[str, float] = field(default_factory=dict)  # e.g. wall_clock_s, peak_memory_gb
    fidelity: str | None = None  # MFBO rung; None = full pipeline
    reports: dict[str, Any] = field(default_factory=dict)  # stage0..stage3 reports, full metrics
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrialRecord":
        d = dict(d)
        d["config"] = CandidateConfig.from_dict(d["config"])
        if d.get("objectives") is not None:
            d["objectives"] = Objectives(**d["objectives"])
        return cls(**d)


class TrialLog:
    """Append-only JSONL log of trials."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def clear(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def append(self, record: TrialRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

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
