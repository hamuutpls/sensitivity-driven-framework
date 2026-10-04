"""The search space, shared by every searcher and defined in one place.

Adding a hyperparameter is a one-line change: append a Param to SEARCH_SPACE. A candidate is a plain dict
of parameter name -> value, validated by SEARCH_SPACE.validate / make.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

PER_CHANNEL = -1  # GPTQ convention: groupsize -1 means one scale per output channel


@dataclass(frozen=True)
class Param:
    name: str
    stage: int  # the stage that consumes it
    default: Any
    low: float | None = None  # continuous range (inclusive) ...
    high: float | None = None
    choices: tuple[Any, ...] | None = None  # ... or a categorical set

    def __post_init__(self):
        if (self.choices is None) == (self.low is None or self.high is None):
            raise ValueError(f"{self.name}: give either low/high or choices")
        if not self.contains(self.default):
            raise ValueError(f"{self.name}: default {self.default!r} outside the space")

    def contains(self, value: Any) -> bool:
        if self.choices is not None:
            return value in self.choices
        return isinstance(value, (int, float)) and self.low <= value <= self.high


class SearchSpace:
    def __init__(self, params: Sequence[Param]):
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise ValueError("duplicate parameter names")
        self.params = tuple(params)
        self._by_name = {p.name: p for p in params}

    def make(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Defaults with `overrides` applied, validated."""
        return self.validate({**{p.name: p.default for p in self.params}, **(overrides or {})})

    def validate(self, values: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(values) - set(self._by_name)
        missing = set(self._by_name) - set(values)
        errors = [f"unknown parameter {n!r}" for n in sorted(unknown)]
        errors += [f"missing parameter {n!r}" for n in sorted(missing)]
        for name, value in values.items():
            p = self._by_name.get(name)
            if p is not None and not p.contains(value):
                bounds = p.choices if p.choices is not None else (p.low, p.high)
                errors.append(f"{name}={value!r} not in {bounds}")
        if errors:
            raise ValueError("invalid candidate: " + "; ".join(errors))
        return dict(values)


SEARCH_SPACE = SearchSpace(
    [
        Param("sensitive_threshold", stage=0, default=0.5, low=0.1, high=0.9),
        Param("prune_ratio_aggressive", stage=0, default=0.3, low=0.0, high=0.6),
        Param("calib_dataset", stage=0, default="wikitext2", choices=("wikitext2", "c4", "pile10k")),
        Param("calib_samples", stage=0, default=64, choices=(16, 32, 64, 128)),
        Param("gptq_groupsize", stage=1, default=128, choices=(32, 64, 128, PER_CHANNEL)),
        Param("smoothquant_alpha", stage=2, default=0.5, low=0.0, high=1.0),
        Param("quarot_k_bits", stage=3, default=4, choices=(2, 3, 4, 8)),
    ]
)
