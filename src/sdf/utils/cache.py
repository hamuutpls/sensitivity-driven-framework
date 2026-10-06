"""Disk cache for expensive artifacts (sensitivity profiles, FP16 baseline metrics, original-method results).

Entries are JSON files keyed by a hash of everything that determines them, so a trial that changes an
unrelated hyperparameter reuses them, and changing any input recomputes them.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable

from sdf.config import config_hash
from sdf.utils.logging import get_logger

log = get_logger(__name__)


class ArtifactCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, kind: str, key: Any) -> Path:
        return self.root / kind / f"{config_hash(key)}.json"

    def get_or_compute(self, kind: str, key: Any, compute: Callable[[], Any]) -> tuple[Any, bool]:
        """Return (value, was_cached)."""
        p = self.path(kind, key)
        if p.exists():
            try:
                value = json.loads(p.read_text(encoding="utf-8"))["value"]
                log.info("cache hit: %s %s", kind, p.name)
                return value, True
            except (json.JSONDecodeError, KeyError):
                log.warning("corrupt cache entry %s ignored", p)
        log.info("cache miss: %s, computing", kind)
        value = compute()
        atomic_write_text(p, json.dumps({"key": key, "value": value}, indent=2, default=str))
        return value, False


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write via a temp file + rename so an interrupted run mid-write never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:  # not the platform default (cp1252 on Windows)
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class JsonFile:
    """Dataclass mixin: `save` (atomic) / `load` as JSON. `to_dict` / `from_dict` default to the dataclass fields
    (unknown keys from older versions are ignored); plans with nested layers override them."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> None:
        atomic_write_text(path, json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
