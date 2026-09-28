"""Disk cache for expensive artifacts (sensitivity profiles, FP16 baseline metrics, original-method results).

Entries are JSON files keyed by a hash of everything that determines them, so a trial that changes an
unrelated hyperparameter reuses them, and changing any input recomputes them.
"""

from __future__ import annotations

import json
import os
import tempfile
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

    def get(self, kind: str, key: Any) -> Any | None:
        p = self.path(kind, key)
        if not p.exists():
            return None
        try:
            entry = json.loads(p.read_text())
        except json.JSONDecodeError:
            log.warning("corrupt cache entry %s ignored", p)
            return None
        return entry["value"]

    def put(self, kind: str, key: Any, value: Any) -> None:
        p = self.path(kind, key)
        atomic_write_text(p, json.dumps({"key": key, "value": value}, indent=2, default=str))

    def get_or_compute(self, kind: str, key: Any, compute: Callable[[], Any]) -> tuple[Any, bool]:
        """Return (value, was_cached)."""
        value = self.get(kind, key)
        if value is not None:
            log.info("cache hit: %s %s", kind, self.path(kind, key).name)
            return value, True
        log.info("cache miss: %s, computing", kind)
        value = compute()
        self.put(kind, key, value)
        return value, False


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write via a temp file + rename so a Colab disconnect mid-write never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
