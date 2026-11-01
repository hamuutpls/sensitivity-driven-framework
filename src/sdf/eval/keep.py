"""Keeping the final model's backend files (Mohammad, 2026-11-01: "save every model produced by each test", "i mean
the model at the end of each run after stage 4"). Each backend exports into a temporary folder that is deleted after
measuring; inside keep_models(folder) the exports of compressed models are moved to <folder>/<backend>/ instead."""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

from sdf.utils.logging import get_logger

log = get_logger(__name__)
_FOLDER: ContextVar[Path | None] = ContextVar("keep_models", default=None)


@contextmanager
def keep_models(folder: str | Path) -> Iterator[Path]:
    """Inside, each backend keeps the export of a compressed model (not the uncompressed row's) in `folder`."""
    token = _FOLDER.set(Path(folder))
    try:
        yield Path(folder)
    finally:
        _FOLDER.reset(token)


def keeping() -> bool:
    """Inside keep_models(): the model has to be built (not read from a cache) for its files to be kept."""
    return _FOLDER.get() is not None


def keep(backend: str, *exports: Path) -> Path | None:
    """Move `exports` (folders or files) to <folder>/<backend>/ inside keep_models(), replacing an older copy; outside
    it, leave them to be deleted. Returns the backend's folder."""
    folder = _FOLDER.get()
    if folder is None:
        return None
    dest = folder / backend
    try:  # the numbers are measured already: a full disk loses the files, never the measurement
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for export in exports:
            if export.exists():
                shutil.move(str(export), str(dest / export.name))
    except OSError as e:
        log.warning("could not keep the %s model in %s: %s", backend, dest, e)
        shutil.rmtree(dest, ignore_errors=True)
        return None
    log.info("kept the %s model in %s", backend, dest)
    return dest
