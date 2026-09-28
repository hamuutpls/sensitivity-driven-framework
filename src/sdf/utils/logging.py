"""Timestamped logging for the whole package (use `get_logger(__name__)`, never print)."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the `sdf` logger once: stderr, plus `log_file` when given (appended, survives restarts)."""
    root = logging.getLogger("sdf")
    root.setLevel(level.upper())
    formatter = logging.Formatter(_FORMAT, _DATEFMT)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
                   for h in root.handlers):
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("sdf") else f"sdf.{name}")
