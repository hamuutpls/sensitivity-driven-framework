"""Run setup shared by every stage: run directory, config snapshot, logging, seeding, cache."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from sdf.config import FrameworkConfig, config_hash
from sdf.utils.cache import ArtifactCache, atomic_write_text
from sdf.utils.logging import get_logger, setup_logging
from sdf.utils.seed import set_seed

log = get_logger(__name__)


@dataclass
class RunContext:
    cfg: FrameworkConfig
    run_id: str
    run_dir: Path
    cache: ArtifactCache


def start_run(cfg: FrameworkConfig) -> RunContext:
    """Create <output_root>/<run_id>/, snapshot the config, start logging to run.log and fix seeds.

    Re-using an existing run_id resumes into the same directory (e.g. after an interrupted run).
    """
    run_id = cfg.run.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}_{config_hash(cfg.to_dict(), length=6)}"
    run_dir = Path(cfg.run.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.run.log_level, run_dir / "run.log")
    atomic_write_text(run_dir / "config.json", json.dumps(cfg.to_dict(), indent=2))
    cache_root = Path(cfg.run.cache_dir) if cfg.run.cache_dir else Path(cfg.run.output_root).parent / "cache"
    log.info("run %s -> %s (cache %s)", run_id, run_dir, cache_root)
    set_seed(cfg.run.seed, cfg.run.deterministic)
    return RunContext(cfg=cfg, run_id=run_id, run_dir=run_dir, cache=ArtifactCache(cache_root))
