"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sdf.config import load_config
from sdf.search_space import SEARCH_SPACE


def _parse_sets(pairs: list[str]) -> dict:
    """--set a.b=value pairs; values are parsed as YAML (numbers, null, lists...)."""
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        out[key] = yaml.safe_load(value)
    return out


def stage0_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage 0: sensitivity profiling and compression planning")
    p.add_argument("--config", type=Path, help="YAML config, e.g. configs/tinyllama.yaml (omit for built-in defaults)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set hyperparams.sensitive_threshold=0.7")
    p.add_argument("--skip-fp16-eval", action="store_true", help="don't measure the FP16 baseline")
    args = p.parse_args(argv)

    from sdf.run import start_run
    from sdf.stage0 import run_stage0

    cfg = load_config(args.config, _parse_sets(args.set))
    candidate = SEARCH_SPACE.make(cfg.hyperparams)
    ctx = start_run(cfg)
    result = run_stage0(ctx, candidate, measure_fp16=not args.skip_fp16_eval)
    for name, path in result.outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    stage0_main()
