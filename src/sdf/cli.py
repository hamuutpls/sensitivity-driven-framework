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
    from sdf.stage0.run import run_stage0

    cfg = load_config(args.config, _parse_sets(args.set))
    candidate = SEARCH_SPACE.make(cfg.hyperparams)
    ctx = start_run(cfg)
    result = run_stage0(ctx, candidate, measure_fp16=not args.skip_fp16_eval)
    for name, path in result.outputs.items():
        print(f"{name}: {path}")



def sweep_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage 0 sweep: several values of every setting, one comparison report")
    p.add_argument("--config", type=Path, help="YAML config, e.g. configs/tinyllama.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    p.add_argument("--grid", action="append", default=[], metavar="NAME=V1,V2",
                   help="values to try for one setting, e.g. --grid sensitive_threshold=0.3,0.5,0.7 "
                        "(default: every choice, or 5 evenly spaced values)")
    p.add_argument("--profile", type=Path, action="append", default=[],
                   help="plan from saved sensitivity_profile.json files instead of profiling (no GPU needed)")
    args = p.parse_args(argv)

    from sdf.run import start_run
    from sdf.stage0.sensitivity import SensitivityProfile
    from sdf.stage0.sweep import run_sweep

    cfg = load_config(args.config, _parse_sets(args.set))
    grid = {}
    for item in args.grid:
        name, sep, values = item.partition("=")
        if not sep:
            raise SystemExit(f"--grid expects name=v1,v2, got {item!r}")
        grid[name] = [yaml.safe_load(v) for v in values.split(",")]
        for v in grid[name]:
            SEARCH_SPACE.make({**cfg.hyperparams, name: v})
    profiles = None
    if args.profile:
        defaults = SEARCH_SPACE.make(cfg.hyperparams)
        profiles = {}
        for path in args.profile:
            prof = SensitivityProfile.load(path)
            key = (prof.meta.get("calib_dataset", defaults["calib_dataset"]),
                   prof.meta.get("calib_samples", defaults["calib_samples"]))
            profiles[key] = prof
    for name, path in run_sweep(start_run(cfg), grid, profiles=profiles).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    stage0_main()
