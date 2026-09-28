"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdf.config import DEFAULT_SEARCH_SPACE, CalibrationConfig, CandidateConfig


def _resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def stage0_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage 0: sensitivity profiling and compression planning")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4", "pile10k"])
    p.add_argument("--n-batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--threshold", type=float, default=0.5, help="sensitivity threshold")
    p.add_argument("--pruning-ratio", type=float, default=0.3, help="robust-layer pruning ratio")
    p.add_argument("--profile", type=Path, help="reuse a saved sensitivity profile instead of profiling")
    p.add_argument("--out", type=Path, default=Path("runs/stage0"))
    args = p.parse_args(argv)

    from sdf.stage0 import SensitivityProfile, profile_sensitivity, run_stage0

    candidate = CandidateConfig(sensitivity_threshold=args.threshold, pruning_ratio=args.pruning_ratio)
    DEFAULT_SEARCH_SPACE.validate(candidate)

    if args.profile:
        profile = SensitivityProfile.load(args.profile)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from sdf.calibration import load_calibration_batches

        calib = CalibrationConfig(
            dataset=args.dataset, n_batches=args.n_batches, batch_size=args.batch_size,
            seq_len=args.seq_len, seed=args.seed,
        )
        device = _resolve_device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(device)
        batches = load_calibration_batches(tokenizer, calib)
        profile = profile_sensitivity(
            model, batches, device=device,
            meta={"model": args.model, "dtype": args.dtype, "calibration": calib.to_dict()},
        )
        profile.save(args.out / "sensitivity_profile.json")

    plan, report = run_stage0(profile, candidate)
    plan.save(args.out / "compression_plan.json")
    (args.out / "stage0_report.json").write_text(json.dumps(report, indent=2))
    print(f"{report['protected']['count']}/{report['num_layers']} layers protected; "
          f"outputs written to {args.out}")


if __name__ == "__main__":
    stage0_main()
