"""Stage 0 — sensitivity profiling and compression planning."""

from __future__ import annotations

from typing import Any

from sdf.config import CandidateConfig
from sdf.stage0.planner import (
    COMPRESSED_BITS,
    PROTECTED_BITS,
    CompressionPlan,
    LayerPlan,
    plan_compression,
    stage0_report,
)
from sdf.stage0.sensitivity import SensitivityProfile, find_decoder_layers, normalize, profile_sensitivity

__all__ = [
    "COMPRESSED_BITS",
    "PROTECTED_BITS",
    "CompressionPlan",
    "LayerPlan",
    "SensitivityProfile",
    "find_decoder_layers",
    "normalize",
    "plan_compression",
    "profile_sensitivity",
    "run_stage0",
    "stage0_report",
]


def run_stage0(profile: SensitivityProfile, config: CandidateConfig) -> tuple[CompressionPlan, dict[str, Any]]:
    """Per-trial Stage 0: plan from a (cached) profile using the candidate's threshold and pruning ratio."""
    plan = plan_compression(profile, config.sensitivity_threshold, config.pruning_ratio)
    return plan, stage0_report(profile, plan)
