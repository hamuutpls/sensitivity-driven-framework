"""Stage 0a: per-layer sensitivity profiling.

Score for decoder layer l, summed over calibration batches b:

    s_l = sum_b sum_{w in layer l} |g_w(b) * w|

i.e. the first-order Taylor estimate of how much the loss changes if the layer's weights are
perturbed (gradient x weight saliency). Scores are then min-max normalised to [0, 1], so 1 is the
most fragile layer and 0 the most robust.

The profile depends only on the model and calibration data, not on any searched parameter, so it
can be computed once and reused by every trial (see SensitivityProfile.save / load).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


@dataclass
class SensitivityProfile:
    scores: list[float]  # normalised, one per decoder layer, in [0, 1]
    raw_scores: list[float]  # accumulated |g * w| before normalisation
    method: str = "grad_x_weight"
    cost: dict[str, Any] = field(default_factory=dict)  # batches, tokens, wall_clock_s, peak_memory_gb
    meta: dict[str, Any] = field(default_factory=dict)  # model name, calibration settings

    @property
    def num_layers(self) -> int:
        return len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SensitivityProfile":
        return cls(**json.loads(Path(path).read_text()))


def find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the model's stack of decoder layers (model.model.layers for Llama-family models)."""
    inner = getattr(model, "model", None)
    if inner is not None and isinstance(getattr(inner, "layers", None), nn.ModuleList):
        return inner.layers
    # Fallback: the longest ModuleList in the model.
    candidates = [m for m in model.modules() if isinstance(m, nn.ModuleList)]
    if not candidates:
        raise ValueError("could not find decoder layers in model")
    return max(candidates, key=len)


def normalize(scores: list[float]) -> list[float]:
    """Min-max normalise to [0, 1]. If all scores are equal every layer gets 0."""
    lo, hi = min(scores), max(scores)
    if hi - lo <= 0:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def profile_sensitivity(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    device: torch.device | str | None = None,
    meta: dict[str, Any] | None = None,
) -> SensitivityProfile:
    """Run the calibration batches through `model` and score every decoder layer.

    Only decoder-layer weights get gradients; embeddings and the LM head are frozen for the
    duration to save memory, and every parameter's requires_grad flag is restored afterwards.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    layers = find_decoder_layers(model)

    layer_params = [list(layer.parameters()) for layer in layers]
    in_layers = {id(p) for ps in layer_params for p in ps}
    saved_flags = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in in_layers)

    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    raw = torch.zeros(len(layers), dtype=torch.float64)
    n_batches = n_tokens = 0
    t0 = time.perf_counter()
    try:
        for input_ids in batches:
            input_ids = input_ids.to(device)
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, labels=input_ids).loss
            loss.backward()
            with torch.no_grad():
                for i, params in enumerate(layer_params):
                    raw[i] += sum(
                        (p.grad.float() * p.float()).abs().sum().item() for p in params if p.grad is not None
                    )
            n_batches += 1
            n_tokens += input_ids.numel()
    finally:
        model.zero_grad(set_to_none=True)
        for p in model.parameters():
            p.requires_grad_(saved_flags[id(p)])
        model.train(was_training)

    if n_batches == 0:
        raise ValueError("no calibration batches given")

    cost: dict[str, Any] = {
        "calibration_batches": n_batches,
        "calibration_tokens": n_tokens,
        "wall_clock_s": time.perf_counter() - t0,
        "peak_memory_gb": (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else None,
    }
    raw_list = raw.tolist()
    return SensitivityProfile(
        scores=normalize(raw_list),
        raw_scores=raw_list,
        cost=cost,
        meta=dict(meta or {}),
    )
