"""Stage 0a: per-layer sensitivity profiling.

Score for decoder layer l, summed over calibration batches b:

    s_l = sum_b sum_{w in layer l} |g_w(b) * w|

i.e. the first-order Taylor estimate of how much the loss changes if the layer's weights are
perturbed (gradient x weight saliency). The profile keeps these raw scores; `normalize` maps them to
[0, 1] (rank by default) when a plan is made, so 1 is the most fragile layer and 0 the most robust.

The profile depends only on the model and calibration data, not on any searched parameter, so it
can be computed once and reused by every trial (see SensitivityProfile.save / load).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


@dataclass
class SensitivityProfile:
    raw_scores: list[float]  # accumulated |g * w|, one per decoder layer (normalise with `normalize`)
    layer_numel: list[int] = field(default_factory=list)  # weights per decoder layer (Linear weights only)
    layer_rows: list[int] = field(default_factory=list)  # output channels per layer (for per-channel scales)
    other_numel: int = 0  # parameters outside the decoder layers + layer norms (kept at baseline precision)
    method: str = "grad_x_weight"
    cost: dict[str, Any] = field(default_factory=dict)  # batches, tokens, wall_clock_s, peak_memory_gb
    meta: dict[str, Any] = field(default_factory=dict)  # model name, calibration settings

    @property
    def num_layers(self) -> int:
        return len(self.raw_scores)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SensitivityProfile":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})  # ignores keys from older versions

    @classmethod
    def load(cls, path: str | Path) -> "SensitivityProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


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


def layer_shapes(model: nn.Module) -> tuple[list[int], list[int], int]:
    """(weights per layer, output channels per layer, parameters outside those Linear weights).

    Only the Linear weights of decoder layers are compressed; embeddings, norms and the LM head stay at
    baseline precision. Tied parameters are counted once.
    """
    layers = find_decoder_layers(model)
    numel, rows, compressible = [], [], set()
    for layer in layers:
        linears = [m for m in layer.modules() if isinstance(m, nn.Linear)]
        numel.append(sum(m.weight.numel() for m in linears))
        rows.append(sum(m.out_features for m in linears))
        compressible.update(id(m.weight) for m in linears)
    unique = {id(p): p for p in model.parameters()}
    other = sum(p.numel() for pid, p in unique.items() if pid not in compressible)
    return numel, rows, other


NORMALIZATIONS = ("rank", "minmax")


def normalize(scores: list[float], method: str = "rank") -> list[float]:
    """Map raw scores to [0, 1]. If all scores are equal there is no signal and every layer gets 0.

    rank (default): position in the sorted order, rank / (n - 1), ties averaged. One outlier layer cannot
        squeeze the rest together, and `sensitive_threshold` t protects roughly the top (1 - t) share of
        layers, so the threshold means the same thing on every model and calibration set.
    minmax: (s - min) / (max - min). Keeps relative magnitudes, but a single low (or high) outlier pushes
        every other layer to one end. On TinyLlama, layer 0 scores 2705 against 6596-8647 for the rest, so
        threshold 0.5 protected 21 of 22 layers.
    """
    if method not in NORMALIZATIONS:
        raise ValueError(f"unknown normalization {method!r}; choose from {NORMALIZATIONS}")
    n = len(scores)
    lo, hi = min(scores), max(scores)
    if hi - lo <= 0 or n == 1:
        return [0.0 for _ in scores]
    if method == "minmax":
        return [(s - lo) / (hi - lo) for s in scores]
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:  # average the ranks of tied scores
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return [r / (n - 1) for r in ranks]


def outlier_layers(raw_scores: list[float], cutoff: float = 3.5) -> list[int]:
    """Layers whose raw score is far from the rest: robust z-score |s - median| / (1.4826 * MAD) > cutoff.

    Median and MAD (median absolute deviation) are used instead of mean and std so the outlier itself
    doesn't hide itself by inflating the spread. 3.5 is the usual cutoff (Iglewicz and Hoaglin).
    """
    n = len(raw_scores)
    if n < 3:
        return []
    ordered = sorted(raw_scores)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    dev = sorted(abs(s - median) for s in raw_scores)
    mad = dev[n // 2] if n % 2 else (dev[n // 2 - 1] + dev[n // 2]) / 2
    if mad == 0:
        return [i for i, s in enumerate(raw_scores) if s != median]
    return [i for i, s in enumerate(raw_scores) if abs(s - median) / (1.4826 * mad) > cutoff]


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
    if not torch.isfinite(raw).all():
        raise FloatingPointError("non-finite sensitivity scores; profile in float32 (stage0.profile_dtype)")

    cost: dict[str, Any] = {
        "calibration_batches": n_batches,
        "calibration_tokens": n_tokens,
        "wall_clock_s": time.perf_counter() - t0,
        "peak_memory_gb": (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else None,
    }
    raw_list = raw.tolist()
    numel, rows, other = layer_shapes(model)
    return SensitivityProfile(
        raw_scores=raw_list,
        layer_numel=numel,
        layer_rows=rows,
        other_numel=other,
        cost=cost,
        meta=dict(meta or {}),
    )
