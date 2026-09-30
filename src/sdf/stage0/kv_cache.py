"""Stage 0c: KV cache sensitivity and plan.

While generating, every decoder layer stores a key and a value vector per past token (the KV cache) so it
does not recompute them. Stage 0 plans three things for it, per layer, and predicts its memory:

1. bits for the cached keys and 2. bits for the cached values, chosen separately. `profile_kv` rounds only
   one layer's keys (or values) to each candidate bit width and measures the calibration perplexity rise.
   `plan_kv_bits` then spends a fixed average bit budget where the measured damage is largest.
   Keys are rounded per channel (across tokens) and values per token, as KIVI / KVQuant do, because keys
   have a few large outlier channels. Keys are rounded before RoPE (KVQuant's pre-RoPE choice), since the
   k_proj output is where a model-agnostic hook can reach them.
3. how many past tokens to keep (token eviction, H2O / SnapKV style). `attention_coverage` measures, per
   layer, the share of attention that the top r share of past tokens receive. A layer whose attention is
   concentrated can drop the rest. This is an oracle upper bound (top-k per query); a real eviction policy
   keeps a bit less.

The profile depends only on the model and calibration text, so it is cached like the weight profile.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch import nn

from sdf.stage0.sensitivity import _mean_loss, find_decoder_layers, round_to_nearest

KINDS = ("key", "value")


@dataclass
class KVProfile:
    bits_options: list[int]
    key_dims: list[int]  # numbers stored per token for keys, per layer (k_proj output size)
    value_dims: list[int]
    key_rise: list[list[float]]  # [layer][bits option] calibration perplexity rise
    value_rise: list[list[float]]
    keep_ratios: list[float]
    coverage: list[list[float]]  # [layer][keep ratio] share of attention on the top keep-ratio tokens
    cost: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        return len(self.key_dims)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KVProfile":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def kv_projections(layer: nn.Module, names: tuple[str, str]) -> tuple[nn.Linear, nn.Linear]:
    """The Linear modules producing a layer's keys and values, found by name suffix (k_proj / v_proj for
    Llama, Mistral, Qwen, OPT). Models with a fused QKV projection need their own names here."""
    found = []
    for name in names:
        mods = [m for n, m in layer.named_modules() if n.split(".")[-1] == name and isinstance(m, nn.Linear)]
        if len(mods) != 1:
            raise ValueError(f"expected one Linear named {name!r} in each decoder layer, found {len(mods)}; "
                             "set stage0.kv_module_names for this model")
        found.append(mods[0])
    return found[0], found[1]


@contextmanager
def quantize_output(module: nn.Module, bits: int, group_size: int, per_channel: bool) -> Iterator[None]:
    """Round `module`'s output (batch, tokens, channels) to `bits` while the context is open: per channel
    across groups of `group_size` tokens, or per token across groups of `group_size` channels."""
    def hook(mod, args, out):
        if per_channel:
            return round_to_nearest(out.transpose(-1, -2), bits, group_size).transpose(-1, -2)
        return round_to_nearest(out, bits, group_size)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def eager_attention(model: nn.Module) -> Iterator[None]:
    """Attention weights are only returned by the plain ("eager") attention code; SDPA / flash skip them."""
    cfg = model.config
    old = getattr(cfg, "_attn_implementation", None)
    cfg._attn_implementation = "eager"
    try:
        yield
    finally:
        cfg._attn_implementation = old


def _coverage(attn: torch.Tensor, keep_ratios: list[float]) -> list[float]:
    """attn (batch, heads, queries, keys). For the second half of the queries (so each sees enough past
    tokens), the share of attention on its top ceil(r * visible) keys, averaged over batch, heads, queries."""
    t = attn.shape[-1]
    q0 = t // 2
    cum = attn[..., q0:, :].float().sort(dim=-1, descending=True).values.cumsum(-1)
    visible = torch.arange(q0, t, device=attn.device) + 1
    out = []
    for r in keep_ratios:
        idx = (torch.ceil(r * visible) - 1).clamp(min=0).long().view(1, 1, -1, 1)
        out.append(cum.gather(-1, idx.expand(*cum.shape[:2], -1, 1)).mean().item())
    return out


@torch.no_grad()
def attention_coverage(model: nn.Module, batches: list[torch.Tensor], keep_ratios: list[float],
                       device: torch.device) -> list[list[float]]:
    total: list[list[float]] | None = None
    with eager_attention(model):
        for b in batches:
            atts = model(input_ids=b.to(device), output_attentions=True).attentions
            if not atts or atts[0] is None:
                raise RuntimeError("the model returned no attention weights; token-budget planning needs them")
            per_layer = [_coverage(a, keep_ratios) for a in atts]
            total = per_layer if total is None else [[x + y for x, y in zip(p, q)] for p, q in zip(total, per_layer)]
    return [[x / len(batches) for x in row] for row in total]


def profile_kv(model: nn.Module, batches: Iterable[torch.Tensor], bits_options: list[int], group_size: int,
               keep_ratios: list[float], module_names: tuple[str, str], device: torch.device | str | None = None,
               meta: dict[str, Any] | None = None) -> KVProfile:
    device = torch.device(device) if device is not None else next(model.parameters()).device
    batches = list(batches)
    if not batches:
        raise ValueError("no calibration batches given")
    projs = [kv_projections(layer, module_names) for layer in find_decoder_layers(model)]
    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    try:
        base = math.exp(_mean_loss(model, batches, device))
        rise: dict[str, list[list[float]]] = {k: [] for k in KINDS}
        for i, kind in enumerate(KINDS):
            for pair in projs:
                row = []
                for bits in bits_options:
                    with quantize_output(pair[i], bits, group_size, per_channel=kind == "key"):
                        row.append(math.exp(_mean_loss(model, batches, device)) - base)
                rise[kind].append(row)
        coverage = attention_coverage(model, batches, keep_ratios, device)
    finally:
        model.train(was_training)
    cost = {"calibration_batches": len(batches), "calibration_tokens": sum(b.numel() for b in batches),
            "wall_clock_s": time.perf_counter() - t0,
            "peak_memory_gb": (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else None}
    return KVProfile(bits_options=list(bits_options), key_dims=[k.out_features for k, _ in projs],
                     value_dims=[v.out_features for _, v in projs], key_rise=rise["key"],
                     value_rise=rise["value"], keep_ratios=list(keep_ratios), coverage=coverage, cost=cost,
                     meta={**(meta or {}), "baseline_ppl": base})


@dataclass(frozen=True)
class KVLayerPlan:
    layer: int
    key_bits: int
    value_bits: int
    keep_ratio: float  # share of past tokens kept in the cache (1.0 = no eviction)


@dataclass(frozen=True)
class KVPlan:
    layers: tuple[KVLayerPlan, ...]
    kind: str  # "uniform" | "sensitivity"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "layers": [asdict(lp) for lp in self.layers]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KVPlan":
        return cls(tuple(KVLayerPlan(**lp) for lp in d["layers"]), d["kind"])

    @classmethod
    def load(cls, path: str | Path) -> "KVPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def uniform_kv_plan(num_layers: int, bits: int) -> KVPlan:
    return KVPlan(tuple(KVLayerPlan(i, bits, bits, 1.0) for i in range(num_layers)), "uniform")


def _monotone(rise: list[float]) -> list[float]:
    """Measured rise per bits option (ascending bits), clipped at 0 and made non-increasing: more bits
    never predicts more damage (small negative or non-monotone values are measurement noise)."""
    out, low = [], math.inf
    for r in rise:
        low = min(low, max(r, 0.0))
        out.append(low)
    return out


def plan_kv_bits(profile: KVProfile, avg_bits: float) -> dict[tuple[int, str], int]:
    """Bits for every (layer, key|value), spending at most `avg_bits` per cached number on average.

    Greedy: start everything at the fewest bits, then repeatedly take the upgrade with the largest drop in
    measured perplexity rise per extra bit, while it fits. Assumes per-tensor damages add up.
    ponytail: greedy, not an exact knapsack; exact only matters with very uneven layer sizes.
    """
    opts = profile.bits_options
    items = {}
    for kind, dims, rises in (("key", profile.key_dims, profile.key_rise),
                              ("value", profile.value_dims, profile.value_rise)):
        for layer, (d, r) in enumerate(zip(dims, rises)):
            items[(layer, kind)] = (d, _monotone(r))
    level = {k: 0 for k in items}
    budget = avg_bits * sum(d for d, _ in items.values()) - opts[0] * sum(d for d, _ in items.values())
    while True:
        best, best_gain = None, 0.0
        for k, (d, r) in items.items():
            for j in range(level[k] + 1, len(opts)):
                extra = (opts[j] - opts[level[k]]) * d
                gain = (r[level[k]] - r[j]) / extra
                if extra <= budget + 1e-9 and gain > best_gain:
                    best, best_gain = (k, j, extra), gain
        if best is None:
            break
        k, j, extra = best
        level[k], budget = j, budget - extra
    return {k: opts[j] for k, j in level.items()}


def keep_ratio_for(coverage: list[float], keep_ratios: list[float], target: float) -> float:
    """The smallest keep ratio whose top tokens get at least `target` of the attention (1.0 if none)."""
    return next((r for r, c in sorted(zip(keep_ratios, coverage)) if c >= target), 1.0)


def plan_kv(profile: KVProfile, avg_bits: float, coverage_target: float | None) -> KVPlan:
    """Per-layer key bits, value bits and (unless coverage_target is None) token budget."""
    bits = plan_kv_bits(profile, avg_bits)
    return KVPlan(tuple(KVLayerPlan(
        i, bits[(i, "key")], bits[(i, "value")],
        1.0 if coverage_target is None else keep_ratio_for(profile.coverage[i], profile.keep_ratios, coverage_target),
    ) for i in range(profile.num_layers)), "sensitivity")


@dataclass(frozen=True)
class KVCost:
    memory_gb: float
    avg_bits: float  # over every cached number that is kept
    kept_share: float  # share of past tokens kept, weighted by numbers per token
    ppl_rise: float  # sum of the measured per-tensor perplexity rises for the chosen bits
    attention_kept: float  # mean over layers of the attention share on the kept tokens
    per_layer_mb: tuple[float, ...]


def predict_kv(plan: KVPlan, profile: KVProfile, context_len: int, batch_size: int, group_size: int,
               group_overhead_bits: int, baseline_bits: int) -> KVCost:
    """Cache size for `batch_size` sequences of `context_len` tokens. Rounded numbers carry one scale and
    zero point (`group_overhead_bits`) per `group_size` numbers; evicted tokens take no space."""
    def rise(rises: list[float], bits: int) -> float:
        return 0.0 if bits >= baseline_bits else _monotone(rises)[profile.bits_options.index(bits)]

    per_layer, kept_numbers, all_numbers, bit_sum, damage, attn = [], 0.0, 0.0, 0.0, 0.0, 0.0
    for lp, kd, vd, kr, vr, cov in zip(plan.layers, profile.key_dims, profile.value_dims, profile.key_rise,
                                       profile.value_rise, profile.coverage):
        tokens = math.ceil(lp.keep_ratio * context_len) * batch_size
        bits = 0.0
        for d, b in ((kd, lp.key_bits), (vd, lp.value_bits)):
            n = tokens * d
            bits += n * b + (n / group_size * group_overhead_bits if b < baseline_bits else 0)
            bit_sum += n * b
            kept_numbers += n
            all_numbers += context_len * batch_size * d
        per_layer.append(bits / 8 / 1e6)
        damage += rise(kr, lp.key_bits) + rise(vr, lp.value_bits)
        attn += 1.0 if lp.keep_ratio >= 1.0 else cov[profile.keep_ratios.index(lp.keep_ratio)]
    return KVCost(memory_gb=sum(per_layer) / 1e3, avg_bits=bit_sum / kept_numbers,
                  kept_share=kept_numbers / all_numbers, ppl_rise=damage, attention_kept=attn / len(plan.layers),
                  per_layer_mb=tuple(per_layer))
