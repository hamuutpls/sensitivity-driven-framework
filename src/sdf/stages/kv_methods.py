"""Stage 3 KV-cache techniques: QuaRot KV, KVQuant (rounding) and H2O, SnapKV, InfiniGen (forgetting words).

Perplexity is measured with one forward pass per text window, so forgetting is simulated per word ("every word",
Mohammad, 2026-10-18): each word attends only to the past words the technique would still hold at that point, under
a budget of ceil(keep_ratio x window length) words per layer (the Stage 0 plan's share, or the papers' 20% in the
standard rows). A dropped word is dropped for a whole key/value head (shared by its group of query heads), since that
is what frees memory. Rounding of cached keys and values is applied in place, as rtn_kv does; nothing is stored
smaller, so memory comes from the plan's prediction and speed is not a compressed cache's speed.

Hooks: rounding of keys before RoPE (KVQuant, plain rounding) is a forward hook on k_proj / v_proj; QuaRot's rotation
(keys after RoPE) and every forgetting rule live in one attention function, `kv_attention_forward`, switched on per
model by `kv_attention`.
"""

from __future__ import annotations

import functools
import math
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

from sdf.stage0.kv_cache import kv_projections
from sdf.stage0.sensitivity import find_decoder_layers, round_to_nearest
from sdf.stages.activation_methods import _orthogonal

SDF_KV = "sdf_kv"  # the attention implementation name registered below
SNAPKV_WINDOW = 32  # SnapKV paper defaults: observation window and max-pooling kernel
SNAPKV_KERNEL = 7
INFINIGEN_PARTIAL = 0.3  # InfiniGen's share of query/key channels used to guess which words matter
KVQUANT_OUTLIERS = 0.01  # KVQuant keeps the largest 1% of numbers (dense-and-sparse) unrounded
KVQUANT_CALIB_TOKENS = 8192  # calibration tokens per layer for KVQuant's ranges and levels


@dataclass(frozen=True)
class LayerKV:
    """What one layer's attention does to its cache: QuaRot rounding (bits < 16, else none) and forgetting."""
    rotate_key_bits: int = 16
    rotate_value_bits: int = 16
    forget: str | None = None  # "h2o" | "snapkv" | "infinigen"
    keep_ratio: float = 1.0


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    """(batch, kv_heads, ...) -> (batch, kv_heads * groups, ...); transformers' repeat_kv takes only 4-D tensors,
    and the keep masks here are 3-D too."""
    return x.repeat_interleave(groups, dim=1)


def group_mean(x: torch.Tensor, groups: int) -> torch.Tensor:
    """(batch, q_heads, ...) -> (batch, kv_heads, ...): mean over the query heads sharing one key/value head."""
    b, h = x.shape[:2]
    return x.reshape(b, h // groups, groups, *x.shape[2:]).mean(2)


@functools.cache
def _rotation(n: int, device: torch.device) -> torch.Tensor:
    """Built once per size and device: a host-to-device copy inside the forward would stop CUDA graph capture."""
    return _orthogonal(n, torch.Generator().manual_seed(0), device)


def quarot_round(x: torch.Tensor, bits: int) -> torch.Tensor:
    """x (batch, heads, tokens, head_dim) rotated by a random Hadamard matrix (Stage 1 QuaRot's), rounded with one
    min/max scale per token and head (QuaRot's asymmetric per-head groups), rotated back. The rotation spreads
    outliers over the head."""
    h = _rotation(x.shape[-1], x.device)
    r = round_to_nearest(x.float() @ h, bits, x.shape[-1])
    return (r @ h.T).to(x.dtype)


def _causal(n: int, device: torch.device) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool, device=device).tril()


def h2o_keep(scores: torch.Tensor, groups: int, budget: int) -> torch.Tensor:
    """H2O: the cache holds `budget` words, half the most recent, half the "heavy hitters" with the most attention
    so far. scores (batch, q_heads, n, n) are attention logits with the causal mask added. Returns keep
    (batch, kv_heads, n, n): the keys each query may use. Evicts one word per new word, greedily, from the attention
    each word actually got (masked), as in the paper's simulation."""
    b, _, n, _ = scores.shape
    keep = _causal(n, scores.device).expand(b, scores.shape[1] // groups, n, n).clone()
    if budget >= n:
        return keep
    recent = max(budget // 2, 1)
    acc = group_mean(scores[:, :, :budget].float().softmax(-1), groups).sum(2)  # nothing evicted before `budget`
    kept = torch.zeros_like(acc, dtype=torch.bool)
    kept[..., :budget] = True
    for i in range(budget, n):
        kept[..., i] = True
        heavy = kept.clone()
        heavy[..., i - recent + 1:] = False  # the recent words are never evicted
        kept.scatter_(-1, acc.masked_fill(~heavy, math.inf).argmin(-1, keepdim=True), False)
        keep[:, :, i] = kept
        p = scores[:, :, i].float().masked_fill(~repeat_kv(kept, groups), -math.inf).softmax(-1)
        acc += group_mean(p, groups)
    return keep


def snapkv_keep(scores: torch.Tensor, groups: int, budget: int, window: int = SNAPKV_WINDOW,
                kernel: int = SNAPKV_KERNEL) -> torch.Tensor:
    """SnapKV, with every word treated as the end of a prompt: keep the last `window` words, plus the earlier words
    the window's queries attend to most (summed over the window, max-pooled over neighbouring words)."""
    b, _, n, _ = scores.shape
    causal = _causal(n, scores.device)
    keep = causal.expand(b, scores.shape[1] // groups, n, n).clone()
    if budget >= n:
        return keep
    w = min(window, budget)
    c = group_mean(scores.float().softmax(-1), groups).cumsum(2)
    votes = c - F.pad(c, (0, 0, w, 0))[:, :, :n]  # row i: attention of queries i-w+1..i
    i, j = torch.arange(n, device=scores.device)[:, None], torch.arange(n, device=scores.device)
    in_window = (j > i - w) & causal
    votes = votes.masked_fill(in_window | ~causal, 0)  # as SnapKV, pool only the words before the window
    votes = F.max_pool1d(votes.reshape(-1, 1, n), kernel, 1, kernel // 2).view_as(votes)
    sel = votes.masked_fill(in_window | ~causal, -math.inf).topk(budget - w, -1).indices
    picked = torch.zeros_like(keep).scatter_(-1, sel, True) & causal  # rows < budget are overwritten below
    keep[:, :, budget:] = (picked | in_window)[:, :, budget:]
    return keep


def infinigen_keep(query: torch.Tensor, key: torch.Tensor, groups: int, budget: int,
                   partial: float = INFINIGEN_PARTIAL) -> torch.Tensor:
    """InfiniGen: guess each query's attention from `partial` of the query/key channels (those with the largest
    |q| x |k| over the words so far), and fetch the word itself plus the `budget` - 1 words with the highest guess.
    shortcut: channels picked from this text's own q/k, not InfiniGen's offline SVD skewing and previous-layer
    speculation; upgrade if the guess quality, not the budget, turns out to decide the result."""
    b, hq, n, d = query.shape
    k = repeat_kv(key, groups)
    c = max(1, round(partial * d))
    q = query.float()
    importance = q.abs().cumsum(2) * k.float().abs().cumsum(2)  # (b, h, n, d): words up to each query only
    channels = torch.zeros_like(q).scatter_(-1, importance.topk(c, -1).indices, 1.0)
    causal = _causal(n, query.device)
    eye = torch.eye(n, dtype=torch.bool, device=query.device)
    guess = group_mean((q * channels) @ k.float().mT, groups).masked_fill(~causal | eye, -math.inf)
    keep = causal.expand(b, hq // groups, n, n).clone()
    if budget >= n:
        return keep
    picked = torch.zeros_like(keep).scatter_(-1, guess.topk(budget - 1, -1).indices, True) | eye
    keep[:, :, budget:] = (picked & causal)[:, :, budget:]
    return keep


def kv_attention_forward(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                         attention_mask: torch.Tensor | None, scaling: float, dropout: float = 0.0,
                         **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager attention (as transformers' eager_attention_forward) with the layer's LayerKV applied: keys/values
    after RoPE rounded the QuaRot way, then the forgetting rule masks the keys each query may use. Forgetting runs
    only when queries and keys are the same text (a perplexity window); decoding steps attend to the whole cache.
    shortcut: per-word masks, not a cache that shrinks; speed is therefore not measured with a smaller cache."""
    spec: LayerKV = module.sdf_kv
    if spec.rotate_key_bits < 16:
        key = quarot_round(key, spec.rotate_key_bits)
    if spec.rotate_value_bits < 16:
        value = quarot_round(value, spec.rotate_value_bits)
    groups = module.num_key_value_groups
    n_q, n_k = query.shape[2], key.shape[2]
    scores = query @ repeat_kv(key, groups).transpose(2, 3) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[..., :n_k]
    elif n_q == n_k:
        scores = scores.masked_fill(~_causal(n_q, scores.device), -math.inf)
    budget = math.ceil(round(spec.keep_ratio * n_k, 9))  # round: 0.7 * 10 is 7.000000000000001
    if spec.forget is not None and n_q == n_k and budget < n_k:
        keep = (infinigen_keep(query, key, groups, budget) if spec.forget == "infinigen"
                else {"h2o": h2o_keep, "snapkv": snapkv_keep}[spec.forget](scores, groups, budget))
        scores = scores.masked_fill(~repeat_kv(keep, groups), -math.inf)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = (probs @ repeat_kv(value, groups)).transpose(1, 2).contiguous()
    return out, probs


AttentionInterface.register(SDF_KV, kv_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register(SDF_KV, eager_mask)


def attention_module(layer: nn.Module) -> nn.Module:
    attn = getattr(layer, "self_attn", None)
    if attn is None or not hasattr(attn, "num_key_value_groups"):
        raise ValueError("Stage 3 needs each decoder layer's attention at .self_attn (Llama, Mistral, Qwen)")
    return attn


@contextmanager
def kv_attention(model: nn.Module, specs: list[LayerKV]) -> Iterator[None]:
    """Run every attention layer through kv_attention_forward with its LayerKV while the context is open."""
    attns = [attention_module(layer) for layer in find_decoder_layers(model)]
    if len(attns) != len(specs):
        raise ValueError(f"{len(specs)} layer settings for {len(attns)} layers")
    cfg = model.config
    old = cfg._attn_implementation
    for a, s in zip(attns, specs):
        a.sdf_kv = s
    cfg._attn_implementation = SDF_KV
    try:
        yield
    finally:
        cfg._attn_implementation = old
        for a in attns:
            del a.sdf_kv


# ------------------------------------------------------------------------------------------------- KVQuant
def kmeans_1d(x: torch.Tensor, k: int, iters: int = 25) -> torch.Tensor:
    """k sorted levels fitting the 1-D sample x (Lloyd's algorithm, started at x's quantiles)."""
    x = x.float().flatten()
    levels = torch.quantile(x, torch.linspace(0, 1, k, device=x.device))
    for _ in range(iters):
        idx = torch.bucketize(x, (levels[1:] + levels[:-1]) / 2)  # nearest level, no (samples x levels) matrix
        total = torch.zeros(k, device=x.device).index_add_(0, idx, x)
        count = torch.zeros(k, device=x.device).index_add_(0, idx, torch.ones_like(x))
        levels = torch.where(count > 0, total / count.clamp(min=1), levels).sort().values
    return levels


def snap_to_levels(u: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Each entry of u replaced by the nearest of the sorted `levels`."""
    mids = (levels[1:] + levels[:-1]) / 2
    return levels[torch.bucketize(u.contiguous(), mids)]


def _normalize(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return 2 * (x - lo) / (hi - lo).clamp(min=1e-8) - 1


def _quantiles(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.float()  # float q, not a tensor: no host-to-device copy inside the forward (CUDA graphs)
    return (torch.quantile(x, KVQUANT_OUTLIERS / 2, dim=dim, keepdim=True),
            torch.quantile(x, 1 - KVQUANT_OUTLIERS / 2, dim=dim, keepdim=True))


@dataclass
class KVQuantLayer:
    key_lo: torch.Tensor  # (channels,) calibrated per-channel range; numbers outside it are outliers
    key_hi: torch.Tensor
    key_levels: dict[int, torch.Tensor]  # bits -> 2**bits levels in [-1, 1] (non-uniform, fitted by k-means)
    value_levels: dict[int, torch.Tensor]


@torch.no_grad()
def calibrate_kvquant(model: nn.Module, batches: list[torch.Tensor], key_bits: list[int], value_bits: list[int],
                      names: tuple[str, str], baseline_bits: int) -> list[KVQuantLayer]:
    """KVQuant's offline step: per layer, the per-channel key range (pre-RoPE) and, for each bit width the layer
    uses, non-uniform levels for keys and for (per-token normalised) values, from the calibration text.
    shortcut: plain k-means, not KVQuant's sensitivity-weighted k-means; upgrade if its levels look worse."""
    device = next(model.parameters()).device
    projs = [kv_projections(layer, names) for layer in find_decoder_layers(model)]
    seen: list[dict[str, list[torch.Tensor]]] = [{"key": [], "value": []} for _ in projs]
    total = sum(b.shape[0] * (b.shape[1] - 1) for b in batches)
    stride = max(1, math.ceil(total / KVQUANT_CALIB_TOKENS))

    def grab(i: int, kind: str) -> Callable:
        def hook(mod, args, out):
            seen[i][kind].append(out[:, 1::stride].reshape(-1, out.shape[-1]).float().cpu())  # first word: sink
        return hook

    with ExitStack() as stack:
        for i, (k, v) in enumerate(projs):
            for mod, kind in ((k, "key"), (v, "value")):
                stack.callback(mod.register_forward_hook(grab(i, kind)).remove)
        for b in batches:
            model(input_ids=b.to(device))
    out = []
    for i, s in enumerate(seen):
        keys, values = torch.cat(s["key"]).to(device), torch.cat(s["value"]).to(device)
        lo, hi = _quantiles(keys, 0)
        ku = _normalize(keys, lo, hi)
        ku = ku[(ku >= -1) & (ku <= 1)]
        vlo, vhi = _quantiles(values, -1)
        vu = _normalize(values, vlo, vhi)
        vu = vu[(vu >= -1) & (vu <= 1)]
        kb, vb = key_bits[i], value_bits[i]
        out.append(KVQuantLayer(lo[0], hi[0], {kb: kmeans_1d(ku, 2 ** kb)} if kb < baseline_bits else {},
                                {vb: kmeans_1d(vu, 2 ** vb)} if vb < baseline_bits else {}))
    return out


def kvquant_round(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """x (batch, tokens, channels) snapped to `levels` inside [lo, hi]; outliers and the first word (the attention
    sink, which KVQuant keeps in full precision) stay as they are."""
    u = _normalize(x.float(), lo, hi)
    q = (snap_to_levels(u.clamp(-1, 1), levels) + 1) / 2 * (hi - lo) + lo
    out = torch.where((u >= -1) & (u <= 1), q, x.float())
    out[:, 0] = x[:, 0].float()
    return out.to(x.dtype)


@contextmanager
def kvquant_hooks(model: nn.Module, layers: list[KVQuantLayer], key_bits: list[int], value_bits: list[int],
                  names: tuple[str, str], baseline_bits: int) -> Iterator[None]:
    """Keys rounded per channel before RoPE with calibrated ranges, values per token with their own range."""
    def key_hook(c: KVQuantLayer, bits: int) -> Callable:
        return lambda mod, args, out: kvquant_round(out, c.key_lo, c.key_hi, c.key_levels[bits])

    def value_hook(c: KVQuantLayer, bits: int) -> Callable:
        def hook(mod, args, out):
            lo, hi = _quantiles(out, -1)
            return kvquant_round(out, lo, hi, c.value_levels[bits])
        return hook

    with ExitStack() as stack:
        for c, kb, vb, layer in zip(layers, key_bits, value_bits, find_decoder_layers(model)):
            k, v = kv_projections(layer, names)
            if kb < baseline_bits:
                stack.callback(k.register_forward_hook(key_hook(c, kb)).remove)
            if vb < baseline_bits:
                stack.callback(v.register_forward_hook(value_hook(c, vb)).remove)
        yield
