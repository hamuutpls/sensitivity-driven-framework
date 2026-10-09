"""Stage 1 activation quantization methods: SmoothQuant, QuaRot, SpinQuant, RPTQ. Weights stay unquantized here.

Each method is fn(model, plan, batches, group_size) -> context manager. It first rewrites `model` in place into an
equivalent model (same logits in full precision; the caller passes a fresh copy), then returns a context manager
that rounds the Linear inputs of every decoder layer to `plan.layers[i].act_bits` while it is open (layers at 16
bits or more are left alone). Rounding is per token with `group_size` channels per scale (`quantize_inputs`),
except RPTQ, which uses static per-cluster scales.

Each method cites the file of its fork (clone in /tmp/forks/<name>) and the clone's commit.
"""

from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from typing import ContextManager, Iterator

import torch
from torch import nn

from sdf.stage0.activation import ActivationPlan, quantize_inputs
from sdf.stage0.sensitivity import find_decoder_layers, minmax_grid, snap

_FP = 16  # bit width at which a layer is not quantized


@contextmanager
def _round_inputs(model: nn.Module, plan: ActivationPlan, group_size: int) -> Iterator[None]:
    with ExitStack() as stack:
        for lp, layer in zip(plan.layers, find_decoder_layers(model)):
            if lp.act_bits < _FP:
                stack.enter_context(quantize_inputs(layer, lp.act_bits, group_size))
        yield


def _decoder_linears(model: nn.Module) -> list[list[nn.Linear]]:
    return [[m for m in layer.modules() if isinstance(m, nn.Linear)] for layer in find_decoder_layers(model)]


@torch.no_grad()
def _run(model: nn.Module, batches: list[torch.Tensor], linears: list[nn.Linear], on_input) -> None:
    """Run `batches` through the model, calling on_input(linear, input) for each of `linears`."""
    handles = [m.register_forward_pre_hook(lambda m, args: on_input(m, args[0])) for m in linears]
    device = next(model.parameters()).device
    try:
        for b in batches:
            model(input_ids=b.to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def _channel_range(model: nn.Module, batches: list[torch.Tensor],
                   linears: list[nn.Linear]) -> dict[nn.Linear, tuple[torch.Tensor, torch.Tensor]]:
    """Per input channel (min, max) of each Linear's input over all calibration tokens."""
    out: dict[nn.Linear, tuple[torch.Tensor, torch.Tensor]] = {}

    def collect(m: nn.Linear, x: torch.Tensor) -> None:
        x = x.reshape(-1, m.in_features).float()
        lo, hi = x.amin(0), x.amax(0)
        out[m] = (torch.minimum(out[m][0], lo), torch.maximum(out[m][1], hi)) if m in out else (lo, hi)

    _run(model, batches, linears, collect)
    return out


# ------------------------------------------------------------------------------------------------ SmoothQuant
# Fork hamuutpls/smoothquant @ c61476d: smoothquant/smooth.py smooth_ln_fcs_llama_like.
@torch.no_grad()
def smoothquant(model: nn.Module, plan: ActivationPlan, batches: list[torch.Tensor], group_size: int,
                alpha: float = 0.5) -> ContextManager[None]:
    """SmoothQuant (Xiao et al., 2023): s_j = max|X_j|^a / max|W_j|^(1-a); x / s is folded into the RMSNorm before
    q/k/v and gate/up and the weight columns are multiplied by s. As in the repo for Llama, o_proj and down_proj
    are not smoothed (nothing before them can absorb the division); they are only rounded. Simplified: the max
    |X_j| comes from `batches` on the original model (the repo uses 512 Pile samples)."""
    layers = find_decoder_layers(model)
    first = [(l.self_attn.q_proj, [l.self_attn.q_proj, l.self_attn.k_proj, l.self_attn.v_proj], l.input_layernorm)
             for l in layers] + [(l.mlp.gate_proj, [l.mlp.gate_proj, l.mlp.up_proj], l.post_attention_layernorm)
                                 for l in layers]
    ranges = _channel_range(model, batches, [f[0] for f in first])
    for probe, fcs, norm in first:
        act = torch.maximum(-ranges[probe][0], ranges[probe][1]).to(norm.weight.dtype)
        w = torch.cat([fc.weight.abs().amax(0, keepdim=True) for fc in fcs]).amax(0).clamp(min=1e-5)
        s = (act.pow(alpha) / w.pow(1 - alpha)).clamp(min=1e-5)
        norm.weight.div_(s)
        for fc in fcs:
            fc.weight.mul_(s)
    return _round_inputs(model, plan, group_size)


# ------------------------------------------------------------------------------------------------ rotations
def _orthogonal(n: int, gen: torch.Generator, device: torch.device | str = "cpu") -> torch.Tensor:
    """Random orthogonal n x n: a Hadamard matrix with random column signs when n is a power of two, otherwise
    a QR-based random orthogonal matrix (Hadamard matrices of other sizes are not built here). The QR runs on
    `device` (a 5632 x 5632 float64 QR takes tens of seconds on the CPU)."""
    if n & (n - 1) == 0:
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)])
        return (h / math.sqrt(n) * (torch.randint(0, 2, (n,), generator=gen) * 2 - 1)).float()
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64).to(device))
    return (q * r.diagonal().sign()).float()


def _rot_out(w: torch.Tensor, q: torch.Tensor) -> None:
    """Rotate the output rows of `w` in blocks of q.shape[0]: W <- Q^T W."""
    n = q.shape[0]
    w.copy_(torch.einsum("ab,nac->nbc", q.to(w.device), w.float().view(-1, n, w.shape[1])).reshape(w.shape))


def _rot_in(w: torch.Tensor, q: torch.Tensor) -> None:
    """Rotate the input columns of `w` in blocks of q.shape[0]: W <- W Q."""
    n = q.shape[0]
    w.copy_((w.float().view(w.shape[0], -1, n) @ q.to(w.device)).reshape(w.shape))


@torch.no_grad()
def _fuse_norms(model: nn.Module) -> None:
    """Fold every RMSNorm weight into the Linears after it and set the norm weight to 1."""
    for l in find_decoder_layers(model):
        for norm, fcs in ((l.input_layernorm, [l.self_attn.q_proj, l.self_attn.k_proj, l.self_attn.v_proj]),
                          (l.post_attention_layernorm, [l.mlp.gate_proj, l.mlp.up_proj])):
            for fc in fcs:
                fc.weight.mul_(norm.weight)
            norm.weight.fill_(1)
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())  # untie from embed_tokens
    model.lm_head.weight.mul_(model.model.norm.weight)
    model.model.norm.weight.fill_(1)


@torch.no_grad()
def _rotate(model: nn.Module, q: torch.Tensor, seed: int) -> None:
    """Rotate the (norm-fused) residual stream by `q` (hidden x hidden), plus the fixed rotations QuaRot adds:
    a per-head rotation on v_proj outputs / o_proj inputs and an online rotation before down_proj."""
    gen = torch.Generator().manual_seed(seed)
    cfg = model.config
    device = model.model.embed_tokens.weight.device
    r2 = _orthogonal(cfg.hidden_size // cfg.num_attention_heads, gen, device)
    online = _orthogonal(cfg.intermediate_size, gen, device)  # one matrix, reused by every layer
    _rot_in(model.model.embed_tokens.weight, q)
    _rot_in(model.lm_head.weight, q)
    for l in find_decoder_layers(model):
        a, m = l.self_attn, l.mlp
        for fc in (a.q_proj, a.k_proj, a.v_proj, m.gate_proj, m.up_proj):
            _rot_in(fc.weight, q)
        _rot_out(a.v_proj.weight, r2)
        _rot_in(a.o_proj.weight, r2)
        _rot_out(a.o_proj.weight, q)
        _rot_in(m.down_proj.weight, online)
        _rot_out(m.down_proj.weight, q)
        m.down_proj.register_buffer("online_rotation", online.to(m.down_proj.weight.device), persistent=False)
        m.down_proj.register_forward_pre_hook(
            lambda mod, args: (args[0] @ mod.online_rotation.to(args[0].dtype),) + tuple(args[1:]))


# Fork hamuutpls/QuaRot @ 5008669: fake_quant/rotation_utils.py (fuse_layer_norms, rotate_*), hadamard_utils.py.
def quarot(model: nn.Module, plan: ActivationPlan, batches: list[torch.Tensor], group_size: int,
           seed: int = 0) -> ContextManager[None]:
    """QuaRot (Ashkboos et al., 2024): fuse RMSNorm weights, rotate the residual stream with a random Hadamard
    matrix, then round the Linear inputs. Simplified: the rotation before down_proj is a random orthogonal matrix
    (intermediate size is not a power of two) and the o_proj input gets only a per-head rotation (the paper's
    cross-head Hadamard is not applied); the rotations are plain matmuls, not fast Hadamard kernels."""
    _fuse_norms(model)
    _rotate(model, _orthogonal(model.config.hidden_size, torch.Generator().manual_seed(seed),
                               model.model.embed_tokens.weight.device), seed + 1)
    return _round_inputs(model, plan, group_size)


# ------------------------------------------------------------------------------------------------ SpinQuant
def _fake_round(z: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Per-token rounding as `round_to_nearest` with a straight-through rounding, so the grid (min/max) carries
    the gradient."""
    n = z.shape[-1]
    g = group_size if 0 < group_size and n % group_size == 0 else n
    z = z.reshape(*z.shape[:-1], n // g, g)
    scale, zero = minmax_grid(z, bits, False)
    t = z / scale + zero
    return ((t + (t.round() - t).detach() - zero) * scale).reshape(*z.shape[:-2], n)


def _cayley_step(r: torch.Tensor, grad: torch.Tensor, lr: float) -> torch.Tensor:
    """One step on the orthogonal group: R <- (I + a/2 A)^-1 (I - a/2 A) R, A = G R^T - R G^T (skew), the step
    normalised so each step turns R by about `lr` (Frobenius)."""
    a = grad @ r.T - r @ grad.T
    a = a * (lr / (a.norm() + 1e-12))
    eye = torch.eye(r.shape[0], device=r.device)
    return torch.linalg.solve(eye + a / 2, (eye - a / 2) @ r)


# Fork hamuutpls/SpinQuant @ 8f47aa3: train_utils/optimizer.py (SGDG, Cayley_loop), same rotation structure as QuaRot.
def spinquant(model: nn.Module, plan: ActivationPlan, batches: list[torch.Tensor], group_size: int,
              steps: int = 100, lr: float = 0.1, max_tokens: int = 2048, seed: int = 0) -> ContextManager[None]:
    """SpinQuant (Liu et al., 2024): QuaRot's structure with the residual rotation R1 learned on the orthogonal
    group by Cayley steps, starting from the QuaRot Hadamard. Simplified: the loss is the squared error between
    rounded and unrounded q/k/v/gate/up inputs (the only inputs R1 changes), summed over layers, on up to
    `max_tokens` calibration tokens per layer (the paper minimises the end loss); plain normalised Cayley steps,
    no momentum; R2 (per head) and the down_proj rotation stay fixed random orthogonal matrices instead of learned."""
    _fuse_norms(model)
    probes, bits = [], []
    for lp, l in zip(plan.layers, find_decoder_layers(model)):
        if lp.act_bits < _FP:
            probes += [l.self_attn.q_proj, l.mlp.gate_proj]
            bits += [lp.act_bits] * 2
    xs: dict[nn.Linear, list[torch.Tensor]] = {}
    _run(model, batches, probes, lambda m, x: xs.setdefault(m, []).append(x.reshape(-1, m.in_features).float()))
    gen = torch.Generator().manual_seed(seed)
    data = []
    for m, b in zip(probes, bits):
        x = torch.cat(xs[m])
        data.append((x[torch.randperm(len(x), generator=gen)[:max_tokens]], b))
    device = data[0][0].device if data else next(model.parameters()).device
    r = _orthogonal(model.config.hidden_size, gen, device)
    best, best_loss = r, float("inf")
    for _ in range(steps if data else 0):
        r.requires_grad_(True)
        loss = sum((_fake_round(x @ r, b, group_size) - x @ r).pow(2).mean() for x, b in data)
        (grad,) = torch.autograd.grad(loss, r)
        r = r.detach()
        if loss.item() < best_loss:
            best, best_loss = r, loss.item()
        r = _cayley_step(r, grad, lr)
    _rotate(model, best, seed + 1)
    return _round_inputs(model, plan, group_size)


# ------------------------------------------------------------------------------------------------ RPTQ
def _kmeans(points: torch.Tensor, k: int, gen: torch.Generator, iters: int = 20) -> torch.Tensor:
    """Cluster label of each row of `points` (Lloyd's algorithm from random rows; empty clusters keep their centre)."""
    centers = points[torch.randperm(len(points), generator=gen)[:k]].clone()
    for _ in range(iters):
        labels = torch.cdist(points, centers).argmin(1)
        for c in range(len(centers)):
            if (labels == c).any():
                centers[c] = points[labels == c].mean(0)
    return torch.cdist(points, centers).argmin(1)


# Fork hamuutpls/RPTQ4LLM @ 96190691: quantize/reorder_utils.py (tensor_calc_reorder_index: k-means on [min, max]).
@torch.no_grad()
def rptq(model: nn.Module, plan: ActivationPlan, batches: list[torch.Tensor], group_size: int,
         clusters: int = 32, seed: int = 0) -> ContextManager[None]:
    """RPTQ (Yuan et al., 2023): channels of each Linear input are clustered by their calibration [min, max] and
    every cluster is rounded with one static min/max scale. Simplified: the channels are not physically reordered
    (that only matters for kernel layout; here a per-channel scale tensor does the same), the clusters come from
    k-means on the original model (the repo also fuses the reorder into LayerNorm / weights), `group_size` is unused
    and only decoder Linear inputs are rounded (no KV cache, as in this stage)."""
    gen = torch.Generator().manual_seed(seed)
    linears = [(lp.act_bits, m) for lp, ms in zip(plan.layers, _decoder_linears(model)) if lp.act_bits < _FP
               for m in ms]
    ranges = _channel_range(model, batches, [m for _, m in linears])
    labels = {}
    for _, m in linears:
        lo, hi = ranges[m]
        labels[m] = _kmeans(torch.stack([lo, hi], 1), min(clusters, m.in_features), gen)

    def make_hook(bits: int, m: nn.Linear):
        lab, (lo, hi) = labels[m], ranges[m]
        k = int(lab.max()) + 1
        c_lo = torch.zeros(k, device=lo.device).scatter_reduce(0, lab, lo, "amin", include_self=False)
        c_hi = torch.zeros(k, device=lo.device).scatter_reduce(0, lab, hi, "amax", include_self=False)
        scale, zero = minmax_grid(torch.stack([c_lo, c_hi], 1), bits, False)
        scale, zero = scale[lab, 0], zero[lab, 0]
        return lambda mod, args: (snap(args[0].float(), scale, zero, bits).to(args[0].dtype),) + tuple(args[1:])

    @contextmanager
    def quantized() -> Iterator[None]:
        handles = [m.register_forward_pre_hook(make_hook(b, m)) for b, m in linears]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    return quantized()


# name: (fn, label, library note, what differs from the fork / paper)
ACTIVATION_METHODS = {
    "smoothquant": (smoothquant, "SmoothQuant", "in repo (torch), after mit-han-lab/smoothquant",
                    "o_proj and down_proj not smoothed (as in the repo for Llama); max from the calibration batches"),
    "quarot": (quarot, "QuaRot", "in repo (torch matmul rotations), after QuaRot",
               "random orthogonal (not Hadamard) before down_proj; per-head rotation only on o_proj input; no fast kernels"),
    "spinquant": (spinquant, "SpinQuant", "in repo (torch, Cayley SGD), after SpinQuant",
                  "only R1 learned, on a proxy loss (rounding error of q/k/v/gate/up inputs) instead of the end loss"),
    "rptq": (rptq, "RPTQ", "in repo (torch, k-means), after RPTQ4LLM",
             "channels not physically reordered (per-channel scale tensor instead); weight-fused reorder omitted"),
}
