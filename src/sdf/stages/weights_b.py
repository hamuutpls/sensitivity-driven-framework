"""More Stage 1 weight quantization methods (simulated: weights keep their dtype but only take the values the real
quantized model would store). Same contract as `gptq_` in weights.py: the per-layer bit width of the plan is used,
layers at or above `baseline_bits` are skipped, nothing is removed.

Each method ports the core algorithm of the user's fork (github.com/hamuutpls/<name>, commit and files in the
comment above it) to plain torch; the docstrings state what is simplified.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import torch
from torch import nn

from sdf.stage0.planner import CompressionPlan, require_bits_only
from sdf.stage0.sensitivity import find_decoder_layers
from sdf.stages.weights import gptq_quantize_, input_hessians

BILLM_FIXED_BITS = True  # BiLLM binarizes every layer the same way: it has no per-layer bit-width knob
VEC = 8  # AQLM group size and QuIP# lattice dimension
_GEN_SEED = 0


def _apply(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], baseline_bits: int, name: str,
           quantize: Callable[[torch.Tensor, torch.Tensor, int], None]) -> None:
    """quantize(weight, hessian, bits) rounds one Linear weight in place, layer by layer on the compressed model."""
    require_bits_only(plan, name)
    for lp, layer in zip(plan.layers, find_decoder_layers(model)):
        if lp.bit_width >= baseline_bits:
            continue
        for m, H in input_hessians(model, layer, batches).items():
            quantize(m.weight, H, lp.bit_width)


def _hinv_chol(H: torch.Tensor, W: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """Upper Cholesky factor of the damped inverse Hessian (as in GPTQ); zeroes the weights of dead inputs."""
    H = H.clone()
    dead = H.diagonal() == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H.diagonal().add_(damp * H.diagonal().mean())
    return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)


@torch.no_grad()
def _feedback_(W: torch.Tensor, H: torch.Tensor, step: Callable[[torch.Tensor, int], torch.Tensor],
               width: int = 1, block: int = 128) -> torch.Tensor:
    """GPTQ error feedback for an arbitrary rounding rule. `W` (out, in, float32) is consumed; `step(cols, c)` rounds
    the `width` columns starting at c (as updated so far). The error of each rounded group is spread over the columns
    after it. Returns the rounded matrix."""
    U = _hinv_chol(H, W)
    Q, E = torch.empty_like(W), torch.empty_like(W)
    cols = W.shape[1]
    for b1 in range(0, cols, block):
        b2 = min(b1 + block, cols)
        for c in range(b1, b2, width):
            e = min(c + width, b2)
            w = W[:, c:e]
            Q[:, c:e] = step(w, c)
            E[:, c:e] = torch.linalg.solve_triangular(U[c:e, c:e], w - Q[:, c:e], upper=True, left=False)
            W[:, e:b2] -= E[:, c:e] @ U[c:e, e:b2]
        W[:, b2:] -= E[:, b1:b2] @ U[b1:b2, b2:]
    return Q


def _orth(n: int, hadamard: bool, device) -> torch.Tensor:
    """Random orthogonal (n, n). Kronecker product of two small ones, so any n works (5632 = 64 x 88).
    hadamard: QuIP#'s randomized Hadamard transform (2**k Hadamard factor, random-orthogonal remainder, random signs);
    otherwise QuIP's butterfly: two random orthogonal (QR of Gaussian) factors, then a random permutation."""
    g = torch.Generator().manual_seed(_GEN_SEED + n)

    def qr(k: int) -> torch.Tensor:
        q, r = torch.linalg.qr(torch.randn(k, k, generator=g, dtype=torch.float64))
        return q * torch.sign(r.diagonal())

    if hadamard:
        a = n & -n  # largest power of two dividing n
        had = torch.ones(1, 1, dtype=torch.float64)
        while had.shape[0] < a:
            had = torch.cat([torch.cat([had, had], 1), torch.cat([had, -had], 1)])
        f1, f2 = had / math.sqrt(a), qr(n // a)
    else:
        a = max(d for d in range(1, math.isqrt(n) + 1) if n % d == 0)
        f1, f2 = qr(a), qr(n // a)
    signs = torch.randint(0, 2, (n,), generator=g).double() * 2 - 1
    q = (torch.kron(f1.contiguous(), f2.contiguous()) * signs).float()
    return (q if hadamard else q[torch.randperm(n, generator=g)]).to(device)


@torch.no_grad()
def _incoherent_(weight: torch.Tensor, H: torch.Tensor, hadamard: bool,
                 quantize: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
    """Replace `weight` by U^T quantize(U W V^T) V, with H rotated to V H V^T (y = W x = U^T (U W V^T)(V x))."""
    W = weight.data.float()
    U, V = _orth(W.shape[0], hadamard, W.device), _orth(W.shape[1], hadamard, W.device)
    Wq = quantize((U @ W @ V.T).contiguous(), V @ H @ V.T)
    weight.data.copy_((U.T @ Wq @ V).to(weight.dtype))


# Fork: QuIP/method.py (commit ac92cfc7a22f6100009e2caf53bb72257d3f3184): preproc (random projection + H
# normalisation) and postproc; the rounding of QuIP/gptq.py (LDLQ-equivalent) is `gptq_quantize_`.
def quip_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
          baseline_bits: int, int_zero: bool = True) -> None:
    """QuIP (Chee et al., 2023, "QuIP: 2-Bit Quantization of LLMs With Guarantees"): incoherence processing, then LDLQ.

    W and the Hessian are rotated by random orthogonal matrices on both sides, so no weight or Hessian direction is
    extreme; the Hessian is first scaled to trace n and given 0.01 I (as the fork does). The rotated weights are
    rounded with LDLQ, which is GPTQ's error feedback (`gptq_quantize_`, min/max grid with `group_size`), and rotated
    back, so the stored weight is the dequantized one.
    Simplified: the orthogonal matrices are a Kronecker product of two QR-of-Gaussian matrices plus a random
    permutation (the fork: butterfly of prime-factor blocks from scipy's special_ortho_group), the rounding grid is
    GPTQ's, no greedy local-search passes after LDLQ."""
    def quantize(Wt, Ht, bits):
        Ht = Ht * (len(Ht) / Ht.trace().clamp(min=1e-8)) + 1e-2 * torch.eye(len(Ht), device=Ht.device)
        gptq_quantize_(Wt, Ht, bits, group_size, int_zero)
        return Wt

    _apply(model, plan, batches, baseline_bits, "QuIP", lambda w, H, bits: _incoherent_(
        w, H, False, lambda Wt, Ht: quantize(Wt, Ht, bits)))


_NORM12 = [[3, 1, 1, 1, 3, 3, 3, 3], [1, 3, 1, 1, 3, 3, 3, 3], [1, 1, 3, 1, 3, 3, 3, 3], [1, 1, 1, 3, 3, 3, 3, 3],
           [3, 3, 3, 1, 3, 3, 1, 1], [3, 3, 3, 1, 3, 1, 3, 1], [3, 3, 3, 1, 1, 3, 3, 1], [3, 3, 3, 1, 3, 1, 1, 3],
           [3, 3, 3, 1, 1, 3, 1, 3], [3, 3, 3, 1, 1, 1, 3, 3], [3, 3, 1, 3, 3, 3, 1, 1], [3, 3, 1, 3, 3, 1, 3, 1],
           [3, 3, 1, 3, 1, 3, 3, 1], [3, 3, 1, 3, 3, 1, 1, 3], [3, 3, 1, 3, 1, 3, 1, 3], [3, 3, 1, 3, 1, 1, 3, 3],
           [3, 1, 3, 3, 3, 3, 1, 1], [3, 1, 3, 3, 3, 1, 3, 1], [3, 1, 3, 3, 1, 3, 3, 1], [3, 1, 3, 3, 3, 1, 1, 3],
           [3, 1, 3, 3, 1, 3, 1, 3], [1, 3, 3, 3, 1, 1, 3, 3], [1, 3, 3, 3, 3, 3, 1, 1], [1, 3, 3, 3, 3, 1, 3, 1],
           [1, 3, 3, 3, 1, 3, 3, 1], [1, 3, 3, 3, 3, 1, 1, 3], [1, 3, 3, 3, 1, 3, 1, 3], [1, 1, 3, 3, 1, 3, 3, 3],
           [3, 3, 1, 1, 3, 3, 3, 1]]  # the 29 norm-12 source vectors of E8P (times 2)


@lru_cache(maxsize=None)
def _e8p(device: str) -> torch.Tensor:
    """The exact E8P codebook (2**16, 8) of the fork: the 227 half-integer vectors of norm^2 <= 10 and the 29 of norm^2
    12 (as absolute values), every sign pattern whose number of flips has the parity of the vector's coordinate sum,
    each shifted by +1/4 and by -1/4."""
    g = torch.arange(4 ** 8)
    a = torch.stack([(g // 4 ** j) % 4 for j in range(8)], 1).float() + .5
    a = torch.cat([a[a.pow(2).sum(1) <= 10], torch.tensor(_NORM12) / 2])
    flips = (torch.arange(256)[:, None] >> torch.arange(8)) & 1
    ok = flips.sum(1)[None] % 2 == (a.sum(1).long() % 2)[:, None]
    cw = (a[:, None] * (1 - 2 * flips)[None].float())[ok]
    return torch.cat([cw + .25, cw - .25]).to(device)


@lru_cache(maxsize=None)
def _e8_one_bit(device: str) -> torch.Tensor:
    """The fork's 1-bit E8 grid (256, 8): 0, the 240 roots of E8 and 15 of the 16 vectors +-2 e_i."""
    ints = torch.zeros(0, 8)
    for i in range(8):
        for j in range(i + 1, 8):
            v = torch.zeros(4, 8)
            v[:, i], v[:, j] = torch.tensor([1., 1, -1, -1]), torch.tensor([1., -1, 1, -1])
            ints = torch.cat([ints, v])
    flips = (torch.arange(256)[:, None] >> torch.arange(8)) & 1
    halves = (.5 * (1 - 2 * flips))[flips.sum(1) % 2 == 0]
    norm4 = torch.cat([2 * torch.eye(8), -2 * torch.eye(8)[:7]])
    return torch.cat([torch.zeros(1, 8), ints, halves, norm4]).to(device)


def _nearest(X: torch.Tensor, C: torch.Tensor, half_sq: torch.Tensor) -> torch.Tensor:
    return torch.cat([(x @ C.T - half_sq).argmax(1) for x in X.split(16384)])


class _E8Residual:
    """Residual vector quantizer on E8 codebooks (the fork's rvq3bit/rvq4bit layout): bits b -> 8b bits per 8-vector,
    in stages of 16 bits (E8P, 2 bits per weight) and a last stage of 8 bits (1-bit E8) when 8b is not a multiple of
    16. Each stage has its own scale, searched on a sample for the smallest squared error."""

    def __init__(self, sample: torch.Tensor, bits: int):
        ks = [16] * (VEC * bits // 16) + ([8] if VEC * bits % 16 else [])
        self.stages, R = [], sample
        for k in ks:
            C = (_e8p if k == 16 else _e8_one_bit)(str(sample.device))
            half_sq = C.pow(2).sum(1) / 2
            base = R.pow(2).mean().sqrt() / C.pow(2).mean().sqrt()
            trials = [(base * g, R - C[_nearest(R / (base * g), C, half_sq)] * (base * g))
                      for g in torch.linspace(.6, 1.4, 5)]
            s, R = min(trials, key=lambda t: t[1].pow(2).sum().item())
            self.stages.append((C, half_sq, s))

    def __call__(self, V: torch.Tensor) -> torch.Tensor:
        out, R = torch.zeros_like(V), V
        for C, half_sq, s in self.stages:
            q = C[_nearest(R / s, C, half_sq)] * s
            out, R = out + q, R - q
        return out


# Fork: quip-sharp/lib/codebook/latticee8_padded12.py, latticee8_padded12_rvq3bit.py, latticee8_padded12_rvq4bit.py
# (codebooks), lib/algo/quip.py (incoherence) (commit 1d8f873e9a2a8b86b12bb1064c312c5689b77d98).
def quipsharp_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
               baseline_bits: int, int_zero: bool = True) -> None:
    """QuIP# (Tseng et al., 2024, "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattices"):
    randomized Hadamard incoherence, then vector quantization of 8-weight blocks on the E8 lattice codebooks of the
    fork (E8P for 2 bits; 3 bits: E8P + a 1-bit E8 residual stage; 4 bits: two E8P stages) with block-wise LDLQ error
    feedback (8 input columns at a time).

    Simplified: (1) the randomized Hadamard transform is Hadamard(2**k) (x) random orthogonal (QR) with random signs
    for the dimension n = 2**k * r (the fork uses Hadamard matrices of size 12, 20, 28 for the odd part); (2) the
    scale of every stage is searched per layer on a 512-vector sample instead of the fork's fixed opt_scale and
    opt_resid_scale; widths other than 2..4 split into 16-bit stages the same way (the fork has no such codebooks);
    (3) the feedback is GPTQ's block form, not the fork's LDL with a buffer; (4) no fine-tuning of the codebooks or
    of the remaining parameters. `group_size` and `int_zero` do not apply (no scalar grid)."""
    def quantize(Wt, Ht, bits):
        V = Wt.reshape(-1, VEC)
        g = torch.Generator().manual_seed(_GEN_SEED)
        q = _E8Residual(V[torch.randperm(len(V), generator=g)[:512].to(V.device)], bits)
        return _feedback_(Wt, Ht, lambda cols, c: q(cols.reshape(-1, VEC)).reshape(cols.shape), width=VEC)

    _apply(model, plan, batches, baseline_bits, "QuIP#", lambda w, H, bits: _incoherent_(
        w, H, True, lambda Wt, Ht: quantize(Wt, Ht, bits)))


@torch.no_grad()
def aqlm_quantize_(weight: torch.Tensor, books: int, iters: int = 12, sample: int = 2 ** 17) -> None:
    """Round `weight` (out, in) in place: each row is divided by its norm, every 8 consecutive inputs become the sum
    of `books` codewords of 2**8, codebook m being k-means of what the books before it left, and the row is
    multiplied by its norm again."""
    scale = weight.data.float().norm(dim=1, keepdim=True) + 1e-9
    V = (weight.data.float() / scale).reshape(-1, VEC)
    g = torch.Generator().manual_seed(_GEN_SEED)
    sub = V[torch.randperm(len(V), generator=g)[:sample].to(V.device)]
    R, Rs = V.clone(), sub.clone()
    for _ in range(books):
        C = Rs[:256].clone() if len(Rs) >= 256 else torch.cat([Rs, Rs.new_zeros(256 - len(Rs), VEC)])
        for _ in range(iters):
            a = _nearest(Rs, C, C.pow(2).sum(1) / 2)
            n = torch.zeros(256, device=C.device).index_add_(0, a, torch.ones_like(a, dtype=torch.float))
            new = torch.zeros_like(C).index_add_(0, a, Rs) / n.clamp(min=1)[:, None]
            C = torch.where(n[:, None] > 0, new, C)
        half_sq = C.pow(2).sum(1) / 2
        R -= C[_nearest(R, C, half_sq)]
        Rs -= C[_nearest(Rs, C, half_sq)]
    weight.data.copy_(((V - R).reshape(weight.shape) * scale).to(weight.dtype))


# Fork: AQLM/src/aq.py (QuantizedWeight scales, init_aq_kmeans) and AQLM/src/kmeans.py (fit_kmeans)
# (commit e79a896ed6656fe4ed06193d42d004e7d0bbdbb2).
def aqlm_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
          baseline_bits: int, int_zero: bool = True) -> None:
    """AQLM (Egiazarian et al., 2024, "Extreme Compression of LLMs via Additive Quantization"): every group of 8
    consecutive input weights is the sum of M codewords, one from each of M codebooks of 2**8 codewords, so M
    codebooks are M bits per weight (a plan width of b bits means M = b), times one scale per output row (the row
    norm, as in the fork with scale_nbits=0).

    Simplified: only the fork's initialisation, greedy residual k-means (random initial centres, 12 iterations
    instead of up to 1000 with convergence check, fitted on a random sample of 2**17 vectors, then every vector gets
    its nearest codeword). The k-means is not weighted by the layer's Hessian; no beam search over the codes, no
    fine-tuning of codebooks, scales or the model. `group_size` and `int_zero` do not apply."""
    _apply(model, plan, batches, baseline_bits, "AQLM", lambda w, H, bits: aqlm_quantize_(w, bits))


@torch.no_grad()
def pbllm_quantize_(weight: torch.Tensor, hessian: torch.Tensor, bits: int, high_bits: int = 8) -> torch.Tensor:
    """Round `weight` (out, in) in place and return the mask of binarized weights: the fraction 1 - r,
    r = (bits - 1) / (high_bits - 1), of the weights with the smallest w^2 / Hinv_jj^2 (over the whole matrix) are
    binarized to mean_row +- scale_row (mean and mean |w - mean| of the row's binarized weights), the others rounded
    to `high_bits` with a per-row min/max grid. Error feedback as in GPTQ."""
    W = weight.data.float().clone()
    U = _hinv_chol(hessian, W)
    score = W.pow(2) / U.diagonal().pow(2)[None]
    mask = torch.zeros(W.numel(), dtype=torch.bool, device=W.device)
    mask[score.flatten().argsort()[:round((1 - (bits - 1) / (high_bits - 1)) * W.numel())]] = True
    mask = mask.reshape(W.shape)
    top = 2 ** high_bits - 1
    lo, hi = W.amin(1, keepdim=True).clamp(max=0), W.amax(1, keepdim=True).clamp(min=0)
    scale = ((hi - lo) / top).clamp(min=1e-12)
    zero = (-lo / scale).round()
    n = mask.sum(1, keepdim=True).clamp(min=1)
    mean = (W * mask).sum(1, keepdim=True) / n
    alpha = ((W - mean).abs() * mask).sum(1, keepdim=True) / n

    def step(w: torch.Tensor, c: int) -> torch.Tensor:
        high = ((w / scale).round() + zero).clamp(0, top).sub(zero) * scale
        return torch.where(mask[:, c:c + 1], torch.sign(w - mean) * alpha + mean, high)

    weight.data.copy_(_feedback_(W, hessian, step).to(weight.dtype))
    return mask


# Fork: PB-LLM/gptq_pb/gptq.py (LowHighGPT.fasterquant), low_quant.py (xnor), high_quant.py
# (commit fe85da943d9df48ab6455d698f75406bb0bfefbc).
def pbllm_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
           baseline_bits: int, int_zero: bool = True) -> None:
    """PB-LLM (Shang et al., 2023, "PB-LLM: Partially Binarized Large Language Models"), post-training version of the
    fork (`--salient_metric hessian`, xnor binarization, 8 high bits): the salient fraction r = (b - 1) / 7 of the
    weights (largest w^2 / Hinv_jj^2) stays at 8 bits and the rest is binarized, so the layer averages
    8 r + (1 - r) = b bits.

    Simplified: the fork computes the row mean and scale of the binarized weights over all columns including the
    zeros of the other weights; here only over the binarized ones; one group per matrix (the fork's default
    groupsize -1; `group_size` and `int_zero` do not apply); the 8-bit grid is per-row min/max with an integer zero.
    No quantization-aware training (the paper's other half)."""
    _apply(model, plan, batches, baseline_bits, "PB-LLM", lambda w, H, bits: pbllm_quantize_(w, H, bits))


def _braq(x: torch.Tensor, mask: torch.Tensor, order: int) -> torch.Tensor:
    """BiLLM's residual binarization of the masked entries of x (rows along dim -2): `order` times, subtract the row
    mean of the residual and binarize it to sign * mean |.| (+ the mean). Leading dims of `mask` are candidates."""
    total = torch.zeros(mask.shape, device=x.device)
    n = mask.sum(-1, keepdim=True).clamp(min=1)
    for _ in range(order):
        res = (x - total) * mask
        mean = res.sum(-1, keepdim=True) / n
        res = (res - mean) * mask
        total = total + (torch.sign(res) * (res.abs().sum(-1, keepdim=True) / n) + mean) * mask
    return total


@torch.no_grad()
def billm_quantize_(weight: torch.Tensor, hessian: torch.Tensor, block: int = 128, up_lim: int = 50) -> None:
    """Round `weight` (out, in) in place the BiLLM way; see `billm_`."""
    W = weight.data.float().clone()
    U = _hinv_chol(hessian, W)
    cols, d = W.shape[1], U.diagonal()
    for b1 in range(0, cols, block):
        b2 = min(b1 + block, cols)
        Wb = W[:, b1:b2]
        target = Wb.pow(2) / d[b1:b2].pow(2)[None]
        top = target.sum(0).topk(min(up_lim, b2 - b1)).indices
        sal = torch.zeros(len(top) - 1, *Wb.shape, dtype=torch.bool, device=W.device)  # candidate i: i + 1 columns
        for i in range(len(sal)):
            sal[i][:, top[:i + 1]] = True
        err = (target - _braq(target, sal, 2) - _braq(target, ~sal, 2)).pow(2).mean((-2, -1))
        m3 = sal[err.argmin()]
        rest = (target * ~m3).abs()
        splits = torch.quantile(rest.flatten()[:2 ** 24], torch.linspace(.1, .9, 81, device=W.device))
        m1 = (rest < splits[:, None, None]) & ~m3
        m2 = ~m3 & ~m1
        err = (target - _braq(target, m1, 1) - _braq(target, m2, 1) - _braq(target, m3, 2)).pow(2).mean((-2, -1))
        m1, m2 = m1[err.argmin()], m2[err.argmin()]
        Q = _braq(Wb, m1, 1) + _braq(Wb, m2, 1) + _braq(Wb, m3, 2)
        E = (Wb - Q) / d[b1:b2][None]
        W[:, b1:b2] = Q
        W[:, b2:] -= E @ U[b1:b2, b2:]
    weight.data.copy_(W.to(weight.dtype))


# Fork: BiLLM/bigptq.py (BRAGPTQ.fasterquant), binary.py (high_order_residual), utils/autosearch.py
# (structural_searching), utils/structure.py, utils/mask.py (commit dc137ebbf62d4b31e8a82ba6bf9e18a51a298dcb).
def billm_(model: nn.Module, plan: CompressionPlan, batches: list[torch.Tensor], group_size: int,
           baseline_bits: int, int_zero: bool = True) -> None:
    """BiLLM (Huang et al., 2024, "BiLLM: Pushing the Limit of Post-Training Quantization for LLMs"): about 1 bit per
    weight. Every layer below `baseline_bits` is quantized the same way, whatever its planned bit width
    (BILLM_FIXED_BITS): the method has no bit-width knob.

    Per block of 128 input columns, as in the fork (salient metric "hessian", partition 3, orders (1, 1, 2)): the
    columns with the largest sum of w^2 / Hinv_jj^2 are salient (their number, 1..49, searched for the smallest
    squared error) and binarized twice (residual approximation); the other weights are split by a break point
    (81 percentiles from 10 to 90 % of the same score, searched the same way) into two groups, each binarized with
    its own row-wise mean and scale. The error of the block is then fed to the columns after it, as in the fork
    (no feedback inside a block, as in the fork).

    Simplified: the searches run batched over the candidates (same values); the quantile uses at most 2**24 values
    of the block; `group_size` and `int_zero` do not apply."""
    _apply(model, plan, batches, baseline_bits, "BiLLM", lambda w, H, bits: billm_quantize_(w, H))


_NOTE = "plain-torch port of the fork's core algorithm, see the docstring in sdf.stages.weights_b"
WEIGHT_METHODS_B = {
    "aqlm": (aqlm_, "AQLM", ("gptq_groupsize",), f"additive codebooks, greedy residual k-means init ({_NOTE})", True,
             "no beam search, no codebook/scale fine-tuning, k-means 12 iterations on a sample, not Hessian-weighted"),
    "quip": (quip_, "QuIP", ("gptq_groupsize",), f"random orthogonal incoherence + LDLQ ({_NOTE})", True,
             "Kronecker QR + permutation instead of the fork's butterfly, no greedy local-search passes"),
    "quipsharp": (quipsharp_, "QuIP#", ("gptq_groupsize",), f"Hadamard incoherence + E8P lattice VQ ({_NOTE})", True,
                  "per-layer searched stage scales instead of fixed opt_scale, Hadamard (x) QR for non-power-of-2 "
                  "dims, GPTQ block feedback, no fine-tuning"),
    "pbllm": (pbllm_, "PB-LLM", ("gptq_groupsize",), f"partial binarization, salient weights at 8 bits ({_NOTE})", True,
              "row mean/scale over binarized weights only, one group per matrix, no quantization-aware training"),
    "billm": (billm_, "BiLLM", ("gptq_groupsize",), f"about 1-bit residual binarization, ignores plan bits ({_NOTE})",
              False, ""),
}
