"""Checks behind the 2026-10-05 review of the Stage 0 report (items 1, 2, 13, 14, 15, 16, 18, 20).

One-off study, kept so the report's numbers can be reproduced. Run from the folder main.py's OUTPUT_ROOT is relative
to (on the host PC: Desktop/Thesis), e.g.  python sensitivity-driven-framework/scripts/review_20261005.py [sections]
Sections: probe (16), prune (1, 2, 20), bits (13; uses probe), act (14), kv (15), hessian (18). Default: all, in
that order.
Writes <OUTPUT_ROOT>/review-20261005/<section>.json; per-window losses go to <section>_nll.json.
"""

from __future__ import annotations

import functools
import json
import math
import random
import statistics
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import build_config  # noqa: E402
from sdf.data import calibration_batches, eval_windows, load_texts  # noqa: E402
from sdf.eval.metrics import perplexity  # noqa: E402
from sdf.run import start_run  # noqa: E402
from sdf.search_space import SEARCH_SPACE  # noqa: E402
from sdf.stage0.activation import (ActivationLayerPlan, ActivationPlan, ActivationProfile,  # noqa: E402
                                   activation_plan_from_weights, plan_activations, predicted_rise,
                                   profile_activations, quantize_inputs, uniform_activation_plan)
from sdf.stage0.kv_cache import _coverage, eager_attention, keep_ratio_for  # noqa: E402
from sdf.stage0.planner import (CompressionPlan, LayerPlan, guarded_layers,  # noqa: E402
                                plan_compression, predict_cost, uniform_plan)
from sdf.stage0.prune_sweep import apply_plan  # noqa: E402
from sdf.stage0.run import _ModelHandle, activation_profile_key, load_profile  # noqa: E402
from sdf.stage0.sensitivity import find_decoder_layers, normalize, profile_by_ablation  # noqa: E402
from sdf.stage0.sweep import _spearman  # noqa: E402
from sdf.utils.env import environment_info, resolve_device  # noqa: E402

SEEDS = [0, 1, 2]
MODES = ("free", "bitmask", "dense")

cfg0 = build_config().with_overrides({"run.run_id": "review-20261005"})
s0 = cfg0.stage0
cand = SEARCH_SPACE.make(cfg0.hyperparams)
GS, PR = cand["gptq_groupsize"], cand["prune_ratio_aggressive"]
ctx0 = start_run(cfg0)
OUT = ctx0.run_dir
device = resolve_device(cfg0.model.device)
handle = _ModelHandle(ctx0, device)
loader = functools.partial(load_texts, cfg0.data.sources)
_windows: list = []
_evals: dict = {}


def ctx_for(seed: int):
    return start_run(cfg0.with_overrides({"run.seed": seed})) if seed else ctx0


def windows():
    if not _windows:
        _windows.extend(eval_windows(loader(cfg0.eval.dataset, "test"), handle.tokenizer, cfg0.eval.seq_len,
                                     cfg0.eval.max_windows))
    return _windows


def removal(seed: int):
    return load_profile(ctx_for(seed), cand, handle, loader, score="layer_removal")[0]


def size(plan: CompressionPlan, profile, mode: str) -> float:
    return predict_cost(plan, profile, GS, s0.group_overhead_bits, s0.baseline_bits, mode).weight_memory_gb


def measure(apply) -> dict:
    """ppl and per-window losses on both halves with `apply(model)` (a context manager) active."""
    m = handle.model(cfg0.model.dtype)
    val, held = windows()
    with apply(m):
        (pv, nv), (ph, nh) = perplexity(m, val, device), perplexity(m, held, device)
    return {"ppl_val": pv, "ppl_heldout": ph, "nll_val": nv, "nll_heldout": nh}


def eval_plan(plan: CompressionPlan) -> dict:
    key = tuple((lp.bit_width, round(lp.pruning_ratio, 6)) for lp in plan.layers)
    if key not in _evals:
        t0 = time.perf_counter()
        _evals[key] = measure(lambda m: apply_plan(m, plan, GS, True, s0.baseline_bits, True))
        print(f"  eval {len(_evals)}: {_evals[key]['ppl_val']:.3f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return _evals[key]


def diff_ci(a: dict, b: dict, half: str, n: int = 2000) -> list[float]:
    """95% paired-bootstrap interval of ppl(a) - ppl(b) over the windows of one half (window-sampling noise)."""
    x, y = a[f"nll_{half}"], b[f"nll_{half}"]
    rng = random.Random(0)
    d = sorted(math.exp(statistics.fmean(x[i] for i in idx)) - math.exp(statistics.fmean(y[i] for i in idx))
               for idx in ([rng.randrange(len(x)) for _ in x] for _ in range(n)))
    return [d[int(0.025 * n)], d[int(0.975 * n)]]


def summary(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("nll")}


def ratio_for(target_gb: float, profile, mode: str) -> float | None:
    """Even pruning ratio whose standard 4-bit plan has `target_gb` predicted size (None if unreachable)."""
    f = lambda r: size(uniform_plan([0.0] * profile.num_layers, s0.uniform_bits, r), profile, mode)  # noqa: E731
    lo, hi = 0.0, 0.99
    if not f(hi) <= target_gb <= f(0.0):
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > target_gb else (lo, mid)
    return hi


def save(name: str, data: dict, nll: dict | None = None) -> None:
    (OUT / f"{name}.json").write_text(json.dumps({"environment": environment_info(), **data}, indent=2),
                                      encoding="utf-8")
    if nll:
        (OUT / f"{name}_nll.json").write_text(json.dumps(nll), encoding="utf-8")
    print(f"wrote {OUT / name}.json", flush=True)


# ------------------------------------------------------------------------------------------- sections
def prune_section() -> None:
    """Items 1, 2, 20: standard + even pruning in one code path; plans vs standard pruned to the same size; seeds."""
    prof0 = removal(0)
    n = prof0.num_layers
    std = {r: eval_plan(uniform_plan([0.0] * n, s0.uniform_bits, r)) for r in (0.0, 0.05, 0.1, 0.2)}
    out = {"standard": {f"{r:g}": {**summary(v), "gb": {m: size(uniform_plan([0.0] * n, 4, r), prof0, m)
                                                         for m in MODES}} for r, v in std.items()}, "plans": []}
    nll = {f"std_{r:g}": v for r, v in std.items()}
    for seed in SEEDS:
        prof = removal(seed)
        scores = normalize(prof.raw_scores, s0.normalization)
        g5, g8 = guarded_layers(prof.raw_scores, 5), guarded_layers(prof.raw_scores, 8)
        bitmask_cost = functools.partial(predict_cost, profile=prof, group_size=GS, sparse_storage="bitmask",
                                         group_overhead_bits=s0.group_overhead_bits, baseline_bits=s0.baseline_bits)
        plans = {"t0.8_g5": plan_compression(scores, 0.8, PR, s0.protected_bits, s0.compressed_bits, g5),
                 "t0.9_g8": plan_compression(scores, 0.9, PR, s0.protected_bits, s0.compressed_bits, g8)}
        for name, plan in plans.items():
            res = eval_plan(plan)
            row = {"seed": seed, "plan": name, "protected": plan.protected_layers, "guard": sorted(plan.guarded_layers),
                   "gb": {m: size(plan, prof, m) for m in MODES}, **summary(res), "vs": {}}
            for mode in ("free", "bitmask"):
                r = ratio_for(row["gb"][mode], prof, mode)
                ref = eval_plan(uniform_plan([0.0] * n, s0.uniform_bits, r)) if r is not None else std[0.0]
                row["vs"][mode] = {"std_prune_ratio": r if r is not None else 0.0,
                                   "std_gb": size(uniform_plan([0.0] * n, 4, r or 0.0), prof, mode),
                                   "std_ppl_val": ref["ppl_val"], "std_ppl_heldout": ref["ppl_heldout"],
                                   "diff_ci_val": diff_ci(res, ref, "val"), "diff_ci_heldout": diff_ci(res, ref, "heldout")}
            nll[f"{name}_seed{seed}"] = res
            out["plans"].append(row)
    save("prune", out, {k: {h: v[f"nll_{h}"] for h in ("val", "heldout")} for k, v in nll.items()})


def bits_section() -> None:
    """Item 13: bits only, no pruning, exactly the standard size: k layers up to 5 bits, k down to 3."""
    prof = removal(0)
    n = prof.num_layers
    probe = OUT / "probe.json"
    orders = {"removal": sorted(range(n), key=lambda i: prof.raw_scores[i], reverse=True)}
    orders["removal_reversed"] = orders["removal"][::-1]
    if probe.exists():  # one-layer 3-bit damage (section "probe"), the measure meant for bits
        rise = json.loads(probe.read_text(encoding="utf-8"))["mean_rise"]["3"]
        orders["one_layer_3bit"] = sorted(range(n), key=lambda i: rise[i], reverse=True)
    for s in range(3):
        orders[f"random{s}"] = random.Random(s).sample(range(n), n)
    std = eval_plan(uniform_plan([0.0] * n, 4, 0.0))
    rows = []
    for k in (2, 4):
        for name, order in orders.items():
            bits = {i: 5 for i in order[:k]} | {i: 3 for i in order[-k:]}
            plan = CompressionPlan(tuple(LayerPlan(i, bits.get(i, 4), 0.0, False, 0.0) for i in range(n)),
                                   "bits_only", None, 0.0)
            assert abs(size(plan, prof, "free") - size(uniform_plan([0.0] * n, 4, 0.0), prof, "free")) < 1e-9
            res = eval_plan(plan)
            rows.append({"k": k, "guide": name, "up_5bit": order[:k], "down_3bit": order[-k:], **summary(res),
                         "diff_ci_val": diff_ci(res, std, "val"), "diff_ci_heldout": diff_ci(res, std, "heldout")})
    save("bits", {"standard": summary(std), "gb": size(uniform_plan([0.0] * n, 4, 0.0), prof, "free"), "rows": rows})


def act_section() -> None:
    """Item 14: activation plans applied to all layers at once (weights untouched), incl. 6-bit baselines."""
    key = activation_profile_key(ctx0, cand)

    def compute():
        b = calibration_batches(loader(cand["calib_dataset"], "train"), handle.tokenizer, s0.act_calib_samples,
                                cfg0.calibration.seq_len, cfg0.calibration.batch_size, cfg0.run.seed)
        return profile_activations(handle.model(s0.profile_dtype), b, s0.act_bits_options, s0.act_group_size,
                                   device=device, meta=key).to_dict()

    prof = ActivationProfile.from_dict(ctx0.cache.get_or_compute("activation_profile", key, compute)[0])
    rem = removal(0)
    n = prof.num_layers
    weight_plan = plan_compression(normalize(rem.raw_scores, s0.normalization), cand["sensitive_threshold"], PR,
                                   s0.protected_bits, s0.compressed_bits, guarded_layers(rem.raw_scores, s0.guard_top_k))
    plans = {"uniform8": uniform_activation_plan(n, 8), "uniform6": uniform_activation_plan(n, 6),
             "uniform4": uniform_activation_plan(n, 4), "measured": plan_activations(prof, s0.act_avg_bits),
             "from_weights": activation_plan_from_weights(weight_plan, 8, 4)}
    for s in range(3):
        top = set(random.Random(s).sample(range(n), n // 2))
        plans[f"random{s}"] = ActivationPlan(tuple(ActivationLayerPlan(i, 8 if i in top else 4, i in top)
                                                   for i in range(n)), "random")

    def apply(plan):
        def ctx(m):
            stack = ExitStack()
            for lp, layer in zip(plan.layers, find_decoder_layers(m)):
                if lp.act_bits < s0.baseline_bits:
                    stack.enter_context(quantize_inputs(layer, lp.act_bits, s0.act_group_size))
            return stack
        return ctx

    rows = {}
    for name, plan in plans.items():
        res = measure(apply(plan))
        rows[name] = {"avg_bits": plan.avg_bits, "predicted_sum_of_one_layer_rises": predicted_rise(plan, prof),
                      "eight_bit_layers": [lp.layer for lp in plan.layers if lp.act_bits == 8], **summary(res)}
        print(f"  act {name}: {res['ppl_val']:.3f}", flush=True)
    save("act", {"fp16_note": "weights unquantized; only Linear inputs rounded", "rows": rows})


def kv_section(n_passages: int = 16) -> None:
    """Item 15: attention coverage (share of attention on each layer's top tokens) at 2,048 vs 512 tokens."""
    m = handle.model(cfg0.model.dtype)
    long = calibration_batches(loader(cand["calib_dataset"], "train"), handle.tokenizer, n_passages, 2048, 1, 0)
    ratios = s0.kv_keep_ratios
    layers = find_decoder_layers(m)
    acc: dict[int, list[list[float]]] = {}

    def hook(i):
        def record(mod, args, out):  # keep the coverage, drop the attention weights at once (memory)
            acc.setdefault(i, []).append(_coverage(out[1], ratios))
            return out[0], None
        return record

    def run(batches) -> list[list[float]]:
        acc.clear()
        hooks = [layer.self_attn.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
        try:
            with eager_attention(m), torch.no_grad():
                for b in batches:
                    m(input_ids=b.to(device))
        finally:
            for h in hooks:
                h.remove()
        return [[statistics.fmean(c[j] for c in acc[i]) for j in range(len(ratios))] for i in range(len(layers))]

    cov = {"2048": run(long), "512": run([b[:, :512] for b in long])}
    keep = {L: [keep_ratio_for(c, ratios, s0.kv_attention_coverage) for c in cov[L]] for L in cov}
    save("kv", {"passages": n_passages, "keep_ratios": ratios, "coverage": cov, "keep_at_95pct": keep})


def probe_section() -> None:
    """Item 16: one-layer compression at 4, 3 and 2 bits, measured on two halves of the 64 passages (noise)."""
    b = calibration_batches(loader(cand["calib_dataset"], "train"), handle.tokenizer, cand["calib_samples"],
                            cfg0.calibration.seq_len, cfg0.calibration.batch_size, cfg0.run.seed)
    halves = [b[: len(b) // 2], b[len(b) // 2:]]
    rem = removal(0).raw_scores
    out = {"rise": {}, "mean_rise": {}, "split_half_agreement": {}, "agreement_with_removal": {}, "seconds": {}}
    for bits in (4, 3, 2):
        t0 = time.perf_counter()
        r = [profile_by_ablation(handle.model(s0.profile_dtype), h, "layer_quant", device, bits, GS).raw_scores
             for h in halves]
        mean = [(x + y) / 2 for x, y in zip(*r)]
        out["rise"][str(bits)], out["mean_rise"][str(bits)] = r, mean
        out["split_half_agreement"][str(bits)] = _spearman(r[0], r[1])
        out["agreement_with_removal"][str(bits)] = _spearman(mean, rem)
        out["seconds"][str(bits)] = time.perf_counter() - t0
        print(f"  probe {bits}-bit: split-half {_spearman(r[0], r[1]):.2f}", flush=True)
    save("probe", out)


def hessian_section(n_layers: int = 4) -> None:
    """Item 18: what one Hessian probe costs, on the first `n_layers` layers (scale x layers/n_layers)."""
    m = handle.model(s0.profile_dtype)
    params = [p for layer in find_decoder_layers(m)[:n_layers] for p in layer.parameters()]
    x = calibration_batches(loader(cand["calib_dataset"], "train"), handle.tokenizer, 1, 512, 1, 0)[0].to(device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    def timed(fn) -> float:
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        return time.perf_counter() - t0

    acc = [torch.zeros(p.shape) for p in params]
    originals = [p.detach().cpu() for p in params]
    t_fb = timed(lambda: m(input_ids=x, labels=x).loss.backward())  # whole model, one forward + backward
    t_d2h = timed(lambda: [a.add_(p.grad.float().cpu()) for a, p in zip(acc, params)])
    t_h2d = timed(lambda: [p.data.copy_(o.to(device)) for p, o in zip(params, originals)])
    m.zero_grad(set_to_none=True)
    scale = len(find_decoder_layers(m)) / n_layers
    save("hessian", {"forward_backward_s": t_fb, "grad_to_cpu_and_add_s_all_layers": t_d2h * scale,
                     "restore_weights_s_all_layers": t_h2d * scale, "per_probe_note": "Hessian = 64 passages x "
                     "(1 + 8 probes); each probe = forward+backward + gradient to CPU and add + restore weights"})


SECTIONS = {"probe": probe_section, "prune": prune_section, "bits": bits_section, "act": act_section,
            "kv": kv_section, "hessian": hessian_section}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(SECTIONS):
        print(f"== {name}", flush=True)
        SECTIONS[name]()
