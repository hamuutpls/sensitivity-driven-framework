"""handoff.md: exactly what Stage 0 hands to every later stage and to the search, in plain language.

report.md says how Stage 0 went; this file is the contract the other stages read: which file each stage loads,
what is in it layer by layer, what it is predicted to cost, and what that stage still has to measure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sdf.reporting.markdown import _column_notes, _table, original_model_lines
from sdf.search_space import PER_CHANNEL, SEARCH_SPACE
from sdf.stage0.activation import (
    ActivationPlan,
    ActivationProfile,
    activation_plan_from_weights,
    predicted_rise,
    uniform_activation_plan,
)
from sdf.stage0.planner import CompressionPlan, PlanCost
from sdf.stage0.sensitivity import SensitivityProfile, outlier_layers
from sdf.utils.cache import atomic_write_text


def _n(v: Any, digits: int = 3) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def _mb(gb: float | None) -> str:
    return "–" if gb is None else f"{gb * 1000:.1f} MB" if gb < 1 else f"{gb:.3f} GB"


def write_handoff(
    path: Path,
    *,
    cfg: Any,
    candidate: dict[str, Any],
    original_model: dict[str, Any],
    profile: SensitivityProfile,
    plan: CompressionPlan,
    budget: CompressionPlan | None,
    no_prune: CompressionPlan | None,
    uniform: CompressionPlan,
    predict: Callable[[CompressionPlan], PlanCost],
    fp16: dict[str, Any],
    guarded: frozenset[int],
    act: ActivationPlan | None,
    act_prof: ActivationProfile | None,
    kv: tuple | None,
) -> Path:
    s0 = cfg.stage0
    n = len(plan.layers)
    lines = [
        "# What Stage 0 hands to each stage",
        "",
        "Stage 0 does not compress anything. It measures which parts of the model are fragile and writes a plan "
        "for every later stage: how many bits each layer's weights get and how much of each layer may be "
        "removed (Stage 1), how many bits the numbers passed between layers get (Stage 2), and how the model's "
        "short-term memory while writing is stored (Stage 3). Stage 4 and the search use the plans' predicted "
        "costs and the uncompressed model's measured numbers as their reference. Every plan below is also a JSON "
        "file next to this one, and each later stage loads that file rather than re-measuring.",
        "",
        "Each stage applies its plan on its own, starting from the original model, so the effect of each kind "
        "of compression can be measured separately. Every stage compares three versions under identical "
        "conditions: the original model, the method used the standard way (no Stage 0), and the method guided "
        "by Stage 0.",
        "",
    ]
    lines += original_model_lines(original_model)

    # ---- at a glance --------------------------------------------------------------------------------------
    rows = [
        ["Stage 1: weights", "bits and share removed per layer; layers never to prune",
         "`compression_plan.json` (+ `_budget_matched`, `_budget_matched_no_prune`)", "`CompressionPlan.load`"],
        ["Stage 2: activations", "bits per layer for the numbers passed between layers",
         "`activation_plan.json`" + (" (+ `activation_profile.json`)" if act_prof else ""), "`ActivationPlan.load`"],
        ["Stage 3: KV cache", "key bits, value bits and share of past words kept per layer",
         "`kv_cache_plan.json` (+ `_bits_only`, `kv_profile.json`)" if kv else "not planned (KV_CACHE off)",
         "`KVPlan.load`" if kv else "–"],
        ["Stage 4: evaluation", "the uncompressed model's measured numbers and the conditions to repeat",
         "`results.json` (row `baseline/fp16`)", "read the JSON"],
        ["Search", "Stage 0's search parameters, cached measurements, cheap predicted costs per trial",
         "`sensitivity_profile.json`, cache folder", "`run_stage0` per trial"],
    ]
    lines += ["## At a glance", "", _table(["Who", "Receives", "File", "How to load"], rows), ""]
    lines += _column_notes([
        ("Who", "The later stage (or the search) that uses this part of the plan."),
        ("Receives", "What the plan tells that stage to do."),
        ("File", "The file Stage 0 writes, in the same folder as this report."),
        ("How to load", "The Python call that reads the file back into a plan object."),
    ])

    # ---- Stage 1 ------------------------------------------------------------------------------------------
    fw, un = predict(plan), predict(uniform)
    lines += [
        "## Stage 1: weights (GPTQ, AWQ, pruning, low-rank)",
        "",
        f"The main plan protects layers with sensitivity at or above {candidate['sensitive_threshold']:.2f} "
        f"({len(plan.protected_layers)} of {n}: kept at {s0.protected_bits} bits, nothing removed) and "
        f"compresses the rest to {s0.compressed_bits} bits with {candidate['prune_ratio_aggressive']:.0%} of "
        "their numbers removed. Two same-size versions fit exactly in the standard method's memory, so accuracy "
        "can be compared size for size. "
        + (f"Layers {', '.join(map(str, sorted(guarded)))} are never pruned in any plan, because removing any one of them alone "
           f"hurts the model most (the {s0.guard_top_k} highest layer-removal scores)."
           if guarded else "No layer is guarded against pruning (GUARD_TOP_K = 0)."),
        "",
    ]
    budget_layers = budget.layers if budget else [None] * n
    no_prune_layers = no_prune.layers if no_prune else [None] * n
    rows = [[lp.layer, f"{lp.sensitivity:.2f}", _n(lp.protected), _n(lp.guarded), lp.bit_width,
             f"{lp.pruning_ratio:.0%}", "–" if bl is None else bl.bit_width,
             "–" if bl is None else f"{bl.pruning_ratio:.0%}", "–" if nl is None else nl.bit_width]
            for lp, bl, nl in zip(plan.layers, budget_layers, no_prune_layers)]
    lines += [_table(["Layer", "Sensitivity", "Protected", "Never pruned", "Bits", "Removed",
                      "Bits (same size)", "Removed (same size)", "Bits (same size, nothing removed)"], rows), ""]
    lines += _column_notes([
        ("Layer", "Position in the model, from 0 at the input end."),
        ("Sensitivity", f"How much the model suffers when this layer changes, from 0 (least) to 1 (most), "
                        f"measured by {profile.method.replace('_', ' ')}."),
        ("Protected", "\"yes\" if the main plan keeps this layer at high precision."),
        ("Never pruned", "\"yes\" if no plan may remove numbers from this layer, whatever its sensitivity."),
        ("Bits", "Bits per number for this layer's weights in the main plan."),
        ("Removed", "Share of this layer's numbers the main plan deletes."),
        ("Bits (same size)", "Bits per number in the plan that fits in the standard method's memory."),
        ("Removed (same size)", "Share deleted in that same-size plan."),
        ("Bits (same size, nothing removed)", "Bits per number in the same-size plan that deletes nothing."),
    ])
    rows = [["Original model (uncompressed)", _mb(fp16.get("predicted_weight_memory_gb")), "16", "0%"],
            [f"Standard method (uniform {s0.uniform_bits}-bit)", _mb(un.weight_memory_gb),
             f"{un.avg_bits_per_weight:.2f}", f"{un.sparsity:.0%}"],
            ["Main plan", _mb(fw.weight_memory_gb), f"{fw.avg_bits_per_weight:.2f}", f"{fw.sparsity:.0%}"]]
    for name, p in (("Same size", budget), ("Same size, nothing removed", no_prune)):
        if p is not None:
            c = predict(p)
            rows.append([name, _mb(c.weight_memory_gb), f"{c.avg_bits_per_weight:.2f}", f"{c.sparsity:.0%}"])
    gs = candidate["gptq_groupsize"]
    lines += [_table(["Version", "Predicted size", "Average bits per number", "Share removed"], rows), ""]
    lines += _column_notes([
        ("Version", "Which plan the row describes."),
        ("Predicted size", f"Memory for the model's numbers, predicted from the plan (group size "
                           f"{'per row' if gs == PER_CHANNEL else gs}). Stage 1 measures the real size."),
        ("Average bits per number", "Storage per number over the whole model, including parts never "
                                    "compressed (16 means uncompressed)."),
        ("Share removed", "Share of all the model's numbers deleted by pruning."),
    ])
    lines += [f"Stage 1 also takes the search parameter `gptq_groupsize` (now {gs}). It must measure what "
              "Stage 0 only predicts: real size, perplexity on both halves of the test text, speed and peak "
              "memory, for each plan against the standard method.", ""]

    # ---- Stage 2 ------------------------------------------------------------------------------------------
    lines += ["## Stage 2: activations (SmoothQuant, QuaRot, RPTQ, SpinQuant)", ""]
    if act is None:
        lines += ["No activation plan was made in this run; see the anomalies in report.md.", ""]
    else:
        hi, lo = max(s0.act_bits_options), min(s0.act_bits_options)
        derived = activation_plan_from_weights(plan, hi, lo)
        how = (f"Each layer's incoming numbers were rounded on their own to each of "
               f"{', '.join(map(str, s0.act_bits_options))} bits on {s0.act_calib_samples} passages, and the rise "
               f"in prediction error was recorded. The plan spends an average of {s0.act_avg_bits:g} bits per "
               "layer, giving the most bits to the layers that suffer most."
               if act_prof is not None else
               "The plan is copied from the weight plan, without a measurement: protected and never-pruned layers "
               "get the most bits, the rest the fewest.")
        lines += [f"{how} Stage 2 applies its smoothing or rotation to every layer and rounds each layer's "
                  "activations to the planned bits.", ""]
        rise = act_prof.rise if act_prof else [None] * n
        rows = [[lp.layer, lp.act_bits, dl.act_bits,
                 "–" if r is None else f"{r[0]:+.4f}", "–" if r is None else f"{r[-1]:+.4f}"]
                for lp, dl, r in zip(act.layers, derived.layers, rise)]
        lines += [_table(["Layer", "Activation bits", "Bits if copied from the weight plan",
                          f"Damage at {lo} bits", f"Damage at {hi} bits"], rows), ""]
        lines += _column_notes([
            ("Layer", "Position in the model, from 0 at the input end."),
            ("Activation bits", "Bits Stage 2 rounds this layer's incoming numbers to."),
            ("Bits if copied from the weight plan", "What the cheaper rule (fragile for weights means fragile "
                                                    "for activations) would give, for comparison."),
            (f"Damage at {lo} bits", "Rise in prediction error (perplexity) when only this layer's incoming "
                                     "numbers are rounded to the fewest bits. Values within about ±0.005 are "
                                     "measurement noise."),
            (f"Damage at {hi} bits", "The same at the most bits."),
        ])
        if act_prof is not None:
            un_a = uniform_activation_plan(n, s0.act_uniform_bits)
            rows = [[f"Standard method (uniform {s0.act_uniform_bits}-bit)", f"{un_a.avg_bits:.2f}",
                     _n(predicted_rise(un_a, act_prof), 4)],
                    ["Plan", f"{act.avg_bits:.2f}", _n(predicted_rise(act, act_prof), 4)],
                    ["Copied from the weight plan", f"{derived.avg_bits:.2f}", _n(predicted_rise(derived, act_prof), 4)]]
            lines += [_table(["Version", "Average bits", "Predicted perplexity rise"], rows), ""]
            lines += _column_notes([
                ("Version", "Which activation plan the row describes."),
                ("Average bits", "Bits per incoming number, averaged over the layers."),
                ("Predicted perplexity rise", "The measured per-layer damages added up for that plan (assumes "
                                              "they add up; Stage 2 measures the real effect). Lower is better; "
                                              "– means the plan uses a bit width that was not measured."),
            ])
        lines += ["Stage 2 adds the search parameter `smoothquant_alpha` to the search space.", ""]

    # ---- Stage 3 ------------------------------------------------------------------------------------------
    lines += ["## Stage 3: KV cache (QuaRot KV, KVQuant, H2O, SnapKV, InfiniGen)", ""]
    if kv is None:
        lines += ["No KV cache plan was made in this run (KV_CACHE is off or the measurement failed).", ""]
    else:
        kprof, kplan, kbits, kuni, kpredict = kv
        lines += [f"While writing, the model keeps notes on every earlier word (the KV cache). The plan sets, per "
                  f"layer, the bits for keys and for values (average {s0.kv_avg_bits or s0.kv_uniform_bits:g}, the "
                  f"standard method's size) and the share of earlier words kept (the fewest that still get "
                  f"{s0.kv_attention_coverage:.0%} of the layer's attention). Measured on {s0.kv_calib_samples} "
                  "passages.", ""]
        rows = [[lp.layer, lp.key_bits, lp.value_bits, f"{lp.keep_ratio:.0%}"] for lp in kplan.layers]
        lines += [_table(["Layer", "Key bits", "Value bits", "Words kept"], rows), ""]
        lines += _column_notes([
            ("Layer", "Position in the model, from 0 at the input end."),
            ("Key bits", "Bits per number for this layer's stored keys (what earlier words are about)."),
            ("Value bits", "Bits per number for this layer's stored values (what earlier words say)."),
            ("Words kept", "Share of earlier words this layer remembers; the rest are forgotten."),
        ])
        c_un, c_fw, c_bits = kpredict(kuni), kpredict(kplan), kpredict(kbits)
        rows = [[f"Standard method (uniform {s0.kv_uniform_bits}-bit, nothing forgotten)", _mb(c_un.memory_gb),
                 f"{c_un.avg_bits:.2f}", f"{c_un.ppl_rise:.4f}"],
                ["Plan (bits + forgetting)", _mb(c_fw.memory_gb), f"{c_fw.avg_bits:.2f}", f"{c_fw.ppl_rise:.4f}"],
                ["Plan, bits only", _mb(c_bits.memory_gb), f"{c_bits.avg_bits:.2f}", f"{c_bits.ppl_rise:.4f}"]]
        lines += [_table(["Version", f"Predicted memory ({s0.kv_context_len} words)", "Average bits",
                          "Predicted perplexity rise from rounding"], rows), ""]
        lines += _column_notes([
            ("Version", "Which KV cache plan the row describes."),
            (f"Predicted memory ({s0.kv_context_len} words)", f"Memory for the notes on one text of "
                                                              f"{s0.kv_context_len} tokens (word pieces)."),
            ("Average bits", "Bits per stored number, averaged over every layer's keys and values."),
            ("Predicted perplexity rise from rounding", "The measured per-layer damages of rounding added up. "
                                                        "Forgetting is not included: Stage 3 measures it."),
        ])
        if c_bits.ppl_rise >= c_un.ppl_rise - 1e-9:
            lines += ["Choosing bits per layer predicts no accuracy gain over the standard method here: at this "
                      "average the measured differences between layers are too small to use. The saving comes "
                      "from forgetting, which Stage 3 has to confirm with a real eviction method.", ""]
        lines += ["Stage 3 adds the search parameter `quarot_k_bits` and checks the cache against `kv_budget_gb`.", ""]

    # ---- Stage 4 ------------------------------------------------------------------------------------------
    ev = cfg.eval
    keys = [("ppl_val", "Perplexity, validation half"), ("ppl_heldout", "Perplexity, held-out half"),
            ("prefill_ms_mean", "Time to read a prompt (ms)"), ("decode_ms_per_token_mean", "Time per word (ms)"),
            ("peak_memory_gb", "Peak graphics-card memory (GB)"), ("predicted_weight_memory_gb", "Size (GB)"),
            ("predicted_kv_memory_gb", "KV cache memory (GB)")]
    rows = [[name, _n(fp16.get(k), 4)] for k, name in keys if k in fp16]
    lines += ["## Stage 4: evaluation across backends", "",
              "Stage 4 runs the compressed versions from Stages 1 to 3 on each backend (HF Transformers, "
              "llama.cpp, vLLM, TensorRT-LLM). From Stage 0 it takes the reference numbers of the uncompressed "
              "model and the conditions they were measured under, so every later number is comparable.", ""]
    lines += [_table(["Reference", "Original model (uncompressed)"], rows), ""]
    lines += _column_notes([("Reference", "What was measured on the uncompressed model."),
                            ("Original model (uncompressed)", "The value measured in this run (– if the "
                                                              "measurement was skipped).")])
    lines += [f"Conditions: {ev.dataset} test split in {ev.seq_len}-token windows"
              f"{'' if ev.max_windows is None else f' (first {ev.max_windows})'}, first half for validation and "
              f"second half held out (never tuned on); latency with a {ev.latency_prompt_len}-token prompt and "
              f"{ev.latency_decode_tokens} generated tokens, {ev.latency_warmup} warm-up and {ev.latency_repeats} "
              f"timed repeats; seed {cfg.run.seed}; backend HF Transformers.", ""]

    # ---- search -------------------------------------------------------------------------------------------
    rows = [[p.name, p.stage, _n(candidate.get(p.name)),
             ", ".join(map(str, p.choices)) if p.choices else f"{p.low} to {p.high}"]
            for p in SEARCH_SPACE.params]
    lines += ["## Search", "",
              "The search tries many settings and keeps the best trade-offs. Stage 0's measurements are cached, "
              f"so a trial that only changes the threshold or the share removed re-plans in milliseconds; a new "
              f"calibration text or passage count costs one new measurement "
              f"({profile.cost.get('wall_clock_s', 0):.0f} s for this profile).", ""]
    lines += [_table(["Parameter", "Used by stage", "This run", "Allowed values"], rows), ""]
    lines += _column_notes([
        ("Parameter", "A setting the search may change."),
        ("Used by stage", "The stage that reads it."),
        ("This run", "The value used for this report."),
        ("Allowed values", "The range or list the search picks from."),
    ])
    lines += ["Per trial Stage 0 gives the search predicted size, average bits, share removed, sensitivity "
              "exposure and KV cache memory without running the model; perplexity and speed come from the "
              "stage that applies the plan. Held-out perplexity is reported for every trial but never optimised.",
              ""]

    # ---- caveats ------------------------------------------------------------------------------------------
    notes = []
    for i in outlier_layers(profile.raw_scores):
        notes.append(f"Layer {i}'s sensitivity score stands far from the others; check it separately in Stage 1.")
    if act is not None and act_prof is None:
        notes.append("The activation plan is copied from the weight plan, not measured.")
    if not notes:
        notes.append("None.")
    lines += ["## Caveats", ""] + [f"- {x}" for x in notes] + [""]
    atomic_write_text(path, "\n".join(lines))
    return path
