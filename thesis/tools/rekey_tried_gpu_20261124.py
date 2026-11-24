"""Rewrite search_tried.csv so its saved tries count on another GPU name (Mohammad 2026-11-24: "an a100 is an a100").
Each row's combination holds a hash of the build settings, which include the GPU name. For every row this recomputes
the combination with the code's own TriedLog.combination under the search's try settings, checks that with the OLD
GPU name it gives exactly the stored text, then writes it with the NEW name. Nothing else in a row changes."""
import csv, json, sys
from pathlib import Path
import torch
csv.field_size_limit(10**9)
src, config_json, old_gpu, new_gpu, out = sys.argv[1:6]
import sdf.stage0.run as s0
from sdf.config import FrameworkConfig
from sdf.run import RunContext
from sdf.search import TriedLog, NO_BACKENDS, tries_need_vllm

cfg = FrameworkConfig.from_dict(json.load(open(config_json)))
s = cfg.search
needs_vllm = tries_need_vllm(cfg)
off = {k: v for k, v in NO_BACKENDS.items() if not (needs_vllm and k == "eval.vllm_python")}
try_tasks = list(s.accuracy_tasks) if cfg.requirement.min_accuracy is not None else []
try_limit = ({"eval.downstream_limit": s.accuracy_try_limit} if try_tasks and s.accuracy_try_limit is not None else {})
tcfg = cfg.with_overrides({**off, "eval.downstream_tasks": try_tasks, **try_limit})
ctxs = {seed: RunContext(tcfg.with_overrides({"run.seed": seed}), "rekey", Path("/nonexistent"), None) for seed in range(4)}
gpu = {"name": old_gpu}
s0.setup = lambda ctx, *a, **k: (torch.device("cuda"), None, None)
torch.cuda.get_device_name = lambda device=None: gpu["name"]

rows = list(csv.DictReader(open(src, newline="", encoding="utf-8")))
fields = list(rows[0])
outrows = []
seeds = {}
for r in rows:
    point = json.loads(r["result"])["point"]
    w = int(r["windows"])
    gpu["name"] = old_gpu
    ctx = next((c for c in ctxs.values() if TriedLog.combination(c, point, w) == r["combination"]), None)
    if ctx is None:  # the seed is part of the key: a row of a seed-1 run matches only with seed 1
        sys.exit(f"row {r['run_id']} #{r['n']}: recomputed combination differs from the stored one; nothing written")
    seeds[r["run_id"]] = ctx.cfg.run.seed
    gpu["name"] = new_gpu
    outrows.append({**r, "combination": TriedLog.combination(ctx, point, w)})
with open(out, "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=fields)
    wr.writeheader()
    wr.writerows(outrows)
print(seeds)
print(f"{len(outrows)} rows checked against {old_gpu} and written for {new_gpu}; distinct {len({r['combination'] for r in outrows})}")
