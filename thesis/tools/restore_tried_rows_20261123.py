"""Rebuild search_tried.csv rows for tries missing from the csv, from trials.json (same fields as TriedLog.add)."""
import csv, json, sys
csv.field_size_limit(10**9)
src_csv, trials_json, run_id, out_csv = sys.argv[1:5]
METRICS = ("ppl_val", "ppl_heldout", "predicted_weight_memory_gb")
rows = list(csv.DictReader(open(src_csv, newline="", encoding="utf-8")))
fields = list(rows[0])
trials = json.load(open(trials_json))

def comb(t):
    s = json.loads(t["model"])["settings"]
    return json.dumps({"settings": s, "windows": t["windows"],
                       **{k: round(v, 6) if isinstance(v, float) else v for k, v in sorted(t["point"].items())}},
                      sort_keys=True)

def row(t, errors):
    keep = {"point": t["point"], "metrics": t["metrics"], "errors": errors, "dir": t["dir"], "model": t["plan_model"]}
    return {"combination": comb(t), "run_id": run_id, "searcher": t["searcher"], "n": t["n"], "windows": t["windows"],
            **t["point"], "status": "ok" if t["ok"] else "failed", "met": t["met"],
            **{m: t["metrics"].get(m) for m in METRICS}, "result": json.dumps(keep, default=str)}

def as_csv(d):
    return {k: str(d[k]) for k in fields}

have = {r["combination"]: r for r in rows}
# check: rebuilding this run's rows already in the csv gives the same text
for t in trials:
    r = have.get(comb(t))
    if r is not None and r["run_id"] == run_id:
        mine = as_csv(row(t, json.loads(r["result"]).get("errors", {})))
        assert mine == {k: r[k] for k in fields}, ("mismatch", t["n"], [k for k in fields if mine[k] != r[k]])
        print("check ok: try", t["n"])
new = [as_csv(row(t, {})) for t in trials if comb(t) not in have and t["built"] and t["ok"]]
print("adding tries", [r["n"] for r in new])
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows + new)
