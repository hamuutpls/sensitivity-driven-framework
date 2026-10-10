"""Run Stage 0 and Stage 1 (quantization only) on Colab with no Google Drive, then print every number.

In a Colab cell (GPU runtime, A100 for comparable rows):

    !git clone -q https://github.com/hamuutpls/sensitivity-driven-framework /content/sdf
    !pip install -q -e "/content/sdf[dev]" bitsandbytes
    %run /content/sdf/scripts/colab_stage1.py

Everything between the RESULTS-BEGIN and RESULTS-END lines is the result: copy it into a local file. Pass
METHODS (a list of Stage 1 method names) before %run to run fewer methods; each row's error is recorded and the
run goes on, so one broken technique never stops the others.
"""

import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
os.chdir(REPO)

import main  # noqa: E402

ALL = ["rtn", "gptq", "awq", "omniquant", "squeezellm", "spqr", "efficientqat", "aqlm", "quip", "quipsharp",
       "pbllm", "billm", "bitsandbytes", "qtip", "abqllm",
       "rtn_act", "smoothquant", "quarot", "spinquant", "rptq"]
methods = globals().get("METHODS") or ALL

main.MODE = "stages"
main.STAGE0_DIR = None
main.STAGE1_METHODS = methods
main.STAGE2_METHODS = []
main.STAGE3_METHODS = []
main.OUTPUT_ROOT = "/content/results"
main.CACHE_DIR = "/content/cache"
main.main()

run_dir = max(glob.glob("/content/results/*/"), key=os.path.getmtime)
data = json.load(open(os.path.join(run_dir, "stage_1", "results.json")))
keep = ("ppl_val", "ppl_heldout", "predicted_weight_memory_gb", "avg_bits_per_weight", "avg_activation_bits",
        "peak_memory_gb", "decode_ms_per_token_mean", "build_time_s")
print("RESULTS-BEGIN", os.path.basename(run_dir.rstrip("/")), "device: Colab", flush=True)
for r in data["rows"]:
    print(json.dumps({"method": r["method"], "variant": r["variant"], "status": r["status"],
                      "error": (r.get("error") or "")[:300],
                      **{k: r["metrics"].get(k) for k in keep}}), flush=True)
print("RESULTS-END", flush=True)
stage0 = json.load(open(os.path.join(run_dir, "stage_0", "compression_plan.json")))
print("PLAN-BEGIN", json.dumps([[lp["layer"], lp["bit_width"]] for lp in stage0["layers"]]), "PLAN-END", flush=True)
joint_w, joint_a = (os.path.join(run_dir, "stage_0", f) for f in ("joint_weight_plan.json", "joint_activation_plan.json"))
if os.path.exists(joint_w) and os.path.exists(joint_a):
    pairs = [[w["bit_width"], a["act_bits"]] for w, a in zip(json.load(open(joint_w))["layers"],
                                                            json.load(open(joint_a))["layers"])]
    row0 = next((r for r in json.load(open(os.path.join(run_dir, "stage_0", "results.json")))["rows"]
                 if r["method"] == "joint_weights_activations"), {})
    print("JOINT-BEGIN", json.dumps({"pairs": pairs, "metrics": row0.get("metrics"),
                                     "separate_plans": row0.get("info", {}).get("separate_plans"),
                                     "interaction": row0.get("info", {}).get("interaction_at_lowest_bits"),
                                     "per_layer_plan": row0.get("info", {}).get("per_layer_plan"),
                                     "refinement": {k: v for k, v in row0.get("info", {}).get("refinement", {}).items()
                                                    if k != "cost"}}),
          "JOINT-END", flush=True)
