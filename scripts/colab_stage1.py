"""Run Stage 0 and Stage 1 (quantization only) on Colab with no Google Drive, then print every number.

In a Colab cell (GPU runtime, A100 for comparable rows):

    !git clone -q https://github.com/hamuutpls/sensitivity-driven-framework /content/sdf
    !pip install -q -e "/content/sdf[dev]" bitsandbytes
    %run /content/sdf/scripts/colab_stage1.py                # every method
    %run /content/sdf/scripts/colab_stage1.py rtn gptq awq   # only these Stage 1 methods

Results and caches are saved where the notebook saves them (/content/thesis_compression/results and /cache, or
D:/Thesis/thesis_compression on a local runtime on the PC), and the lines between RESULTS-BEGIN and RESULTS-END (with
PLAN and JOINT) are also written to stage1_summary.txt in the run's folder. At the end the results are packed into
/content/thesis_results.tgz, which the host PC session copies to D:/Thesis. Each row's error is recorded and the run
goes on, so one broken technique never stops the others.
(%run starts the script in a fresh namespace, so a METHODS list set in the notebook is not seen; colab_stage1_joint.py
sets METHODS itself and exec()s this file.)
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
from sdf.utils.colab import pack_results, results_root  # noqa: E402

ALL = ["rtn", "gptq", "awq", "omniquant", "squeezellm", "spqr", "efficientqat", "aqlm", "quip", "quipsharp",
       "pbllm", "billm", "bitsandbytes", "qtip", "abqllm",
       "rtn_act", "smoothquant", "quarot", "spinquant", "rptq"]
methods = sys.argv[1:] or globals().get("METHODS") or ALL
ROOT = results_root()

main.MODE = "stages"
main.STAGE0_DIR = None
main.STAGE1_METHODS = methods
main.STAGE2_METHODS = []
main.STAGE3_METHODS = []
main.OUTPUT_ROOT = f"{ROOT}/results"
main.CACHE_DIR = f"{ROOT}/cache"
main.main()

run_dir = max(glob.glob(f"{ROOT}/results/*/"), key=os.path.getmtime)
summary = open(os.path.join(run_dir, "stage1_summary.txt"), "w")  # closed in the finally below, even on an error


def emit(*parts) -> None:
    """Print a result line and keep it in stage1_summary.txt."""
    print(*parts, flush=True)
    print(*parts, file=summary, flush=True)


try:
    data = json.load(open(os.path.join(run_dir, "stage_1", "results.json")))
    keep = ("ppl_val", "ppl_heldout", "predicted_weight_memory_gb", "avg_bits_per_weight", "avg_activation_bits",
            "peak_memory_gb", "decode_ms_per_token_mean", "build_time_s")
    emit("RESULTS-BEGIN", os.path.basename(run_dir.rstrip("/")), "device: Colab")
    for r in data["rows"]:
        emit(json.dumps({"method": r["method"], "variant": r["variant"], "status": r["status"],
                         "error": (r.get("error") or "")[:300],
                         **{k: r["metrics"].get(k) for k in keep}}))
    emit("RESULTS-END")
    stage0 = json.load(open(os.path.join(run_dir, "stage_0", "compression_plan.json")))
    emit("PLAN-BEGIN", json.dumps([[lp["layer"], lp["bit_width"]] for lp in stage0["layers"]]), "PLAN-END")
    joint_w, joint_a = (os.path.join(run_dir, "stage_0", f) for f in ("joint_weight_plan.json", "joint_activation_plan.json"))
    if os.path.exists(joint_w) and os.path.exists(joint_a):
        pairs = [[w["bit_width"], a["act_bits"]] for w, a in zip(json.load(open(joint_w))["layers"],
                                                                json.load(open(joint_a))["layers"])]
        row0 = next((r for r in json.load(open(os.path.join(run_dir, "stage_0", "results.json")))["rows"]
                     if r["method"] == "joint_weights_activations"), {})
        emit("JOINT-BEGIN", json.dumps({"pairs": pairs, "metrics": row0.get("metrics"),
                                        "separate_plans": row0.get("info", {}).get("separate_plans"),
                                        "interaction": row0.get("info", {}).get("interaction_at_lowest_bits"),
                                        "per_layer_plan": row0.get("info", {}).get("per_layer_plan"),
                                        "refinement": {k: v for k, v in row0.get("info", {}).get("refinement", {}).items()
                                                       if k != "cost"}}),
             "JOINT-END")
finally:
    summary.close()
print("packed:", pack_results(ROOT) or "already saved in D:/Thesis")
