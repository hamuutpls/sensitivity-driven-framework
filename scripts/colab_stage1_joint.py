"""Stage 1 with weights and activations quantized together (W<bits>A<bits> per layer from Stage 0), on Colab
with no Google Drive. Same output lines as colab_stage1.py (RESULTS-BEGIN ... RESULTS-END, PLAN-BEGIN ... PLAN-END).

    !git clone -q https://github.com/hamuutpls/sensitivity-driven-framework /content/sdf
    !pip install -q -e "/content/sdf[dev]" bitsandbytes
    %run /content/sdf/scripts/colab_stage1_joint.py

Rows per pair: standard W4A8 on every layer ("<w>_with_<a>/original"), our method with the separate Stage 0 weight
and activation plans ("<w>_with_<a>/framework"), our method with the joint plan ("<w>_with_<a>_joint/framework") and
standard W8A8 on every layer ("<w>_with_<a>_w8a8/original"). JOINT-BEGIN ... JOINT-END prints the joint pairs.
Left out because they did not work in the 2026-10-09 run: aqlm, billm, qtip, abqllm (weights), rptq (activations).
"""

import os

WEIGHTS = ["rtn", "gptq", "awq", "omniquant", "squeezellm", "spqr", "efficientqat", "quip", "quipsharp", "pbllm",
           "bitsandbytes"]
PAIRS = [(w, "rtn_act") for w in WEIGHTS] + [("rtn", "smoothquant"), ("gptq", "quarot"), ("gptq", "spinquant")]
METHODS = [n for w, a in PAIRS for n in (f"{w}_with_{a}", f"{w}_with_{a}_w8a8")]

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_stage1.py")).read())
