"""Perplexity against WxAy per layer from a saved Stage 0 run folder (no model needed).

    python scripts/joint_curves.py <run>/stage_0 [out_dir] [model name] [plan name for the legend]

Reads joint_profile.json, joint_weight_plan.json and joint_activation_plan.json and writes joint_curves.csv plus
the figures (joint_curves_all_layers.png, joint_curve_layerNN.png) to out_dir/joint_curves, as a run does
(out_dir defaults to the stage_0 folder).
"""

import sys
from pathlib import Path

from sdf.stage0 import joint_curves
from sdf.stage0.activation import ActivationPlan
from sdf.stage0.joint import JointProfile
from sdf.stage0.planner import CompressionPlan

if len(sys.argv) < 2:
    sys.exit(__doc__)
src = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else src
out.mkdir(parents=True, exist_ok=True)
profile = JointProfile.load(src / "joint_profile.json")
weights = CompressionPlan.load(src / "joint_weight_plan.json")
acts = ActivationPlan.load(src / "joint_activation_plan.json")
picks = [(lp.bit_width, al.act_bits) for lp, al in zip(weights.layers, acts.layers)]
print(joint_curves.write_csv(profile, picks, out / "joint_curves.csv"))
for f in joint_curves.draw(profile, picks, out / "joint_curves", *sys.argv[3:5]):
    print(f)
