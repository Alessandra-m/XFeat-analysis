"""Truncated variants of the space-to-depth x4 model (deploy/s2d_net.py, kp="conv") for timing by subtraction.

Same method as deploy/export_parts.py (REPORT_rknn.md 6.6), finer in the late blocks, which take most of
the time after the s2d rewrite (6.10). All variants take the s2d input (1, 16, 120, 160):

  name          what is inside (prefix of the network)            difference gives
  q0_min        avg_pool 2x2 on the input only                    fixed cost of a run
  q1_block1     block1 + skip1 + add                              block1        = q1 - q0
  q2_block2     ... + block2                                      block2        = q2 - q1
  q3_block3     ... + block3 (x3, 64x60x80)                       block3        = q3 - q2
  q4_block4     ... + block4 (x4, 64x30x40)                       block4        = q4 - q3
  q5_block5     ... + block5 (x5, 64x15x20)                       block5        = q5 - q4
  q6_fpn        ... + 2x Resize to 60x80 + x3 + x4 + x5           FPN           = q6 - q5
  q7_fusion     ... + block_fusion (feats)                        block_fusion  = q7 - q6
  q8_desc       ... + heatmap_head (feats, heatmap)               heatmap_head  = q8 - q7
  q9_kpts       keypoint branch only (2x2/s2 conv + head)         keypoint branch alone
  full model    deploy/models/s2d/xfeat_s2d4_conv_640x480_i8.rknn  keypoint branch in the model = full - q8

    conda activate xfeat
    python -m deploy.export_parts_s2d            # -> deploy/models/parts_s2d/*.onnx
Timing on the board -> deploy/logs/parts_s2d_time.txt -> python -m deploy.summary_parts --s2d deploy/logs/parts_s2d_time.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import HERE

import torch.nn as nn
import torch.nn.functional as F

from deploy.export_onnx import DEFAULT_WEIGHTS, DEFAULT_IMAGE, load_net
from deploy.export_parts import export
from deploy.preprocess import preprocess, s2d
from deploy.s2d_net import XFeatS2D

OUT_DIR = os.path.join(HERE, "deploy", "models", "parts_s2d")
STAGES = ["min", "block1", "block2", "block3", "block4", "block5", "fpn", "fusion", "desc"]


class PrefixS2D(nn.Module):
    def __init__(self, m, upto):
        super().__init__()
        self.m, self.upto = m, upto

    def forward(self, x):                                   # x: (1, 16, H/4, W/4)
        m, n, u = self.m, self.m.net, self.upto
        if u == "min":
            return F.avg_pool2d(x, 2)
        if u == "kpts":
            return m.kp_rest(m.kp0(x))
        x = m.block1(x) + m.skip1(x)
        if u == "block1":
            return x
        x = n.block2(x)
        if u == "block2":
            return x
        x3 = n.block3(x)
        if u == "block3":
            return x3
        x4 = n.block4(x3)
        if u == "block4":
            return x4
        x5 = n.block5(x4)
        if u == "block5":
            return x5
        s = F.interpolate(x4, m.size, mode="bilinear") + F.interpolate(x5, m.size, mode="bilinear") + x3
        if u == "fpn":
            return s
        feats = n.block_fusion(s)
        if u == "fusion":
            return feats
        return feats, n.heatmap_head(feats)                  # u == "desc"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()
    h, w = args.height, args.width
    os.makedirs(OUT_DIR, exist_ok=True)

    net = load_net(DEFAULT_WEIGHTS)
    m = XFeatS2D(net, h, w, factor=4, kp="conv").eval()
    x = s2d(preprocess(DEFAULT_IMAGE, h, w)[1], 4)          # (1, 16, H/4, W/4)

    print(f"exporting to {os.path.relpath(OUT_DIR, HERE)}/")
    for i, u in enumerate(STAGES):
        export(PrefixS2D(m, u), x, "image_s2d", os.path.join(OUT_DIR, f"q{i}_{u}.onnx"))
    export(PrefixS2D(m, "kpts"), x, "image_s2d", os.path.join(OUT_DIR, "q9_kpts.onnx"))


if __name__ == "__main__":
    main()
