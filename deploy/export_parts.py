"""Export truncated XFeat variants to time each part on the NPU by subtraction.

The per-layer profile does not work on our board (RKNN_LOG_LEVEL=4 / COLLECT_PERF crash inside
librknnrt 2.5.0 with driver 0.9.8), but plain rknn_benchmark timing is stable. So we time
growing prefixes of the network and take differences:

  name            input              what is inside                               difference gives
  p0_min          gray 1x480x640     avg_pool 8x8 only (reads the input once)     fixed cost of a run
  p1_block1       gray 1x480x640     block1 + skip1 + add                         block1 = p1 - p0
  p2_block2       gray 1x480x640     ... + block2                                 block2 = p2 - p1
  p3_block3       gray 1x480x640     ... + block3                                 block3 = p3 - p2
  p4_desc         gray 1x480x640     full descriptor path: feats + heatmap        block4/5+FPN+heads = p4 - p3
  p5_kpts         gray 1x480x640     keypoint branch only (unshuffle conv + head) keypoint branch
  gray_nopre      gray 1x480x640     gray model without mean+InstanceNorm         pre-ops  = gray - gray_nopre
  gray_nopost     raw  3x480x640     gray model without L2/softmax/slice          post-ops = gray - gray_nopost

and for reference the full models we already have: xfeat_640x480.onnx (= "xfeat"),
xfeat_grayscale_480_640.onnx (= "gray"). Pre/post are subtracted within the gray family, because the
gray model's keypoint branch (merged 8x8 conv) differs from ours.
Differences are approximate: the compiler fuses and schedules each graph on its own.
Timing on the board -> deploy/logs/parts_time.txt -> python -m deploy.summary_parts

    conda activate xfeat
    python -m deploy.export_parts                 # -> deploy/models/parts/*.onnx
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import HERE

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from deploy.export_onnx import DEFAULT_WEIGHTS, DEFAULT_IMAGE, XFeatBackbone, load_net
from deploy.preprocess import load_raw, preprocess
from deploy.xfeat_gray import XFeatGray

OUT_DIR = os.path.join(HERE, "deploy", "models", "parts")


class Prefix(nn.Module):
    """First `upto` stages of the backbone, or one of the two full branches."""
    def __init__(self, bb, upto):
        super().__init__()
        self.bb, self.upto = bb, upto

    def forward(self, x):                                  # x: normalized (1, 1, H, W)
        n = self.bb.net
        if self.upto == "min":
            return F.avg_pool2d(x, 8)
        if self.upto == "kpts":
            return n.keypoint_head(F.conv2d(x, self.bb.unshuffle_w, stride=8))
        x = n.block1(x) + n.skip1(x)
        if self.upto == "block1":
            return x
        x = n.block2(x)
        if self.upto == "block2":
            return x
        x3 = n.block3(x)
        if self.upto == "block3":
            return x3
        x4 = n.block4(x3)
        x5 = n.block5(x4)
        x4 = F.interpolate(x4, self.bb.size, mode="bilinear")
        x5 = F.interpolate(x5, self.bb.size, mode="bilinear")
        feats = n.block_fusion(x3 + x4 + x5)
        return feats, n.heatmap_head(feats)                # upto == "desc"


def export(model, x, input_name, path):
    model.eval()
    with torch.no_grad():
        ref = model(torch.from_numpy(x))
    ref = [ref] if isinstance(ref, torch.Tensor) else list(ref)
    names = [f"out{i}" for i in range(len(ref))]
    torch.onnx.export(model, (torch.from_numpy(x),), path, input_names=[input_name], output_names=names,
                      opset_version=13, dynamo=False)
    m = onnx.load(path)
    m.ir_version = 8                                       # readable by the older onnx in the rknn env
    onnx.checker.check_model(m)
    onnx.save(m, path)
    out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {input_name: x})
    diff = max(float(np.abs(a.numpy() - b).max()) for a, b in zip(ref, out))
    shapes = " ".join(str(tuple(o.shape)) for o in out)
    print(f"  {os.path.basename(path):24s} in {tuple(x.shape)} -> {shapes}  torch vs ort max|diff|={diff:.1e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()
    h, w = args.height, args.width
    os.makedirs(OUT_DIR, exist_ok=True)

    net = load_net(DEFAULT_WEIGHTS)
    bb = XFeatBackbone(net, h, w).eval()
    _, norm = preprocess(DEFAULT_IMAGE, h, w)              # (1, 1, H, W) normalized gray
    raw = load_raw(DEFAULT_IMAGE, h, w)                    # (1, 3, H, W) raw BGR

    print(f"exporting to {os.path.relpath(OUT_DIR, HERE)}/")
    for i, upto in enumerate(["min", "block1", "block2", "block3", "desc"]):
        export(Prefix(bb, upto), norm, "image", os.path.join(OUT_DIR, f"p{i}_{upto}.onnx"))
    export(Prefix(bb, "kpts"), norm, "image", os.path.join(OUT_DIR, "p5_kpts.onnx"))
    export(XFeatGray(net, with_pre=False), norm, "image", os.path.join(OUT_DIR, "gray_nopre.onnx"))
    export(XFeatGray(net, with_post=False), raw, "INPUT_1", os.path.join(OUT_DIR, "gray_nopost.onnx"))


if __name__ == "__main__":
    main()
