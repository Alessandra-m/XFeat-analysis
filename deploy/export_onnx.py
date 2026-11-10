"""Export the NPU part of XFeat (backbone + heads) to ONNX with a fixed input size.

Differences from XFeatModel.forward (both are numerically equivalent):
  1. mean over channels + InstanceNorm2d(1) are NOT in the graph -> done on CPU (preprocess()).
  2. _unfold2d(x, 8) is replaced by a fixed 8x8/stride-8 conv with one-hot weights
     (== pixel_unshuffle), which is NPU-friendly and exact in INT8.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import HERE, XFEAT_DIR

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.model import XFeatModel

DEFAULT_WEIGHTS = os.path.join(XFEAT_DIR, "weights", "xfeat.pt")
DEFAULT_IMAGE = os.path.join(HERE, "data", "hpatches-sequences-release", "v_tabletop", "1.ppm")
OUT_DIR = os.path.join(HERE, "deploy", "models")


def load_net(path):
    """xfeat.pt -> state_dict of the original model; *.pth -> whole pruned module (torch.save(net))."""
    if path.endswith(".pt"):
        net = XFeatModel()
        net.load_state_dict(torch.load(path, map_location="cpu"))
    else:
        net = torch.load(path, map_location="cpu", weights_only=False)
    return net.eval()


class XFeatBackbone(nn.Module):
    def __init__(self, net, height, width):
        super().__init__()
        self.net = net
        self.size = (height // 8, width // 8)                # static -> no Shape/Slice in the graph
        unshuffle = torch.eye(64).reshape(64, 1, 8, 8)
        self.register_buffer("unshuffle_w", unshuffle)

    def forward(self, x):                                    # x: (1, 1, H, W), already normalized
        n = self.net
        x1 = n.block1(x)
        x2 = n.block2(x1 + n.skip1(x))
        x3 = n.block3(x2)
        x4 = n.block4(x3)
        x5 = n.block5(x4)
        size = self.size
        x4 = F.interpolate(x4, size, mode="bilinear")
        x5 = F.interpolate(x5, size, mode="bilinear")
        feats = n.block_fusion(x3 + x4 + x5)
        heatmap = n.heatmap_head(feats)
        keypoints = n.keypoint_head(F.conv2d(x, self.unshuffle_w, stride=8))
        return feats, keypoints, heatmap


def preprocess(path, h, w):
    """Returns (raw, normalized) float32 arrays of shape (1, 1, h, w)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    norm = (img - img.mean()) / np.sqrt(img.var() + 1e-5)   # == InstanceNorm2d(1), eps=1e-5
    return img[None, None], norm[None, None].astype(np.float32)


def max_diff(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_WEIGHTS, help="xfeat.pt or pruned *.pth")
    ap.add_argument("--name", default="xfeat", help="output name prefix")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="test image for the output check")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()
    assert args.height % 32 == 0 and args.width % 32 == 0, "H and W must be multiples of 32"

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.name}_{args.width}x{args.height}.onnx")

    net = load_net(args.model)
    model = XFeatBackbone(net, args.height, args.width).eval()
    raw, norm = preprocess(args.image, args.height, args.width)

    with torch.no_grad():
        ref = [t.numpy() for t in net(torch.from_numpy(raw))]        # original forward
        wrp = [t.numpy() for t in model(torch.from_numpy(norm))]     # exported graph in torch

    print("[1] original vs wrapper (torch):")
    for name, a, b in zip(["feats", "keypoints", "heatmap"], ref, wrp):
        print(f"    {name:9s} shape={tuple(a.shape)}  max|diff|={max_diff(a, b):.2e}")

    torch.onnx.export(
        model, (torch.from_numpy(norm),), out,
        input_names=["image"], output_names=["feats", "keypoints", "heatmap"],
        opset_version=args.opset, dynamo=False,
    )
    m = onnx.load(out)
    m.ir_version = 8               # keep the file readable by the older onnx in the rknn env
    onnx.checker.check_model(m)
    onnx.save(m, out)

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"image": norm})
    print("[2] torch vs onnxruntime:")
    for name, a, b in zip(["feats", "keypoints", "heatmap"], wrp, ort_out):
        print(f"    {name:9s} max|diff|={max_diff(a, b):.2e}")

    ops = sorted({n.op_type for n in m.graph.node})
    params = sum(p.numel() for name, p in net.named_parameters() if not name.startswith("fine_matcher"))
    print(f"[3] saved {out}")
    print(f"    size={os.path.getsize(out) / 1e6:.2f} MB, params={params}, opset={args.opset}")
    print(f"    ops: {', '.join(ops)}")


if __name__ == "__main__":
    main()
