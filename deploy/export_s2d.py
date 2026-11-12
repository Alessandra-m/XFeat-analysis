"""Export the space-to-depth variants of XFeat (deploy/s2d_net.py) to ONNX + board benchmark inputs.

    conda activate xfeat
    python -m deploy.export_s2d          # -> deploy/models/s2d/xfeat_s2d{2,4}_{conv,cpu}_640x480.onnx

Every variant is checked against the ORIGINAL XFeatModel (with its own normalization) on a real image:
the rewrite is exact, so the difference must be at float32 rounding level (~1e-5), like deploy/export_onnx.py.
Also writes NHWC float32 inputs for rknn_benchmark: bench_s2d{2,4,8}_nhwc.npy.
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

from deploy.export_onnx import DEFAULT_WEIGHTS, DEFAULT_IMAGE, load_net
from deploy.preprocess import preprocess, s2d
from deploy.s2d_net import XFeatS2D

OUT_DIR = os.path.join(HERE, "deploy", "models", "s2d")
OUT_NAMES = ["feats", "keypoints", "heatmap"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_WEIGHTS)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()
    h, w = args.height, args.width
    assert h % 32 == 0 and w % 32 == 0
    os.makedirs(OUT_DIR, exist_ok=True)

    net = load_net(args.model)
    raw, norm = preprocess(DEFAULT_IMAGE, h, w)
    with torch.no_grad():
        ref = [t.numpy() for t in net(torch.from_numpy(raw))]      # original forward, incl. normalization

    for f in (2, 4):
        for kp in ("conv", "cpu"):
            model = XFeatS2D(net, h, w, factor=f, kp=kp).eval()
            xs = [s2d(norm, f)] + ([s2d(norm, 8)] if kp == "cpu" else [])
            in_names = ["image_s2d"] + (["image_s2d8"] if kp == "cpu" else [])
            out = os.path.join(OUT_DIR, f"xfeat_s2d{f}_{kp}_{w}x{h}.onnx")
            torch.onnx.export(model, tuple(torch.from_numpy(x) for x in xs), out,
                              input_names=in_names, output_names=OUT_NAMES, opset_version=13, dynamo=False)
            m = onnx.load(out)
            m.ir_version = 8                                        # readable by the older onnx in the rknn env
            onnx.checker.check_model(m)
            onnx.save(m, out)

            res = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, dict(zip(in_names, xs)))
            diff = "  ".join(f"{n} {np.abs(a - b).max():.1e}" for n, a, b in zip(OUT_NAMES, ref, res))
            ins = " + ".join(str(x.shape[1:]) for x in xs)
            print(f"{os.path.basename(out):30s} in {ins:28s} original vs onnxruntime max|diff|: {diff}")

    if (h, w) != (480, 640):                                       # bench inputs only for the 640x480 board model
        return
    for f in (2, 4, 8):                                            # rknn_benchmark reads NHWC
        p = os.path.join(OUT_DIR, f"bench_s2d{f}_nhwc.npy")
        np.save(p, np.ascontiguousarray(s2d(norm, f).transpose(0, 2, 3, 1)))
    print(f"bench inputs -> {os.path.relpath(OUT_DIR, HERE)}/bench_s2d{{2,4,8}}_nhwc.npy")


if __name__ == "__main__":
    main()
