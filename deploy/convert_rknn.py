"""ONNX -> RKNN for RK3588 (FP16 or INT8) + check in the PC simulator against onnxruntime."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import HERE

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN

from deploy.preprocess import preprocess

DEFAULT_IMAGE = os.path.join(HERE, "data", "hpatches-sequences-release", "v_tabletop", "1.ppm")
NAMES = ["feats", "keypoints", "heatmap"]


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def desc_cosine(a, b):
    """Mean per-cell cosine of L2-normalized 64-D descriptors -- what matching actually uses."""
    a = a.reshape(a.shape[1], -1).T
    b = b.reshape(b.shape[1], -1).T
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return float((a * b).sum(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--dtype", choices=["fp16", "i8"], default="fp16")
    ap.add_argument("--dataset", default=None, help="calibration dataset.txt (needed for i8)")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--accuracy_analysis", action="store_true", help="per-layer comparison (slow)")
    args = ap.parse_args()
    if args.dtype == "i8" and not args.dataset:
        ap.error("--dataset is required for --dtype i8")

    out = args.onnx.replace(".onnx", f"_{args.dtype}.rknn")

    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3588", optimization_level=3)   # no mean/std: input is already normalized
    assert rknn.load_onnx(model=args.onnx) == 0, "load_onnx failed"
    assert rknn.build(do_quantization=(args.dtype == "i8"), dataset=args.dataset) == 0, "build failed"
    assert rknn.export_rknn(out) == 0, "export failed"
    print(f"--> saved {out} ({os.path.getsize(out) / 1e6:.2f} MB)")

    # reference: onnxruntime on CPU, same normalized input
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    h, w = sess.get_inputs()[0].shape[2:]
    _, norm = preprocess(args.image, h, w)
    ref = sess.run(None, {"image": norm})

    assert rknn.init_runtime() == 0, "init_runtime failed"            # no target -> simulator
    sim = rknn.inference(inputs=[norm], data_format="nchw")

    print(f"--> simulator ({args.dtype}) vs onnxruntime:")
    for name, r, s in zip(NAMES, ref, sim):
        s = s.reshape(r.shape)
        print(f"    {name:9s} cos={cosine(r, s):.5f}  max|diff|={np.abs(r - s).max():.3e}  ref range=[{r.min():.2f}, {r.max():.2f}]")
    print(f"    descriptors: mean per-cell cosine = {desc_cosine(ref[0], sim[0].reshape(ref[0].shape)):.5f}")

    if args.accuracy_analysis:
        npy = os.path.join(os.path.dirname(out), "_aa_input.npy")
        np.save(npy, norm)                                   # NCHW, same as the calibration .npy
        snap = os.path.join(os.path.dirname(out), "snapshot_" + os.path.basename(out).replace(".rknn", ""))
        rknn.accuracy_analysis(inputs=[npy], output_dir=snap)
        print(f"--> per-layer report: {snap}/error_analysis.txt")

    rknn.release()


if __name__ == "__main__":
    main()
