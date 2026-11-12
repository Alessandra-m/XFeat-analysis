"""ONNX -> RKNN for RK3588 (FP16 or INT8) + check in the PC simulator against onnxruntime."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import HERE

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN

from deploy.preprocess import model_inputs

DEFAULT_IMAGE = os.path.join(HERE, "data", "hpatches-sequences-release", "v_tabletop", "1.ppm")


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
    ap.add_argument("--quant-algo", choices=["normal", "mmse", "kl_divergence"], default="normal",
                    help="INT8 calibration algorithm")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--accuracy_analysis", action="store_true", help="per-layer comparison (slow)")
    ap.add_argument("--verbose", action="store_true",
                    help="full toolkit log incl. the compiler's per-op cycle estimate table; "
                         "it goes to the console, redirect it: ... --verbose > log.txt 2>&1, "
                         "then python -m deploy.parse_cycles log.txt")
    args = ap.parse_args()
    suffix = args.dtype + ("" if args.quant_algo == "normal" else f"_{args.quant_algo}")
    out = args.onnx.replace(".onnx", f"_{suffix}.rknn")

    rknn = RKNN(verbose=args.verbose)
    rknn.config(target_platform="rk3588", optimization_level=3,   # no mean/std: input is already normalized
                quantized_algorithm=args.quant_algo)
    assert rknn.load_onnx(model=args.onnx) == 0, "load_onnx failed"
    assert rknn.build(do_quantization=(args.dtype == "i8"), dataset=args.dataset) == 0, "build failed"
    assert rknn.export_rknn(out) == 0, "export failed"
    print(f"--> saved {out} ({os.path.getsize(out) / 1e6:.2f} MB)")

    # reference: onnxruntime on CPU, same inputs. Input type follows the model (deploy/preprocess.model_inputs):
    # 1 channel -> normalized gray, 3 -> raw BGR 0..255 (gray model), f*f -> normalized gray, space-to-depth by f
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inps = sess.get_inputs()
    xs = model_inputs(args.image, [i.shape for i in inps])
    norm = xs[0]                                                      # for --accuracy_analysis
    ref = sess.run(None, {i.name: x for i, x in zip(inps, xs)})
    names = [o.name for o in sess.get_outputs()]

    assert rknn.init_runtime() == 0, "init_runtime failed"            # no target -> simulator
    sim = rknn.inference(inputs=xs, data_format=["nchw"] * len(xs))

    shapes = ", ".join(f"{i.name} {tuple(x.shape)}" for i, x in zip(inps, xs))
    print(f"--> simulator ({args.dtype}) vs onnxruntime, input {shapes}:")
    for name, r, s in zip(names, ref, sim):
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
