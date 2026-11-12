"""Build the INT8 calibration set: normalized HPatches frames saved as .npy + dataset.txt for rknn.build()."""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import HERE

import numpy as np

from deploy.preprocess import load_gray, load_raw, normalize, s2d

HPATCHES = os.path.join(HERE, "data", "hpatches-sequences-release")
CALIB_DIR = os.path.join(HERE, "deploy", "calib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--raw", action="store_true",
                    help="raw BGR (1, 3, H, W) 0..255 for the gray model; default: normalized gray (1, 1, H, W)")
    ap.add_argument("--s2d", type=int, nargs="+", default=None, metavar="F",
                    help="space-to-depth model inputs (deploy/s2d_net.py), one per factor, e.g. --s2d 4 "
                         "or --s2d 4 8 for two inputs; each line of dataset.txt lists one .npy per input")
    args = ap.parse_args()

    seqs = sorted(glob.glob(os.path.join(HPATCHES, "*")))
    idx = np.linspace(0, len(seqs) - 1, args.num).round().astype(int)    # evenly over i_* and v_* sequences
    tag = "_raw" if args.raw else ("_s2d" + "_".join(map(str, args.s2d)) if args.s2d else "")
    out_dir = os.path.join(CALIB_DIR, f"{args.width}x{args.height}" + tag)
    os.makedirs(out_dir, exist_ok=True)

    lines = []
    for i in idx:
        src = os.path.join(seqs[i], "1.ppm")
        name = os.path.basename(seqs[i])
        if args.raw:
            xs = {"": load_raw(src, args.height, args.width)}                       # (1, 3, H, W), 0..255
        else:
            x = normalize(load_gray(src, args.height, args.width))[None, None]      # (1, 1, H, W) = NCHW, as the ONNX input
            xs = {f"_f{f}": s2d(x, f) for f in args.s2d} if args.s2d else {"": x}
        paths = []
        for suffix, x in xs.items():
            dst = os.path.join(out_dir, name + suffix + ".npy")
            np.save(dst, x)
            paths.append(dst)
        lines.append(" ".join(paths))                       # several inputs: space-separated on one line

    txt = os.path.join(out_dir, "dataset.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} samples -> {txt}")


if __name__ == "__main__":
    main()
