"""Build the INT8 calibration set: normalized HPatches frames saved as .npy + dataset.txt for rknn.build()."""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import HERE

import numpy as np

from deploy.preprocess import load_gray, normalize

HPATCHES = os.path.join(HERE, "data", "hpatches-sequences-release")
CALIB_DIR = os.path.join(HERE, "deploy", "calib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    seqs = sorted(glob.glob(os.path.join(HPATCHES, "*")))
    idx = np.linspace(0, len(seqs) - 1, args.num).round().astype(int)    # evenly over i_* and v_* sequences
    out_dir = os.path.join(CALIB_DIR, f"{args.width}x{args.height}")
    os.makedirs(out_dir, exist_ok=True)

    lines = []
    for i in idx:
        src = os.path.join(seqs[i], "1.ppm")
        x = normalize(load_gray(src, args.height, args.width))[None, None]          # (1, 1, H, W) = NCHW, as the ONNX input
        dst = os.path.join(out_dir, os.path.basename(seqs[i]) + ".npy")
        np.save(dst, x)
        lines.append(dst)

    txt = os.path.join(out_dir, "dataset.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} samples -> {txt}")


if __name__ == "__main__":
    main()
