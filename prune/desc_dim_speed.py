import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import XFEAT_DIR

import numpy as np
import torch
import torch.nn.functional as F

from modules.xfeat import XFeat


def timed(fn, sync, runs, warmup):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync()
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            fn()
            sync()
            ts.append((time.perf_counter() - t) * 1000.0)
    return float(np.median(ts))


def parse_args():
    p = argparse.ArgumentParser(description="Cost of the descriptor-size-dependent steps (MNN, descriptor "
                                            "interpolation) for D = 64 / 48 / 32: forecast for the optional "
                                            "64 -> 48/32 config of the fine-tuning stage")
    p.add_argument("--dims", type=int, nargs="+", default=[64, 48, 32])
    p.add_argument("--devices", nargs="+", default=["cpu", "cuda"], choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--vga-w", type=int, default=640)
    p.add_argument("--vga-h", type=int, default=480)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    xf = XFeat(top_k=args.top_k)
    n, h8, w8 = args.top_k, args.vga_h // 8, args.vga_w // 8

    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
        # keypoints inside the image, as detectAndCompute passes them to the interpolator
        kpts = torch.stack([torch.randint(0, args.vga_w, (1, n)), torch.randint(0, args.vga_h, (1, n))],
                           -1).float().to(device)
        print(f"\n{device.upper()}" + (f", {torch.get_num_threads()} threads" if device == "cpu" else "")
              + f" | top_k {n} | median of {args.runs} runs, ms per pair")
        print(f"{'D':>4}{'MNN sparse':>12}{'MNN XFeat*':>12}{'interp x2':>11}{'sum':>9}{'saved':>9}")
        base = None
        for d in args.dims:
            d0 = F.normalize(torch.randn(n, d, device=device), dim=-1)
            d1 = F.normalize(torch.randn(n, d, device=device), dim=-1)
            fmap = torch.randn(1, d, h8, w8, device=device)
            mnn = timed(lambda: xf.match(d0, d1, min_cossim=-1), sync, args.runs, args.warmup)
            bmm = timed(lambda: xf.batch_match(d0[None, :n - 1], d1[None, :n - 1]), sync, args.runs, args.warmup)
            itp = 2 * timed(lambda: F.normalize(xf.interpolator(fmap, kpts, H=args.vga_h, W=args.vga_w), dim=-1),
                            sync, args.runs, args.warmup)
            total = mnn + itp
            base = base or total
            print(f"{d:>4}{mnn:>12.2f}{bmm:>12.2f}{itp:>11.2f}{total:>9.2f}{base - total:>9.2f}")
    print("\nsparse pair = 2 extractions + MNN: 'sum' covers MNN and the two descriptor interpolations;"
          "\nthe network itself changes only in its last 1x1 conv (64 -> D), negligible.")


if __name__ == "__main__":
    main()
