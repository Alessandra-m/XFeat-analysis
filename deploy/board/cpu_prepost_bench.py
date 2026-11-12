"""Time on the board CPU the operations we want to move out of the NPU graph (numpy only).

    python3 cpu_prepost_bench.py            # on the board, needs python3-numpy

pre  : raw BGR (480, 640, 3) uint8 -> mean over channels -> (x - mean) / std    (gray model's ReduceMean + InstanceNorm)
post : softmax over 65 channels + drop dustbin, L2-normalize 64-D feats          (gray model's softmax/slice + L2)
s2d  : space-to-depth of the normalized frame by 4 -> (16, 120, 160) and by 8 -> (64, 60, 80)
"""
import time

import numpy as np

H, W = 480, 640
N = 50


def bench(name, fn, *args):
    fn(*args)                                             # warm-up
    t = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn(*args)
        t.append((time.perf_counter() - t0) * 1e3)
    t = np.array(t)
    print(f"{name:42s} median {np.median(t):6.2f} ms   min {t.min():6.2f}   max {t.max():6.2f}")


def pre(bgr):                                             # == x.mean(dim=1) + InstanceNorm2d(1), eps=1e-5
    g = bgr.astype(np.float32).mean(axis=2)
    return (g - g.mean()) / np.sqrt(g.var() + 1e-5)


def post(feats, logits):
    e = np.exp(logits - logits.max(axis=0, keepdims=True))
    prob = (e / e.sum(axis=0, keepdims=True))[:64]
    desc = feats / np.maximum(np.linalg.norm(feats, axis=0, keepdims=True), 1e-12)
    return prob, desc


def s2d(x, r):                                            # (H, W) -> (r*r, H/r, W/r), same order as pixel_unshuffle
    h, w = x.shape
    return np.ascontiguousarray(x.reshape(h // r, r, w // r, r).transpose(1, 3, 0, 2).reshape(r * r, h // r, w // r))


def main():
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    norm = pre(bgr).astype(np.float32)
    feats = rng.standard_normal((64, H // 8, W // 8)).astype(np.float32)
    logits = rng.standard_normal((65, H // 8, W // 8)).astype(np.float32)

    print(f"numpy {np.__version__}, {N} runs each, frame {W}x{H}")
    bench("pre: mean RGB + normalize", pre, bgr)
    bench("post: softmax(65)[:64] + L2(64)", post, feats, logits)
    bench("s2d by 4 -> (16, 120, 160)", s2d, norm, 4)
    bench("s2d by 8 -> (64, 60, 80)", s2d, norm, 8)


if __name__ == "__main__":
    main()
