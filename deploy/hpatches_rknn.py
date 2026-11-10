"""HPatches evaluation of the deployed XFeat backbone: onnxruntime (float reference) or RKNN simulator.

Protocol = benchmark/hpatches_eval.py (same pairs, resize, MAGSAC, metrics, summary). Differences:
  * the network runs at a FIXED input size: landscape images -> W x H model, portrait -> H x W model;
  * network outputs come from --backend; post-processing is XFeat's own code (detectAndCompute), unchanged.
Stage 1 extracts keypoints + descriptors for every image and caches them (one RKNN instance at a time);
stage 2 matches cached features (MNN, min_cossim=-1, as xfeat.match_xfeat) and computes the metrics.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # post-processing on CPU, as on the board

import argparse
import json
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import HERE, RESULTS_DIR                  # also puts XFeat on sys.path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from benchmark.hpatches_eval import (list_pairs, resize_short, load_pair, magsac, corner_error, mma,
                                     summarize, print_summary, environment, to_json)
from deploy.preprocess import normalize
from modules.interpolator import InterpolateSparse2d
from modules.xfeat import XFeat

MODELS_DIR = os.path.join(HERE, "deploy", "models")
CALIB_DIR = os.path.join(HERE, "deploy", "calib")
CACHE_DIR = os.path.join(HERE, "deploy", "cache")

XF = XFeat(weights=None)       # only its post-processing / matching methods are used, not its network
NEAREST = InterpolateSparse2d("nearest")
BILINEAR = InterpolateSparse2d("bilinear")


# ---------------------------------------------------------------- backends: (1,1,H,W) -> 3 arrays
class OnnxNet:
    def __init__(self, onnx_path, dtype=None, dataset=None, quant_algo="normal"):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def __call__(self, x):
        return self.sess.run(None, {"image": x})

    def release(self):
        pass


class RknnNet:
    """The PC simulator only runs a model built in the same session, so build from ONNX here
    with exactly the settings of deploy/convert_rknn.py."""
    def __init__(self, onnx_path, dtype, dataset, quant_algo="normal"):
        from rknn.api import RKNN
        self.rknn = RKNN(verbose=False)
        self.rknn.config(target_platform="rk3588", optimization_level=3, quantized_algorithm=quant_algo)
        assert self.rknn.load_onnx(model=onnx_path) == 0, "load_onnx failed"
        assert self.rknn.build(do_quantization=(dtype == "i8"), dataset=dataset) == 0, "build failed"
        assert self.rknn.init_runtime() == 0, "init_runtime failed"

    def __call__(self, x):
        return self.rknn.inference(inputs=[x], data_format="nchw")

    def release(self):
        self.rknn.release()


# ---------------------------------------------------------------- XFeat post-processing (B = 1)
@torch.inference_mode()
def postprocess(feats, kpts, heat, rh, rw, top_k, threshold=0.05):
    """Same steps as XFeat.detectAndCompute after `M1, K1, H1 = self.net(x)`."""
    M1 = F.normalize(torch.from_numpy(feats), dim=1)
    K1 = torch.from_numpy(kpts)
    H1 = torch.from_numpy(heat)
    _H1, _W1 = H1.shape[-2] * 8, H1.shape[-1] * 8

    K1h = XF.get_kpts_heatmap(K1)
    mkpts = XF.NMS(K1h, threshold=threshold, kernel_size=5)
    scores = (NEAREST(K1h, mkpts, _H1, _W1) * BILINEAR(H1, mkpts, _H1, _W1)).squeeze(-1)
    scores[torch.all(mkpts == 0, dim=-1)] = -1

    idxs = torch.argsort(-scores)
    mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :top_k]
    mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :top_k]
    mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
    scores = torch.gather(scores, -1, idxs)[:, :top_k]

    desc = F.normalize(XF.interpolator(M1, mkpts, H=_H1, W=_W1), dim=-1)
    mkpts = mkpts * torch.tensor([rw, rh]).view(1, 1, -1)
    valid = scores[0] > 0
    return mkpts[0][valid].numpy().astype(np.float32), desc[0][valid].numpy().astype(np.float32)


# ---------------------------------------------------------------- stage 1: extraction + cache
def read_image(root, seq, i, short):
    img = cv2.imread(os.path.join(root, seq, f"{i}.ppm"))
    return resize_short(img, short)[0]                       # identical to load_pair()


def model_size(img, args):
    """(W, H) of the model used for this image: landscape/square -> W x H, portrait -> H x W."""
    h, w = img.shape[:2]
    return (args.height, args.width) if h > w else (args.width, args.height)


def cache_path(cache, seq, i):
    return os.path.join(cache, f"{seq}_{i}.npz")


def extract(args, images, cache):
    groups = {}
    for seq, i in images:
        if not os.path.exists(cache_path(cache, seq, i)):
            groups.setdefault(model_size(read_image(args.data_dir, seq, i, args.resize), args), []).append((seq, i))
    if not groups:
        print("all features cached")
        return {}

    net_cls = RknnNet if args.backend == "rknn" else OnnxNet
    used = {}
    for (W, H), todo in groups.items():
        onnx_path = os.path.join(MODELS_DIR, f"{args.name}_{W}x{H}.onnx")
        dataset = os.path.join(CALIB_DIR, f"{W}x{H}", "dataset.txt")
        for p in [onnx_path] + ([dataset] if args.backend == "rknn" and args.dtype == "i8" else []):
            if not os.path.exists(p):
                sys.exit(f"missing {p}")
        print(f"--> {args.backend} {W}x{H}: {len(todo)} images, model {os.path.relpath(onnx_path, HERE)}")
        used[f"{W}x{H}"] = {"onnx": os.path.relpath(onnx_path, HERE), "images": len(todo)}

        net = net_cls(onnx_path, args.dtype, dataset, args.quant_algo)
        shapes = [(1, 64, H // 8, W // 8), (1, 65, H // 8, W // 8), (1, 1, H // 8, W // 8)]
        t0 = time.time()
        for n, (seq, i) in enumerate(todo, 1):
            img = read_image(args.data_dir, seq, i, args.resize)
            h, w = img.shape[:2]
            gray = img.astype(np.float32).mean(axis=2)                      # XFeat: x.mean(dim=1)
            x = normalize(cv2.resize(gray, (W, H), interpolation=cv2.INTER_LINEAR))[None, None]
            feats, kpts, heat = [np.asarray(o, np.float32).reshape(s) for o, s in zip(net(x), shapes)]
            k, d = postprocess(feats, kpts, heat, rh=h / H, rw=w / W, top_k=args.top_k)
            np.savez(cache_path(cache, seq, i), kpts=k, desc=d)
            if n % 50 == 0 or n == len(todo):
                print(f"    {n}/{len(todo)} images, {time.time() - t0:.0f} s")
        net.release()
    return used


# ---------------------------------------------------------------- stage 2: matching + metrics
@torch.inference_mode()
def match_cached(cache, seq, i0, i1):
    a, b = np.load(cache_path(cache, seq, i0)), np.load(cache_path(cache, seq, i1))
    idx0, idx1 = XF.match(torch.from_numpy(a["desc"]), torch.from_numpy(b["desc"]), min_cossim=-1)
    return a["kpts"][idx0.numpy()].astype(np.float64), b["kpts"][idx1.numpy()].astype(np.float64)


def parse_args():
    p = argparse.ArgumentParser(description="HPatches evaluation of the XFeat backbone via ONNX / RKNN simulator")
    p.add_argument("--backend", choices=["onnx", "rknn"], required=True)
    p.add_argument("--dtype", choices=["fp16", "i8"], default="fp16", help="rknn only")
    p.add_argument("--quant-algo", choices=["normal", "mmse", "kl_divergence"], default="normal", help="rknn i8 only")
    p.add_argument("--name", default="xfeat", help="model prefix: deploy/models/<name>_<W>x<H>.onnx")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--data-dir", default=os.path.join(HERE, "data", "hpatches-sequences-release"))
    p.add_argument("--subset", choices=["all", "108"], default="all")
    p.add_argument("--resize", type=int, default=480)
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--ransac-thr", type=float, default=3.0)
    p.add_argument("--ransac-iters", type=int, default=10000)
    p.add_argument("--ransac-conf", type=float, default=0.999)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--max-seqs", type=int, default=0, help="debug: use only N sequences")
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    tag = f"{args.name}_{args.backend}" + (f"_{args.dtype}" if args.backend == "rknn" else "")
    if args.backend == "rknn" and args.dtype == "i8" and args.quant_algo != "normal":
        tag += f"_{args.quant_algo}"
    cache = os.path.join(args.cache_dir, f"{tag}_r{args.resize}_k{args.top_k}")
    os.makedirs(cache, exist_ok=True)

    pairs = list_pairs(args.data_dir, args.subset, args.max_seqs)
    images = sorted({(s, 1) for s, _ in pairs} | set(pairs))
    print(f"{len(pairs)} pairs, {len(images)} images, backend {tag}, cache {os.path.relpath(cache, HERE)}")

    t = time.time()
    used = extract(args, images, cache)
    t_extract = time.time() - t

    records = []
    for seq, i in pairs:
        img0, img1, H_gt = load_pair(args.data_dir, seq, i, args.resize)
        h0, w0 = img0.shape[:2]
        pts0, pts1 = match_cached(cache, seq, 1, i)
        errs, inliers = [], []
        for seed in args.seeds:
            H_est, n_in = magsac(pts0, pts1, args.ransac_thr, args.ransac_iters, args.ransac_conf, seed)
            errs.append(corner_error(H_est, H_gt, w0, h0))
            inliers.append(n_in)
        records.append({"seq": seq, "pair": f"1-{i}", "size0_hw": [h0, w0],
                        "size1_hw": list(img1.shape[:2]), "n_matches": int(len(pts0)),
                        "mma": mma(pts0, pts1, H_gt), "corner_error": errs, "n_inliers": inliers})

    summary = summarize(records, args.seeds)
    print_summary(summary, args.seeds)

    out = args.out or os.path.join(
        RESULTS_DIR, f"hpatches_{tag}_{args.subset}_r{args.resize}_thr{args.ransac_thr:g}.json")
    info = {"backend": args.backend, "dtype": args.dtype if args.backend == "rknn" else "fp32",
            "model_input_wh": [args.width, args.height], "models": used, "extract_time_s": t_extract,
            "matcher": "XFeat detectAndCompute post-processing + XFeat.match (MNN, min_cossim=-1)"}
    result = {"config": vars(args), "environment": environment(info),
              "thresholds": {"mma": list(range(1, 11)), "auc": [3, 5, 10], "mha": [3, 5, 7]},
              "summary": summary, "pairs": records}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(to_json(result), f, indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
