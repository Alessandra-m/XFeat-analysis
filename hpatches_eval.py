import argparse
import json
import os
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

from paths import RESULTS_DIR, XFEAT_DIR, HERE

MMA_THR = list(range(1, 11))
AUC_THR = [3, 5, 10]
MHA_THR = [3, 5, 7]

# high-resolution sequences excluded in the 108-scene variant (D2-Net and later works)
EXCLUDED_108 = ["i_contruction", "i_crownnight", "i_dc", "i_pencils", "i_whitebuilding",
                "v_artisans", "v_astronautis", "v_talent"]

# XFeat paper (CVPR 2024), Table 3, MHA in %, sparse XFeat
PAPER_MHA = {"i": {3: 95.0, 5: 98.1, 7: 98.8}, "v": {3: 68.6, 5: 81.1, 7: 86.1}}



def list_pairs(root, subset, max_seqs=0):
    seqs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and d[:2] in ("i_", "v_"))
    if subset == "108":
        seqs = [s for s in seqs if s not in EXCLUDED_108]
    if max_seqs > 0:
        # keep both types when debugging on a few sequences
        i_seqs = [s for s in seqs if s.startswith("i_")][:(max_seqs + 1) // 2]
        v_seqs = [s for s in seqs if s.startswith("v_")][:max_seqs // 2]
        seqs = i_seqs + v_seqs
    return [(s, i) for s in seqs for i in range(2, 7)]


def resize_short(img, short):
    """Resize so that the shorter side equals `short`. Returns the image and its scale matrix."""
    h, w = img.shape[:2]
    if short <= 0:
        return img, np.eye(3)
    s = short / min(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    out = cv2.resize(img, (nw, nh), interpolation=interp)
    # separate x / y factors: rounding makes them slightly different
    return out, np.diag([nw / w, nh / h, 1.0])


def load_pair(root, seq, i, short):
    d = os.path.join(root, seq)
    img0 = cv2.imread(os.path.join(d, "1.ppm"))
    img1 = cv2.imread(os.path.join(d, f"{i}.ppm"))
    if img0 is None or img1 is None:
        raise FileNotFoundError(f"cannot read images of {seq}, pair 1-{i}")
    H = np.loadtxt(os.path.join(d, f"H_1_{i}"))
    img0, S0 = resize_short(img0, short)
    img1, S1 = resize_short(img1, short)
    return img0, img1, S1 @ H @ np.linalg.inv(S0)



def make_matcher(name, top_k):
    """Return (fn(img0, img1) -> (mkpts0, mkpts1), info dict)."""
    if name == "xfeat":
        import torch
        from modules.xfeat import XFeat

        xfeat = XFeat(top_k=top_k)

        def fn(img0, img1):
            with torch.inference_mode():
                return xfeat.match_xfeat(img0, img1, top_k=top_k, min_cossim=-1)

        return fn, {"torch": torch.__version__, "device": str(xfeat.dev),
                    "matcher": "xfeat.match_xfeat (MNN, min_cossim=-1)"}

    if name == "orb":
        # sanity baseline: same pipeline with a classic detector, no torch needed
        orb = cv2.ORB_create(nfeatures=top_k)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        def fn(img0, img1):
            k0, d0 = orb.detectAndCompute(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), None)
            k1, d1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
            if d0 is None or d1 is None:
                return np.zeros((0, 2)), np.zeros((0, 2))
            m = bf.match(d0, d1)
            return (np.float64([k0[x.queryIdx].pt for x in m]).reshape(-1, 2),
                    np.float64([k1[x.trainIdx].pt for x in m]).reshape(-1, 2))

        return fn, {"matcher": "cv2.ORB + BFMatcher crossCheck (MNN)"}

    raise ValueError(name)



def reprojection_errors(pts0, pts1, H):
    proj = cv2.perspectiveTransform(pts0.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    return np.linalg.norm(proj - pts1, axis=1)


def mma(pts0, pts1, H):
    if len(pts0) == 0:
        return {t: 0.0 for t in MMA_THR}
    err = reprojection_errors(pts0, pts1, H)
    return {t: float((err <= t).mean()) for t in MMA_THR}


def magsac(pts0, pts1, thr, max_iters, confidence, seed):
    """MAGSAC with explicit parameters. cv2.findHomography(..., USAC_MAGSAC) has a fixed
    internal seed, so cv2.setRNGSeed does not change it; UsacParams.randomGeneratorState does."""
    if len(pts0) < 4:
        return None, 0
    p = cv2.UsacParams()
    p.threshold = thr
    p.maxIterations = max_iters
    p.confidence = confidence
    p.score = cv2.SCORE_METHOD_MAGSAC
    p.sampler = cv2.SAMPLING_UNIFORM
    p.loMethod = cv2.LOCAL_OPTIM_SIGMA
    p.final_polisher = cv2.MAGSAC
    p.randomGeneratorState = seed
    try:
        H, mask = cv2.findHomography(pts0.astype(np.float32), pts1.astype(np.float32), p)
    except cv2.error:
        return None, 0
    if H is None:
        return None, 0
    return H, int(mask.sum())


def corner_error(H_est, H_gt, w, h):
    if H_est is None:
        return np.inf
    c = np.float64([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    d = cv2.perspectiveTransform(c, H_est) - cv2.perspectiveTransform(c, H_gt)
    return float(np.linalg.norm(d, axis=-1).mean())


def error_auc(errors, thresholds):
    """Copied from XFeat modules/eval/megadepth1500.py (importing it needs poselib).
    Failed pairs (inf) stay in the denominator and count as misses."""
    trapz = getattr(np, "trapezoid", None) or np.trapz
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))
    aucs = {}
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs[thr] = float(trapz(y, x) / thr)
    return aucs


def mha(errors, thresholds):
    e = np.asarray(errors)
    return {t: float((e <= t).mean()) for t in thresholds}



def mean_std(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def summarize(records, seeds):
    out = {}
    splits = {"i": [r for r in records if r["seq"].startswith("i_")],
              "v": [r for r in records if r["seq"].startswith("v_")],
              "all": records}
    for name, recs in splits.items():
        if not recs:
            continue
        s = {"pairs": len(recs),
             "matches_mean": float(np.mean([r["n_matches"] for r in recs])),
             "mma": {t: float(np.mean([r["mma"][t] for r in recs])) for t in MMA_THR}}
        per_seed = []
        for k, _ in enumerate(seeds):
            errs = [r["corner_error"][k] for r in recs]
            per_seed.append({"auc": error_auc(errs, AUC_THR), "mha": mha(errs, MHA_THR),
                             "failures": int(np.sum(~np.isfinite(errs))),
                             "inliers_mean": float(np.mean([r["n_inliers"][k] for r in recs]))})
        s["auc"] = {t: mean_std([p["auc"][t] for p in per_seed]) for t in AUC_THR}
        s["mha"] = {t: mean_std([p["mha"][t] for p in per_seed]) for t in MHA_THR}
        s["failures"] = mean_std([p["failures"] for p in per_seed])
        s["inliers_mean"] = mean_std([p["inliers_mean"] for p in per_seed])
        out[name] = s
    return out


def print_summary(summary, seeds):
    print(f"\nMMA, % (share of matches with reprojection error <= t px, mean over pairs)")
    print(f"{'split':<6}{'pairs':>6}{'matches':>9}" + "".join(f"{'@' + str(t):>7}" for t in MMA_THR))
    for name, s in summary.items():
        print(f"{name:<6}{s['pairs']:>6}{s['matches_mean']:>9.0f}"
              + "".join(f"{100 * s['mma'][t]:>7.1f}" for t in MMA_THR))

    pm = "+-" if len(seeds) > 1 else ""
    print(f"\nhomography, % (mean{' +- std' if pm else ''} over {len(seeds)} MAGSAC seed(s))")
    print(f"{'split':<6}" + "".join(f"{'AUC@' + str(t):>13}" for t in AUC_THR)
          + "".join(f"{'MHA@' + str(t):>13}" for t in MHA_THR) + f"{'inliers':>9}{'fail':>6}")

    def cell(v):
        txt = f"{100 * v['mean']:.1f}"
        if pm:
            txt += f"+-{100 * v['std']:.1f}"
        return f"{txt:>13}"

    for name, s in summary.items():
        print(f"{name:<6}" + "".join(cell(s["auc"][t]) for t in AUC_THR)
              + "".join(cell(s["mha"][t]) for t in MHA_THR)
              + f"{s['inliers_mean']['mean']:>9.0f}{s['failures']['mean']:>6.1f}")

    print("\nMHA vs XFeat paper (Table 3), %")
    print(f"{'split':<6}" + "".join(f"{'@' + str(t) + ' paper':>11}{'ours':>7}" for t in MHA_THR))
    for name in ("i", "v"):
        if name in summary:
            print(f"{name:<6}" + "".join(
                f"{PAPER_MHA[name][t]:>11.1f}{100 * summary[name]['mha'][t]['mean']:>7.1f}"
                for t in MHA_THR))



def git_commit(path):
    # a submodule has its own .git file/dir; without it git would report the parent repo
    if not os.path.exists(os.path.join(path, ".git")):
        return None
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def environment(matcher_info):
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "processor": platform.processor() or platform.machine(),
           "opencv": cv2.__version__, "numpy": np.__version__,
           "repo_commit": git_commit(HERE), "xfeat_commit": git_commit(XFEAT_DIR)}
    env.update(matcher_info)
    return env


def to_json(obj):
    """inf is not valid JSON: store failed corner errors as null."""
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    return obj



def parse_args():
    p = argparse.ArgumentParser(description="HPatches evaluation: MMA, homography AUC and MHA")
    p.add_argument("--data-dir", default=os.environ.get(
        "HPATCHES_DIR", os.path.join(HERE, "data", "hpatches-sequences-release")))
    p.add_argument("--matcher", choices=["xfeat", "orb"], default="xfeat")
    p.add_argument("--subset", choices=["all", "108"], default="all",
                   help="all 116 sequences or 108 without the high-resolution ones")
    p.add_argument("--resize", type=int, default=480,
                   help="shorter image side in px, 0 keeps the original resolution")
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--ransac-thr", type=float, default=3.0, help="MAGSAC threshold, px")
    p.add_argument("--ransac-iters", type=int, default=10000)
    p.add_argument("--ransac-conf", type=float, default=0.999)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--cpu", action="store_true", help="hide CUDA devices from torch")
    p.add_argument("--max-seqs", type=int, default=0, help="debug: use only N sequences")
    p.add_argument("--out", default=None, help="output JSON (default: results/hpatches_<matcher>_...json)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    pairs = list_pairs(args.data_dir, args.subset, args.max_seqs)
    if not pairs:
        sys.exit(f"no sequences found in {args.data_dir}")
    print(f"{len(pairs)} pairs from {len(pairs) // 5} sequences, matcher {args.matcher}, "
          f"resize short side {args.resize or 'off'}, MAGSAC thr {args.ransac_thr}, seeds {args.seeds}")

    match_fn, matcher_info = make_matcher(args.matcher, args.top_k)

    records = []
    t_start = time.time()
    for n, (seq, i) in enumerate(pairs, 1):
        img0, img1, H_gt = load_pair(args.data_dir, seq, i, args.resize)
        h0, w0 = img0.shape[:2]

        t = time.perf_counter()
        pts0, pts1 = match_fn(img0, img1)
        t_match = time.perf_counter() - t

        # matching is deterministic, only MAGSAC depends on the seed
        errs, inliers = [], []
        for seed in args.seeds:
            H_est, n_in = magsac(pts0, pts1, args.ransac_thr, args.ransac_iters, args.ransac_conf, seed)
            errs.append(corner_error(H_est, H_gt, w0, h0))
            inliers.append(n_in)

        records.append({"seq": seq, "pair": f"1-{i}", "size0_hw": [h0, w0],
                        "size1_hw": list(img1.shape[:2]), "n_matches": int(len(pts0)),
                        "mma": mma(pts0, pts1, H_gt), "corner_error": errs,
                        "n_inliers": inliers, "match_time_s": t_match})

        if n % 50 == 0 or n == len(pairs):
            print(f"  {n}/{len(pairs)} pairs, {time.time() - t_start:.0f} s")

    summary = summarize(records, args.seeds)
    print_summary(summary, args.seeds)

    out = args.out or os.path.join(
        RESULTS_DIR, f"hpatches_{args.matcher}_{args.subset}_r{args.resize}_thr{args.ransac_thr:g}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    result = {"config": vars(args), "environment": environment(matcher_info),
              "thresholds": {"mma": MMA_THR, "auc": AUC_THR, "mha": MHA_THR},
              "summary": summary, "pairs": records}
    with open(out, "w") as f:
        json.dump(to_json(result), f, indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
