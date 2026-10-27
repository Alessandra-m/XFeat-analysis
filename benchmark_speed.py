import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

from paths import RESULTS_DIR, XFEAT_DIR, HERE


def git_commit(path):
    if not os.path.exists(os.path.join(path, ".git")):
        return None
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def git_dirty(path):
    """True if tracked code differs from HEAD; results/ is ignored, runs rewrite it."""
    if not os.path.exists(os.path.join(path, ".git")):
        return None
    try:
        out = subprocess.check_output(
            ["git", "-C", path, "status", "--porcelain", "--untracked-files=no",
             "--", ".", ":(exclude)results"], stderr=subprocess.DEVNULL, text=True)
        return bool(out.strip())
    except Exception:
        return None


def stats(times_ms):
    a = np.asarray(times_ms, dtype=np.float64)
    return {"runs": int(a.size),
            "mean_ms": float(a.mean()), "std_ms": float(a.std()),
            "median_ms": float(np.median(a)), "min_ms": float(a.min()),
            "max_ms": float(a.max()), "p95_ms": float(np.percentile(a, 95)),
            "fps_mean": float(1000.0 / a.mean())}


def parse_args():
    p = argparse.ArgumentParser(description="XFeat inference-speed benchmark (VGA)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--width", type=int, default=640, help="input width, px (VGA=640)")
    p.add_argument("--height", type=int, default=480, help="input height, px (VGA=480)")
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--runs", type=int, default=100, help="measured runs (>=100 for the task)")
    p.add_argument("--warmup", type=int, default=10, help="warm-up runs, not measured")
    p.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads to pin; 0 keeps the default and just records it")
    p.add_argument("--seed", type=int, default=0, help="seed for the synthetic input")
    p.add_argument("--out", default=None, help="output JSON (default: results/speed_<device>_*.json)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    from modules.xfeat import XFeat

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    n_threads = torch.get_num_threads()

    torch.manual_seed(args.seed)
    xfeat = XFeat(top_k=args.top_k)
    assert xfeat.dev.type == args.device, f"expected {args.device}, got {xfeat.dev}"
    cuda = args.device == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    x0 = torch.randint(0, 256, (1, 3, args.height, args.width), dtype=torch.float32)
    x1 = torch.randint(0, 256, (1, 3, args.height, args.width), dtype=torch.float32)

    def run_extract():
        with torch.inference_mode():
            xfeat.detectAndCompute(x0, top_k=args.top_k)

    def run_extract_match():
        with torch.inference_mode():
            xfeat.match_xfeat(x0, x1, top_k=args.top_k, min_cossim=-1)

    with torch.inference_mode():
        d0 = xfeat.detectAndCompute(x0, top_k=args.top_k)[0]["descriptors"]
        d1 = xfeat.detectAndCompute(x1, top_k=args.top_k)[0]["descriptors"]

    def run_match():
        with torch.inference_mode():
            xfeat.match(d0, d1, min_cossim=-1)

    print(f"XFeat {args.device.upper()} benchmark | input {args.width}x{args.height} | top_k {args.top_k} "
          f"| threads {n_threads} | warmup {args.warmup} | runs {args.runs}")

    results = {}
    for label, fn in (("extract", run_extract), ("match", run_match),
                      ("extract_match", run_extract_match)):
        for _ in range(args.warmup): 
            fn()
        sync()
        times_ms = []
        for _ in range(args.runs):
            t = time.perf_counter()
            fn()
            sync()
            times_ms.append((time.perf_counter() - t) * 1000.0)
        s = stats(times_ms)
        results[label] = s
        print(f"  {label:<14} {s['mean_ms']:7.2f} +- {s['std_ms']:5.2f} ms  "
              f"(median {s['median_ms']:.2f}, p95 {s['p95_ms']:.2f}, {s['fps_mean']:.1f} FPS)")


    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "processor": platform.processor() or platform.machine(),
           "cpu_count_logical": os.cpu_count(), "torch_threads": n_threads,
           "torch": torch.__version__, "numpy": np.__version__,
           "device": str(xfeat.dev),
           "gpu": torch.cuda.get_device_name(0) if cuda else None,
           "cuda": torch.version.cuda if cuda else None,
           "repo_commit": git_commit(HERE), "repo_dirty": git_dirty(HERE),
           "xfeat_commit": git_commit(XFEAT_DIR)}

    out = args.out or os.path.join(
        RESULTS_DIR, f"speed_{args.device}_{args.width}x{args.height}_t{n_threads}_k{args.top_k}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    payload = {"config": vars(args), "environment": env,
               "resolution": {"width": args.width, "height": args.height},
               "timings_ms": results}
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
