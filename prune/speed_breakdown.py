import argparse
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import XFEAT_DIR

import numpy as np
import torch

import prune_grid_eval as pg
import prune_nonuniform as pn
from prune_xfeat import load_model

STEPS = ("net", "extract", "match", "pair")


def make_steps(xf, x0, x1, top_k):
    """net = forward only; extract = detectAndCompute; match = MNN; pair = match_xfeat."""
    with torch.inference_mode():
        xin, _, _ = xf.preprocess_tensor(x0)
        d0 = xf.detectAndCompute(x0, top_k=top_k)[0]["descriptors"]
        d1 = xf.detectAndCompute(x1, top_k=top_k)[0]["descriptors"]

    def net():
        with torch.inference_mode():
            xf.net(xin)

    def extract():
        with torch.inference_mode():
            xf.detectAndCompute(x0, top_k=top_k)

    def match():
        with torch.inference_mode():
            xf.match(d0, d1, min_cossim=-1)

    def pair():
        with torch.inference_mode():
            xf.match_xfeat(x0, x1, top_k=top_k, min_cossim=-1)

    return {"net": net, "extract": extract, "match": match, "pair": pair}


def profile_extract(xf, x0, top_k, runs, rows):
    from torch.profiler import profile, ProfilerActivity
    with torch.inference_mode():
        for _ in range(10):
            xf.detectAndCompute(x0, top_k=top_k)
        with profile(activities=[ProfilerActivity.CPU]) as p:
            for _ in range(runs):
                xf.detectAndCompute(x0, top_k=top_k)
    print(f"\nprofile of detectAndCompute (baseline, {runs} calls):")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=rows))


def compare_steps(xfs, rounds, runs, warmup, top_k, w, h):
    """Per-step pair time (instrumented match_xfeat from analysis/profile.py) on the real test pair:
    shows where the pair saving comes from inside the pipeline."""
    import analysis.profile as pf
    x0, x1 = pf.load_pair(w, h, "cpu")
    acc = {k: {} for k in xfs}
    for r in range(rounds):
        for k in (list(xfs) if r % 2 == 0 else list(xfs)[::-1]):
            res = pf.time_pipeline(pf.sparse_pair, xfs[k], x0, x1, top_k, lambda: None, runs, warmup)
            for step, v in res["steps_ms"].items():
                acc[k].setdefault(step, []).append(v)
            acc[k].setdefault("pair", []).append(res["total_ms"])
    print(f"\nper-step sparse pair on ref/tgt, median of {rounds} rounds x {runs} runs, ms")
    print(f"{'step':<12}{'baseline':>10}{'pruned':>10}{'saved':>8}")
    for step in acc["baseline"]:
        b, p = np.median(acc["baseline"][step]), np.median(acc["pruned"][step])
        print(f"{step:<12}{b:>10.2f}{p:>10.2f}{b - p:>8.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="CPU time breakdown of the pair: net / extract / match / pair, "
                                            "baseline vs non-uniform pruned net")
    p.add_argument("--config", nargs="+",
                   default=["block1:0.15", "block2:0.15", "block3:0.15", "block4:0.2", "block5:0.6"],
                   help="stage:ratio list for the pruned variant")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--rounds", type=int, default=7, help="rounds; variants alternate order each round")
    p.add_argument("--runs", type=int, default=40, help="timed runs per step per round")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--vga-w", type=int, default=640)
    p.add_argument("--vga-h", type=int, default=480)
    p.add_argument("--profile", action="store_true", help="also profile detectAndCompute by operator")
    p.add_argument("--steps", action="store_true",
                   help="also compare the pair step by step (net / NMS / interpolation / MNN) on the test pair")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    example = torch.randn(1, 3, args.vga_h, args.vga_w)
    config = pn.parse_config(args.config)

    nets = {"baseline": load_model("cpu"),
            "pruned": pn.build_nonuniform(config, "cpu", example)[0]}
    xfs = {k: pg.make_xfeat_with_net(v, args.top_k, "cpu") for k, v in nets.items()}

    torch.manual_seed(0)
    x0 = torch.randint(0, 256, (1, 3, args.vga_h, args.vga_w), dtype=torch.float32)
    x1 = torch.randint(0, 256, (1, 3, args.vga_h, args.vga_w), dtype=torch.float32)
    steps = {k: make_steps(xf, x0, x1, args.top_k) for k, xf in xfs.items()}

    times = {k: {s: [] for s in STEPS} for k in steps}
    for r in range(args.rounds):
        order = list(steps) if r % 2 == 0 else list(steps)[::-1]
        for k in order:
            for s, fn in steps[k].items():
                for _ in range(args.warmup):
                    fn()
                ts = []
                for _ in range(args.runs):
                    t = time.perf_counter()
                    fn()
                    ts.append((time.perf_counter() - t) * 1000.0)
                times[k][s].append(np.mean(ts))
        print(f"round {r+1}/{args.rounds} done", flush=True)

    print(f"\nconfig: {' '.join(args.config)} | threads {args.threads} | top_k {args.top_k}")
    print(f"median over {args.rounds} rounds x {args.runs} runs, ms (std across rounds)")
    print(f"{'variant':<10}" + "".join(f"{s:>14}" for s in STEPS) + f"{'2*ext+match':>14}")
    med = {}
    for k in times:
        med[k] = {s: float(np.median(v)) for s, v in times[k].items()}
        sd = {s: float(np.std(v)) for s, v in times[k].items()}
        cells = "".join(f"{med[k][s]:>8.1f}+-{sd[s]:<4.1f}" for s in STEPS)
        print(f"{k:<10}{cells}{2*med[k]['extract'] + med[k]['match']:>12.1f}")

    b, p = med["baseline"], med["pruned"]
    print("\nsaving, ms: " + ", ".join(f"{s} {b[s] - p[s]:.2f}" for s in STEPS))
    print(f"net only:    {b['net']:.1f} -> {p['net']:.1f} ms ({100*(1 - p['net']/b['net']):.0f}% faster)")
    print(f"extract FPS: {1e3/b['extract']:.2f} -> {1e3/p['extract']:.2f} "
          f"({100*(b['extract']/p['extract'] - 1):+.1f}%)")
    print(f"share of baseline pair (2*extract + match): 2*extract "
          f"{100*2*b['extract']/(2*b['extract'] + b['match']):.0f}%, "
          f"of it 2*net {100*2*b['net']/(2*b['extract'] + b['match']):.0f}%, "
          f"match {100*b['match']/(2*b['extract'] + b['match']):.0f}%")

    if args.profile:
        profile_extract(xfs["baseline"], x0, args.top_k, 30, 12)
    if args.steps:
        compare_steps(xfs, 6, args.runs, args.warmup, args.top_k, args.vga_w, args.vga_h)


if __name__ == "__main__":
    main()
