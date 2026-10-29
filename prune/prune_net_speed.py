import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import PRUNE_DIR

import numpy as np
import torch

import prune_grid_eval as pg
import prune_nonuniform as pn
from prune_xfeat import STAGES

# fine-tuning candidates (Part IV.5), pruned in one pass with the FPN outputs protected
CONFIGS = {"A": {"block5": 0.5},
           "B": {"block5": 0.5, "block2": 0.25},
           "C": {"block5": 0.5, "block2": 0.5, "block3": 0.1},
           "D": {"block5": 0.7},
           "D (rejected: block5+block4)": {"block5": 0.5, "block4": 0.5},
           "nonuniform": {"block1": 0.15, "block2": 0.15, "block3": 0.15, "block4": 0.2, "block5": 0.6}}


def build_variants(stages, ratios, example):
    """(name, kind, net, macs_off): baseline, the single-stage grid and the configs."""
    out = [("baseline", "baseline", *pg.build_pruned_net(None, 0.0, "cpu", example)[:2])]
    for s in stages:
        for r in ratios:
            net, macs_off, _ = pg.build_pruned_net(s, r, "cpu", example)
            out.append((f"{s}@{r:g}", "grid", net, macs_off))
    for name, cfg in CONFIGS.items():
        net, info = pn.build_nonuniform(cfg, "cpu", example, protect_fpn=True)
        out.append((f"config {name}" if name != "nonuniform" else "nonuniform (FPN protected)",
                    "config", net, info["macs_off"]))
    net, info = pn.build_nonuniform(CONFIGS["nonuniform"], "cpu", example, protect_fpn=False)
    out.append(("nonuniform (FPN cut)", "config", net, info["macs_off"]))
    return out


def time_net(net, x, sync, runs, warmup):
    with torch.inference_mode():
        for _ in range(warmup):
            net(x)
        sync()
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            net(x)
            sync()
            ts.append((time.perf_counter() - t) * 1000.0)
    return float(np.mean(ts))


def parse_args():
    p = argparse.ArgumentParser(description="Forward time of the pruned nets only (no NMS / matching)")
    p.add_argument("--devices", nargs="+", default=["cpu", "cuda"], choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--stages", nargs="+", default=STAGES)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    p.add_argument("--rounds", type=int, default=5, help="rounds; every round times all variants in shuffled order")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--vga-w", type=int, default=640)
    p.add_argument("--vga-h", type=int, default=480)
    p.add_argument("--out", default=os.path.join(PRUNE_DIR, "prune_net_speed.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    example = torch.randn(1, 3, args.vga_h, args.vga_w)
    variants = build_variants(args.stages, args.ratios, example)

    rows = [{"variant": n, "kind": k, "macs_off_pct": round(m, 2)} for n, k, _, m in variants]
    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            print("cuda not available, skipped")
            continue
        sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
        x = torch.zeros(1, 3, args.vga_h, args.vga_w, device=device)
        nets = [net.to(device).eval() for _, _, net, _ in variants]
        times = [[] for _ in variants]
        for rep in range(args.rounds):
            for idx in np.random.default_rng(rep).permutation(len(variants)):
                times[idx].append(time_net(nets[idx], x, sync, args.runs, args.warmup))
            print(f"{device}: round {rep+1}/{args.rounds} done", flush=True)
        base = float(np.median(times[0]))
        for row, ts in zip(rows, times):
            med = float(np.median(ts))
            row[f"{device}_net_ms"] = round(med, 3)
            row[f"{device}_net_ms_std"] = round(float(np.std(ts)), 3)
            row[f"{device}_net_saved_pct"] = round(100 * (1 - med / base), 1)
        for net in nets:
            net.to("cpu")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = [c for c in rows[0] if c.endswith(("_net_ms", "_net_saved_pct"))]
    print(f"\nnet forward only, VGA, median of {args.rounds} rounds x {args.runs} runs "
          f"(CPU {args.threads} threads)")
    print(f"{'variant':<28}{'MACs off':>9}" + "".join(f"{c:>20}" for c in cols))
    for r in rows:
        print(f"{r['variant']:<28}{r['macs_off_pct']:>8.1f}%" + "".join(f"{r[c]:>20}" for c in cols))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
