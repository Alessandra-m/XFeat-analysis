import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import PRUNE_DIR

import numpy as np
import torch

import benchmark.hpatches_eval as he
import prune_grid_eval as pg
from prune_xfeat import STAGES


def parse_args():
    p = argparse.ArgumentParser(description="Per-stage pruning sensitivity profile")
    p.add_argument("--data-dir", default=os.environ.get(
        "HPATCHES_DIR", os.path.join(he.HERE, "data", "hpatches-sequences-release")))
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--stages", nargs="+", default=STAGES)
    p.add_argument("--ratios", type=float, nargs="+",
                   default=[0.15, 0.3, 0.4, 0.5, 0.6, 0.7],
                   help="fine ratio grid applied to every stage")
    p.add_argument("--safe-drop", type=float, default=2.0,
                   help="max allowed MMA@3 drop (p.p.) to still call a level 'safe'")
    p.add_argument("--subset", choices=["all", "108"], default="all")
    p.add_argument("--resize", type=int, default=480)
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--ransac-thr", type=float, default=3.0)
    p.add_argument("--ransac-iters", type=int, default=10000)
    p.add_argument("--ransac-conf", type=float, default=0.999)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--with-speed", action="store_true", help="also measure CPU extract FPS")
    p.add_argument("--speed-threads", type=int, default=4)
    p.add_argument("--speed-runs", type=int, default=50)
    p.add_argument("--vga-w", type=int, default=640)
    p.add_argument("--vga-h", type=int, default=480)
    p.add_argument("--out", default=os.path.join(PRUNE_DIR, "prune_sensitivity.csv"))
    return p.parse_args()


def eval_point(stage, ratio, args, device, example):
    """Prune one (stage, ratio) and return a metrics dict."""
    net, macs_off, _ = pg.build_pruned_net(stage, ratio, device, example)
    xf = pg.make_xfeat_with_net(net, args.top_k, args.device)
    summ = pg.eval_quality(xf, args.data_dir, args.subset, args.resize, args.top_k,
                           args.ransac_thr, args.ransac_iters, args.ransac_conf, args.seeds)
    row = {"stage": stage or "baseline", "ratio": ratio, "macs_off": round(macs_off, 2),
           "mma_all_3": round(100 * summ['all']['mma'][3], 2),
           "mma_v_3": round(100 * summ['v']['mma'][3], 2),
           "mha_all_3": round(100 * summ['all']['mha'][3]['mean'], 2),
           "mha_v_3": round(100 * summ['v']['mha'][3]['mean'], 2),
           "inliers_all": round(summ['all']['inliers_mean']['mean'], 0)}
    if args.with_speed:
        spd, _ = pg.eval_speed(net, args.vga_w, args.vga_h, args.top_k,
                               args.speed_threads, args.speed_runs, 10, 0)
        row["cpu_extract_fps"] = round(spd['extract']['fps'], 2)
    del net
    return row


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    example = torch.randn(1, 3, args.vga_h, args.vga_w, device=device)

    fields = ["stage", "ratio", "macs_off", "mma_all_3", "mma_v_3",
              "mha_all_3", "mha_v_3", "inliers_all"]
    if args.with_speed:
        fields.append("cpu_extract_fps")

    rows = []

    def write():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    print("=== baseline ===")
    base = eval_point(None, 0.0, args, device, example)
    base_mma = base["mma_all_3"]
    rows.append(base)
    print(f"  MMA@3 {base_mma}  MHA@3 {base['mha_all_3']}")
    write()

    for stage in args.stages:
        print(f"\n########## {stage} ##########")
        for ratio in args.ratios:
            t0 = time.time()
            r = eval_point(stage, ratio, args, device, example)
            rows.append(r)
            drop = base_mma - r["mma_all_3"]
            flag = "" if drop <= args.safe_drop else "  <-- past safe-drop"
            sp = f"  extractFPS {r['cpu_extract_fps']}" if args.with_speed else ""
            print(f"  r={ratio:<4}  MACs-{r['macs_off']:4.1f}%  "
                  f"MMA@3 {r['mma_all_3']:5.1f} (drop {drop:4.1f})  "
                  f"MHA@3 {r['mha_all_3']:5.1f}  MHA_v {r['mha_v_3']:5.1f}  "
                  f"inl {r['inliers_all']:5.0f}{sp}  [{time.time()-t0:.0f}s]{flag}")
            write()

    # safe budget per stage: largest ratio whose MMA@3 drop stays within --safe-drop
    print(f"\n=== sensitivity map (MMA@3 drop <= {args.safe_drop} p.p. vs baseline {base_mma}) ===")
    print(f"{'stage':<8}{'safe ratio':>11}{'MACs off':>10}{'cliff ratio':>12}")
    for stage in args.stages:
        srows = sorted((r for r in rows if r["stage"] == stage), key=lambda r: r["ratio"])
        safe_ratio, safe_macs, cliff_ratio = 0.0, 0.0, None
        for r in srows:
            if base_mma - r["mma_all_3"] <= args.safe_drop:
                safe_ratio, safe_macs = r["ratio"], r["macs_off"]
            elif cliff_ratio is None:
                cliff_ratio = r["ratio"]
        print(f"{stage:<8}{safe_ratio:>11}{safe_macs:>9.1f}%"
              f"{(cliff_ratio if cliff_ratio is not None else '>max'):>12}")

    print(f"\nsaved {args.out}  ({len(rows)} rows)")
    print("\nSafe ratio = how far this stage can be pruned before MMA@3 drops past the threshold.")
    print("Cliff ratio = first level where it does. Use safe ratios to build a non-uniform config.")


if __name__ == "__main__":
    main()
