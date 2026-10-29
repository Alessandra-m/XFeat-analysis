import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import XFEAT_DIR, PRUNE_DIR

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp

from prune_xfeat import load_model, ignored_layers, stage_conv_layers
import prune_grid_eval as pg
import benchmark.hpatches_eval as he


def _first_attr(mod, names):
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise AttributeError(f"none of {names} found in {mod.__name__}")


def make_importance(crit):
    imp = tp.importance
    if crit == "l1":
        cls = _first_attr(imp, ["MagnitudeImportance", "GroupMagnitudeImportance", "GroupNormImportance"])
        return cls(p=1)
    if crit == "l2":
        cls = _first_attr(imp, ["MagnitudeImportance", "GroupMagnitudeImportance", "GroupNormImportance"])
        return cls(p=2)
    if crit == "random":
        cls = _first_attr(imp, ["RandomImportance"])
        return cls()
    if crit == "taylor":
        cls = _first_attr(imp, ["GroupTaylorImportance", "TaylorImportance"])
        return cls()
    raise ValueError(crit)


def make_pruner(model, example, importance, ratio, ign):
    cls = _first_attr(tp.pruner, ["MetaPruner", "MagnitudePruner"])
    return cls(model, example_inputs=example, importance=importance,
               pruning_ratio=ratio, ignored_layers=ign, round_to=1)


def build_pruned(stage, ratio, crit, device, example, seed=0,
                 taylor_batches=4, vga=(640, 480)):
    torch.manual_seed(seed)
    model = load_model(device)
    target = set(stage_conv_layers(model, stage))
    ign = ignored_layers(model)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m not in target:
            ign.append(m)

    importance = make_importance(crit)
    pruner = make_pruner(model, example, importance, ratio, ign)
    base_macs, _ = tp.utils.count_ops_and_params(model, example)

    if crit == "taylor":

        model.zero_grad(set_to_none=True)
        for p in model.parameters():
            p.requires_grad_(True)
        for _ in range(taylor_batches):
            x = torch.randint(0, 256, (1, 3, vga[1], vga[0]),
                              dtype=torch.float32, device=device)
            feats, kpts, heat = model(x)
            loss = feats.pow(2).mean()
            loss.backward()

    for group in pruner.step(interactive=True):
        group.prune()
    macs, _ = tp.utils.count_ops_and_params(model, example)

    model.eval()
    with torch.inference_mode():
        f, k, h = model(example)
    assert f.shape[1] == 64 and k.shape[1] == 65 and h.shape[1] == 1, "interface broken"
    return model, 100 * (1 - macs / base_macs)


def parse_point(s):
    stage, ratio = s.split(":")
    return stage, float(ratio)


def parse_args():
    p = argparse.ArgumentParser(description="Compare pruning importance criteria")
    p.add_argument("--data-dir", default=os.environ.get(
        "HPATCHES_DIR", os.path.join(he.HERE, "data", "hpatches-sequences-release")))
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--points", nargs="+", default=["block2:0.5", "block3:0.25", "block3:0.5"],
                   help="stage:ratio operating points where pruning hurts")
    p.add_argument("--criteria", nargs="+", default=["l1", "l2", "random"],
                   choices=["l1", "l2", "random", "taylor"])
    p.add_argument("--with-taylor", action="store_true", help="also run taylor (needs backward)")
    p.add_argument("--rand-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--subset", choices=["all", "108"], default="all")
    p.add_argument("--resize", type=int, default=480)
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--ransac-thr", type=float, default=3.0)
    p.add_argument("--ransac-iters", type=int, default=10000)
    p.add_argument("--ransac-conf", type=float, default=0.999)
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="MAGSAC seeds for eval")
    p.add_argument("--speed-threads", type=int, default=4)
    p.add_argument("--speed-runs", type=int, default=50)
    p.add_argument("--vga-w", type=int, default=640)
    p.add_argument("--vga-h", type=int, default=480)
    p.add_argument("--out", default=os.path.join(PRUNE_DIR, "prune_criteria.csv"))
    return p.parse_args()


def eval_variant(net, args):
    xf = pg.make_xfeat_with_net(net, args.top_k, args.device)
    summ = pg.eval_quality(xf, args.data_dir, args.subset, args.resize, args.top_k,
                           args.ransac_thr, args.ransac_iters, args.ransac_conf, args.seeds)
    spd, _ = pg.eval_speed(net, args.vga_w, args.vga_h, args.top_k,
                           args.speed_threads, args.speed_runs, 10, 0)
    return summ, spd


def main():
    import csv
    args = parse_args()
    crits = list(args.criteria)
    if args.with_taylor and "taylor" not in crits:
        crits.append("taylor")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    example = torch.randn(1, 3, args.vga_h, args.vga_w, device=device)
    points = [parse_point(s) for s in args.points]

    fields = ["stage", "ratio", "criterion", "macs_off",
              "mma_all_3", "mma_all_3_std", "mma_v_3",
              "mha_all_3", "mha_all_3_std", "mha_v_3", "mha_v_3_std",
              "inliers_all", "cpu_extract_fps", "note"]
    rows = []

    for stage, ratio in points:
        print(f"\n########## {stage}@{ratio} ##########")
        for crit in crits:
            seeds = args.rand_seeds if crit == "random" else [0]
            per = []
            macs_off = None
            for sd in seeds:
                net, macs_off = build_pruned(stage, ratio, crit, device, example, seed=sd,
                                             vga=(args.vga_w, args.vga_h))
                summ, spd = eval_variant(net, args)
                per.append((100*summ['all']['mma'][3], 100*summ['v']['mma'][3],
                            100*summ['all']['mha'][3]['mean'], 100*summ['v']['mha'][3]['mean'],
                            summ['all']['inliers_mean']['mean'], spd['extract']['fps']))
                del net
            a = np.array(per)
            mean = a.mean(axis=0)
            std = a.std(axis=0)                 # spread across seeds (only meaningful for random)
            note = f"avg of {len(seeds)} seeds" if crit == "random" else (
                "surrogate-loss proxy" if crit == "taylor" else "")
            if len(seeds) > 1:
                print(f"  {crit:7s} MACs-{macs_off:4.1f}%  "
                      f"MMA@3 {mean[0]:5.1f}+-{std[0]:.1f}  MMA_v {mean[1]:5.1f}+-{std[1]:.1f}  "
                      f"MHA@3 {mean[2]:5.1f}+-{std[2]:.1f}  MHA_v {mean[3]:5.1f}+-{std[3]:.1f}  "
                      f"inl {mean[4]:5.0f}  {note}")
            else:
                print(f"  {crit:7s} MACs-{macs_off:4.1f}%  "
                      f"MMA@3 {mean[0]:5.1f}  MMA_v {mean[1]:5.1f}  "
                      f"MHA@3 {mean[2]:5.1f}  MHA_v {mean[3]:5.1f}  "
                      f"inl {mean[4]:5.0f}  {note}")
            rows.append({"stage": stage, "ratio": ratio, "criterion": crit,
                         "macs_off": round(macs_off, 2),
                         "mma_all_3": round(mean[0], 2), "mma_all_3_std": round(std[0], 2),
                         "mma_v_3": round(mean[1], 2),
                         "mha_all_3": round(mean[2], 2), "mha_all_3_std": round(std[2], 2),
                         "mha_v_3": round(mean[3], 2), "mha_v_3_std": round(std[3], 2),
                         "inliers_all": round(mean[4], 0),
                         "cpu_extract_fps": round(mean[5], 2), "note": note})
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)

    print(f"\nsaved {args.out}  ({len(rows)} rows)")
    print("\nHow to read: compare L1 vs random at the same point. If L1's MMA/MHA is clearly")
    print("higher than random, the L1 criterion genuinely selects less-important channels.")


if __name__ == "__main__":
    main()
