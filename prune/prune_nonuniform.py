import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import XFEAT_DIR, PRUNE_DIR  # noqa: E402

import numpy as np                 
import torch                       # noqa: E402
import torch.nn as nn              # noqa: E402
import torch_pruning as tp         # noqa: E402

import benchmark.hpatches_eval as he   # noqa: E402
import prune_grid_eval as pg           # noqa: E402  make_xfeat_with_net, eval_quality, eval_speed
from prune_xfeat import load_model, ignored_layers  # noqa: E402


def build_nonuniform(config, device, example, verbose=False):
    """config: dict {stage_name: ratio}. Prune all named stages in one pass."""
    model = load_model(device)

    # map each named stage module -> its ratio; global default stays 0.0
    ratio_dict = {getattr(model, s): r for s, r in config.items() if r > 0}
    ign = ignored_layers(model)        # protect descriptor output + both heads

    importance = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs=example, importance=importance,
        pruning_ratio=0.0,                  # untouched stages: no pruning
        pruning_ratio_dict=ratio_dict,       # per-stage overrides
        ignored_layers=ign, round_to=1,
    )

    before = {n: (m.in_channels, m.out_channels)
              for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    base_macs, base_params = tp.utils.count_ops_and_params(model, example)

    n_groups = 0
    for group in pruner.step(interactive=True):
        if verbose:
            print(group.details())
        group.prune()
        n_groups += 1

    macs, params = tp.utils.count_ops_and_params(model, example)
    after = {n: (m.in_channels, m.out_channels)
             for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}

    model.eval()
    with torch.inference_mode():
        f, k, h = model(example)
    ok = (f.shape[1] == 64 and k.shape[1] == 65 and h.shape[1] == 1)
    if not ok:
        raise SystemExit("interface broken (feats/kpts/heat shape changed)")

    changed = [(n, before[n][1], after[n][1]) for n in before if before[n][1] != after[n][1]]
    info = {"macs_off": 100 * (1 - macs / base_macs),
            "params_off": 100 * (1 - params / base_params),
            "n_groups": n_groups, "macs_g": macs / 1e9, "changed": changed}
    return model, info


def parse_config(items):
    cfg = {}
    for it in items:
        stage, ratio = it.split(":")
        cfg[stage] = float(ratio)
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Non-uniform per-stage pruning in one pass")
    p.add_argument("--data-dir", default=os.environ.get(
        "HPATCHES_DIR", os.path.join(he.HERE, "data", "hpatches-sequences-release")))
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--config", nargs="+", required=True,
                   help="stage:ratio list, e.g. block1:0.15 block5:0.6")
    p.add_argument("--subset", choices=["all", "108"], default="all")
    p.add_argument("--resize", type=int, default=480)
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--ransac-thr", type=float, default=3.0)
    p.add_argument("--ransac-iters", type=int, default=10000)
    p.add_argument("--ransac-conf", type=float, default=0.999)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--with-speed", action="store_true")
    p.add_argument("--speed-threads", type=int, default=4)
    p.add_argument("--speed-runs", type=int, default=50)
    p.add_argument("--vga-w", type=int, default=640)
    p.add_argument("--vga-h", type=int, default=480)
    p.add_argument("--verbose", action="store_true", help="print DepGraph groups")
    p.add_argument("--save", default=None, help="path (under PRUNE_DIR if relative) to save pruned model")
    p.add_argument("--out", default=os.path.join(PRUNE_DIR, "prune_nonuniform.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    example = torch.randn(1, 3, args.vga_h, args.vga_w, device=device)
    config = parse_config(args.config)

    print("config:", ", ".join(f"{s}:{r}" for s, r in config.items()))
    net, info = build_nonuniform(config, device, example, verbose=args.verbose)
    print(f"  MACs off: {info['macs_off']:.1f}%  |  params off: {info['params_off']:.1f}%  "
          f"|  groups: {info['n_groups']}")
    print("  changed conv outputs:")
    for n, bo, ao in info["changed"]:
        print(f"    {n:24s} {bo:3d} -> {ao:3d}")

    # ---- combined quality ----
    xf = pg.make_xfeat_with_net(net, args.top_k, args.device)
    t0 = time.time()
    summ = pg.eval_quality(xf, args.data_dir, args.subset, args.resize, args.top_k,
                           args.ransac_thr, args.ransac_iters, args.ransac_conf, args.seeds)
    print(f"\n  quality ({time.time()-t0:.0f}s):")
    print(f"    MMA@3 all {100*summ['all']['mma'][3]:.1f}  i {100*summ['i']['mma'][3]:.1f}  "
          f"v {100*summ['v']['mma'][3]:.1f}")
    print(f"    MHA@3 all {100*summ['all']['mha'][3]['mean']:.1f}  "
          f"i {100*summ['i']['mha'][3]['mean']:.1f}  v {100*summ['v']['mha'][3]['mean']:.1f}")
    print(f"    inliers {summ['all']['inliers_mean']['mean']:.0f}")

    row = {"config": ";".join(f"{s}:{r}" for s, r in config.items()),
           "macs_off": round(info["macs_off"], 2), "params_off": round(info["params_off"], 2),
           "mma_all_3": round(100*summ['all']['mma'][3], 2),
           "mma_i_3": round(100*summ['i']['mma'][3], 2),
           "mma_v_3": round(100*summ['v']['mma'][3], 2),
           "mha_all_3": round(100*summ['all']['mha'][3]['mean'], 2),
           "mha_i_3": round(100*summ['i']['mha'][3]['mean'], 2),
           "mha_v_3": round(100*summ['v']['mha'][3]['mean'], 2),
           "inliers_all": round(summ['all']['inliers_mean']['mean'], 0),
           "cpu_extract_fps": "", "cpu_match_fps": ""}

    if args.with_speed:
        spd, _ = pg.eval_speed(net, args.vga_w, args.vga_h, args.top_k,
                               args.speed_threads, args.speed_runs, 10, 0)
        row["cpu_extract_fps"] = round(spd['extract']['fps'], 2)
        row["cpu_match_fps"] = round(spd['extract_match']['fps'], 2)
        print(f"    CPU extract {spd['extract']['fps']:.1f} FPS | "
              f"extract+match {spd['extract_match']['fps']:.1f} FPS")

    # append to CSV (keep previous configs)
    fields = ["config", "macs_off", "params_off", "mma_all_3", "mma_i_3", "mma_v_3",
              "mha_all_3", "mha_i_3", "mha_v_3", "inliers_all", "cpu_extract_fps", "cpu_match_fps"]
    existing = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = list(csv.DictReader(f))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing + [row])
    print(f"\n  appended to {args.out}")

    if args.save:
        path = args.save if os.path.isabs(args.save) else os.path.join(PRUNE_DIR, args.save)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(net, path)   # full module: pruned shapes won't load into a fresh XFeatModel
        print(f"  saved pruned model -> {path}")


if __name__ == "__main__":
    main()
