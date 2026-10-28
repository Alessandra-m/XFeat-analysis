
import argparse
import copy
import csv
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from common.paths import XFEAT_DIR, PRUNE_DIR

import numpy as np
import torch

from modules.xfeat import XFeat

import benchmark.hpatches_eval as he
from prune_xfeat import load_model, ignored_layers, stage_conv_layers, STAGES
import torch_pruning as tp
import torch.nn as nn



def build_pruned_net(stage, ratio, device, example):
    model = load_model(device)
    if ratio <= 0 or stage is None:
        base_macs, _ = tp.utils.count_ops_and_params(model, example)
        return model, 0.0, base_macs
    importance = tp.importance.MagnitudeImportance(p=1)
    target = set(stage_conv_layers(model, stage))
    ign = ignored_layers(model)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m not in target:
            ign.append(m)
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs=example, importance=importance,
        pruning_ratio=ratio, ignored_layers=ign, round_to=1)
    base_macs, _ = tp.utils.count_ops_and_params(model, example)
    for group in pruner.step(interactive=True):
        group.prune()
    macs, _ = tp.utils.count_ops_and_params(model, example)
    model.eval()
    with torch.inference_mode():
        f, k, h = model(example)
    assert f.shape[1] == 64 and k.shape[1] == 65 and h.shape[1] == 1, "interface broken"
    return model, 100 * (1 - macs / base_macs), macs


def make_xfeat_with_net(net, top_k, device):
    """An XFeat wrapper whose backbone is the given (pruned) net."""
    xf = XFeat(top_k=top_k)
    xf.dev = torch.device(device)
    xf.net = net.to(xf.dev).eval()
    return xf


# ----- quality: reuse hpatches_eval metric pipeline on the pruned matcher -----

def eval_quality(xf, data_dir, subset, resize, top_k, ransac_thr, ransac_iters, ransac_conf, seeds):
    pairs = he.list_pairs(data_dir, subset, 0)

    def match_fn(img0, img1):
        with torch.inference_mode():
            return xf.match_xfeat(img0, img1, top_k=top_k, min_cossim=-1)

    records = []
    for seq, i in pairs:
        img0, img1, H_gt = he.load_pair(data_dir, seq, i, resize)
        h0, w0 = img0.shape[:2]
        pts0, pts1 = match_fn(img0, img1)
        errs, inl = [], []
        for seed in seeds:
            H_est, n_in = he.magsac(pts0, pts1, ransac_thr, ransac_iters, ransac_conf, seed)
            errs.append(he.corner_error(H_est, H_gt, w0, h0))
            inl.append(n_in)
        records.append({"seq": seq, "pair": f"1-{i}", "size0_hw": [h0, w0],
                        "size1_hw": list(img1.shape[:2]), "n_matches": int(len(pts0)),
                        "mma": he.mma(pts0, pts1, H_gt), "corner_error": errs,
                        "n_inliers": inl, "match_time_s": 0.0})
    return he.summarize(records, seeds)


# ----- speed: fixed VGA, CPU, fixed threads; extract / extract+match, mean/std -----

def eval_speed(net, width, height, top_k, threads, runs, warmup, seed):
    prev = torch.get_num_threads()
    if threads > 0:
        torch.set_num_threads(threads)
    try:
        xf = make_xfeat_with_net(net, top_k, "cpu")
        torch.manual_seed(seed)
        x0 = torch.randint(0, 256, (1, 3, height, width), dtype=torch.float32)
        x1 = torch.randint(0, 256, (1, 3, height, width), dtype=torch.float32)

        def extract():
            with torch.inference_mode():
                xf.detectAndCompute(x0, top_k=top_k)

        def extract_match():
            with torch.inference_mode():
                xf.match_xfeat(x0, x1, top_k=top_k, min_cossim=-1)

        out = {}
        for label, fn in (("extract", extract), ("extract_match", extract_match)):
            for _ in range(warmup):
                fn()
            ts = []
            for _ in range(runs):
                t = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t) * 1000.0)
            a = np.asarray(ts)
            out[label] = {"mean_ms": float(a.mean()), "std_ms": float(a.std()),
                          "fps": float(1000.0 / a.mean())}
        return out, torch.get_num_threads()
    finally:
        torch.set_num_threads(prev)


def parse_args():
    p = argparse.ArgumentParser(description="XFeat pruning grid: quality + speed, no fine-tuning")
    p.add_argument("--data-dir", default=os.environ.get(
        "HPATCHES_DIR", os.path.join(he.HERE, "data", "hpatches-sequences-release")))
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="device for quality eval")
    p.add_argument("--subset", choices=["all", "108"], default="all")
    p.add_argument("--resize", type=int, default=480)
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--ransac-thr", type=float, default=3.0)
    p.add_argument("--ransac-iters", type=int, default=10000)
    p.add_argument("--ransac-conf", type=float, default=0.999)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--stages", nargs="+", default=STAGES)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    p.add_argument("--speed-threads", type=int, default=4)
    p.add_argument("--speed-runs", type=int, default=50, help="timed runs per repeat")
    p.add_argument("--speed-warmup", type=int, default=10)
    p.add_argument("--speed-repeats", type=int, default=5,
                   help="speed rounds; each round times every variant in shuffled order")
    p.add_argument("--vga-w", type=int, default=640)
    p.add_argument("--vga-h", type=int, default=480)
    p.add_argument("--out", default=os.path.join(PRUNE_DIR, "prune_grid.csv"))
    p.add_argument("--save-models", action="store_true", help="also save each pruned .pth")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    example = torch.randn(1, 3, args.vga_h, args.vga_w, device=device)

    # variants: baseline first, then the grid
    variants = [(None, 0.0)] + [(s, r) for s in args.stages for r in args.ratios]

    fields = ["stage", "ratio", "macs_off_pct",
              "mma_all_3", "mma_all_5", "mma_i_3", "mma_v_3",
              "auc_all_3", "auc_all_5", "mha_all_3", "mha_i_3", "mha_v_3",
              "matches_all", "inliers_all",
              "cpu_extract_ms", "cpu_extract_ms_std", "cpu_extract_fps", "cpu_extract_fps_std",
              "cpu_match_ms", "cpu_match_ms_std", "cpu_match_fps", "cpu_match_fps_std",
              "speed_threads", "speed_repeats"]
    rows, nets = [], []

    def write_csv():
        # write incrementally so a crash mid-grid keeps finished rows
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    # ----- pass 1: build + quality for every variant -----
    for stage, ratio in variants:
        tag = "baseline" if stage is None else f"{stage}@{ratio}"
        print(f"\n=== {tag} ===")
        net, macs_off, _ = build_pruned_net(stage, ratio, device, example)
        print(f"  MACs off: {macs_off:.1f}%")

        xf = make_xfeat_with_net(net, args.top_k, args.device)
        t0 = time.time()
        summ = eval_quality(xf, args.data_dir, args.subset, args.resize, args.top_k,
                            args.ransac_thr, args.ransac_iters, args.ransac_conf, args.seeds)
        print(f"  quality done in {time.time()-t0:.0f}s | "
              f"MMA@3 all {100*summ['all']['mma'][3]:.1f} | "
              f"MHA@3 all {100*summ['all']['mha'][3]['mean']:.1f}")

        rows.append({
            "stage": stage or "baseline", "ratio": ratio, "macs_off_pct": round(macs_off, 2),
            "mma_all_3": round(100*summ['all']['mma'][3], 2),
            "mma_all_5": round(100*summ['all']['mma'][5], 2),
            "mma_i_3": round(100*summ['i']['mma'][3], 2),
            "mma_v_3": round(100*summ['v']['mma'][3], 2),
            "auc_all_3": round(100*summ['all']['auc'][3]['mean'], 2),
            "auc_all_5": round(100*summ['all']['auc'][5]['mean'], 2),
            "mha_all_3": round(100*summ['all']['mha'][3]['mean'], 2),
            "mha_i_3": round(100*summ['i']['mha'][3]['mean'], 2),
            "mha_v_3": round(100*summ['v']['mha'][3]['mean'], 2),
            "matches_all": round(summ['all']['matches_mean'], 0),
            "inliers_all": round(summ['all']['inliers_mean']['mean'], 0),
        })
        nets.append(net)

        if args.save_models and stage is not None:
            os.makedirs(PRUNE_DIR, exist_ok=True)
            torch.save(net, os.path.join(PRUNE_DIR, f"{stage}_{int(ratio*100)}.pth"))
        write_csv()

    # ----- pass 2: speed in rounds; every round times all variants in a shuffled order,
    # so machine drift (thermal, background load) hits all variants alike, not just one -----
    times = {k: [[] for _ in variants] for k in ("extract", "extract_match")}
    for rep in range(args.speed_repeats):
        order = np.random.default_rng(rep).permutation(len(variants))
        print(f"\n--- speed round {rep+1}/{args.speed_repeats} ---")
        for idx in order:
            spd, used_threads = eval_speed(nets[idx], args.vga_w, args.vga_h, args.top_k,
                                           args.speed_threads, args.speed_runs, args.speed_warmup, 0)
            for k in times:
                times[k][idx].append(spd[k]["mean_ms"])
            print(f"  {rows[idx]['stage']}@{rows[idx]['ratio']}: "
                  f"extract {spd['extract']['mean_ms']:.1f} ms | "
                  f"extract+match {spd['extract_match']['mean_ms']:.1f} ms")

        # median over rounds is robust to a single noisy round; std shows the spread
        for idx, row in enumerate(rows):
            for k, col in (("extract", "cpu_extract"), ("extract_match", "cpu_match")):
                ms = np.asarray(times[k][idx])
                fps = 1000.0 / ms
                row[col + "_ms"] = round(float(np.median(ms)), 2)
                row[col + "_ms_std"] = round(float(ms.std()), 2)
                row[col + "_fps"] = round(float(np.median(fps)), 2)
                row[col + "_fps_std"] = round(float(fps.std()), 2)
            row["speed_threads"] = used_threads
            row["speed_repeats"] = rep + 1
        write_csv()

    print(f"\nsaved {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
