import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import RESULTS_DIR, XFEAT_DIR

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from modules.xfeat import XFeat
from modules.interpolator import InterpolateSparse2d

REF_PATH = os.path.join(XFEAT_DIR, "assets", "ref.png")
TGT_PATH = os.path.join(XFEAT_DIR, "assets", "tgt.png")

BRANCHES = [("skip1", "skip1 (AvgPool + 1x1)"),
            ("block1", "block1 (1/1 -> 1/4)"),
            ("block2", "block2 (1/4)"),
            ("block3", "block3 (1/8)"),
            ("block4", "block4 (1/16)"),
            ("block5", "block5 (1/32)"),
            ("block_fusion", "descriptor head (FPN + block_fusion)"),
            ("heatmap_head", "reliability head"),
            ("keypoint_head", "keypoint head")]
RESOLUTIONS = [(640, 480), (800, 600), (1024, 768), (1280, 720)]

GROUPS = {"net": "сеть",
          "preprocess": "постобработка", "heatmap": "постобработка", "nms": "постобработка",
          "scores_topk": "постобработка", "desc_interp": "постобработка",
          "resize": "постобработка", "dense_topk": "постобработка", "output": "постобработка",
          "mnn": "матчинг MNN",
          "refine_mlp": "refinement", "refine_post": "refinement"}
GROUP_ORDER = ["сеть", "постобработка", "матчинг MNN", "refinement"]


def layer_macs(module, inputs, output):
    if isinstance(module, nn.Conv2d):
        k = module.kernel_size[0] * module.kernel_size[1]
        return output.numel() // output.shape[0] * module.in_channels // module.groups * k
    if isinstance(module, nn.Linear):
        return output.numel() // output.shape[-1] * module.in_features * module.out_features
    return 0


def branch_macs(net, h, w):
    """MACs of every Conv2d, summed per top-level branch, for one 1x1xHxW forward."""
    macs = {name: 0 for name, _ in BRANCHES}
    handles = []
    for name, _ in BRANCHES:
        for m in getattr(net, name).modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                def hook(mod, inp, out, name=name):
                    macs[name] += layer_macs(mod, inp, out)
                handles.append(m.register_forward_hook(hook))
    x = torch.zeros(1, 1, h, w, device=next(net.parameters()).device)
    with torch.inference_mode():
        net(x)
    for hd in handles:
        hd.remove()
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:   # cross-check, FLOPs = 2 * MACs
        net(x)
    return macs, fc.get_total_flops() // 2


def fine_matcher_macs_per_match(net):
    return sum(m.in_features * m.out_features for m in net.fine_matcher if isinstance(m, nn.Linear))


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def net_input_hw(w, h):
    """preprocess_tensor floors H and W to multiples of 32."""
    return (h // 32) * 32, (w // 32) * 32


def dualscale_hw(w, h, scales=(0.6, 1.3)):
    """Network input sizes of the two XFeat* scales (interpolate, then floor to 32)."""
    out = []
    for s in scales:
        x = F.interpolate(torch.zeros(1, 1, h, w), scale_factor=s, mode="bilinear", align_corners=False)
        out.append(net_input_hw(x.shape[-1], x.shape[-2]))
    return out


class Marks:
    """Accumulates wall time per step; sync() makes CUDA timings honest."""
    def __init__(self, sync):
        self.sync, self.acc = sync, {}

    def start(self):
        self.sync()
        self.t = time.perf_counter()

    def mark(self, step):
        self.sync()
        t = time.perf_counter()
        self.acc[step] = self.acc.get(step, 0.0) + (t - self.t) * 1000.0
        self.t = t


def sparse_extract(xf, x, top_k, tm):
    """detectAndCompute (xfeat.py) split into steps."""
    x, rh1, rw1 = xf.preprocess_tensor(x)
    B, _, _H1, _W1 = x.shape
    tm.mark("preprocess")
    M1, K1, H1 = xf.net(x)
    tm.mark("net")
    M1 = F.normalize(M1, dim=1)
    K1h = xf.get_kpts_heatmap(K1)
    tm.mark("heatmap")
    mkpts = xf.NMS(K1h, threshold=xf.detection_threshold, kernel_size=5)
    tm.mark("nms")
    _nearest = InterpolateSparse2d("nearest")
    _bilinear = InterpolateSparse2d("bilinear")
    scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
    scores[torch.all(mkpts == 0, dim=-1)] = -1
    idxs = torch.argsort(-scores)
    mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :top_k]
    mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :top_k]
    mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
    scores = torch.gather(scores, -1, idxs)[:, :top_k]
    tm.mark("scores_topk")
    feats = xf.interpolator(M1, mkpts, H=_H1, W=_W1)
    feats = F.normalize(feats, dim=-1)
    mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, 1, -1)
    valid = scores > 0
    out = {"keypoints": mkpts[0][valid[0]], "descriptors": feats[0][valid[0]]}
    tm.mark("desc_interp")
    return out


def sparse_pair(xf, x0, x1, top_k, tm):
    """match_xfeat: two extractions + MNN."""
    out0 = sparse_extract(xf, x0, top_k, tm)
    out1 = sparse_extract(xf, x1, top_k, tm)
    idx0, idx1 = xf.match(out0["descriptors"], out1["descriptors"], min_cossim=-1)
    tm.mark("mnn")
    res = out0["keypoints"][idx0].cpu().numpy(), out1["keypoints"][idx1].cpu().numpy()
    tm.mark("output")
    return res, {"refined_in": 0}


def dense_extract(xf, x, top_k, tm):
    """extractDense (xfeat.py) split into net / top-k."""
    if top_k < 1:
        top_k = 100_000_000
    x, rh1, rw1 = xf.preprocess_tensor(x)
    tm.mark("preprocess")
    M1, K1, H1 = xf.net(x)
    tm.mark("net")
    B, C, _H1, _W1 = M1.shape
    xy1 = (xf.create_xy(_H1, _W1, M1.device) * 8).expand(B, -1, -1)
    M1 = M1.permute(0, 2, 3, 1).reshape(B, -1, C)
    H1 = H1.permute(0, 2, 3, 1).reshape(B, -1)
    _, top_k = torch.topk(H1, k=min(len(H1[0]), top_k), dim=-1)
    feats = torch.gather(M1, 1, top_k[..., None].expand(-1, -1, 64))
    mkpts = torch.gather(xy1, 1, top_k[..., None].expand(-1, -1, 2))
    mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, -1)
    tm.mark("dense_topk")
    return mkpts, feats


def dualscale(xf, x, top_k, tm, s1=0.6, s2=1.3):
    """extract_dualscale (xfeat.py)."""
    x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode="bilinear")
    x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode="bilinear")
    tm.mark("resize")
    mkpts_1, feats_1 = dense_extract(xf, x1, int(top_k * 0.20), tm)
    mkpts_2, feats_2 = dense_extract(xf, x2, int(top_k * 0.80), tm)
    mkpts = torch.cat([mkpts_1 / s1, mkpts_2 / s2], dim=1)
    sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1 / s1)
    sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1 / s2)
    out = {"keypoints": mkpts, "descriptors": torch.cat([feats_1, feats_2], dim=1),
           "scales": torch.cat([sc1, sc2], dim=1)}
    tm.mark("dense_topk")
    return out


def star_pair(xf, x0, x1, top_k, tm, fine_conf=0.25):
    """match_xfeat_star: dual-scale dense extraction, MNN, refinement (refine_matches)."""
    out0 = dualscale(xf, x0, top_k, tm)
    out1 = dualscale(xf, x1, top_k, tm)
    idx0, idx1 = xf.batch_match(out0["descriptors"], out1["descriptors"])[0]
    tm.mark("mnn")
    feats1 = out0["descriptors"][0][idx0]
    feats2 = out1["descriptors"][0][idx1]
    mkpts_0 = out0["keypoints"][0][idx0]
    mkpts_1 = out1["keypoints"][0][idx1]
    sc0 = out0["scales"][0][idx0]
    offsets = xf.net.fine_matcher(torch.cat([feats1, feats2], dim=-1))
    tm.mark("refine_mlp")
    conf = F.softmax(offsets * 3, dim=-1).max(dim=-1)[0]
    offsets = xf.subpix_softmax2d(offsets.view(-1, 8, 8))
    mkpts_0 = mkpts_0 + offsets * sc0[:, None]
    good = conf > fine_conf
    matches = torch.cat([mkpts_0[good], mkpts_1[good]], dim=-1)
    tm.mark("refine_post")
    res = matches[:, :2].cpu().numpy(), matches[:, 2:].cpu().numpy()
    tm.mark("output")
    return res, {"refined_in": int(len(idx0))}


def time_pipeline(fn, xf, x0, x1, top_k, sync, runs, warmup):
    with torch.inference_mode():
        for _ in range(warmup):
            tm = Marks(sync)
            tm.start()
            fn(xf, x0, x1, top_k, tm)
        per_step, totals, info = {}, [], None
        for _ in range(runs):
            tm = Marks(sync)
            tm.start()
            _, info = fn(xf, x0, x1, top_k, tm)
            for k, v in tm.acc.items():
                per_step.setdefault(k, []).append(v)
            totals.append(sum(tm.acc.values()))
    steps = {k: float(np.median(v)) for k, v in per_step.items()}
    return {"steps_ms": steps, "total_ms": float(np.median(totals)),
            "total_std_ms": float(np.std(totals)), **info}


def time_branches(net, x, sync, runs, warmup):
    """Forward-hook wall time of each top-level branch inside one net(x) call."""
    acc, t0 = {}, {}

    def pre(name):
        def f(m, inp):
            sync()
            t0[name] = time.perf_counter()
        return f

    def post(name):
        def f(m, inp, out):
            sync()
            acc.setdefault(name, []).append((time.perf_counter() - t0[name]) * 1000.0)
        return f

    names = [n for n, _ in BRANCHES]
    handles = []
    for n in names:
        mod = getattr(net, n)
        handles += [mod.register_forward_pre_hook(pre(n)), mod.register_forward_hook(post(n))]
    totals = []
    with torch.inference_mode():
        for i in range(warmup + runs):
            sync()
            t = time.perf_counter()
            net(x)
            sync()
            if i >= warmup:
                totals.append((time.perf_counter() - t) * 1000.0)
    for hd in handles:
        hd.remove()
    out = {n: float(np.median(acc[n][warmup:])) for n in names}
    out["total"] = float(np.median(totals))
    out["other"] = out["total"] - sum(out[n] for n in names)   # InstanceNorm, upsample, unfold, sums
    return out


def make_xfeat(device, top_k):
    xf = XFeat(top_k=top_k)
    xf.dev = torch.device(device)
    xf.net = xf.net.to(xf.dev).eval()
    return xf


def load_pair(w, h, device):
    imgs = []
    for p in (REF_PATH, TGT_PATH):
        img = cv2.resize(cv2.imread(p), (w, h), interpolation=cv2.INTER_AREA)
        imgs.append((torch.tensor(img).permute(2, 0, 1)[None].float() / 255).to(device))
    return imgs


def check_identical(xf, x0, x1, top_k):
    """The instrumented copies must reproduce the library calls exactly."""
    noop = Marks(lambda: None)
    noop.start()
    with torch.inference_mode():
        a = xf.match_xfeat(x0, x1, top_k=top_k, min_cossim=-1)
        b, _ = sparse_pair(xf, x0, x1, top_k, noop)
        c = xf.match_xfeat_star(x0, x1, top_k=top_k)
        d, _ = star_pair(xf, x0, x1, top_k, noop)
    ok = all(np.array_equal(p, q) for p, q in zip(a + c, b + d))
    return ok, len(a[0]), len(c[0])


def group_ms(steps):
    g = {k: 0.0 for k in GROUP_ORDER}
    for s, v in steps.items():
        g[GROUPS[s]] += v
    return g


def parse_args():
    p = argparse.ArgumentParser(description="XFeat cost profile: MACs/params per branch, CPU/GPU time split")
    p.add_argument("--devices", nargs="+", default=["cpu", "cuda"], choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--top-k", type=int, default=4096)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--plot", action="store_true", help="save results/profile_costs.png")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    net = make_xfeat("cpu", args.top_k).net
    per_match = fine_matcher_macs_per_match(net)
    params = {n: n_params(getattr(net, n)) for n, _ in BRANCHES}
    params["fine_matcher"] = n_params(net.fine_matcher)
    total_params = n_params(net)
    cnn_params = total_params - params["fine_matcher"]

    rows = []
    print(f"MACs per branch (GMACs) and parameters | fine_matcher: {per_match/1e6:.3f} MMACs per match")
    hdr = "".join(f"{f'{w}x{h}':>12}" for w, h in RESOLUTIONS)
    print(f"{'branch':<40}{hdr}{'share':>8}{'params':>10}{'p.share':>9}")
    macs_by_res = {}
    for w, h in RESOLUTIONS:
        hh, ww = net_input_hw(w, h)
        macs, fc_total = branch_macs(net, hh, ww)
        assert abs(sum(macs.values()) - fc_total) <= 1e-6 * fc_total, (sum(macs.values()), fc_total)
        macs_by_res[(w, h)] = macs
    vga = macs_by_res[RESOLUTIONS[0]]
    vga_total = sum(vga.values())
    for name, label in BRANCHES:
        cells = "".join(f"{macs_by_res[r][name]/1e9:>12.3f}" for r in RESOLUTIONS)
        print(f"{label:<40}{cells}{100*vga[name]/vga_total:>7.1f}%{params[name]/1e3:>9.1f}k"
              f"{100*params[name]/cnn_params:>8.1f}%")
    totals = "".join(f"{sum(macs_by_res[r].values())/1e9:>12.3f}" for r in RESOLUTIONS)
    print(f"{'total CNN (cross-checked with FlopCounterMode)':<40}{totals}{'100%':>8}{cnn_params/1e3:>9.1f}k")
    print(f"{'fine_matcher (refinement MLP)':<40}{'per match':>12}{per_match/1e6:>11.3f}M"
          f"{'':>32}{params['fine_matcher']/1e3:>9.1f}k  ({100*params['fine_matcher']/total_params:.1f}% of all)")
    ins = "  ".join(f"{w}x{h}->{net_input_hw(w, h)[1]}x{net_input_hw(w, h)[0]}" for w, h in RESOLUTIONS)
    print(f"network input after preprocess (WxH): {ins}")

    for w, h in RESOLUTIONS:
        hh, ww = net_input_hw(w, h)
        for name, label in BRANCHES:
            rows.append({"resolution": f"{w}x{h}", "net_input": f"{ww}x{hh}", "branch": name,
                         "macs": macs_by_res[(w, h)][name], "params": params[name]})
    rows.append({"resolution": "per match", "net_input": "", "branch": "fine_matcher",
                 "macs": per_match, "params": params["fine_matcher"]})
    with open(os.path.join(RESULTS_DIR, "profile_macs.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0]))
        wcsv.writeheader()
        wcsv.writerows(rows)

    (h1, w1), (h2, w2) = dualscale_hw(args.width, args.height)
    star_macs = sum(branch_macs(net, h1, w1)[0].values()) + sum(branch_macs(net, h2, w2)[0].values())
    print(f"\nper image at {args.width}x{args.height}: sparse net {vga_total/1e9:.3f} GMACs | "
          f"XFeat* net at {w1}x{h1} + {w2}x{h2} = {star_macs/1e9:.3f} GMACs "
          f"({star_macs/vga_total:.2f}x sparse)")

    report = {"config": vars(args), "macs_vga": vga, "params": params,
              "fine_matcher_macs_per_match": per_match, "devices": {}}
    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            print("\ncuda not available, skipped")
            continue
        sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
        xf = make_xfeat(device, args.top_k)
        x0, x1 = load_pair(args.width, args.height, device)
        ok, n_sparse, n_star = check_identical(xf, x0, x1, args.top_k)
        if not ok:
            raise SystemExit("instrumented pipeline differs from match_xfeat / match_xfeat_star")

        hh, ww = net_input_hw(args.width, args.height)
        br = time_branches(xf.net, torch.zeros(1, 3, hh, ww, device=device), sync, args.runs, args.warmup)
        sp = time_pipeline(sparse_pair, xf, x0, x1, args.top_k, sync, args.runs, args.warmup)
        st = time_pipeline(star_pair, xf, x0, x1, args.top_k, sync, args.runs, args.warmup)
        report["devices"][device] = {"branches_ms": br, "sparse": sp, "star": st,
                                     "matches_sparse": n_sparse, "matches_star": n_star}

        tag = f"{device.upper()}" + (f", {torch.get_num_threads()} threads" if device == "cpu" else "")
        print(f"\n===== {tag} | {args.width}x{args.height}, top_k {args.top_k}, "
              f"median of {args.runs} runs | instrumented == library: OK =====")
        print(f"net forward by branch (one image), ms -- MACs share vs time share")
        for name, label in BRANCHES:
            print(f"  {label:<40}{br[name]:>7.2f} ms  time {100*br[name]/br['total']:5.1f}%  "
                  f"MACs {100*vga[name]/vga_total:5.1f}%")
        print(f"  {'other (InstanceNorm, upsample, unfold, sums)':<40}{br['other']:>7.2f} ms  "
              f"time {100*br['other']/br['total']:5.1f}%")
        print(f"  {'net total':<40}{br['total']:>7.2f} ms")

        for title, res, n_net in (("sparse  (match_xfeat)", sp, 2), ("XFeat*  (match_xfeat_star)", st, 4)):
            g = group_ms(res["steps_ms"])
            tot = sum(g.values())
            print(f"\n{title}: pair {res['total_ms']:.1f} +- {res['total_std_ms']:.1f} ms "
                  f"({n_net} net calls per pair"
                  + (f", {res['refined_in']} matches refined)" if res["refined_in"] else ")"))
            for step, v in sorted(res["steps_ms"].items(), key=lambda kv: -kv[1]):
                print(f"    {step:<12}{v:>8.2f} ms  {100*v/tot:5.1f}%   [{GROUPS[step]}]")
            print("  groups: " + " | ".join(f"{k} {g[k]:.1f} ms ({100*g[k]/tot:.0f}%)" for k in GROUP_ORDER))

        print(f"\nceiling of pruning on {tag} (pair time saved if a branch became free)")
        print(f"  {'pruned part':<34}{'sparse ms':>10}{'%pair':>7}{'XFeat* ms':>11}{'%pair':>7}")
        sp_tot, st_tot = sum(sp["steps_ms"].values()), sum(st["steps_ms"].values())
        net_sp, net_st = sp["steps_ms"]["net"], st["steps_ms"]["net"]
        lines = [(label, br[name] / br["total"]) for name, label in BRANCHES[:6]]
        lines.append(("whole backbone (skip1+block1..5)", sum(br[n] for n, _ in BRANCHES[:6]) / br["total"]))
        lines.append(("whole CNN (backbone + heads)", 1.0))
        for label, frac in lines:
            a, b = net_sp * frac, net_st * frac
            print(f"  {label:<34}{a:>10.1f}{100*a/sp_tot:>6.1f}%{b:>11.1f}{100*b/st_tot:>6.1f}%")
        mlp = st["steps_ms"]["refine_mlp"]
        print(f"  {'fine_matcher (refinement MLP)':<34}{'-':>10}{'-':>7}{mlp:>11.1f}{100*mlp/st_tot:>6.1f}%")

    out = os.path.join(RESULTS_DIR, f"profile_{args.width}x{args.height}_k{args.top_k}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nsaved {out} and {os.path.join(RESULTS_DIR, 'profile_macs.csv')}")

    if args.plot:
        plot_costs(report, os.path.join(RESULTS_DIR, "profile_costs.png"))


SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def plot_costs(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vga, params = report["macs_vga"], report["params"]
    names = [n for n, _ in BRANCHES]
    labels = [lab for _, lab in BRANCHES]
    cnn_params = sum(params[n] for n in names)
    macs_share = [100 * vga[n] / sum(vga.values()) for n in names]
    par_share = [100 * params[n] / cnn_params for n in names]
    series = [("MACs", macs_share), ("параметры CNN", par_share)]
    if "cpu" in report["devices"]:
        br = report["devices"]["cpu"]["branches_ms"]
        series.append(("время сети на CPU", [100 * br[n] / br["total"] for n in names]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    y = np.arange(len(names))[::-1]
    hgt = 0.8 / len(series)
    for i, (lab, vals) in enumerate(series):
        yy = y + (len(series) - 1) / 2 * hgt - i * hgt
        ax1.barh(yy, vals, height=hgt * 0.86, color=SERIES[i], label=lab)
        for v, yv in zip(vals, yy):
            if v >= 8:
                ax1.text(v + 0.6, yv, f"{v:.0f}%", va="center", fontsize=7.5, color=INK2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, color=INK, fontsize=9)
    ax1.set_xlabel("доля, %", color=INK2)
    ax1.set_title("CNN по веткам: MACs, параметры и время сети (VGA)", color=INK, fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)

    bars = []
    for dev in ("cpu", "cuda"):
        if dev not in report["devices"]:
            continue
        d = report["devices"][dev]
        name = "CPU 4 потока" if dev == "cpu" else "GPU"
        for key, title in (("sparse", "sparse"), ("star", "XFeat*")):
            g = group_ms(d[key]["steps_ms"])
            bars.append((f"{title}, {name}\n{d[key]['total_ms']:.1f} мс на пару", g))
    yb = np.arange(len(bars))[::-1]
    seen = set()
    for (lab, g), yv in zip(bars, yb):
        tot = sum(g.values())
        left = 0.0
        for i, k in enumerate(GROUP_ORDER):
            w = 100 * g[k] / tot
            if w <= 0:
                continue
            ax2.barh(yv, w, left=left, height=0.62, color=SERIES[i], edgecolor=SURFACE, linewidth=2,
                     label=None if k in seen else k)
            seen.add(k)
            ink = "white" if i == 0 else INK
            if w >= 7:
                ax2.text(left + w / 2, yv, f"{g[k]:.1f} мс\n{w:.0f}%", ha="center", va="center",
                         fontsize=8, color=ink)
            elif w >= 4:
                ax2.text(left + w / 2, yv, f"{w:.0f}%", ha="center", va="center", fontsize=8, color=ink)
            left += w
    ax2.set_yticks(yb)
    ax2.set_yticklabels([b[0] for b in bars], color=INK, fontsize=9)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("доля времени пары, %", color=INK2)
    ax2.set_title("Время пары: сеть, постобработка, матчинг, refinement", color=INK, fontsize=11, loc="left")
    handles, lbls = ax2.get_legend_handles_labels()
    order = [lbls.index(k) for k in GROUP_ORDER if k in lbls]
    ax2.legend([handles[i] for i in order], [lbls[i] for i in order], frameon=False, fontsize=9,
               ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), labelcolor=INK)

    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main()
