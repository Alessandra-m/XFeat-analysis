import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from paths import XFEAT_DIR
from modules.model import BasicLayer
from modules.xfeat import XFeat

REF_PATH = os.path.join(XFEAT_DIR, "assets", "ref.png")
TOP_K = 4096
H_IN, W_IN = 576, 800

# known homography: scale + perspective, no rotation
H_GT = np.array([[0.9, 0.05, 30.0],
                 [0.02, 0.95, 20.0],
                 [1e-5, 2e-5, 1.0]])


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def conv_macs(net):
    """MACs of every conv in the backbone, grouped by output scale."""
    stats = {}

    def hook(module, inputs, output):
        conv = module.layer[0]
        k = conv.kernel_size[0] * conv.kernel_size[1]
        macs = output.shape[2] * output.shape[3] * conv.in_channels * conv.out_channels * k
        scale = H_IN // output.shape[2]
        c, m = stats.get(scale, (conv.out_channels, 0))
        stats[scale] = (max(c, conv.out_channels), m + macs)

    handles = []
    for name in ["block1", "block2", "block3", "block4", "block5"]:
        for layer in getattr(net, name):
            if isinstance(layer, BasicLayer):
                handles.append(layer.register_forward_hook(hook))
    with torch.inference_mode():
        net(torch.zeros(1, 1, H_IN, W_IN, device=next(net.parameters()).device))
    for h in handles:
        h.remove()
    return stats


def report_macs(net):
    stats = conv_macs(net)
    total = sum(m for _, m in stats.values())
    print("backbone compute by scale (input 576x800)")
    print(f"{'scale':<8}{'max C':<8}{'GMACs':<10}{'share':<8}")
    for scale in sorted(stats):
        c, m = stats[scale]
        print(f"1/{scale:<6}{c:<8}{m / 1e9:<10.3f}{100 * m / total:<6.1f}%")
    print(f"total   {'':<8}{total / 1e9:.3f}")

    # same 3x3 conv at 1/1: what if it had 64 channels like a VGG-style stem
    hw = H_IN * W_IN
    print("\none 3x3 conv at full resolution:")
    for cin, cout in [(1, 4), (4, 64), (64, 64)]:
        print(f"  {cin:>2} -> {cout:<3} {hw * cin * cout * 9 / 1e9:.3f} GMACs")


def reprojection_errors(pts0, pts1, H):
    proj = cv2.perspectiveTransform(pts0.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    return np.linalg.norm(proj - pts1, axis=1)


def summarize(name, err):
    print(f"{name:<28}{len(err):>6}{np.median(err):>10.2f}"
          f"{100 * (err < 1).mean():>9.1f}%{100 * (err < 3).mean():>9.1f}%")


def coarse_and_refined(xfeat, img0, img1):
    """XFeat* matches with and without the refinement offsets."""
    out0 = xfeat.detectAndComputeDense(xfeat.parse_input(img0), top_k=TOP_K)
    out1 = xfeat.detectAndComputeDense(xfeat.parse_input(img1), top_k=TOP_K)
    idx0, idx1 = xfeat.batch_match(out0["descriptors"], out1["descriptors"])[0]

    f0, f1 = out0["descriptors"][0][idx0], out1["descriptors"][0][idx1]
    k0, k1 = out0["keypoints"][0][idx0], out1["keypoints"][0][idx1]
    sc0 = out0["scales"][0][idx0]

    logits = xfeat.net.fine_matcher(torch.cat([f0, f1], dim=-1))
    conf = F.softmax(logits * 3, dim=-1).max(dim=-1)[0]
    offsets = xfeat.subpix_softmax2d(logits.view(-1, 8, 8))
    good = conf > 0.25

    coarse = k0[good].cpu().numpy(), k1[good].cpu().numpy()
    refined = (k0 + offsets * sc0[:, None])[good].cpu().numpy(), k1[good].cpu().numpy()
    return coarse, refined


def report_refinement(xfeat, img):
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, H_GT, (w, h))

    print("\nmatching ref -> warp(ref, H_gt), error vs ground-truth homography, px")
    print(f"{'method':<28}{'N':>6}{'median':>10}{'<1px':>10}{'<3px':>10}")

    with torch.inference_mode():
        m0, m1 = xfeat.match_xfeat(img, warped, top_k=TOP_K)
        summarize("sparse (detector + MNN)", reprojection_errors(m0, m1, H_GT))

        coarse, refined = coarse_and_refined(xfeat, img, warped)
        summarize("XFeat* coarse (8x8 grid)", reprojection_errors(*coarse, H_GT))
        summarize("XFeat* refined", reprojection_errors(*refined, H_GT))

        # refinement moves points of image 0, so measure error in image 0
        H_inv = np.linalg.inv(H_GT)
        e_c = reprojection_errors(coarse[1], coarse[0], H_inv)
        e_r = reprojection_errors(refined[1], refined[0], H_inv)
        print(f"\nerror measured in image 0: coarse median {np.median(e_c):.2f}, "
              f"refined median {np.median(e_r):.2f}")


def report_reliability(xfeat, img):
    """Match every 8x8 cell (no top-k by reliability) and bin accuracy by reliability."""
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, H_GT, (w, h))

    with torch.inference_mode():
        k0, d0 = xfeat.extractDense(xfeat.parse_input(img), top_k=0)
        k1, d1 = xfeat.extractDense(xfeat.parse_input(warped), top_k=0)
        idx0, idx1 = xfeat.batch_match(d0, d1)[0]

        x, _, _ = xfeat.preprocess_tensor(img)
        _, _, rel = xfeat.net(x)
        rel = rel[0, 0].flatten()

    pts0, pts1 = k0[0][idx0].cpu().numpy(), k1[0][idx1].cpu().numpy()
    # extractDense keeps cells sorted by reliability, recover each cell's value
    _, order = torch.sort(rel, descending=True)
    rel_matched = rel[order][idx0].cpu().numpy()

    # coarse points are cell corners, so allow one cell of error
    correct = reprojection_errors(pts0, pts1, H_GT) < 8
    print(f"\nall {len(rel)} cells, {len(pts0)} MNN matches; accuracy (<8px) by reliability quartile")
    qs = np.quantile(rel_matched, [0, 0.25, 0.5, 0.75, 1])
    for i in range(4):
        sel = (rel_matched >= qs[i]) & (rel_matched <= qs[i + 1])
        print(f"  Q{i + 1} reliability {qs[i]:.3f}-{qs[i + 1]:.3f}: {100 * correct[sel].mean():5.1f}%")

    # how many cells of each quartile survive the mutual nearest neighbour check
    rel_all = rel[order].cpu().numpy()
    mutual = np.zeros(len(rel_all), dtype=bool)
    mutual[idx0.cpu().numpy()] = True
    qs = np.quantile(rel_all, [0, 0.25, 0.5, 0.75, 1])
    print("share of cells with a mutual match, by reliability quartile over all cells")
    for i in range(4):
        sel = (rel_all >= qs[i]) & (rel_all <= qs[i + 1])
        print(f"  Q{i + 1} reliability {qs[i]:.3f}-{qs[i + 1]:.3f}: {100 * mutual[sel].mean():5.1f}%")


def main():
    xfeat = XFeat()
    img = load_image(REF_PATH)

    report_macs(xfeat.net)
    report_refinement(xfeat, img)
    report_reliability(xfeat, img)


if __name__ == "__main__":
    main()
