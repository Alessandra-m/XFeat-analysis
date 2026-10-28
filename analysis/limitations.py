import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from common.paths import XFEAT_DIR
from analysis.decisions import H_GT, coarse_and_refined
from modules.xfeat import XFeat

REF_PATH = os.path.join(XFEAT_DIR, "assets", "ref.png")
TOP_K = 4096
ANGLES = [0, 10, 20, 30, 45, 60, 90, 180]
SHIFTS = [0.0, 0.25, 0.5, 0.75]


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def reprojection_errors(pts0, pts1, H):
    proj = cv2.perspectiveTransform(pts0.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    return np.linalg.norm(proj - pts1, axis=1)


def texture(h, w, seed=0):
    t = np.random.default_rng(seed).standard_normal((h, w)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), 3)
    return 100 * t / np.abs(t).max()


def encode(t, mode):
    img = np.full(t.shape + (3,), 128, dtype=np.float32)
    if mode == "gray":
        img += t[..., None]
    else:
        img[..., 2] += t
        img[..., 1] -= t
    return img.clip(0, 255).astype(np.uint8)


def report_grayscale(xfeat, img):
    print("1. grayscale input")

    with torch.inference_mode():
        x, _, _ = xfeat.preprocess_tensor(img)
        f_bgr, _, _ = xfeat.net(x)
        f_rgb, _, _ = xfeat.net(x[:, [2, 0, 1]])
    rel = ((f_bgr - f_rgb).norm() / f_bgr.norm()).item()
    print(f"  descriptors BGR vs GBR permutation: relative diff = {rel:.1e}")

    t = texture(576, 800)
    for mode in ["gray", "hue"]:
        im = encode(t, mode)
        moved = cv2.warpAffine(im, np.float32([[1, 0, 16], [0, 1, 8]]), (800, 576),
                               borderMode=cv2.BORDER_REFLECT)
        H = np.array([[1, 0, 16], [0, 1, 8], [0, 0, 1]], dtype=np.float64)
        gray_std = im.astype(np.float32).mean(axis=2).std()
        with torch.inference_mode():
            n = len(xfeat.detectAndCompute(im, top_k=TOP_K)[0]["keypoints"])
            m0, m1 = xfeat.match_xfeat(im, moved, top_k=TOP_K)
        c, p = correct_matches(m0, m1, H)
        print(f"  texture in {mode:<5} std of channel mean {gray_std:5.1f}, "
              f"keypoints {n:>5}, matches {len(m0):>5}, correct {c:>5} ({p:.1f}%)")


def rotation_homography(angle, w, h):
    A = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return np.vstack([A, [0, 0, 1]])


def correct_matches(m0, m1, H, thr=3.0):
    if len(m0) == 0:
        return 0, 0.0
    ok = reprojection_errors(m0, m1, H) < thr
    return int(ok.sum()), 100 * ok.mean()


def report_rotation(xfeat, img):
    print("\n2. rotation (ref -> rotated ref), correct = error < 3 px")
    print(f"  {'angle':>6}{'sparse N':>10}{'correct':>9}{'%':>7}{'XFeat* N':>11}{'correct':>9}{'%':>7}")
    h, w = img.shape[:2]
    for angle in ANGLES:
        H = rotation_homography(angle, w, h)
        rotated = cv2.warpAffine(img, H[:2], (w, h))
        with torch.inference_mode():
            s0, s1 = xfeat.match_xfeat(img, rotated, top_k=TOP_K)
            d0, d1 = xfeat.match_xfeat_star(img, rotated, top_k=TOP_K)
        sc, sp = correct_matches(s0, s1, H)
        dc, dp = correct_matches(d0, d1, H)
        print(f"  {angle:>6}{len(s0):>10}{sc:>9}{sp:>6.1f}%{len(d0):>11}{dc:>9}{dp:>6.1f}%")


def report_subpixel(xfeat, img):
    print("\n3. subpixel accuracy: ref -> ref shifted by dx px along x")
    img = img[:576, :800]
    h, w = img.shape[:2]

    with torch.inference_mode():
        kpts = xfeat.detectAndCompute(img, top_k=TOP_K)[0]["keypoints"].cpu().numpy()
    frac = np.abs(kpts - np.round(kpts)).max()
    print(f"  sparse keypoints: max distance to integer pixel = {frac:.3f}")

    print(f"  {'dx':>6}{'sparse median err':>20}{'integer disp.':>15}{'XFeat* median err':>20}"
          f"{'XFeat* mean signed err (x, y)':>32}")
    for dx in SHIFTS:
        A = np.float32([[1, 0, dx], [0, 1, 0]])
        shifted = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        H = np.vstack([A, [0, 0, 1]]).astype(np.float64)
        with torch.inference_mode():
            s0, s1 = xfeat.match_xfeat(img, shifted, top_k=TOP_K)
            d0, d1 = xfeat.match_xfeat_star(img, shifted, top_k=TOP_K)
        es = reprojection_errors(s0, s1, H)
        ed = reprojection_errors(d0, d1, H)
        disp = s1 - s0
        integer = 100 * np.all(np.abs(disp - np.round(disp)) < 1e-6, axis=1).mean()
        signed = (d1 - cv2.perspectiveTransform(d0.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)).mean(0)
        print(f"  {dx:>6.2f}{np.median(es):>20.2f}{integer:>14.1f}%{np.median(ed):>20.2f}"
              f"{'':>14}({signed[0]:+.2f}, {signed[1]:+.2f})")


def report_refinement_center(xfeat, img):
    print("\n4. refinement zero bin: official center 4 vs center 3")

    with torch.inference_mode():
        _, _, bins = coarse_and_refined(xfeat, img, img)
    vals, counts = np.unique(bins, return_counts=True)
    top = np.argsort(-counts)[:3]
    print("  identical pair, most frequent argmax bins (x, y): " + ", ".join(
        f"({vals[i] % 8}, {vals[i] // 8}) {100 * counts[i] / len(bins):.1f}%" for i in top))

    h, w = img.shape[:2]
    crop = img[:576, :800]
    A = np.float32([[1, 0, 0.5], [0, 1, 0]])
    pairs = [
        ("homography H_gt", img, cv2.warpPerspective(img, H_GT, (w, h)), H_GT),
        ("shift dx = 0.5", crop,
         cv2.warpAffine(crop, A, (800, 576), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT),
         np.vstack([A, [0, 0, 1]]).astype(np.float64)),
        ("identical pair", img, img, np.eye(3)),
    ]
    print(f"  {'pair':<18}{'center':>8}{'median':>10}{'<1px':>10}")
    for name, im0, im1, H in pairs:
        for center in [4, 3]:
            with torch.inference_mode():
                _, refined, _ = coarse_and_refined(xfeat, im0, im1, center=center)
            err = reprojection_errors(*refined, H)
            print(f"  {name:<18}{center:>8}{np.median(err):>10.2f}{100 * (err < 1).mean():>9.1f}%")


def main():
    xfeat = XFeat()
    img = load_image(REF_PATH)

    report_grayscale(xfeat, img)
    report_rotation(xfeat, img)
    report_subpixel(xfeat, img)
    report_refinement_center(xfeat, img)


if __name__ == "__main__":
    main()
