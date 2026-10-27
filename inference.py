import os

import cv2
import numpy as np
import torch

from paths import RESULTS_DIR, XFEAT_DIR
from modules.xfeat import XFeat

REF_PATH = os.path.join(XFEAT_DIR, "assets", "ref.png")
TGT_PATH = os.path.join(XFEAT_DIR, "assets", "tgt.png")
TOP_K = 4096


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def detect(xfeat, img):
    out = xfeat.detectAndCompute(img, top_k=TOP_K)[0]
    return out["keypoints"], out["descriptors"], out["scores"]


def reliability_map(xfeat, img):
    x, _, _ = xfeat.preprocess_tensor(img)
    feats, kpt_logits, heatmap = xfeat.net(x)
    return feats, kpt_logits, heatmap


def estimate_homography(pts_ref, pts_tgt):
    H, mask = cv2.findHomography(
        pts_ref, pts_tgt, cv2.USAC_MAGSAC, 3.5, maxIters=1000, confidence=0.999
    )
    return H, mask.ravel().astype(bool)


def draw_matches(img_ref, img_tgt, pts_ref, pts_tgt):
    kp_ref = [cv2.KeyPoint(float(x), float(y), 5) for x, y in pts_ref]
    kp_tgt = [cv2.KeyPoint(float(x), float(y), 5) for x, y in pts_tgt]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(pts_ref))]
    return cv2.drawMatches(
        img_ref, kp_ref, img_tgt, kp_tgt, matches, None,
        matchColor=(0, 255, 0), flags=2,
    )


def save_reliability(heatmap, img, path, overlay_path):
    rel = heatmap[0, 0].detach().cpu().numpy()
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
    h, w = img.shape[:2]
    rel = cv2.resize(rel, (w, h), interpolation=cv2.INTER_LINEAR)
    rel = (rel * 255).astype(np.uint8)
    cv2.imwrite(path, rel)
    color = cv2.applyColorMap(rel, cv2.COLORMAP_JET)
    cv2.imwrite(overlay_path, cv2.addWeighted(img, 0.5, color, 0.5, 0))


def main():
    xfeat = XFeat()

    img_ref = load_image(REF_PATH)
    img_tgt = load_image(TGT_PATH)

    kpts_ref, desc_ref, scores_ref = detect(xfeat, img_ref)
    feats, kpt_logits, heatmap = reliability_map(xfeat, img_ref)
    save_reliability(
        heatmap, img_ref,
        os.path.join(RESULTS_DIR, "inference_reliability.png"),
        os.path.join(RESULTS_DIR, "inference_reliability_overlay.png"),
    )

    mkpts_ref, mkpts_tgt = xfeat.match_xfeat_star(img_ref, img_tgt, top_k=TOP_K)
    H, inliers = estimate_homography(mkpts_ref, mkpts_tgt)

    vis = draw_matches(img_ref, img_tgt, mkpts_ref[inliers], mkpts_tgt[inliers])
    cv2.imwrite(os.path.join(RESULTS_DIR, "inference_matches.png"), vis)


if __name__ == "__main__":
    main()
