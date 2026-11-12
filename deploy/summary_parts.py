"""Cost of each XFeat part on the NPU, by subtraction of rknn_benchmark timings (see deploy/export_parts.py).

Input: the board log with blocks like
    == p1_block1_i8 core_mask=1
    ...
    Avg Time 3.21ms, Avg FPS = ...

    python -m deploy.summary_parts deploy/logs/parts_time.txt              # original model (6.6)
    python -m deploy.summary_parts --s2d deploy/logs/parts_s2d_time.txt    # s2d x4 model (deploy/export_parts_s2d.py)
"""
import argparse
import re

# (label, minuend, subtrahend); None = take the model time as is
ROWS = [
    ("fixed cost of a run (avg_pool only)", "p0_min", None),
    ("block1 + skip1 (480x640 -> 120x160)", "p1_block1", "p0_min"),
    ("block2 (120x160)", "p2_block2", "p1_block1"),
    ("block3 (60x80)", "p3_block3", "p2_block2"),
    ("block4 + block5 + FPN + fusion + heatmap", "p4_desc", "p3_block3"),
    ("keypoint branch alone (unshuffle 8x8 + head)", "p5_kpts", "p0_min"),
    ("keypoint branch inside the full model", "xfeat", "p4_desc"),
    ("full model, ours", "xfeat", None),
    ("gray: pre-ops in graph (mean + InstanceNorm)", "gray", "gray_nopre"),
    ("gray: post-ops in graph (L2 + softmax + slice)", "gray", "gray_nopost"),
    ("full model, gray", "gray", None),
]

ROWS_S2D = [
    ("fixed cost of a run (avg_pool only)", "q0_min", None),
    ("block1 + skip1 (s2d, 120x160)", "q1_block1", "q0_min"),
    ("block2 (120x160)", "q2_block2", "q1_block1"),
    ("block3 (60x80)", "q3_block3", "q2_block2"),
    ("block4 (30x40)", "q4_block4", "q3_block3"),
    ("block5 (15x20)", "q5_block5", "q4_block4"),
    ("FPN: 2x Resize + add (60x80)", "q6_fpn", "q5_block5"),
    ("block_fusion (60x80)", "q7_fusion", "q6_fpn"),
    ("heatmap_head (60x80)", "q8_desc", "q7_fusion"),
    ("keypoint branch alone", "q9_kpts", "q0_min"),
    ("keypoint branch inside the full model", "s2d4", "q8_desc"),
    ("full model, s2d x4", "s2d4", None),
]


def parse(path):
    times, cur = {}, None
    for line in open(path, errors="replace"):
        m = re.match(r"== (\S+) core_mask=(\d+)", line)
        if m:
            cur = (re.sub(r"_(i8|fp16)(_mmse)?$", "", m[1]), int(m[2]))
            continue
        m = re.search(r"Avg Time ([\d.]+)ms", line)
        if m and cur:
            times[cur] = float(m[1])
            cur = None
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--s2d", action="store_true", help="parts of the s2d x4 model (full model label: s2d4)")
    args = ap.parse_args()
    t = parse(args.log)
    rows, full = (ROWS_S2D, "s2d4") if args.s2d else (ROWS, "xfeat")
    masks = sorted({c for _, c in t})
    head = "".join(f"{f'core_mask={c}, ms':>18s}" for c in masks)
    print(f"{'part':48s}{head}")
    for label, a, b in rows:
        cells = []
        for c in masks:
            if (a, c) not in t or (b is not None and (b, c) not in t):
                cells.append(f"{'—':>18s}")
                continue
            v = t[(a, c)] - (t[(b, c)] if b else 0.0)
            share = f" ({v / t[(full, c)] * 100:3.0f}%)" if b and (full, c) in t and not a.startswith("gray") else ""
            cells.append(f"{v:>11.2f}{share:>7s}")
        print(f"{label:48s}{''.join(cells)}")
    print(f"\nshare = % of the full model '{full}'. Differences are approximate (each graph is compiled on its own).")


if __name__ == "__main__":
    main()
