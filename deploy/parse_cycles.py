"""Per-op cycle estimate of the RKNN compiler, grouped by XFeat block.

The table comes from `convert_rknn.py --verbose` (rknn-toolkit2 prints it during build()).
These are the compiler's estimates, NOT a measurement: totals are lower than the board time
(Resize/Add/Softmax get 0 cycles), but the shares between layers are a useful first guess.
Measured per-layer time on the board: RKNN_PERF=1 ./rknn_benchmark ... (deploy/board/).

    python -m deploy.parse_cycles deploy/logs/build_xfeat_i8_mmse.txt [more logs ...]
"""
import argparse
import collections
import re

ROW = re.compile(r"(\d+)\s+(\S+)\s+(\S+)\s+(NPU|CPU)\s+(.*?)\s+(\d+)/(\d+)/(\d+)\s+(\d+)\s+(\S+)")
GROUPS = ["keypoint_head", "heatmap_head", "block_fusion", "block1", "block2", "block3", "block4", "block5", "skip1"]


def parse(path):
    rows, on = [], False
    for line in open(path, errors="replace"):
        line = re.sub(r"^D RKNN: \[[^\]]*\] ", "", line.rstrip())
        if line.startswith("ID") and "Cycles" in line:
            on, rows = True, []                      # keep the last table in the log
            continue
        if on and line.startswith("----") and rows:
            on = False
            continue
        m = ROW.match(line) if on else None
        if m:
            rows.append(dict(op=m[2], target=m[4], ddr=int(m[6]), npu=int(m[7]), total=int(m[8]),
                             rw_kb=int(m[9]), name=m[10].split(":", 1)[-1]))
    return rows


def group(name):
    for g in GROUPS:
        if g in name:
            return g
    if "Conv_split" in name or "Conv_select" in name:
        return "unshuffle 8x8 (keypoints)"
    if any(s in name for s in ("norm", "ReduceMean", "INPUT_1")):
        return "input + norm"
    if "Softmax" in name or "Slice" in name:
        return "softmax + slice"
    if "Resize" in name or "Add" in name:
        return "FPN resize/add"
    if "lpnorm" in name:
        return "L2 norm"
    return name


def report(path, top):
    rows = parse(path)
    if not rows:
        print(f"{path}: no cycle table (was convert_rknn.py run with --verbose and output redirected?)")
        return
    total = sum(r["total"] for r in rows)
    cpu = [r["op"] for r in rows if r["target"] == "CPU" and "Operator" not in r["op"]]
    print(f"\n=== {path}\n    total cycles {total:,} | ops {len(rows)} | ops on CPU: {cpu or 'none'}")
    g = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = group(r["name"])
        g[k][0] += r["total"]
        g[k][1] += r["rw_kb"]
    print(f"    {'part':28s} {'share':>6s} {'cycles':>11s} {'RW, MB':>7s}")
    for k, (c, rw) in sorted(g.items(), key=lambda x: -x[1][0]):
        if c or rw > 500:
            print(f"    {k:28s} {c / total * 100:5.1f}% {c:>11,} {rw / 1024:7.1f}")
    print(f"    top {top} ops (bound = which of DDR/NPU cycles is larger):")
    for r in sorted(rows, key=lambda r: -r["total"])[:top]:
        bound = "DDR" if r["ddr"] > r["npu"] else "NPU"
        print(f"    {r['total'] / total * 100:5.1f}%  {r['op']:12s} {bound}  {r['name'][:70]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()
    for p in args.logs:
        report(p, args.top)


if __name__ == "__main__":
    main()
