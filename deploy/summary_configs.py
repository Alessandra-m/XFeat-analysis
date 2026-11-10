"""Summary table: baseline XFeat and pruned configs on RK3588 -- HPatches quality (simulator) + NPU speed (board).

Quality: results/hpatches_<name>_<variant>_all_r480_thr3.json (deploy/hpatches_rknn.py).
Speed:   board log of rknn_benchmark with lines "== <name>_<dtype> core_mask=<c>" followed by "Avg Time X ms".
MACs:    prune/results/prune_net_speed.csv (column macs_off_pct; "config A" -> config_A).
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import HERE, RESULTS_DIR

NAMES = ["xfeat", "config_A", "config_B", "config_C", "config_D"]
VARIANTS = [("onnx", "onnx fp32", None), ("rknn_fp16", "rknn fp16", "fp16"), ("rknn_i8_mmse", "rknn i8 mmse", "i8_mmse")]


def load_quality(name, variant, suffix):
    path = os.path.join(RESULTS_DIR, f"hpatches_{name}_{variant}_{suffix}.json")
    if not os.path.exists(path):
        return None
    s = json.load(open(path))["summary"]
    a, i, v = s["all"], s["i"], s["v"]
    return {"matches": a["matches_mean"], "mma3": 100 * a["mma"]["3"],
            "auc3": 100 * a["auc"]["3"]["mean"], "auc5": 100 * a["auc"]["5"]["mean"], "auc10": 100 * a["auc"]["10"]["mean"],
            "mha3_i": 100 * i["mha"]["3"]["mean"], "mha3_v": 100 * v["mha"]["3"]["mean"]}


def load_speed(log):
    """{('config_A', 'fp16', 1): ms, ...}"""
    speed, key = {}, None
    if not log or not os.path.exists(log):
        return speed
    for line in open(log):
        m = re.match(r"==\s*(\S+?)_(fp16|i8_mmse|i8)\s+core_mask=(\d+)", line)
        if m:
            key = (m.group(1), m.group(2), int(m.group(3)))
            continue
        m = re.search(r"Avg Time\s+([\d.]+)\s*ms", line)
        if m and key:
            speed[key] = float(m.group(1))
            key = None
    return speed


def load_macs():
    path = os.path.join(HERE, "prune", "results", "prune_net_speed.csv")
    macs = {"xfeat": 0.0}
    if os.path.exists(path):
        for row in csv.DictReader(open(path)):
            if row["kind"] == "config" and re.fullmatch(r"config [A-D]", row["variant"]):
                macs[row["variant"].replace(" ", "_")] = float(row["macs_off_pct"])
    return macs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-log", default=os.path.join(HERE, "deploy", "logs", "bench_configs.txt"))
    p.add_argument("--suffix", default="all_r480_thr3")
    p.add_argument("--cores", type=int, nargs="+", default=[1, 7], help="core_mask values to show")
    args = p.parse_args()

    speed, macs = load_speed(args.bench_log), load_macs()
    base = {v: load_quality("xfeat", v, args.suffix) for v, _, _ in VARIANTS}

    cols = "".join(f"  ms c{c}" for c in args.cores)
    head = f"{'model':9s} {'variant':13s} {'MACs-%':>6s} {'match':>6s} {'MMA@3':>6s} {'AUC@3':>6s} {'AUC@5':>6s} {'AUC@10':>6s} {'MHA@3i':>6s} {'MHA@3v':>6s} {'dAUC@3':>6s} {'dMMA@3':>6s}{cols}"
    print(head)
    print("-" * len(head))
    for name in NAMES:
        rows = [(v, l, d, load_quality(name, v, args.suffix)) for v, l, d in VARIANTS]
        rows = [r for r in rows if r[3] is not None]
        if not rows:
            continue
        for variant, label, dtype, q in rows:
            b = base[variant]
            d_auc = f"{q['auc3'] - b['auc3']:+6.1f}" if b else f"{'':6s}"
            d_mma = f"{q['mma3'] - b['mma3']:+6.1f}" if b else f"{'':6s}"
            sp = "".join(f"{speed[(name, dtype, c)]:7.2f}" if dtype and (name, dtype, c) in speed else f"{'-':>7s}"
                         for c in args.cores)
            m = macs.get(name)
            print(f"{name:9s} {label:13s} {(f'{m:.1f}' if m is not None else '-'):>6s} {q['matches']:6.0f} {q['mma3']:6.1f} "
                  f"{q['auc3']:6.1f} {q['auc5']:6.1f} {q['auc10']:6.1f} {q['mha3_i']:6.1f} {q['mha3_v']:6.1f} "
                  f"{d_auc} {d_mma}{sp}")
        print()
    print("dAUC@3 / dMMA@3: difference to xfeat with the same variant (effect of pruning alone).")
    print("ms cN: rknn_benchmark Avg Time, core_mask N (1 = one core, 7 = three cores).")


if __name__ == "__main__":
    main()
