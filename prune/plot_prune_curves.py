import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from common.paths import PRUNE_DIR


STAGE_COLORS = {"block1": "#D85A30", "block2": "#BA7517", "block3": "#534AB7",
                "block4": "#1D9E75", "block5": "#185FA5"}


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # numbers come back as strings -> cast the columns we use
    for r in rows:
        for k, v in r.items():
            if k not in ("stage",):
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    pass
        r["ratio_pct"] = 100 * r["ratio"]   # share of channels removed in the stage, %
    return rows


def baseline(rows):
    for r in rows:
        if r["stage"] == "baseline":
            return r
    raise SystemExit("no baseline row in CSV")


def stage_series(rows, stage, xcol, ycol, base):
    """Points for one stage: (0, baseline) + every pruned variant, sorted by x."""
    pts = [(0.0, base[ycol])]
    pts += [(r[xcol], r[ycol]) for r in rows if r["stage"] == stage]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def stages_in(rows):
    return [s for s in STAGE_COLORS if any(r["stage"] == s for r in rows)]


RATIO_LABEL = "Удалено каналов в стадии, %"


def ratio_ticks(ax, rows):
    ax.set_xticks([0] + sorted({r["ratio_pct"] for r in rows if r["stage"] != "baseline"}))


def plot_quality(rows, base, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, ycol, name in [(axes[0], "mma_all_3", "MMA@3"),
                           (axes[1], "auc_all_3", "AUC@3"),
                           (axes[2], "mha_all_3", "MHA@3")]:
        for s in stages_in(rows):
            x, y = stage_series(rows, s, "ratio_pct", ycol, base)
            ax.plot(x, y, marker="o", color=STAGE_COLORS[s], label=s, lw=1.8, ms=5)
        ax.axhline(base[ycol], color="#888", ls="--", lw=1, label="baseline")
        ratio_ticks(ax, rows)
        ax.set_xlabel(RATIO_LABEL)
        ax.set_ylabel(name + ", %")
        ax.set_title(name + " vs доля удалённых каналов")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved", out)


def plot_speed(rows, base, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ycol, name in [(axes[0], "cpu_extract_fps", "Extract FPS"),
                           (axes[1], "cpu_match_fps", "Extract+match FPS")]:
        scol = ycol + "_std"   # spread over speed repeats; absent in old CSVs
        has_std = scol in base
        for s in stages_in(rows):
            x, y = stage_series(rows, s, "ratio_pct", ycol, base)
            if has_std:
                _, e = stage_series(rows, s, "ratio_pct", scol, base)
                ax.errorbar(x, y, yerr=e, marker="o", color=STAGE_COLORS[s], label=s,
                            lw=1.8, ms=5, capsize=3)
            else:
                ax.plot(x, y, marker="o", color=STAGE_COLORS[s], label=s, lw=1.8, ms=5)
        ax.axhline(base[ycol], color="#888", ls="--", lw=1, label="baseline")
        if has_std:
            ax.axhspan(base[ycol] - base[scol], base[ycol] + base[scol], color="#888", alpha=0.15)
        ratio_ticks(ax, rows)
        ax.set_xlabel(RATIO_LABEL)
        ax.set_ylabel("FPS (CPU), медиана по повторам" if has_std else "FPS (CPU)")
        ax.set_title(name + " vs доля удалённых каналов")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved", out)


def plot_tradeoff(rows, base, out, ycol="mma_all_3"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for s in stages_in(rows):
        pts = [(r["macs_off_pct"], base[ycol] - r[ycol]) for r in rows if r["stage"] == s]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", color=STAGE_COLORS[s], label=s, lw=1.8, ms=6)
    ax.axhline(0, color="#888", lw=1)
    ax.set_xlabel("Срезано MACs сети (backbone + головы), %")
    ax.set_ylabel("Потеря MMA@3 относительно baseline, п.п.")
    ax.set_title("Сделка: сколько качества теряем за процент MACs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description="Plot pruning-grid curves from the CSV")
    ap.add_argument("--csv", default=os.path.join(PRUNE_DIR, "prune_grid.csv"))
    ap.add_argument("--outdir", default=PRUNE_DIR)
    args = ap.parse_args()

    rows = load_csv(args.csv)
    base = baseline(rows)
    os.makedirs(args.outdir, exist_ok=True)

    plot_quality(rows, base, os.path.join(args.outdir, "prune_quality.png"))
    plot_speed(rows, base, os.path.join(args.outdir, "prune_speed.png"))
    plot_tradeoff(rows, base, os.path.join(args.outdir, "prune_tradeoff.png"))


if __name__ == "__main__":
    main()
