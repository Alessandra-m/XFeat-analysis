import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import XFEAT_DIR

import torch
import torch.nn as nn
import torch_pruning as tp

from prune.prune_xfeat import load_model, ignored_layers, stage_conv_layers, STAGES


def prune_one_stage(model, stage, ratio, example):
    """Prune a single stage in place; return (macs_off, params_off, changed_layers, n_groups)."""
    importance = tp.importance.MagnitudeImportance(p=1)

    target = set(stage_conv_layers(model, stage))
    ign = ignored_layers(model)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m not in target:
            ign.append(m)

    prefix = stage + "."
    before = {n: (m.in_channels, m.out_channels)
              for n, m in model.named_modules()
              if isinstance(m, nn.Conv2d) and n.startswith(prefix)}

    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs=example, importance=importance,
        pruning_ratio=ratio, ignored_layers=ign, round_to=1,
    )

    base_macs, base_params = tp.utils.count_ops_and_params(model, example)
    n_groups = 0
    for group in pruner.step(interactive=True):
        group.prune()
        n_groups += 1
    macs, params = tp.utils.count_ops_and_params(model, example)

    after = {n: (m.in_channels, m.out_channels)
             for n, m in model.named_modules()
             if isinstance(m, nn.Conv2d) and n.startswith(prefix)}
    changed = [f"{n.split('.')[-3]}:{before[n][1]}->{after[n][1]}"
               for n in before if before[n][1] != after[n][1]]

    model.eval()
    with torch.inference_mode():
        feats, kpts, heat = model(example)
    ok = (feats.shape[1] == 64 and kpts.shape[1] == 65 and heat.shape[1] == 1)

    return {"macs_off": 100 * (1 - macs / base_macs),
            "params_off": 100 * (1 - params / base_params),
            "changed": changed, "n_groups": n_groups, "ok": ok,
            "macs_g": macs / 1e9}


def main():
    ap = argparse.ArgumentParser(description="Per-stage MACs reduction recon sweep")
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    example = torch.randn(1, 3, args.height, args.width, device=device)

    base = load_model(device)
    base_macs, base_params = tp.utils.count_ops_and_params(base, example)
    print(f"baseline: {base_macs/1e9:.3f} GMACs, {base_params/1e3:.1f}k params "
          f"| input {args.width}x{args.height} | ratio {args.ratio}\n")

    print(f"{'stage':<8}{'MACs off':>9}{'params off':>12}{'groups':>8}  changed conv outputs")
    rows = []
    for stage in STAGES:
        model = load_model(device)   # fresh copy each time -- stages measured independently
        r = prune_one_stage(model, stage, args.ratio, example)
        flag = "" if r["ok"] else "  [BROKEN]"
        print(f"{stage:<8}{r['macs_off']:>8.1f}%{r['params_off']:>11.1f}%{r['n_groups']:>8}  "
              f"{', '.join(r['changed']) or '(none)'}{flag}")
        rows.append((stage, r))
        del model

    print("\nnote: block3/4/5 outputs are frozen by the FPN sum (x3+x4+x5 -> 64-D descriptor),")
    print("so pruning them in isolation only trims inner layers -> small MACs reduction.")


if __name__ == "__main__":
    main()
