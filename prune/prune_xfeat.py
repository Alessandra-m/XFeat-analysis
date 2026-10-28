import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from common.paths import XFEAT_DIR

import torch
import torch.nn as nn
import torch_pruning as tp

from modules.model import XFeatModel


WEIGHTS = os.path.join(XFEAT_DIR, "weights", "xfeat.pt")
STAGES = ["block1", "block2", "block3", "block4", "block5"]


def load_model(device):
    model = XFeatModel().to(device).eval()
    sd = torch.load(WEIGHTS, map_location=device)
    model.load_state_dict(sd)
    return model


def descriptor_output_conv(model):
    """The last Conv2d of block_fusion produces the 64-D descriptor -- its output channels
    are the network's interface and must never be pruned."""
    convs = [m for m in model.block_fusion if isinstance(m, nn.Conv2d)]
    return convs[-1]


def ignored_layers(model):
    """Everything whose OUTPUT channel count must stay fixed."""
    ign = [descriptor_output_conv(model)]
    # both heads: their first conv consumes a fixed-width input, their last conv is an interface
    for head in (model.heatmap_head, model.keypoint_head):
        ign += [m for m in head if isinstance(m, nn.Conv2d)]
    return ign


def stage_conv_layers(model, stage):
    block = getattr(model, stage)
    return [m for m in block.modules() if isinstance(m, nn.Conv2d)]


def main():
    ap = argparse.ArgumentParser(description="Structural L1 pruning of one XFeat stage")
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--ratio", type=float, required=True, help="fraction of channels to remove, e.g. 0.25")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=None, help="path to save the pruned model (full module via torch.save)")
    ap.add_argument("--verbose", action="store_true", help="print each coupled group (group.details())")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = load_model(device)

    example = torch.randn(1, 3, args.height, args.width, device=device)

    # L1 (magnitude, p=1) importance -> structural channel pruning
    importance = tp.importance.MagnitudeImportance(p=1)

    # Prune ONLY the chosen stage: restrict the target layers to that block's convs.
    target_convs = set(stage_conv_layers(model, args.stage))
    ign = ignored_layers(model)
    # also ignore every conv that is NOT in the target stage, so only this stage is cut
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m not in target_convs:
            ign.append(m)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example,
        importance=importance,
        pruning_ratio=args.ratio,
        ignored_layers=ign,
        round_to=1,
    )

    # record conv shapes of the target stage BEFORE pruning (for the before/after table)
    prefix = args.stage + "."
    before = {name: (m.in_channels, m.out_channels)
              for name, m in model.named_modules()
              if isinstance(m, nn.Conv2d) and name.startswith(prefix)}

    base_macs, base_params = tp.utils.count_ops_and_params(model, example)

    # interactive=True yields each coupled group; we prune it right away in the SAME pass.
    # Do NOT also call pruner.step() afterwards -- that double-counts the iterative budget
    # and triggers "Pruning exceed the maximum iterative steps, no pruning will be performed".
    n_groups = 0
    for group in pruner.step(interactive=True):
        if args.verbose:
            print(group.details())
        group.prune()
        n_groups += 1

    macs, params = tp.utils.count_ops_and_params(model, example)

    model.eval()
    with torch.inference_mode():
        feats, kpts, heat = model(example)
    ok = (feats.shape[1] == 64 and kpts.shape[1] == 65 and heat.shape[1] == 1)

    after = {name: (m.in_channels, m.out_channels)
             for name, m in model.named_modules()
             if isinstance(m, nn.Conv2d) and name.startswith(prefix)}
    print(f"\nstage {args.stage}  convs (in->out), * = changed:")
    for name in before:
        bi, bo = before[name]
        ai, ao = after.get(name, (bi, bo))
        mark = " *" if (ai, ao) != (bi, bo) else ""
        print(f"  {name:22s} {bi:3d}->{bo:3d}   =>  {ai:3d}->{ao:3d}{mark}")
    print(f"  coupled groups pruned: {n_groups}")

    print(f"\nstage {args.stage}  ratio {args.ratio}")
    print(f"  params: {base_params/1e3:.1f}k -> {params/1e3:.1f}k  ({100*(1-params/base_params):.1f}% off)")
    print(f"  MACs:   {base_macs/1e9:.3f}G -> {macs/1e9:.3f}G  ({100*(1-macs/base_macs):.1f}% off)")
    print(f"  output shapes: feats {tuple(feats.shape)}, keypoints {tuple(kpts.shape)}, heatmap {tuple(heat.shape)}")
    print(f"  interface preserved (feats=64, kpts=65, heat=1): {'OK' if ok else 'BROKEN'}")
    if not ok:
        raise SystemExit("output interface broken -- pruning coupled a protected layer")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save(model, args.out)
        print(f"  saved pruned model -> {args.out}")


if __name__ == "__main__":
    main()