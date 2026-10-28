import os
import sys

import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from common.paths import XFEAT_DIR
from modules.xfeat import XFeat

REF_PATH = os.path.join(XFEAT_DIR, "assets", "ref.png")
TOP_K = 4096

BACKBONE = ["skip1", "block1", "block2", "block3", "block4", "block5"]
BRANCHES = {
    "descriptor head": "block_fusion",
    "reliability head": "heatmap_head",
    "keypoint head": "keypoint_head",
}


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def register_hooks(net, names, records):
    handles = []
    for name in names:
        def hook(module, inputs, output, name=name):
            records.append((name, tuple(inputs[0].shape), tuple(output.shape)))
        handles.append(getattr(net, name).register_forward_hook(hook))
    return handles


def scale(shape, H, W):
    h, w = shape[-2:]
    if (h, w) == (H, W):
        return "1/1"
    return f"1/{H // h}" if H % h == 0 and W // w == H // h else f"{h}x{w}"


def print_table(title, records, H, W):
    print(f"\n{title}")
    print(f"{'module':<16}{'input':<22}{'output':<22}{'in scale':<10}{'out scale':<10}")
    for name, s_in, s_out in records:
        print(f"{name:<16}{str(s_in):<22}{str(s_out):<22}"
              f"{scale(s_in, H, W):<10}{scale(s_out, H, W):<10}")


def check(name, cond):
    print(f"[{'OK' if cond else 'FAIL'}] {name}")


def main():
    xfeat = XFeat()
    net = xfeat.net
    img = load_image(REF_PATH)

    x, _, _ = xfeat.preprocess_tensor(img)
    _, _, H, W = x.shape
    print(f"image {img.shape} -> network input {tuple(x.shape)}")

    records = []
    handles = register_hooks(net, BACKBONE + list(BRANCHES.values()), records)
    with torch.inference_mode():
        feats, kpt_logits, heatmap = net(x)
    for h in handles:
        h.remove()

    by_name = {name: (s_in, s_out) for name, s_in, s_out in records}
    print_table("backbone", [r for r in records if r[0] in BACKBONE], H, W)
    print_table("heads", [r for r in records if r[0] not in BACKBONE], H, W)

    # keypoint head eats the unfolded grayscale image, not backbone features
    with torch.inference_mode():
        gray = net.norm(x.mean(dim=1, keepdim=True))
        unfolded = net._unfold2d(gray, ws=8)
    print(f"\nkeypoint head input: gray {tuple(gray.shape)} -> unfold 8x8 {tuple(unfolded.shape)}")

    print("FPN fusion: block3 {} + up(block4 {}) + up(block5 {}) -> block_fusion {}".format(
        by_name["block3"][1], by_name["block4"][1], by_name["block5"][1], by_name["block_fusion"][1]))

    rec_fine = []
    handles = register_hooks(net, ["fine_matcher"], rec_fine)
    with torch.inference_mode():
        out = xfeat.detectAndComputeDense(xfeat.parse_input(img), top_k=TOP_K)
        idx0, idx1 = xfeat.batch_match(out["descriptors"], out["descriptors"])[0]
        xfeat.refine_matches(out, out, [(idx0, idx1)], batch_idx=0)
    for h in handles:
        h.remove()
    _, s_in, s_out = rec_fine[0]
    print(f"fine matcher: {s_in} -> {s_out} (reshaped to N x 8 x 8 offsets)")

    sparse = xfeat.detectAndCompute(img, top_k=TOP_K)[0]
    print(f"sparse: keypoints {tuple(sparse['keypoints'].shape)}, "
          f"descriptors {tuple(sparse['descriptors'].shape)}, scores {tuple(sparse['scores'].shape)}")

    print("\nchecks against the paper")
    channels = [net.block1[i].layer[0].out_channels for i in range(4)]
    channels += [by_name[b][1][1] for b in ["block2", "block3", "block4"]]
    channels += [net.block5[0].layer[0].out_channels]
    unique = sorted(set(channels), key=channels.index)
    print(f"backbone widths: {channels} -> {unique}")
    check("backbone channels {4, 8, 24, 64, 64, 128}",
          unique == [4, 8, 24, 64, 128] and by_name["block4"][1][1] == 64)
    check("pyramid 1/8, 1/16, 1/32",
          [scale(by_name[b][1], H, W) for b in ["block3", "block4", "block5"]] == ["1/8", "1/16", "1/32"])
    check("keypoint head H/8 x W/8 x 65", tuple(kpt_logits.shape) == (1, 65, H // 8, W // 8))
    check("descriptors H/8 x W/8 x 64", tuple(feats.shape) == (1, 64, H // 8, W // 8))
    check("reliability H/8 x W/8 x 1", tuple(heatmap.shape) == (1, 1, H // 8, W // 8))
    check("unfold 8x8 gives 64 channels at 1/8", tuple(unfolded.shape) == (1, 64, H // 8, W // 8))

    # keypoint heatmap after softmax + depth-to-space is full resolution
    kh = xfeat.get_kpts_heatmap(kpt_logits)
    check(f"keypoint heatmap after depth-to-space {tuple(kh.shape)} = 1 x H x W",
          tuple(kh.shape) == (1, 1, H, W))


if __name__ == "__main__":
    main()
