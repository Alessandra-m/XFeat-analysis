"""PyTorch module behind deploy/models/xfeat_grayscale_480_640.onnx (the "gray model").

Reconstructed from the ONNX graph and checked against it (max|diff| <= 6e-6 on all outputs).
Weights are the original xfeat.pt; the only rewrite is that _unfold2d(x, 8) + the first 1x1 conv
of keypoint_head are merged into one Conv2d(1, 64, 8, stride=8) (same weights, reshaped).

Compared with deploy/export_onnx.py (XFeatBackbone) the gray model keeps inside the graph:
  * pre:  mean over RGB + InstanceNorm2d(1)           (input: raw (1, 3, H, W), 0..255)
  * post: L2-normalize feats, softmax(keypoints)[:, :64]
The flags `with_pre` / `with_post` switch these parts off (used by deploy/export_parts.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class XFeatGray(nn.Module):
    def __init__(self, net, with_pre=True, with_post=True):
        super().__init__()                     # net: XFeatModel with xfeat.pt loaded, eval()
        self.net = net
        self.with_pre = with_pre
        self.with_post = with_post
        kh = net.keypoint_head
        self.kp0 = nn.Conv2d(1, 64, 8, stride=8, bias=False)
        self.kp0.weight.data = kh[0].layer[0].weight.data.reshape(64, 1, 8, 8).clone()
        self.kp_rest = nn.Sequential(kh[0].layer[1], kh[0].layer[2], *kh[1:])

    def forward(self, x):
        n = self.net
        if self.with_pre:                      # x: raw (1, 3, H, W); else already normalized (1, 1, H, W)
            x = n.norm(x.mean(dim=1, keepdim=True))
        x1 = n.block1(x)
        x2 = n.block2(x1 + n.skip1(x))
        x3 = n.block3(x2)
        x4 = n.block4(x3)
        x5 = n.block5(x4)
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode="bilinear")
        feats = n.block_fusion(x3 + x4 + x5)
        heatmap = n.heatmap_head(feats)
        kpts = self.kp_rest(self.kp0(x))
        if self.with_post:
            return F.normalize(feats, dim=1), F.softmax(kpts, 1)[:, :64], heatmap
        return feats, kpts, heatmap
