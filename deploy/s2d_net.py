"""XFeat backbone rewritten for space-to-depth input: exact, same weights, no retraining.

Why: on the RK3588 NPU block1 (480x640 -> 120x160) takes 50% of the network time and the keypoint
branch (8x8 unshuffle at 480x640) another 24% (REPORT_rknn.md, 6.6). These layers have 1-8 channels,
but the NPU computes convolutions in blocks of 32 channels, so almost all of the work is padding.
Space-to-depth moves pixels into channels: the same data, 4x or 16x fewer pixels, channels that fill
the alignment.

Space-to-depth by f (== torch.pixel_unshuffle): a (C, H, W) tensor becomes (C*f*f, H/f, W/f), channel
c*f*f + i*f + j holds the pixels with row % f == i and col % f == j ("phase" (i, j)).

A 3x3 conv with stride s and padding 1 on a full-resolution tensor stored at factor f_in is computed
exactly by a 3x3 conv with stride 1 and padding 1 on the phase grid, with output at factor f_out = f_in / s:
output pixel Y = a*f_out + q reads input rows y = s*Y + dy - 1 = a*f_in + t, t = s*q + dy - 1,
i.e. phase-grid offset floor(t / f_in) in {-1, 0, 1} and input phase t mod f_in. Zero padding of the
original image is the zero padding of the phase grid, so borders are exact too.

Variants (factor = 2 or 4 is the CPU space-to-depth of the normalized frame):
  factor 4: input (1, 16, H/4, W/4); block1.0 16->64, block1.1 64->32, block1.2 32->32, block1.3 32->24
  factor 2: input (1, 4,  H/2, W/2); block1.0 4->16,  block1.1 16->8, block1.2 / block1.3 as in the original
  kp="conv": keypoint branch from the same input (2x2/s2 conv for factor 4, 4x4/s4 for factor 2,
             merged with the first 1x1 conv of keypoint_head, like the gray model does at factor 1)
  kp="cpu":  second input (1, 64, H/8, W/8) = space-to-depth by 8 done on the CPU; head starts with its 1x1 conv
Outputs are the same as deploy/export_onnx.py: feats (raw), keypoints (65 logits), heatmap.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def fold_bn(conv, bn):
    """Conv2d(bias=False) + BatchNorm2d(affine=False), eval mode -> (weight, bias)."""
    s = torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight.data / s.view(-1, 1, 1, 1)
    b = -bn.running_mean / s
    if conv.bias is not None:
        b = b + conv.bias.data / s
    return w, b


def phase_conv(w, b, stride, f_in):
    """3x3 conv (pad 1, `stride`) on a full-res tensor stored at factor f_in -> equivalent 3x3 conv
    (pad 1, stride 1) on the phase grid. Returns (weight, bias, f_out)."""
    co, ci, kh, kw = w.shape
    assert (kh, kw) == (3, 3) and f_in % stride == 0
    f_out = f_in // stride
    W = torch.zeros(co * f_out * f_out, ci * f_in * f_in, 3, 3, dtype=w.dtype)
    for qy in range(f_out):
        for qx in range(f_out):
            o = torch.arange(co) * f_out * f_out + qy * f_out + qx          # output channels of this phase
            for dy in range(3):
                ty = stride * qy + dy - 1
                ky, py = ty // f_in + 1, ty % f_in                          # phase-grid tap, input phase row
                for dx in range(3):
                    tx = stride * qx + dx - 1
                    kx, px = tx // f_in + 1, tx % f_in
                    i = torch.arange(ci) * f_in * f_in + py * f_in + px     # input channels of that phase
                    W[o[:, None], i[None, :], ky, kx] += w[:, :, dy, dx]
    B = b.repeat_interleave(f_out * f_out)
    return W, B, f_out


def block_conv(w, f):
    """Conv on a (C*f*f, H/f, W/f) phase tensor that equals: unshuffle the full-res 1-channel image
    by 8 and apply a 1x1 conv `w` (co, 64) -- i.e. kernel 8/f, stride 8/f. Requires C == 1."""
    k = 8 // f
    co = w.shape[0]
    W = torch.zeros(co, f * f, k, k, dtype=w.dtype)
    for u in range(k):
        for v in range(k):
            for i in range(f):
                for j in range(f):
                    W[:, i * f + j, u, v] = w[:, (u * f + i) * 8 + (v * f + j)]   # unfold order: row*8 + col
    return W, k


def conv_layer(w, b, stride=1, padding=1):
    c = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2], stride=stride, padding=padding, bias=True)
    c.weight.data, c.bias.data = w.clone(), b.clone()
    return c


class XFeatS2D(nn.Module):
    def __init__(self, net, height, width, factor=4, kp="conv"):
        super().__init__()
        assert factor in (2, 4) and kp in ("conv", "cpu")
        self.net, self.factor, self.kp = net, factor, kp
        self.size = (height // 8, width // 8)

        # block1 at the phase grid of `factor`; stops once the output is back at factor 1
        layers, f = [], factor
        for basic in net.block1:
            conv, bn = basic.layer[0], basic.layer[1]
            w, b = fold_bn(conv, bn)
            s = conv.stride[0]
            if f > 1:
                w, b, f = phase_conv(w, b, s, f)
                layers += [conv_layer(w, b), nn.ReLU()]
            else:
                layers += [conv_layer(w, b, stride=s), nn.ReLU()]
        assert f == 1
        self.block1 = nn.Sequential(*layers)

        # skip1 = AvgPool(4) + conv1x1(1 -> 24): from the phase grid
        sw, sb = net.skip1[1].weight.data, net.skip1[1].bias.data         # (24, 1, 1, 1), (24,)
        if factor == 4:      # mean of the 16 phases = average over each 4x4 block
            self.skip1 = conv_layer(sw.repeat(1, 16, 1, 1) / 16, sb, padding=0)
        else:                # factor 2: 2x2/s2 over 4 phases = 4x4 block
            self.skip1 = conv_layer(sw.repeat(1, 4, 2, 2) / 16, sb, stride=2, padding=0)

        # keypoint branch: first BasicLayer of keypoint_head (conv1x1 64->64 + BN + ReLU), then the rest
        kh = net.keypoint_head
        w0, b0 = fold_bn(kh[0].layer[0], kh[0].layer[1])                   # (64, 64, 1, 1)
        if kp == "conv":
            wk, k = block_conv(w0.reshape(64, 64), factor)
            self.kp0 = conv_layer(wk, b0, stride=k, padding=0)
        else:
            self.kp0 = conv_layer(w0, b0, padding=0)                        # input: CPU s2d by 8
        self.kp_rest = nn.Sequential(nn.ReLU(), *kh[1:])

    def forward(self, x, kp_in=None):
        n = self.net
        x2 = n.block2(self.block1(x) + self.skip1(x))
        x3 = n.block3(x2)
        x4 = n.block4(x3)
        x5 = n.block5(x4)
        x4 = F.interpolate(x4, self.size, mode="bilinear")
        x5 = F.interpolate(x5, self.size, mode="bilinear")
        feats = n.block_fusion(x3 + x4 + x5)
        heatmap = n.heatmap_head(feats)
        keypoints = self.kp_rest(self.kp0(x if self.kp == "conv" else kp_in))
        return feats, keypoints, heatmap
