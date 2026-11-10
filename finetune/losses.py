import torch
import torch.nn.functional as F


def dual_softmax_loss(X, Y, temp=0.2):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp(conf_matrix12).max(dim=-1)[0]
        conf21 = torch.exp(conf_matrix21).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device=X.device)

    loss = F.nll_loss(conf_matrix12, target) + \
           F.nll_loss(conf_matrix21, target)

    return loss, conf


def keypoint_loss(heatmap, target):
    # Compute L1 loss
    L1_loss = F.l1_loss(heatmap, target)
    return L1_loss * 3.0


def desc_distill_loss(fs, ft):
    """Dense descriptor maps B×64×H×W: (cosine term, MSE relative to the teacher's energy).
    Cosine fixes the direction used by sparse matching, MSE the raw scale that
    XFeat* and fine_matcher read."""
    cos = (1.0 - F.cosine_similarity(fs, ft, dim=1)).mean()
    mse = F.mse_loss(fs, ft) / ft.pow(2).mean()
    return cos, mse


def at_points(feats, b, pts):
    """N×C descriptors of batch item b at coarse (x, y) points."""
    return feats[b, :, pts[:, 1].long(), pts[:, 0].long()].permute(1, 0)


def distill_losses(s1, s2, t1, t2, positives):
    """s1/s2, t1/t2: (feats, keypoints, heatmap) of student and teacher on both views.
    positives: per-item N×4 coarse correspondences. Returns a dict of unweighted terms."""
    cos1, mse1 = desc_distill_loss(s1[0], t1[0])
    cos2, mse2 = desc_distill_loss(s2[0], t2[0])
    rel = (keypoint_loss(s1[2], t1[2]) + keypoint_loss(s2[2], t2[2])) / 2

    gt, cross = [], []
    for b, pos in enumerate(positives):
        pts1, pts2 = pos[:, :2], pos[:, 2:]
        ms1, ms2 = at_points(s1[0], b, pts1), at_points(s2[0], b, pts2)
        mt1, mt2 = at_points(t1[0], b, pts1), at_points(t2[0], b, pts2)
        gt.append(dual_softmax_loss(ms1, ms2)[0])
        cross.append((dual_softmax_loss(ms1, mt2)[0] + dual_softmax_loss(mt1, ms2)[0]) / 2)

    return {"cos": (cos1 + cos2) / 2, "mse": (mse1 + mse2) / 2, "rel": rel,
            "cross": torch.stack(cross).mean(), "gt": torch.stack(gt).mean()}
