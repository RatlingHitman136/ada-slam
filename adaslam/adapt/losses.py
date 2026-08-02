"""Depth and pose losses. In both, the scale estimate is deliberately NOT detached (9.3)."""
import torch.nn.functional as F


def median_scale(pred, gt, mask):
    """Median ratio, NOT detached: detaching rewards a shrinking prediction and collapses it."""
    return gt[mask].median() / pred[mask].median().clamp(min=1e-6)


def depth_loss(pred_depth, gt_depth, mask, cfg, scale=None):
    """Masked, scale-aligned L1 in DEPTH space, and the scale it used.

    Depth, not disparity: VGGT's head emits depth and HI-SLAM2 inverts it itself, unconditionally.
    A `scale` from pose_loss is a depth scale, so it applies directly.
    """
    if mask.sum() < cfg.min_mask_pixels:
        return pred_depth.sum() * 0.0, None
    p, g = pred_depth.clamp(min=1e-3), gt_depth.clamp(min=1e-3)
    s = median_scale(p, g, mask) if scale is None else scale
    return (g[mask] - s * p[mask]).abs().mean(), s


def pose_loss(pred_enc, gt_enc):
    """Translation (independently norm'd) + quaternion, over the non-reference frames."""
    if pred_enc.shape[0] < 2:
        z = pred_enc.sum() * 0.0
        return z, z, None
    tp, tg = pred_enc[1:, :3], gt_enc[1:, :3]
    # as in median_scale: a DETACHED normaliser lets the translations collapse at no loss cost
    np_, ng = tp.norm(dim=-1).mean().clamp(min=1e-6), tg.norm(dim=-1).mean().clamp(min=1e-6)
    l_t = F.huber_loss(tp / np_, tg / ng)

    qp = F.normalize(pred_enc[1:, 3:7], dim=-1)
    qg = F.normalize(gt_enc[1:, 3:7], dim=-1)
    l_r = (1.0 - (qp * qg).sum(-1).abs()).mean()      # abs handles quaternion sign ambiguity
    return l_t, l_r, (ng / np_).detach()
