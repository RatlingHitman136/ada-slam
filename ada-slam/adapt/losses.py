"""Depth and pose losses.

Both are scale-invariant by construction, and in both the scale estimate is deliberately NOT
detached. That is the single most fragile property in this package - see the comments below.
"""
import torch.nn.functional as F


def median_scale(pred, gt, mask):
    """Median ratio. Deliberately NOT detached.

    Detaching makes the loss only *look* scale-invariant: the optimiser then sees a gradient that
    rewards shrinking the prediction, even though rescaling is a no-op once the scale is recomputed
    on the next forward pass. Letting the gradient flow through the ratio makes the loss genuinely
    invariant, so it can only push shape, never overall magnitude.
    """
    return gt[mask].median() / pred[mask].median().clamp(min=1e-6)


def depth_loss(pred_depth, gt_depth, mask, cfg, scale=None):
    """Returns (loss, depth-space scale). The returned scale is always expressed in DEPTH terms,
    even when the loss is computed in disparity, so callers can compare it against the pose scale."""
    if mask.sum() < cfg.min_mask_pixels:
        return pred_depth.sum() * 0.0, None
    p, g = pred_depth.clamp(min=1e-3), gt_depth.clamp(min=1e-3)
    inv = cfg.depth_space == 'disparity'
    if inv:
        p, g = 1.0 / p, 1.0 / g
    # a scale supplied by the caller (coupled_scale) is a depth scale; in disparity space the
    # equivalent multiplier is its reciprocal
    s = median_scale(p, g, mask) if scale is None else (1.0 / scale if inv else scale)
    loss = (g[mask] - s * p[mask]).abs().mean()
    return loss, (1.0 / s if inv else s)


def pose_loss(pred_enc, gt_enc):
    """Translation (independently norm'd) + quaternion, over the non-reference frames."""
    if pred_enc.shape[0] < 2:
        z = pred_enc.sum() * 0.0
        return z, z, None
    tp, tg = pred_enc[1:, :3], gt_enc[1:, :3]
    # same reasoning as median_scale: normalising by a *detached* predicted norm lets the
    # translations collapse toward zero at no loss cost. Keep the gradient in the normaliser.
    np_, ng = tp.norm(dim=-1).mean().clamp(min=1e-6), tg.norm(dim=-1).mean().clamp(min=1e-6)
    l_t = F.huber_loss(tp / np_, tg / ng)

    qp = F.normalize(pred_enc[1:, 3:7], dim=-1)
    qg = F.normalize(gt_enc[1:, 3:7], dim=-1)
    l_r = (1.0 - (qp * qg).sum(-1).abs()).mean()      # abs handles quaternion sign ambiguity
    return l_t, l_r, (ng / np_).detach()
