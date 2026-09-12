"""Depth and pose losses. In both, the scale estimate is deliberately NOT detached (9.3)."""
import torch.nn.functional as F


def median_scale(pred, gt, mask):
    """Median ratio, NOT detached: detaching rewards a shrinking prediction and collapses it."""
    return gt[mask].median() / pred[mask].median().clamp(min=1e-6)


def mean_scale(pred, gt, mask):
    """Mean ratio, NOT detached - the dense-gradient twin of median_scale.

    Same quantity in intent, different estimator, and the difference is the gradient's SUPPORT.
    torch.median routes its gradient to the single selected element, so a term built on
    median_scale reaches the model through ONE pixel per sample - fine for depth_loss, which uses
    the scale only to align a residual measured over every pixel, and fatal for a term whose
    ENTIRE gradient is the scale, as online/config.py:lambda_cons is. The mean ratio spreads it
    over every masked pixel.

    Undetached for the same reason median_scale is: detaching rewards a shrinking prediction.
    """
    return gt[mask].mean() / pred[mask].mean().clamp(min=1e-6)


def depth_loss(pred_depth, gt_depth, mask, cfg, scale=None):
    """Masked, scale-aligned L1 in DEPTH space, and the scale it used.

    Depth, not disparity: VGGT's head emits depth and HI-SLAM2 inverts it itself, unconditionally.
    A `scale` from pose_loss is a depth scale, so it applies directly.

    THREE things `scale` can be, and they are not variations on one idea:
      None   fit median_scale here, per sample. The loss is then EXACTLY scale-invariant - a
             global rescale of `pred_depth` leaves it bit-identical - so it carries no gradient
             about absolute scale at all. Every arm before normalize_target ran this way.
      a tensor  the pose head's gauge (coupled_scale) or a batch-pooled one.
      1.0    the target already carries its own gauge (config.py:normalize_target), so re-fitting
             one here would divide out the very quantity being supervised. This is the only
             setting under which the depth term can move the prediction's absolute scale.
    """
    if mask.sum() < cfg.min_mask_pixels:
        return pred_depth.sum() * 0.0, None
    p, g = pred_depth.clamp(min=1e-3), gt_depth.clamp(min=1e-3)
    s = median_scale(p, g, mask) if scale is None else scale
    return (g[mask] - s * p[mask]).abs().mean(), s


def relative_loss(loss, gt_depth, mask):
    """`loss` as a fraction of the frame's MEDIAN TARGET DEPTH - a unit-free fit measure.

    depth_loss is in the tracker's own depth units, and those shrink along a run as the SLAM
    solution's scale drifts: measured on rellis_00000, one run's first-step loss falls 0.029 ->
    0.007 across units 0..209 with no change in how well it fits (the same loss expressed in GT
    metres is flat). A threshold on the RAW loss is therefore an implicit schedule over the
    sequence rather than a statement about fit - it stops adapting late and calls that a decision.

    Dividing by the same frame's median target depth cancels the unit and leaves "mean absolute
    error as a fraction of how far the scene is". Over the same run that reads 0.0195 -> 0.0262,
    i.e. flat to ~1.3x, which is what makes one fixed threshold meaningful end to end.

    None when there is nothing to measure, so callers can distinguish "fits well" from "no data".
    """
    if loss is None or not mask.any():
        return None
    return float(loss) / float(gt_depth[mask].median().clamp(min=1e-6))


def pose_loss(pred_enc, gt_enc, absolute=False):
    """Translation + quaternion over the non-reference frames. `absolute` picks the gauge.

    absolute=False (the default, every arm before gauge_pose): each translation set is divided by
    its OWN mean norm before the comparison, so the term is scale-free - it constrains direction
    and rotation and says nothing about how far the camera moved.

    absolute=True is VGGT's own camera translation term. It presumes the caller has already put
    gt's translations in the same gauge as the depth target (config.py:gauge_pose), which is what
    VGGT's training does: normalize_camera_extrinsics_and_points_batch divides camera translations
    and depth maps by ONE avg_scale, and camera_loss_single then compares translations directly -
    `(pred[..., :3] - gt[..., :3]).abs()` - with no re-normalisation anywhere. Scale agreement
    between the two heads is a property of the data preparation, not a term in the loss, which is
    why VGGT needs no coupled_scale.

    L1 RATHER THAN HUBER IS PART OF THE SWITCH, not an incidental second change. At these
    magnitudes - translations land near 0.4-0.7 once divided by the gauge - huber_loss's delta=1
    transition never fires, so it would be silently L2, and VGGT's own note records that L1 was
    more stable than smooth-L1 and L2 for exactly this term.

    pose_scale is returned either way. Under gauge_pose it is no longer a gauge anyone consumes
    (coupled_scale is refused alongside normalize_target) but it stays a useful diagnostic: with
    depth_scale pinned to 1.0, the logged scale_ratio becomes 1/pose_scale, i.e. how far the pose
    head's translation scale sits from the gauge the target imposes.
    """
    if pred_enc.shape[0] < 2:
        z = pred_enc.sum() * 0.0
        return z, z, None
    tp, tg = pred_enc[1:, :3], gt_enc[1:, :3]
    # as in median_scale: a DETACHED normaliser lets the translations collapse at no loss cost
    np_, ng = tp.norm(dim=-1).mean().clamp(min=1e-6), tg.norm(dim=-1).mean().clamp(min=1e-6)
    l_t = (tp - tg).abs().mean() if absolute else F.huber_loss(tp / np_, tg / ng)

    qp = F.normalize(pred_enc[1:, 3:7], dim=-1)
    qg = F.normalize(gt_enc[1:, 3:7], dim=-1)
    l_r = (1.0 - (qp * qg).sum(-1).abs()).mean()      # abs handles quaternion sign ambiguity
    return l_t, l_r, (ng / np_).detach()

