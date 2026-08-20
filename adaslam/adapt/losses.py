"""Depth and pose losses. In both, the scale estimate is deliberately NOT detached (9.3)."""
import torch
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


# ---------------------------------------------------------------- the JDSA-matched loss
# One (4, H, W) weight stack per (size, device, dtype). The grid is pure geometry - it depends on
# nothing but the image shape - so building it once per resolution costs nothing and keeps it off
# the per-sample path.
_GRID_CACHE = {}


def grid_weights_t(hw, device, dtype):
    """Bilinear weights of a 2x2 scale grid over an (H, W) image -> (4, H, W), on `device`.

    The torch mirror of priortest/metrics.py:grid_weights, which in turn reproduces
    geom/ba.py:get_prior_depth_aligned's meshgrid exactly. Same corner ORDER as that function
    ([y0x0, y0x1, y1x0, y1x1]) - the order differs from the CUDA kernel's, which is harmless
    because the fit and the evaluation below share this one basis, but it means the four numbers
    are not directly comparable with video.dscales.
    """
    key = (tuple(hw), str(device), dtype)
    if key not in _GRID_CACHE:
        h, w = hw
        fy = torch.linspace(0, 1 - 1e-6, h, device=device, dtype=dtype)[:, None]
        fx = torch.linspace(0, 1 - 1e-6, w, device=device, dtype=dtype)[None, :]
        _GRID_CACHE[key] = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy),
                                        (1 - fx) * fy, fx * fy])
    return _GRID_CACHE[key]


def _ba_lattice(pred, gt, mask, stream_hw):
    """The three maps on the (H/8, W/8) grid BA actually reads - depth_video.py:73's `[3::8, 3::8]`.

    Reproduces the deployment chain: the prior leaves VGGT at vggt_hw, VggtPrior interpolates it to
    the stream size, and depth_video.py point-subsamples THAT by 8 with an offset of 3. Bilinear for
    the prediction (as VggtPrior does - bicubic can overshoot to negative depth), nearest for the
    target and the mask, so a hole is never averaged into its neighbours.
    """
    p = F.interpolate(pred[None, None], stream_hw, mode='bilinear', align_corners=False)[0, 0]
    g = F.interpolate(gt[None, None], stream_hw, mode='nearest')[0, 0]
    m = F.interpolate(mask[None, None].float(), stream_hw, mode='nearest')[0, 0] > 0.5
    return p[3::8, 3::8], g[3::8, 3::8], m[3::8, 3::8]


def jdsa_loss(pred_depth, gt_depth, mask, cfg, stream_hw=None):
    """The residual JDSA's 2x2 disparity grid CANNOT absorb, and the grid it fitted.

    depth_loss aligns with one median scalar in DEPTH; the solver aligns with a 4-DOF bilinear field
    in DISPARITY, refit every BA iteration (geom/ba.py:161-196). Every error the grid can absorb is
    free at deployment, so training against the median-aligned error both spends capacity on what
    the solver discards and never sees the part it keeps. This fits the SAME family the solver fits
    and returns what survives it:

        q = 1/pred   d = 1/gt              A[p,k] = q_p * w_k(p)
        s = argmin |d - A s|^2             L = mean |d - A s|      (or squared)

    `s` is NOT detached. With jdsa_norm='l2' that makes no difference to the gradient - the fit
    minimises the same objective the loss measures, so dL/ds = 0 at the optimum (envelope theorem)
    and the extra path contributes zero. With 'l1' the two differ and the attached form is the
    correct one.

    NOTE the invariance: L(c*pred) = L(pred) EXACTLY, because A -> cA and s -> s/c. This loss is
    blind to any bilinear multiplicative field in disparity - deliberately, because that is what the
    solver is blind to. It therefore does NOT restore a gradient on per-frame scale, and should not:
    the evidence that per-frame scale consistency matters did not survive (corr(CV_depth, ATE) =
    +0.02 over 17 arms, and E3's placebo scored the same as lambda=0).

    Returns (loss, s) - `s` is the fitted 4-vector, for logging.
    """
    if mask.sum() < cfg.min_mask_pixels:
        return pred_depth.sum() * 0.0, None
    if cfg.jdsa_lattice == 'ba':
        if stream_hw is None:
            raise ValueError("jdsa_lattice='ba' needs the tracking resolution; the caller must "
                             'pass stream_hw (SceneData.stream_hw / LiveSampler.stream_hw)')
        pred_depth, gt_depth, mask = _ba_lattice(pred_depth, gt_depth, mask, tuple(stream_hw))
        if mask.sum() < 8:                 # 4 unknowns; below this the grid is not determined
            return pred_depth.sum() * 0.0, None

    w4 = grid_weights_t(pred_depth.shape[-2:], pred_depth.device, pred_depth.dtype)
    q = 1.0 / pred_depth.clamp(min=1e-3)
    d = 1.0 / gt_depth.clamp(min=1e-3)

    A = (w4 * q[None])[:, mask].T.double()          # (N, 4)
    y = d[mask].double()                            # (N,)
    AtA, Aty = A.T @ A, A.T @ y
    # RELATIVE ridge: A scales with the prediction's units, so a fixed absolute damping would mean
    # something different on every scene. Scaling by the mean diagonal makes jdsa_ridge unit-free.
    eye = torch.eye(4, device=A.device, dtype=A.dtype)
    s = torch.linalg.solve(AtA + cfg.jdsa_ridge * (AtA.diagonal().mean() + 1e-12) * eye, Aty)

    r = (y - A @ s).to(pred_depth.dtype)
    return (r.abs().mean() if cfg.jdsa_norm == 'l1' else r.pow(2).mean()), s.detach()
