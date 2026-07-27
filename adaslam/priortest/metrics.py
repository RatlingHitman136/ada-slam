"""Scoring a depth prior: three alignments, and what the differences between them mean.

HI-SLAM2 never consumes the prior raw. JDSA fits a 2x2 bilinear scale grid per keyframe and
MULTIPLIES (geom/ba.py:get_prior_depth_aligned is `depth_prior * mscales_bi` - scale only, no
shift). So one prediction yields three very different errors depending how much of that freedom you
grant, and the differences are the diagnosis:

    per-frame scale   median ratio, one per frame     pure shape error, scale-free
    2x2 JDSA grid     least squares, one per frame    the error the pipeline CANNOT absorb
    global scale      median ratio, one per sequence  shape error PLUS cross-frame scale drift

    consistency index = L1_global / L1_per-frame      the cross-frame drift, as a single number
    scale CV          = std/mean of the per-frame scales, the same drift measured directly

The consistency index is ARCHITECTURE.md 10.2's "first number to check" generalised to every arm:
Omnidata on Replica room0 reads 0.0735 / 0.2078, a 2.8x blow-up under one global scale, and THAT
blow-up is what this research track exists to fix. An adapter trained on SLAM depth should drive it
toward 1.0.

Scale+shift alignment is deliberately NOT offered. It is the MiDaS convention and it would flatter
Omnidata with a degree of freedom the pipeline can never exploit.

CAVEAT worth repeating wherever these numbers are read: the JDSA-grid residual is a LOWER bound on
what JDSA actually leaves behind. The real solver fits that grid jointly with poses and depths
against photometric residuals, not in closed form against ground truth, so it does no better than
this and usually worse.

Every per-frame value here is independent of the seen/unseen split - the global scale is fitted over
ALL frames and only then are the errors split, the same choice end2end/metrics.py documents
("fitting per half would hide exactly the drift we are looking for"). That is what lets aggregate()
re-split a cached frames.csv for free.
"""
import numpy as np

from ..extract.accuracy import align_scale

# One row of frames.csv. Order matters: it is the CSV's column order.
FRAME_FIELDS = ('idx', 'n_valid', 'scale', 'l1_perframe', 'l1_jdsa', 'l1_global',
                'absrel', 'delta125')


def grid_weights(hw):
    """Bilinear weights of a 2x2 scale grid over an (H, W) image -> (4, H, W).

    Reproduces geom/ba.py:get_prior_depth_aligned's meshgrid exactly: linspace(0, 1-1e-6) along
    each axis of a 2x2 grid, so the four corners are the grid nodes and the interior is bilinear.
    """
    h, w = hw
    fy = np.linspace(0, 1 - 1e-6, h, dtype=np.float64)[:, None]     # rows
    fx = np.linspace(0, 1 - 1e-6, w, dtype=np.float64)[None, :]     # cols
    return np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])


def fit_jdsa_grid(pred, gt, w4):
    """Least-squares 2x2 scale grid, fitted in DISPARITY - where JDSA fits it.

    depth_video.py:70-73 stores the prior as 1/depth and JDSA scales that, so the family the
    pipeline can absorb is `disp * S` with S bilinear. In depth that is `depth / S`, which is NOT
    bilinear in depth - fitting in depth space would be a different family.

    `pred`, `gt`, `w4` are already flattened to the valid pixels. Returns the aligned DEPTH.
    """
    dp, dg = 1.0 / pred, 1.0 / gt
    A = w4 * dp                                    # (4, N): disp * each corner's weight
    s, *_ = np.linalg.lstsq(A.T, dg, rcond=None)
    disp = np.maximum(A.T @ s, 1e-9)
    return 1.0 / disp


def score_frame(idx, pred, gt, cfg, rng):
    """One frame -> a dict of FRAME_FIELDS with l1_global still missing, plus the kept samples.

    l1_global cannot be filled in yet: its scale is fitted across every frame, so it arrives in
    finish_global() once the sequence is done.
    """
    valid = (gt >= cfg.eval_min_depth) & (gt <= cfg.eval_max_depth) & (pred > 0) & np.isfinite(pred)
    n_valid = int(valid.sum())
    if n_valid < 16:                               # too few pixels for a median to mean anything
        return None, None

    w4 = grid_weights(gt.shape)[:, valid]
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)

    if cfg.eval_samples_per_frame and n_valid > cfg.eval_samples_per_frame:
        keep = rng.choice(n_valid, cfg.eval_samples_per_frame, replace=False)
        p, g, w4 = p[keep], g[keep], w4[:, keep]

    s = align_scale(p, g)                          # the same median ratio export.txt uses
    aligned = s * p
    row = {'idx': idx, 'n_valid': n_valid, 'scale': s,
           'l1_perframe': float(np.abs(g - aligned).mean()),
           'l1_jdsa': float(np.abs(g - fit_jdsa_grid(p, g, w4)).mean()),
           'l1_global': np.nan,                    # filled by finish_global
           'absrel': float((np.abs(g - aligned) / g).mean()),
           'delta125': float((np.maximum(aligned / g, g / aligned) < 1.25).mean())}
    return row, (p, g)


def finish_global(rows, samples):
    """Fit ONE scale over every frame's samples, then fill each row's l1_global.

    Fitted over the whole sequence and never per population - a scale refitted per half would
    absorb exactly the drift the consistency index exists to expose.
    """
    if not rows:
        return rows, float('nan')
    s = align_scale(np.concatenate([p for p, _ in samples]),
                    np.concatenate([g for _, g in samples]))
    for row, (p, g) in zip(rows, samples):
        row['l1_global'] = float(np.abs(g - s * p).mean())
    return rows, s


def aggregate(rows, split_at):
    """{'all': block, 'seen': block, 'unseen': block} - the ONLY place the split is used.

    Every input value is split-independent, so this is pure arithmetic over subsets: changing which
    adapter defines the boundary re-aggregates a cached frames.csv in milliseconds instead of
    re-running inference.
    """
    def block(sel):
        sub = [r for r in rows if sel(r['idx'])]
        if not sub:
            return None
        out = {'n': len(sub)}
        for key in ('l1_perframe', 'l1_jdsa', 'l1_global', 'absrel', 'delta125'):
            out[key] = float(np.mean([r[key] for r in sub]))
        scales = np.array([r['scale'] for r in sub], dtype=np.float64)
        out['scale_mean'] = float(scales.mean())
        out['scale_cv'] = float(scales.std() / scales.mean()) if scales.mean() else float('nan')
        # The headline: how much worse one scale for the whole sequence is than one per frame.
        # Floored rather than merely guarded against zero - a prior whose per-frame error is a
        # micrometre has no shape error for the drift to be relative TO, and the ratio there is an
        # artefact of float noise, not a measurement (a synthetic perfect-shape input prints ~1e15).
        out['consistency_index'] = (out['l1_global'] / out['l1_perframe']
                                    if out['l1_perframe'] > 1e-6 else float('nan'))
        return out

    res = {'all': block(lambda i: True)}
    if split_at is not None:
        res['seen'] = block(lambda i: i < split_at)
        res['unseen'] = block(lambda i: i >= split_at)
    return res
