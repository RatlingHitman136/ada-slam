"""Scoring a depth prior: three alignments, whose differences are the diagnosis (9.2.2).

    per-frame scale   median ratio, one per frame     pure shape error, scale-free
    2x2 JDSA grid     least squares, one per frame    the error the pipeline CANNOT absorb
    global scale      median ratio, one per sequence  shape error PLUS cross-frame scale drift

    consistency index = L1_global / L1_per-frame; scale CV = the same drift, measured directly

Scale+shift is deliberately NOT offered: it is the MiDaS convention and would flatter Omnidata
with a freedom the pipeline cannot exploit. The JDSA-grid row is a LOWER bound on what JDSA leaves
behind - the real solver fits that grid against photometric residuals, not against GT.

Every per-frame value is split-independent, which is what lets aggregate() re-split for free.
"""
import numpy as np

from ..extract.accuracy import align_scale

# One row of frames.csv. Order matters: it is the CSV's column order.
FRAME_FIELDS = ('idx', 'n_valid', 'scale', 'l1_perframe', 'l1_jdsa', 'l1_global',
                'absrel', 'delta125')


def grid_weights(hw):
    """Bilinear weights of a 2x2 scale grid over an (H, W) image -> (4, H, W).

    Reproduces geom/ba.py:get_prior_depth_aligned's meshgrid exactly.
    """
    h, w = hw
    fy = np.linspace(0, 1 - 1e-6, h, dtype=np.float64)[:, None]     # rows
    fx = np.linspace(0, 1 - 1e-6, w, dtype=np.float64)[None, :]     # cols
    return np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])


def fit_jdsa_grid(pred, gt, w4):
    """Least-squares 2x2 scale grid, fitted in DISPARITY - where JDSA fits it.

    The pipeline absorbs `disp * S` with S bilinear; in depth that is `depth / S`, a different
    family. Inputs are flattened to the valid pixels; returns the aligned DEPTH.
    """
    dp, dg = 1.0 / pred, 1.0 / gt
    A = w4 * dp                                    # (4, N): disp * each corner's weight
    s, *_ = np.linalg.lstsq(A.T, dg, rcond=None)
    disp = np.maximum(A.T @ s, 1e-9)
    return 1.0 / disp


def score_frame(idx, pred, gt, cfg, rng):
    """One frame -> a FRAME_FIELDS dict, l1_global left to finish_global, plus the kept samples."""
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

    Never per population: a scale refitted per half would absorb the very drift this exposes.
    """
    if not rows:
        return rows, float('nan')
    s = align_scale(np.concatenate([p for p, _ in samples]),
                    np.concatenate([g for _, g in samples]))
    for row, (p, g) in zip(rows, samples):
        row['l1_global'] = float(np.abs(g - s * p).mean())
    return rows, s


def aggregate(rows, split_at):
    """{'all', 'seen', 'unseen'} blocks - the ONLY place the split is used, pure arithmetic."""
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
        # the headline. Floored, not just zero-guarded: with no shape error to be relative TO the
        # ratio is float noise, not a measurement (a perfect-shape input prints ~1e15).
        out['consistency_index'] = (out['l1_global'] / out['l1_perframe']
                                    if out['l1_perframe'] > 1e-6 else float('nan'))
        return out

    res = {'all': block(lambda i: True)}
    if split_at is not None:
        res['seen'] = block(lambda i: i < split_at)
        res['unseen'] = block(lambda i: i >= split_at)
    return res
