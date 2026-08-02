"""The depth-source accuracy table - the first number to read after an extract run (10.2).

The GAP between the per-frame and global columns is the diagnostic this track targets: a source
that gets worse under one global scale is cross-frame inconsistent.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def align_scale(pred, gt):
    """Median-ratio scale on flat arrays - SLAM units are arbitrary, GT is metric."""
    return float(np.median(gt) / np.median(pred))


def l1_per_frame(pairs):
    """One scale fitted per keyframe."""
    return np.mean([np.abs(g - align_scale(p, g) * p).mean() for g, p in pairs])


def l1_global(pairs):
    """One scale for the whole sequence, averaged as l1_per_frame - so only the fit differs."""
    s = align_scale(np.concatenate([p for _, p in pairs]),
                    np.concatenate([g for g, _ in pairs]))
    return np.mean([np.abs(g - s * p).mean() for g, p in pairs])


def report_accuracy(x, cfg):
    """Print the table. No-op without cfg.gt_depths. Reads nothing off disk but the GT depth."""
    from geom.ba import get_prior_depth_aligned
    if cfg.gt_depths is None:
        return None

    K, H, W = x.shape
    gtfiles = sorted(os.listdir(cfg.gt_depths))

    # JDSA-aligned prior, via geom/ba.py's own scale field; 1/8-res, upsampled to compare
    prior_al, _ = get_prior_depth_aligned(torch.from_numpy(x.npz['disps_prior']).cuda(),
                                          torch.from_numpy(x.npz['dscales']).cuda())
    prior_al = F.interpolate(prior_al[:, None], size=(H, W), mode='bilinear',
                             align_corners=False)[:, 0]
    prior_depth = (1.0 / prior_al.clamp(min=1e-6)).cpu().numpy()

    pairs = {k: [] for k in ('slam', 'prior')}
    for i in range(K):
        idx = int(x.tstamp[i])
        gt = cv2.imread(os.path.join(cfg.gt_depths, gtfiles[idx]),
                        cv2.IMREAD_ANYDEPTH) / cfg.depth_png_scale
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        valid = (gt > 0) & x.mask[i]
        if valid.sum() == 0:
            continue

        for name, pred in (('slam', x.depth[i]), ('prior', prior_depth[i])):
            v = valid & (pred > 0)
            if v.sum() > 0:
                pairs[name].append((gt[v], pred[v]))

    print(f'\nscale-aligned depth L1 (m) vs GT, masked, over {len(pairs["slam"])} keyframes')
    print(f'  {"source":<34} {"per-frame":>10} {"global":>10}')
    print(f'  {"-" * 56}')
    for name, label in (('slam', 'SLAM depth (1/disps_up)'),
                        ('prior', 'JDSA-aligned Omnidata prior')):
        if pairs[name]:
            print(f'  {label:<34} {l1_per_frame(pairs[name]):>10.4f} '
                  f'{l1_global(pairs[name]):>10.4f}')
        else:
            print(f'  {label:<34} {"n/a":>10} {"n/a":>10}')
    return pairs
