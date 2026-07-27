"""The depth-source accuracy table - the first number to read after an extract run.

Three candidate supervision sources scored against GT, in two columns: one scale fitted per
keyframe, and one scale for the whole sequence. The GAP between those columns is the diagnostic
this whole research track targets - a source that gets worse under a single global scale is
cross-frame inconsistent. On Replica room0 the Omnidata row reads 0.0735 / 0.2078, a 2.8x blow-up;
SLAM and Gaussian-rendered depth both improve instead (ARCHITECTURE.md 10.2).
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
    """One scale for the whole sequence, then averaged the same way as l1_per_frame.

    Averaging per frame in both keeps the only difference the scale fit itself, so the gap between
    the two columns isolates cross-frame scale drift rather than frame weighting.
    """
    s = align_scale(np.concatenate([p for _, p in pairs]),
                    np.concatenate([g for g, _ in pairs]))
    return np.mean([np.abs(g - s * p).mean() for g, p in pairs])


def report_accuracy(run_dir, x, cfg):
    """Print the table. No-op when cfg.gt_depths is None - there is nothing to score against.

    Takes the HI-SLAM2 run directory (extract/<exp>/full), not the experiment above it: the
    Gaussian-rendered row is read straight out of that run's renders/.
    """
    from geom.ba import get_prior_depth_aligned
    if cfg.gt_depths is None:
        return None

    K, H, W = x.shape
    gtfiles = sorted(os.listdir(cfg.gt_depths))

    # JDSA-aligned Omnidata prior, reusing geom/ba.py's bilinear scale field.
    # Inherently 1/8-res in the pipeline; bilinearly upsampled here so all three are comparable.
    prior_al, _ = get_prior_depth_aligned(torch.from_numpy(x.npz['disps_prior']).cuda(),
                                          torch.from_numpy(x.npz['dscales']).cuda())
    prior_al = F.interpolate(prior_al[:, None], size=(H, W), mode='bilinear',
                             align_corners=False)[:, 0]
    prior_depth = (1.0 / prior_al.clamp(min=1e-6)).cpu().numpy()

    pairs = {k: [] for k in ('slam', 'rendered', 'prior')}
    for i in range(K):
        idx = int(x.tstamp[i])
        gt = cv2.imread(os.path.join(cfg.gt_depths, gtfiles[idx]),
                        cv2.IMREAD_ANYDEPTH) / cfg.depth_png_scale
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        valid = (gt > 0) & x.mask[i]
        if valid.sum() == 0:
            continue

        srcs = [('slam', x.depth[i]), ('prior', prior_depth[i])]
        rf = f'{run_dir}/renders/depth_after_opt/{idx:06d}.png'
        if os.path.exists(rf):
            srcs.append(('rendered',
                         cv2.imread(rf, cv2.IMREAD_ANYDEPTH) / cfg.depth_png_scale))

        for name, pred in srcs:
            v = valid & (pred > 0)
            if v.sum() > 0:
                pairs[name].append((gt[v], pred[v]))

    print(f'\nscale-aligned depth L1 (m) vs GT, masked, over {len(pairs["slam"])} keyframes')
    print(f'  {"source":<34} {"per-frame":>10} {"global":>10}')
    print(f'  {"-" * 56}')
    for name, label in (('slam', 'SLAM depth (1/disps_up)'),
                        ('rendered', f'Gaussian-rendered ({len(pairs["rendered"])} kf)'),
                        ('prior', 'JDSA-aligned Omnidata prior')):
        if pairs[name]:
            print(f'  {label:<34} {l1_per_frame(pairs[name]):>10.4f} '
                  f'{l1_global(pairs[name]):>10.4f}')
        else:
            print(f'  {label:<34} {"n/a":>10} {"n/a":>10}')
    return pairs
