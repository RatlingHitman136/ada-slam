"""Export the depth produced by `demo.py --dump_slam_depth` into training-ready files.

Reads slam_depth.npz (written in Hi2.terminate() right after global BA) and produces per-keyframe
depth / mask / image files plus a TUM pose list. With --gtdepthdir it also reports scale-aligned
depth L1 for the SLAM depth, the Gaussian-rendered depth and the JDSA-aligned Omnidata prior, so
the three can be compared as supervision targets.

`--depth_source` picks which of them is exported:

    rendered (default)  renders/depth_after_opt/, the Gaussian map's expected depth
    slam                1/disps_up straight from the npz

Rendered wins on measured accuracy (0.0133 vs 0.0324 m global-scale L1 on Replica room0) and on
consistency: it is rendered from the post-refinement trajectory that traj_full.txt also holds,
while the npz is deliberately dumped *before* gs.finalize() overwrites video.poses (hi2.py:155).
Both write float32 .npy, so only the directory name differs downstream.

Depths stay in raw SLAM units (mean depth ~= 2.0, set by video.normalize() with
scale_multiplier: 2.0). They are not metric; the comparison below fits a global scale first.
"""
import os    # nopep8
import sys   # nopep8
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hislam2'))   # nopep8
import argparse

import cv2
import lietorch
import numpy as np
import torch
import torch.nn.functional as F
import droid_backends

from lietorch import SE3
from geom.ba import get_prior_depth_aligned


def confidence_mask(poses, disps, intrinsics_full, filter_thresh, min_count):
    """Multi-view consistency mask, following util/droid_visualization.py:104-110.

    droid_backends.depth_filter counts, per pixel, how many of 6 temporal neighbours agree on the
    reprojected disparity. The kernel bounds-checks against disps.size(0), so the arrays must be
    sliced to the real keyframe count - otherwise trailing keyframes match unused buffer slots
    that still hold the initial 1.0.
    """
    K = disps.shape[0]
    ix = torch.arange(K, device='cuda', dtype=torch.long)
    thresh = filter_thresh * torch.ones(K, device='cuda', dtype=torch.float)
    count = droid_backends.depth_filter(poses, disps, intrinsics_full / 8.0, ix, thresh)
    return (count >= min_count) & (disps > .5 * disps.mean(dim=[1, 2], keepdim=True))


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
    s = align_scale(np.concatenate([p for _, p in pairs]), np.concatenate([g for g, _ in pairs]))
    return np.mean([np.abs(g - s * p).mean() for g, p in pairs])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str, required=True, help="run output dir holding slam_depth.npz")
    parser.add_argument("--gtdepthdir", type=str, default=None, help="optional GT depths, 16-bit scaled by 6553.5")
    parser.add_argument("--filter_thresh", type=float, default=0.005, help="disparity agreement threshold")
    parser.add_argument("--min_count", type=int, default=2, help="min agreeing neighbours out of 6")
    parser.add_argument("--no_export", action="store_true", help="only report metrics, write nothing")
    parser.add_argument("--depth_source", choices=('rendered', 'slam'), default='rendered',
                        help="which depth to export as the training target: the Gaussian-rendered "
                             "depth (default, ~2.4x closer to GT and consistent with traj_full.txt) "
                             "or the tracker's own 1/disps_up")
    args = parser.parse_args()

    d = np.load(f'{args.result}/slam_depth.npz')
    tstamp, intrinsics = d['tstamp'], d['intrinsics']
    K, H, W = d['disps_up'].shape
    print(f"{K} keyframes, {H}x{W}, intrinsics fx={intrinsics[0]:.2f} fy={intrinsics[1]:.2f} "
          f"cx={intrinsics[2]:.2f} cy={intrinsics[3]:.2f}")

    poses = torch.from_numpy(d['poses']).cuda().contiguous()
    disps = torch.from_numpy(d['disps']).cuda().contiguous()
    intr = torch.from_numpy(intrinsics).cuda().contiguous()

    # 1/8-res consistency mask, nearest-upsampled to full res for use with disps_up
    mask_low = confidence_mask(poses, disps, intr, args.filter_thresh, args.min_count)
    mask = F.interpolate(mask_low[:, None].float(), size=(H, W), mode='nearest')[:, 0].cpu().numpy() > 0.5
    print(f"confidence mask (thresh={args.filter_thresh}, min_count={args.min_count}): "
          f"{100.0 * mask.mean():.1f}% of pixels kept")

    depth = 1.0 / np.clip(d['disps_up'], 1e-6, None)
    depth[~np.isfinite(depth)] = 0.0

    if not args.no_export:
        # 'rendered' is the Gaussian map's expected depth after the colour refinement. It is the
        # better target on two counts: measurably closer to GT (0.0133 vs 0.0324 m on Replica
        # room0), and rendered from the SAME post-refinement trajectory that traj_full.txt holds,
        # whereas 1/disps_up is dumped before the refinement overwrites video.poses (hi2.py:155).
        ddir, mdir = f'depth_{args.depth_source}', f'mask_{args.depth_source}'
        for sub in (ddir, mdir, 'image'):
            os.makedirs(f'{args.result}/{sub}', exist_ok=True)

        kept, missing = [], []
        for i in range(K):
            idx = int(tstamp[i])
            if args.depth_source == 'rendered':
                rf = f'{args.result}/renders/depth_after_opt/{idx:06d}.png'
                if not os.path.exists(rf):
                    missing.append(idx)
                    continue
                # dequantize once here so downstream keeps a single float32 .npy loader
                dep = cv2.imread(rf, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 6553.5
            else:
                dep = depth[i].astype(np.float32)

            np.save(f'{args.result}/{ddir}/{idx:06d}.npy', dep)
            cv2.imwrite(f'{args.result}/{mdir}/{idx:06d}.png',
                        ((mask[i] & (dep > 0)) * 255).astype(np.uint8))
            rgb = d['images'][i].transpose(1, 2, 0)          # stored RGB (mono_stream converts)
            cv2.imwrite(f'{args.result}/image/{idx:06d}.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            kept.append(i)

        if not kept:
            raise SystemExit(f"no {args.depth_source} depth found - "
                             f"{args.result}/renders/depth_after_opt/ is empty or absent, so the "
                             f"run probably died before eval_rendering")
        if missing:
            print(f"WARNING: {len(missing)} of {K} keyframes have no render and were skipped: "
                  f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")

        # only the exported keyframes: lora_adapt_vggt.py takes its keyframe list from this file
        # and would then look for depth files that were never written
        # same convention as demo.py save_trajectory: TUM, camera-to-world
        poses_wc = SE3(poses).inv().data.cpu().numpy()
        np.savetxt(f'{args.result}/poses_slam.txt',
                   np.concatenate([tstamp[kept][:, None], poses_wc[kept]], axis=1))
        print(f"wrote {ddir}/ {mdir}/ image/ poses_slam.txt to {args.result} "
              f"({len(kept)} keyframes)")

    if args.gtdepthdir is None:
        return

    # ---- compare the three candidate supervision sources against GT ----
    gtfiles = sorted(os.listdir(args.gtdepthdir))

    # JDSA-aligned Omnidata prior, reusing geom/ba.py's bilinear scale field.
    # Inherently 1/8-res in the pipeline; bilinearly upsampled here so all three are comparable.
    prior_al, _ = get_prior_depth_aligned(torch.from_numpy(d['disps_prior']).cuda(),
                                          torch.from_numpy(d['dscales']).cuda())
    prior_al = F.interpolate(prior_al[:, None], size=(H, W), mode='bilinear', align_corners=False)[:, 0]
    prior_depth = (1.0 / prior_al.clamp(min=1e-6)).cpu().numpy()

    pairs = {k: [] for k in ('slam', 'rendered', 'prior')}
    for i in range(K):
        idx = int(tstamp[i])
        gt = cv2.imread(os.path.join(args.gtdepthdir, gtfiles[idx]), cv2.IMREAD_ANYDEPTH) / 6553.5
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        valid = (gt > 0) & mask[i]
        if valid.sum() == 0:
            continue

        srcs = [('slam', depth[i]), ('prior', prior_depth[i])]
        rf = f'{args.result}/renders/depth_after_opt/{idx:06d}.png'
        if os.path.exists(rf):
            srcs.append(('rendered', cv2.imread(rf, cv2.IMREAD_ANYDEPTH) / 6553.5))

        for name, pred in srcs:
            v = valid & (pred > 0)
            if v.sum() > 0:
                pairs[name].append((gt[v], pred[v]))

    print(f"\nscale-aligned depth L1 (m) vs GT, masked, over {len(pairs['slam'])} keyframes")
    print(f"  {'source':<34} {'per-frame':>10} {'global':>10}")
    print(f"  {'-' * 56}")
    for name, label in (('slam', 'SLAM depth (1/disps_up)'),
                        ('rendered', f"Gaussian-rendered ({len(pairs['rendered'])} kf)"),
                        ('prior', 'JDSA-aligned Omnidata prior')):
        if pairs[name]:
            print(f"  {label:<34} {l1_per_frame(pairs[name]):>10.4f} {l1_global(pairs[name]):>10.4f}")
        else:
            print(f"  {label:<34} {'n/a':>10} {'n/a':>10}")


if __name__ == '__main__':
    main()
