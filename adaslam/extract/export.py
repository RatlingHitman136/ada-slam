"""slam_depth.npz -> depth_<src>/ mask_<src>/ image/ poses_slam.txt.

Hi2.terminate() dumps the tracker's own state at the one instant where disps, disps_up and poses
are mutually consistent (hi2.py:155); this turns that dump into the per-keyframe files the adapt
stage trains on.

Loading is split from writing because the accuracy table (accuracy.py) needs the same arrays but
writes nothing: `load_export` then `report_accuracy`, skipping `write_keyframes`, scores an
existing dump without touching the files on disk.
"""
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ExportInputs:
    """Everything both the writer and the accuracy table read out of one slam_depth.npz."""
    npz: object            # the loaded archive, for images / disps_prior / dscales
    tstamp: np.ndarray     # (K,) frame index per keyframe
    poses: torch.Tensor    # (K,7) world->cam on cuda
    depth: np.ndarray      # (K,H,W) 1/disps_up, non-finite zeroed
    mask: np.ndarray       # (K,H,W) bool, the multi-view consistency mask at full res
    shape: tuple           # (K, H, W)


def confidence_mask(poses, disps, intrinsics_full, cfg):
    """Multi-view consistency mask, following util/droid_visualization.py:104-110.

    droid_backends.depth_filter counts, per pixel, how many of 6 temporal neighbours agree on the
    reprojected disparity. The kernel bounds-checks against disps.size(0), so the arrays must be
    sliced to the real keyframe count - otherwise trailing keyframes match unused buffer slots
    that still hold the initial 1.0.
    """
    import droid_backends
    K = disps.shape[0]
    ix = torch.arange(K, device='cuda', dtype=torch.long)
    thresh = cfg.mask_filter_thresh * torch.ones(K, device='cuda', dtype=torch.float)
    count = droid_backends.depth_filter(poses, disps, intrinsics_full / 8.0, ix, thresh)
    return (count >= cfg.mask_min_count) & \
           (disps > cfg.mask_min_disp_ratio * disps.mean(dim=[1, 2], keepdim=True))


def load_export(out, cfg):
    """Read out/slam_depth.npz and build the mask. Writes nothing."""
    d = np.load(f'{out}/slam_depth.npz')
    tstamp, intrinsics = d['tstamp'], d['intrinsics']
    K, H, W = d['disps_up'].shape
    print(f'{K} keyframes, {H}x{W}, intrinsics fx={intrinsics[0]:.2f} fy={intrinsics[1]:.2f} '
          f'cx={intrinsics[2]:.2f} cy={intrinsics[3]:.2f}')

    poses = torch.from_numpy(d['poses']).cuda().contiguous()
    disps = torch.from_numpy(d['disps']).cuda().contiguous()
    intr = torch.from_numpy(intrinsics).cuda().contiguous()

    # 1/8-res consistency mask, nearest-upsampled to full res for use with disps_up
    mask_low = confidence_mask(poses, disps, intr, cfg)
    mask = F.interpolate(mask_low[:, None].float(), size=(H, W),
                         mode='nearest')[:, 0].cpu().numpy() > 0.5
    print(f'confidence mask (thresh={cfg.mask_filter_thresh}, min_count={cfg.mask_min_count}): '
          f'{100.0 * mask.mean():.1f}% of pixels kept')

    depth = 1.0 / np.clip(d['disps_up'], 1e-6, None)
    depth[~np.isfinite(depth)] = 0.0
    return ExportInputs(npz=d, tstamp=tstamp, poses=poses, depth=depth, mask=mask,
                        shape=(K, H, W))


def write_keyframes(out, x, cfg):
    """Write depth_<src>/ mask_<src>/ image/ poses_slam.txt. Returns the exported keyframe rows.

    'rendered' is the Gaussian map's expected depth after the colour refinement. It is the better
    target on two counts: measurably closer to GT (0.0133 vs 0.0324 m on Replica room0), and
    rendered from the SAME post-refinement trajectory that traj_full.txt holds, whereas
    1/disps_up is dumped before the refinement overwrites video.poses (hi2.py:155).
    """
    from lietorch import SE3
    K, _, _ = x.shape
    ddir, mdir = f'depth_{cfg.depth_source}', f'mask_{cfg.depth_source}'
    for sub in (ddir, mdir, 'image'):
        os.makedirs(f'{out}/{sub}', exist_ok=True)

    kept, missing = [], []
    for i in range(K):
        idx = int(x.tstamp[i])
        if cfg.depth_source == 'rendered':
            rf = f'{out}/renders/depth_after_opt/{idx:06d}.png'
            if not os.path.exists(rf):
                missing.append(idx)
                continue
            # dequantize once here so downstream keeps a single float32 .npy loader
            dep = cv2.imread(rf, cv2.IMREAD_ANYDEPTH).astype(np.float32) / cfg.depth_png_scale
        else:
            dep = x.depth[i].astype(np.float32)

        np.save(f'{out}/{ddir}/{idx:06d}.npy', dep)
        cv2.imwrite(f'{out}/{mdir}/{idx:06d}.png',
                    ((x.mask[i] & (dep > 0)) * 255).astype(np.uint8))
        rgb = x.npz['images'][i].transpose(1, 2, 0)      # stored RGB (mono_stream converts)
        cv2.imwrite(f'{out}/image/{idx:06d}.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        kept.append(i)

    if not kept:
        raise SystemExit(f'no {cfg.depth_source} depth found - {out}/renders/depth_after_opt/ is '
                         f'empty or absent, so the run probably died before eval_rendering')
    if missing:
        print(f'WARNING: {len(missing)} of {K} keyframes have no render and were skipped: '
              f'{missing[:8]}{" ..." if len(missing) > 8 else ""}')

    # only the exported keyframes: the adapt stage takes its keyframe list from this file and
    # would otherwise look for depth files that were never written.
    # same convention as save_trajectory: TUM, camera-to-world
    poses_wc = SE3(x.poses).inv().data.cpu().numpy()
    np.savetxt(f'{out}/poses_slam.txt',
               np.concatenate([x.tstamp[kept][:, None], poses_wc[kept]], axis=1))
    print(f'wrote {ddir}/ {mdir}/ image/ poses_slam.txt to {out} ({len(kept)} keyframes)')
    return kept


def export_slam_depth(out, cfg):
    """load -> write -> accuracy table. Returns the number of keyframes exported."""
    from .accuracy import report_accuracy
    x = load_export(out, cfg)
    kept = write_keyframes(out, x, cfg)
    report_accuracy(out, x, cfg)
    return len(kept)
