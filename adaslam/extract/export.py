"""slam_depth.npz -> depth_slam/ mask_slam/ image/ poses_slam.txt.

Hi2.terminate() dumps the tracker's own state at the one instant where disps, disps_up and poses
are mutually consistent (hi2.py:155); this turns that dump into the per-keyframe files the adapt
stage trains on.

Two directories, and every function here says which it takes. `run_dir` is the untouched HI-SLAM2
run (extract/<exp>/full), which is where the npz is; `exp_dir` is the experiment above it, which
holds the handoff artifacts and nothing else. The split is what lets full/ be deleted afterwards
without breaking adapt.

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

from ..common import DEPTH_DIR, MASK_DIR, extract_run_dir


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


def load_export(run_dir, cfg):
    """Read run_dir/slam_depth.npz and build the mask. Writes nothing."""
    d = np.load(f'{run_dir}/slam_depth.npz')
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


def write_keyframes(exp_dir, x, cfg):
    """Write depth_slam/ mask_slam/ image/ poses_slam.txt into exp_dir. Returns the kept rows.

    One target: 1/disps_up, the tracker's own depth, straight out of the npz. There was a second
    once - the Gaussian map's expected depth, read back out of run_dir/renders/ - and it was the
    more accurate of the two (0.0133 vs 0.0324 m on Replica room0). It went with the terminate-time
    render (SlamConfig.render_eval), which is the only thing that ever produced it.
    """
    from lietorch import SE3
    K, _, _ = x.shape
    for sub in ('image', DEPTH_DIR, MASK_DIR):
        os.makedirs(f'{exp_dir}/{sub}', exist_ok=True)

    for i in range(K):
        idx = int(x.tstamp[i])
        dep = x.depth[i].astype(np.float32)
        np.save(f'{exp_dir}/{DEPTH_DIR}/{idx:06d}.npy', dep)
        cv2.imwrite(f'{exp_dir}/{MASK_DIR}/{idx:06d}.png',
                    ((x.mask[i] & (dep > 0)) * 255).astype(np.uint8))
        # a record of which keyframes were exported, not a handoff artifact - SceneData indexes
        # the full colour directory by frame number, so it cannot read a keyframes-only folder
        rgb = x.npz['images'][i].transpose(1, 2, 0)      # stored RGB (mono_stream converts)
        cv2.imwrite(f'{exp_dir}/image/{idx:06d}.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # poses_slam.txt is the keyframe LIST every later stage works from: adapt reads the indices
    # here and then opens depth_slam/<idx>.npy for each (adapt/data.py), so a keyframe listed but
    # absent is a crash. Every keyframe in the npz gets a file above, so the list is all of them.
    # Same convention as save_trajectory: TUM, camera-to-world.
    rows = list(range(K))
    poses_wc = SE3(x.poses).inv().data.cpu().numpy()
    np.savetxt(f'{exp_dir}/poses_slam.txt',
               np.concatenate([x.tstamp[rows][:, None], poses_wc[rows]], axis=1))
    print(f'wrote {DEPTH_DIR}/ {MASK_DIR}/ image/ poses_slam.txt to '
          f'{exp_dir} ({len(rows)} keyframes)')
    return rows


def export_slam_depth(exp_dir, cfg):
    """load -> write -> accuracy table. Returns the number of keyframes exported.

    Takes the EXPERIMENT directory; the run it reads from is exp_dir/full.
    """
    from .accuracy import report_accuracy
    x = load_export(extract_run_dir(exp_dir), cfg)
    kept = write_keyframes(exp_dir, x, cfg)
    report_accuracy(x, cfg)
    return len(kept)
