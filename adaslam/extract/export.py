"""slam_depth.npz -> depth_<src>/ mask_<src>/ image/ poses_slam.txt.

Hi2.terminate() dumps the tracker's own state at the one instant where disps, disps_up and poses
are mutually consistent (hi2.py:155); this turns that dump into the per-keyframe files the adapt
stage trains on.

Two directories, and every function here says which it takes. `run_dir` is the untouched HI-SLAM2
run (extract/<exp>/full), which is where the npz and the renders are; `exp_dir` is the experiment
above it, which holds the handoff artifacts and nothing else. The split is what lets full/ be
deleted afterwards without breaking adapt.

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

from ..common import extract_run_dir


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


def write_keyframes(exp_dir, run_dir, x, cfg):
    """Write depth_<src>/ mask_<src>/ image/ poses_slam.txt into exp_dir. Returns the kept rows.

    EVERY source in cfg.depth_sources is written, in one pass over the keyframes. All of them are
    handoff artifacts: which one supervises is AdaptConfig.depth_source's decision, made later and
    changed in a second, whereas a source that was never exported costs another SLAM run.

    'rendered' is the Gaussian map's expected depth after the colour refinement, read back out of
    run_dir's PNGs. It is the better target on two counts: measurably closer to GT (0.0133 vs
    0.0324 m on Replica room0), and rendered from the SAME post-refinement trajectory that
    traj_full.txt holds, whereas 1/disps_up is dumped before the refinement overwrites video.poses
    (hi2.py:155).
    """
    from lietorch import SE3
    K, _, _ = x.shape
    os.makedirs(f'{exp_dir}/image', exist_ok=True)
    for src in cfg.depth_sources:
        for sub in (f'depth_{src}', f'mask_{src}'):
            os.makedirs(f'{exp_dir}/{sub}', exist_ok=True)

    kept = {src: [] for src in cfg.depth_sources}
    missing = {src: [] for src in cfg.depth_sources}
    for i in range(K):
        idx = int(x.tstamp[i])
        for src in cfg.depth_sources:
            if src == 'rendered':
                rf = f'{run_dir}/renders/depth_after_opt/{idx:06d}.png'
                if not os.path.exists(rf):
                    missing[src].append(idx)
                    continue
                # dequantize once here so downstream keeps a single float32 .npy loader
                dep = cv2.imread(rf, cv2.IMREAD_ANYDEPTH).astype(np.float32) / cfg.depth_png_scale
            else:
                dep = x.depth[i].astype(np.float32)

            np.save(f'{exp_dir}/depth_{src}/{idx:06d}.npy', dep)
            cv2.imwrite(f'{exp_dir}/mask_{src}/{idx:06d}.png',
                        ((x.mask[i] & (dep > 0)) * 255).astype(np.uint8))
            kept[src].append(i)
        # a record of which keyframes were exported, not a handoff artifact - SceneData indexes
        # the full colour directory by frame number, so it cannot read a keyframes-only folder
        rgb = x.npz['images'][i].transpose(1, 2, 0)      # stored RGB (mono_stream converts)
        cv2.imwrite(f'{exp_dir}/image/{idx:06d}.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # A source with nothing in it is dropped rather than fatal: a run that died before
    # eval_rendering has no renders at all, and its depth_slam/ is still perfectly usable.
    live = [s for s in cfg.depth_sources if kept[s]]
    for src in (s for s in cfg.depth_sources if not kept[s]):
        print(f'WARNING: no {src} depth for any keyframe - that source is not exported. For '
              f'"rendered" this means {run_dir}/renders/depth_after_opt/ is empty or absent, so '
              f'the run probably died before eval_rendering.')
    if not live:
        raise SystemExit(f'nothing exported: none of {cfg.depth_sources} produced depth for any '
                         f'of the {K} keyframes in {run_dir}')
    for src in live:
        if missing[src]:
            print(f'WARNING: {len(missing[src])} of {K} keyframes have no {src} depth: '
                  f'{missing[src][:8]}{" ..." if len(missing[src]) > 8 else ""}')

    # poses_slam.txt is the keyframe LIST every later stage works from: adapt reads the indices
    # here and then opens depth_<src>/<idx>.npy for whichever source it was configured with
    # (adapt/data.py:71), so a keyframe listed but absent from that source is a crash. Taking the
    # intersection is what makes one file valid for every source written.
    # Same convention as save_trajectory: TUM, camera-to-world.
    rows = sorted(set.intersection(*(set(kept[s]) for s in live)))
    if not rows:
        raise SystemExit(f'no keyframe has depth in all of {live} at once, so poses_slam.txt '
                         f'would be empty - export a single source instead')
    dropped = sorted(set.union(*(set(kept[s]) for s in live)) - set(rows))
    if dropped:
        print(f'{len(dropped)} keyframes are missing from at least one source and so are left out '
              f'of poses_slam.txt: {[int(x.tstamp[i]) for i in dropped[:8]]}'
              f'{" ..." if len(dropped) > 8 else ""}')

    poses_wc = SE3(x.poses).inv().data.cpu().numpy()
    np.savetxt(f'{exp_dir}/poses_slam.txt',
               np.concatenate([x.tstamp[rows][:, None], poses_wc[rows]], axis=1))
    print(f'wrote {" ".join(f"depth_{s}/ mask_{s}/" for s in live)} image/ poses_slam.txt to '
          f'{exp_dir} ({len(rows)} keyframes)')
    return rows


def export_slam_depth(exp_dir, cfg):
    """load -> write -> accuracy table. Returns the number of keyframes exported.

    Takes the EXPERIMENT directory; the run it reads from is exp_dir/full.
    """
    from .accuracy import report_accuracy
    run_dir = extract_run_dir(exp_dir)
    x = load_export(run_dir, cfg)
    kept = write_keyframes(exp_dir, run_dir, x, cfg)
    report_accuracy(run_dir, x, cfg)
    return len(kept)
