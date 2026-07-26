"""run_extract - stage 1 end to end: generated config -> SLAM -> export."""
import os
import time

from runtime import free_vram, tee
from slam import write_tracking_config

from .export import export_slam_depth


def run_extract(runner, cfg, out, length, base_config, skip_existing=False):
    """SLAM over the first `length` frames with the depth dump on, then export what it wrote.

    `base_config` is the experiment's tracking YAML; this is the ONLY run that gets the keyframe
    knobs layered on top of it, written to out/extract_config.yaml so the run is self-documenting.
    Returns the number of keyframes exported.
    """
    tracking_cfg = write_tracking_config(out, base_config,
                                         motion_thresh=cfg.kf_motion_thresh,
                                         init_thresh=cfg.kf_init_thresh,
                                         keyframe_thresh=cfg.kf_redundant_thresh,
                                         covis_thresh=cfg.kf_covis_thresh)

    # the npz standing in for the whole stage, export included: re-exporting is cheap, but a
    # skipped stage that still rewrote depth_<src>/ would quietly re-do work the caller asked to
    # reuse. Change DEPTH_SOURCE and you must let the stage run.
    if skip_existing and os.path.exists(f'{out}/slam_depth.npz'):
        print(f'{out}/slam_depth.npz exists - skipping the SLAM run')
        return None

    t0 = time.time()
    # gtdepthdir stays None: eval_utils.py:50-52 zeroes the rendered depth wherever GT is
    # invalid, and on real sensors (TUM: 24% holes, on exactly the hard surfaces) that would
    # both shrink the training set and tie its mask to where the Kinect happened to work.
    # cfg.gt_depths reaches the accuracy table below instead, which masks on (gt > 0) & mask.
    n_kf = runner.run(out, tracking_cfg, length, cfg.buffer,
                      gtdepthdir=None, dump_slam_depth=True).n_kf
    print(f'=== SLAM done in {time.time()-t0:.0f}s: {n_kf} keyframes over {length} '
          f'frames (1 per {length/max(n_kf,1):.1f}). For more, lower kf_redundant_thresh '
          f'({cfg.kf_redundant_thresh}) first, then kf_motion_thresh '
          f'({cfg.kf_motion_thresh}) - the redundancy gate binds')
    if n_kf >= cfg.buffer:
        print(f'WARNING: keyframe count hit buffer ({cfg.buffer})')

    with tee(f'{out}/export.txt'):
        n_exported = export_slam_depth(out, cfg)
    free_vram('extract')
    print(f'{n_exported} keyframes exported to {out}/depth_{cfg.depth_source}/')
    return n_exported
