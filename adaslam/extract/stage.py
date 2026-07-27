"""run_extract - stage 1 end to end: generated config -> SLAM -> export."""
import os
import shutil
import time

from ..common import DEPTH_DIR, HANDOFF_UP, extract_run_dir
from ..print_utils import tee
from ..runtime import free_vram
from ..slam import write_tracking_config

from .export import export_slam_depth


def handoff_paths(out):
    """What a finished export leaves at the top level of an extract experiment directory.

    Used to decide whether an already-tracked run still needs its export re-run: any of these
    missing re-runs it, which costs only reading the npz back.
    """
    return [f'{out}/poses_slam.txt', f'{out}/{DEPTH_DIR}',
            *(f'{out}/{f}' for f in HANDOFF_UP)]


def run_extract(runner, cfg, out, length, base_config, skip_existing=False):
    """SLAM over the first `length` frames with the depth dump on, then export what it wrote.

    `out` is the EXPERIMENT directory. The run itself goes into out/full and stays there whole;
    only the handoff artifacts reach the top level, so full/ can be deleted afterwards to reclaim
    the Gaussian map without breaking the adapt stage.

    `base_config` is the experiment's tracking YAML; this is the ONLY run that gets the keyframe
    knobs layered on top of it, written to out/full/extract_config.yaml so the run is
    self-documenting. Returns the number of keyframes exported, or None if nothing was done.
    """
    run_dir = extract_run_dir(out)
    tracking_cfg = write_tracking_config(run_dir, base_config,
                                         motion_thresh=cfg.kf_motion_thresh,
                                         init_thresh=cfg.kf_init_thresh,
                                         keyframe_thresh=cfg.kf_redundant_thresh,
                                         covis_thresh=cfg.kf_covis_thresh)

    # The two halves are skipped independently: re-exporting is cheap (it reads the npz back), so
    # it re-runs whenever a handoff artifact is missing even though the SLAM run is reused.
    if skip_existing and os.path.exists(f'{run_dir}/slam_depth.npz'):
        print(f'{run_dir}/slam_depth.npz exists - skipping the SLAM run')
        stale = [p for p in handoff_paths(out) if not os.path.exists(p)]
        if not stale:
            return None
        print(f'but re-exporting: {" ".join(os.path.basename(p) for p in stale)} missing')
    else:
        t0 = time.time()
        # gtdepthdir stays None: eval_utils.py:50-52 zeroes the rendered depth wherever GT is
        # invalid, and on real sensors (TUM: 24% holes, on exactly the hard surfaces) that would
        # both shrink the training set and tie its mask to where the Kinect happened to work. Moot
        # while render_eval is off, and stated anyway so it stays true if it goes back on.
        # cfg.gt_depths reaches the accuracy table below instead, which masks on (gt > 0) & mask.
        n_kf = runner.run(run_dir, tracking_cfg, length, cfg.buffer,
                          gtdepthdir=None, dump_slam_depth=True).n_kf
        print(f'=== SLAM done in {time.time()-t0:.0f}s: {n_kf} keyframes over {length} '
              f'frames (1 per {length/max(n_kf,1):.1f}). For more, lower kf_redundant_thresh '
              f'({cfg.kf_redundant_thresh}) first, then kf_motion_thresh '
              f'({cfg.kf_motion_thresh}) - the redundancy gate binds')
        if n_kf >= cfg.buffer:
            print(f'WARNING: keyframe count hit buffer ({cfg.buffer})')

    # Copied, not moved or symlinked: full/ has to stay a complete HI-SLAM2 run, and the point of
    # the split is that deleting it leaves the adapt stage's inputs intact. Both are small text.
    for name in HANDOFF_UP:
        if not os.path.exists(f'{run_dir}/{name}'):
            raise SystemExit(f'{run_dir}/{name} is missing, so that run never reached '
                             f'save_trajectory - delete {run_dir} and let the SLAM run repeat')
        shutil.copyfile(f'{run_dir}/{name}', f'{out}/{name}')

    with tee(f'{out}/export.txt'):
        n_exported = export_slam_depth(out, cfg)
    free_vram('extract')
    print(f'{n_exported} keyframes exported to {out}/{DEPTH_DIR}/')
    return n_exported
