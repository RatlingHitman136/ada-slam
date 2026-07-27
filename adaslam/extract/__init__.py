"""Stage 1: HI-SLAM2's own depth, exported as training data (ARCHITECTURE.md 9.2.1).

    from adaslam.extract import ExtractConfig, run_extract

    run_extract(runner, ExtractConfig(...), out, length, base_config)

produces, under the experiment directory `out`:

    depth_slam/%06d.npy     per-keyframe training depth, float32, SLAM units (1/disps_up)
    mask_slam/%06d.png      multi-view consistency mask & depth > 0
    image/%06d.jpg          the matching keyframe RGB (a record; SceneData reads the full colour
                            directory, indexed by frame number, not this keyframes-only one)
    poses_slam.txt          the exported keyframes, TUM c2w - adapt takes its keyframe list here
    traj_full.txt           every frame's pose, copied up from full/ - adapt's actual poses
    intrinsics.npy          fx fy cx cy at the tracker's resolution, copied up from full/
    export.txt              the depth accuracy table
    full/                   the untouched HI-SLAM2 run: extract_config.yaml (what it was told to
                            do), slam_depth.npz (Hi2's post-global-BA dump), the trajectories and
                            3dgs_final.ply - plus renders/ and psnr/ if SlamConfig.render_eval

Only the top level is the handoff to adapt, so full/ can be deleted afterwards to reclaim the
Gaussian map without breaking the stage after it.

Loading is split from writing, so re-exporting an existing slam_depth.npz without re-running SLAM -
or having the accuracy table without the files - is a call into the modules themselves. load_export
takes the RUN directory (out/full), where the npz is; the other two take the arrays it returns:

    from adaslam.extract.export import load_export, write_keyframes
    from adaslam.extract.accuracy import report_accuracy

geom.ba and droid_backends live in hislam2/, which adaslam/__init__.py put on sys.path before this
file could run.
"""
from .config import ExtractConfig
from .stage import run_extract

__all__ = ['ExtractConfig', 'run_extract']
