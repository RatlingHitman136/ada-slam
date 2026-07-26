"""Stage 1: HI-SLAM2's own depth, exported as training data (ARCHITECTURE.md 9.2.1).

    from adaslam.extract import ExtractConfig, run_extract

    run_extract(runner, ExtractConfig(...), out, length, base_config)

produces, under `out`:

    extract_config.yaml     what the run was told to do (inherit_from + the keyframe knobs)
    slam_depth.npz          Hi2's post-global-BA dump
    depth_<src>/%06d.npy    per-keyframe training depth, float32, SLAM units
    mask_<src>/%06d.png     multi-view consistency mask & depth > 0
    image/%06d.jpg          the matching keyframe RGB
    poses_slam.txt          the exported keyframes, TUM c2w - adapt takes its keyframe list here
    export.txt              the depth-source accuracy table

Loading is split from writing, so re-exporting an existing slam_depth.npz without re-running SLAM -
or having the accuracy table without the files - is a call into the modules themselves:

    from adaslam.extract.export import load_export, write_keyframes
    from adaslam.extract.accuracy import report_accuracy

geom.ba and droid_backends live in hislam2/, which adaslam/__init__.py put on sys.path before this
file could run.
"""
from .config import ExtractConfig
from .stage import run_extract

__all__ = ['ExtractConfig', 'run_extract']
