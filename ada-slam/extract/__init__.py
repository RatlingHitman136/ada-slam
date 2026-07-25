"""Stage 1: HI-SLAM2's own depth, exported as training data (ARCHITECTURE.md 9.2.1).

    from extract import ExtractConfig, run_extract

    run_extract(runner, ExtractConfig(...), out, length, base_config)

produces, under `out`:

    extract_config.yaml     what the run was told to do (inherit_from + the keyframe knobs)
    slam_depth.npz          Hi2's post-global-BA dump
    depth_<src>/%06d.npy    per-keyframe training depth, float32, SLAM units
    mask_<src>/%06d.png     multi-view consistency mask & depth > 0
    image/%06d.jpg          the matching keyframe RGB
    poses_slam.txt          the exported keyframes, TUM c2w - adapt takes its keyframe list here
    export.txt              the depth-source accuracy table

load_export / write_keyframes / report_accuracy are exposed separately so a caller can have the
table without the files (scripts/export_slam_depth.py --no_export).
"""
import os    # nopep8
import sys   # nopep8

# The irreducible four lines: `paths` is itself a sibling module, so ada-slam/ has to reach
# sys.path before it can be imported. Everything after this goes through paths.bootstrap.
_ADA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # <repo>/ada-slam
if _ADA not in sys.path:
    sys.path.insert(0, _ADA)

from paths import HISLAM2, bootstrap                                # noqa: E402

bootstrap(HISLAM2)   # geom.ba and droid_backends live in hislam2/

from .accuracy import align_scale, l1_global, l1_per_frame, report_accuracy   # noqa: E402
from .config import ExtractConfig                                   # noqa: E402
from .export import confidence_mask, export_slam_depth, load_export, write_keyframes  # noqa: E402
from .stage import run_extract                                      # noqa: E402

__all__ = ['ExtractConfig', 'align_scale', 'confidence_mask', 'export_slam_depth', 'l1_global',
           'l1_per_frame', 'load_export', 'report_accuracy', 'run_extract', 'write_keyframes']
