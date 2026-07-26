"""ExtractConfig - the training-data run.

No field carries a default. Fields SlamConfig already declares are not repeated: run_extract
receives both.
"""
from dataclasses import dataclass
from typing import Optional

from ..common import DEPTH_SOURCES


@dataclass(frozen=True)
class ExtractConfig:
    """Keyframe production, then the export.

    The four thresholds are EXTRACT-ONLY and that is the whole point: they go into a generated
    tracking YAML that only this run is given, so denser training data can never be mistaken for a
    tracking change in the A/B comparison. Any of them may be None = inherit the base config.

    Two gates decide the keyframe count and the SECOND usually wins. MotionFilter proposes one
    once the mean flow since the last exceeds kf_motion_thresh (motion_filter.py:112-113), then
    TrackFrontend deletes it again if it lands closer than kf_redundant_thresh to its neighbour
    (track_frontend.py:49-52, and :93-99 during init, where it prunes on that alone). So lowering
    kf_motion_thresh by itself just proposes keyframes that are immediately pruned. Measured over
    204 TUM frames:
        (motion, redundant) = (2.4, 4.0) -> 43 kf   (1.2, 4.0) -> 45 kf   (1.2, 1.5) -> 83 kf
    To densify, lower BOTH; kf_redundant_thresh is the one that moves the number.
    """
    # ---------------------------------------------------------------- keyframe production
    kf_motion_thresh: Optional[float]     # motion_filter.thresh
    kf_init_thresh: Optional[float]       # the same gate before initialisation
    kf_redundant_thresh: Optional[float]  # frontend.keyframe_thresh - the gate that binds
    kf_covis_thresh: Optional[float]      # backend.covis_thresh; extras in terminate(). LOWER=more
    buffer: int                           # hard cap; MUST exceed the count (no overflow guard)
    # ---------------------------------------------------------------- export
    depth_source: str                     # 'rendered' (Gaussian expected) | 'slam' (1/disps_up)
    depth_png_scale: float                # 16-bit depth PNG scale used across the repo
    mask_filter_thresh: float             # depth_filter disparity agreement
    mask_min_count: int                   # min agreeing neighbours out of 6
    mask_min_disp_ratio: float            # drop pixels below this fraction of the frame's mean
    # The accuracy table's reference ONLY - it must never reach Hi2. eval_utils.py:50-52 zeroes
    # rendered depth wherever GT is invalid, and on TUM's ~24% holes that would both shrink the
    # training set and tie its mask to where the Kinect happened to work. Hence a field here and
    # a separate, per-call gtdepthdir argument on SlamRunner.run (ARCHITECTURE.md 9.3).
    gt_depths: Optional[str]

    def __post_init__(self):
        if self.depth_source not in DEPTH_SOURCES:
            raise ValueError(f'depth_source={self.depth_source!r} is not one of {DEPTH_SOURCES}')
        if self.buffer <= 0:
            raise ValueError(f'buffer={self.buffer} must be > 0')
        if self.mask_min_count < 0:
            raise ValueError(f'mask_min_count={self.mask_min_count} must be >= 0')
        if not 0.0 <= self.mask_min_disp_ratio < 1.0:
            raise ValueError(f'mask_min_disp_ratio={self.mask_min_disp_ratio} must be in [0, 1)')
