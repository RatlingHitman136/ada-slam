"""ExtractConfig - the training-data run. Fields SlamConfig declares are not repeated."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExtractConfig:
    """Keyframe production, then the export.

    The four thresholds are EXTRACT-ONLY: they go into a generated tracking YAML only this run is
    given, so denser training data cannot look like a tracking change in the comparison.
    """
    # ---------------------------------------------------------------- keyframe production
    # Two gates decide the count and the second usually wins: MotionFilter proposes, TrackFrontend
    # prunes. Over 204 TUM frames (motion, redundant) = (2.4, 4.0) -> 43 kf, (1.2, 4.0) -> 45,
    # (1.2, 1.5) -> 83. To densify, lower BOTH. Any threshold may be None = inherit the base config.
    kf_motion_thresh: Optional[float]     # motion_filter.thresh
    kf_init_thresh: Optional[float]       # the same gate before initialisation
    kf_redundant_thresh: Optional[float]  # frontend.keyframe_thresh - the gate that binds
    kf_covis_thresh: Optional[float]      # backend.covis_thresh; extras in terminate(). LOWER=more
    buffer: int                           # hard cap; MUST exceed the count (no overflow guard)
    # ---------------------------------------------------------------- export
    depth_png_scale: float                # 16-bit depth PNG scale used across the repo
    mask_filter_thresh: float             # depth_filter disparity agreement
    mask_min_count: int                   # min agreeing neighbours out of 6
    mask_min_disp_ratio: float            # drop pixels below this fraction of the frame's mean
    gt_depths: Optional[str]              # the accuracy table ONLY - must never reach Hi2 (9.3)

    def __post_init__(self):
        if self.buffer <= 0:
            raise ValueError(f'buffer={self.buffer} must be > 0')
        if self.mask_min_count < 0:
            raise ValueError(f'mask_min_count={self.mask_min_count} must be >= 0')
        if not 0.0 <= self.mask_min_disp_ratio < 1.0:
            raise ValueError(f'mask_min_disp_ratio={self.mask_min_disp_ratio} must be in [0, 1)')
