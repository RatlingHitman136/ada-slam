"""SlamConfig - the stream and the tracker (9.2.1)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SlamConfig:
    """What is identical across every run of an experiment; what differs is a run() argument.

    Primitives only - spawn rebuilds every config literal in the reader child. No gtdepthdir:
    routing GT depth into Hi2 corrupts a run whose renders become training data (9.3), so it is a
    per-run argument nothing can inherit by accident.
    """
    weights: str          # pretrained_models/droid.pth -> Hi2's DroidNet
    colors: str           # image dir, walked sorted; the filenames supply the timestamps
    calib: str            # one line: 'fx fy cx cy [k1 k2 p1 p2 ...]'
    start: int            # first frame index
    undistort: bool       # reader-only; undistort offline instead (10.1)
    crop_border: int      # reader-only, likewise
    stream_res: int       # tracking pixel budget - common.stream_resize's argument
    render_eval: bool     # hi2.py's eval_rendering -> renders/ + psnr/; nothing here reads them

    def __post_init__(self):
        if self.start < 0:
            raise ValueError(f'start={self.start} must be >= 0')
        if self.crop_border < 0:
            raise ValueError(f'crop_border={self.crop_border} must be >= 0')
        if self.stream_res <= 0:
            raise ValueError(f'stream_res={self.stream_res} must be > 0')
