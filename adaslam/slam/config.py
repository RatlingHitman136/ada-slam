"""SlamConfig - the stream and the tracker (9.2.1)."""
from dataclasses import dataclass
from typing import Optional


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
    # THE WINDOW: which frames this experiment is about. Half-open, [start, stop), so stop is the
    # first frame NOT run and (0, None) is the whole sequence. Not to be confused with run()'s
    # `length`, which is a per-CALL cap WITHIN the window - the extract stage runs a prefix of it
    # while the arms run all of it, which is why that one cannot live here.
    start: int            # first frame index
    stop: Optional[int]   # one past the last; None = to the end of the sequence
    undistort: bool       # reader-only; undistort offline instead (10.1)
    crop_border: int      # reader-only, likewise
    stream_res: int       # tracking pixel budget - common.stream_resize's argument
    render_eval: bool     # hi2.py's eval_rendering -> renders/ + psnr/; nothing here reads them

    def __post_init__(self):
        if self.start < 0:
            raise ValueError(f'start={self.start} must be >= 0')
        if self.stop is not None and self.stop <= self.start:
            raise ValueError(f'stop={self.stop} must be > start={self.start} (the window is '
                             f'half-open, [start, stop)); None means "to the end"')
        if self.crop_border < 0:
            raise ValueError(f'crop_border={self.crop_border} must be >= 0')
        if self.stream_res <= 0:
            raise ValueError(f'stream_res={self.stream_res} must be > 0')
