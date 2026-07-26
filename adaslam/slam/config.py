"""SlamConfig - the stream and the tracker.

No field carries a default, as in adapt/config.py: whoever drives a run states every value, so a
knob is written down in exactly one place per entry point.

The split is deliberate and load-bearing. What is IDENTICAL across every run of one experiment
lives here and is handed to SlamRunner once; what DIFFERS per run - the tracking YAML, the output
dir, length, buffer, gtdepthdir, dump_slam_depth - is an argument to SlamRunner.run(). That is
what makes it visible at the call site that the extract run gets a generated config and the A/B
arms get the base one (ARCHITECTURE.md 9.2.1), rather than that difference hiding inside an
object.

Note what is NOT here: gtdepthdir. Routing GT depth into Hi2 corrupts a run whose renders become
training data (eval_utils.py:50-52 zeroes rendered depth wherever GT is invalid - 9.3), so it is
a per-run argument that each caller has to state, not a field it could inherit by accident.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SlamConfig:
    """One sequence, one tracker. Pickled by value into the reader child, so keep it primitives.

    Never make a field a computed path or an open handle: torch.multiprocessing's 'spawn' start
    method re-executes the driver module in the child, and every config literal in it is rebuilt.
    """
    weights: str          # pretrained_models/droid.pth -> Hi2's DroidNet
    colors: str           # image directory, walked sorted; the filenames supply the timestamps
    calib: str            # one line: 'fx fy cx cy [k1 k2 p1 p2 ...]'
    start: int            # first frame index
    undistort: bool       # reader-only. 10.1 says undistort offline instead: split_render_metrics
    crop_border: int      # reader-only, likewise    re-derives the GT frame with a resize alone
    stream_res: int       # tracking resolution budget - common.stream_resize's argument

    def __post_init__(self):
        if self.start < 0:
            raise ValueError(f'start={self.start} must be >= 0')
        if self.crop_border < 0:
            raise ValueError(f'crop_border={self.crop_border} must be >= 0')
        if self.stream_res <= 0:
            raise ValueError(f'stream_res={self.stream_res} must be > 0')
