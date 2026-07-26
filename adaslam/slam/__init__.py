"""Driving HI-SLAM2 (ARCHITECTURE.md 9.2.1) - the single interface every stage runs it through.

    from slam import SlamConfig, SlamRunner

    runner = SlamRunner(SlamConfig(weights='pretrained_models/droid.pth', colors=..., calib=...,
                                   start=0, undistort=False, crop_border=0, stream_res=341*640))
    res = runner.run(out, config='config/tum_config.yaml', length=200, buffer=500,
                     dump_slam_depth=True)                       # the extract run
    res = runner.run(out, config, length, buffer, gtdepthdir=DEPTHS, prior=VggtPrior(...))  # an arm

This package is the only code in the repo besides demo.py that talks to `Hi2` or `MotionFilter`.
Keeping it that way is what makes "one interface" a checkable property rather than a convention -
see runner.py's docstring for the grep.

Import cost is deliberate: torch.multiprocessing, numpy and tqdm arrive at import time, but hi2,
motion_filter, lietorch and yaml are imported inside the functions that need them, because the
reader Process is started with 'spawn' and re-imports this package in the child.
"""
import os    # nopep8
import sys   # nopep8

# The irreducible four lines: `paths` is itself a sibling module, so ada-slam/ has to reach
# sys.path before it can be imported. Everything after this goes through paths.bootstrap.
_ADA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # <repo>/ada-slam
if _ADA not in sys.path:
    sys.path.insert(0, _ADA)

from paths import HISLAM2, bootstrap                              # noqa: E402

bootstrap(HISLAM2)   # `common`/`runtime` are siblings; hi2 and motion_filter live in hislam2/

from .config import SlamConfig                                    # noqa: E402
from .runner import HI2_ARGS, SlamResult, SlamRunner, save_trajectory   # noqa: E402
from .stream import mono_stream                                   # noqa: E402
from .tracking_config import write_tracking_config                # noqa: E402

__all__ = ['HI2_ARGS', 'SlamConfig', 'SlamResult', 'SlamRunner', 'mono_stream',
           'save_trajectory', 'write_tracking_config']
