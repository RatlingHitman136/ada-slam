"""Driving HI-SLAM2 (ARCHITECTURE.md 9.2.1) - the single interface every stage runs it through.

    from adaslam.slam import SlamConfig, SlamRunner

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

Besides SlamRunner it exposes PriorProbe, which runs a depth prior over frames WITHOUT a SLAM run.
The prior test needs that, and the stock prior is a MotionFilter method, so it has to live here.

hi2 and motion_filter live in hislam2/, which adaslam/__init__.py put on sys.path before this file
could run.
"""
from .config import SlamConfig
from .prior_probe import PriorProbe
from .runner import SlamResult, SlamRunner
from .tracking_config import write_tracking_config

__all__ = ['PriorProbe', 'SlamConfig', 'SlamResult', 'SlamRunner', 'write_tracking_config']
