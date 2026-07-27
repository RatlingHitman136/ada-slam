"""Test kind 2: comparing depth-prior generators directly (ARCHITECTURE.md 9.2.2).

    from adaslam.priortest import PriorTestConfig, run_prior_test

    run_prior_test(slam_cfg, PriorTestConfig(...), out_root)

No SLAM run: each prior is evaluated against ground-truth depth frame by frame through
slam.PriorProbe, which calls the same extractor a real arm would. Minutes per arm, not forty.

It exists because the end2end test cannot attribute its own null result (9.4): "swapping the prior
changed nothing" is either "the new prior is no better" or "HI-SLAM2 is insensitive to the way it is
better", and telling those apart needs the priors measured before SLAM touches them. metrics.py's
three alignments - per-frame scale, the 2x2 grid JDSA can absorb, and one global scale - are what
separate them.

Arm directories are named by end2end's rule, imported not restated, so
outputs/test/{end2end,prior}/<scene>/omni/ are the same generator measured two ways.
"""
from .config import PriorTestConfig
from .stage import run_prior_test

__all__ = ['PriorTestConfig', 'run_prior_test']
