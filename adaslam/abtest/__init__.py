"""Stage 3: the A/B arms and the metrics harness (ARCHITECTURE.md 9.2.1).

    from adaslam.abtest import TestConfig, run_ab_test

    run_ab_test(runner, slam_cfg, TestConfig(...), arm_config, adapter, split_at)

One full-sequence SLAM run per arm, differing ONLY in the depth prior, then evo ATE -> TSDF ->
Sim(3) align -> eval_recon plus render metrics recomputed per frame, all split seen/unseen at the
frame the adapter's training data ended.

Named `abtest`, not `test`: back when ada-slam/ was itself on sys.path, a package called `test`
there would have shadowed CPython's stdlib one. As `adaslam.test` it no longer would - the name is
kept because renaming it now would buy nothing.

report.py's two functions are pure formatting over evaluate()'s dicts, so they can be re-run
against ab_results.json files already on disk without a GPU:

    from adaslam.abtest.report import compare, print_report

gaussian.utils.loss_utils and midas.omnidata live in hislam2/, which adaslam/__init__.py put on
sys.path before this file could run.
"""
from .config import TestConfig
from .prior import VggtPrior
from .stage import run_ab_test

__all__ = ['TestConfig', 'VggtPrior', 'run_ab_test']
