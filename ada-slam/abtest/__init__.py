"""Stage 3: the A/B arms and the metrics harness (ARCHITECTURE.md 9.2.1).

    from abtest import TestConfig, run_ab_test

    run_ab_test(runner, slam_cfg, TestConfig(...), arm_config, adapter, split_at)

One full-sequence SLAM run per arm, differing ONLY in the depth prior, then evo ATE -> TSDF ->
Sim(3) align -> eval_recon plus render metrics recomputed per frame, all split seen/unseen at the
frame the adapter's training data ended.

Not named `test/`: ada-slam/ is on sys.path, and a package called `test` there would shadow
CPython's stdlib one.

report.py's two functions are pure formatting over evaluate()'s dicts, so they can be re-run
against ab_results.json files already on disk without a GPU.
"""
import os    # nopep8
import sys   # nopep8

# The irreducible four lines: `paths` is itself a sibling module, so ada-slam/ has to reach
# sys.path before it can be imported. Everything after this goes through paths.bootstrap.
_ADA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # <repo>/ada-slam
if _ADA not in sys.path:
    sys.path.insert(0, _ADA)

from paths import HISLAM2, bootstrap                                # noqa: E402

bootstrap(HISLAM2)   # gaussian.utils.loss_utils and midas.omnidata live in hislam2/

from .config import ARM_DIRS, ARM_NAMES, TestConfig                 # noqa: E402
from .metrics import evaluate, run_ate, run_mesh, split_render_metrics    # noqa: E402
from .prior import VggtPrior                                        # noqa: E402
from .report import compare, print_report                           # noqa: E402
from .stage import make_prior, run_ab_test                          # noqa: E402

__all__ = ['ARM_DIRS', 'ARM_NAMES', 'TestConfig', 'VggtPrior', 'compare', 'evaluate',
           'make_prior', 'print_report', 'run_ab_test', 'run_ate', 'run_mesh',
           'split_render_metrics']
