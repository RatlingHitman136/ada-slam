"""Stage 3: comparing depth-prior generators end to end (ARCHITECTURE.md 9.2.1).

    from adaslam.end2end import End2EndConfig, run_end2end_test

    run_end2end_test(runner, slam_cfg, End2EndConfig(...), out_root, arm_config, split_at)

One full-sequence SLAM run per entry in `priors`, differing ONLY in the depth prior, then evo ATE
-> TSDF -> Sim(3) align -> eval_recon plus render metrics recomputed per frame, all split
seen/unseen at the frame the adapter's training data ended.

An arm is a prior GENERATOR - 'omnidata', 'vggt_base', or the handoff directory of any adapt run or
one of its checkpoints - and its output directory is INFERRED from that (config.py:arm_name), never
typed. So an arm is standalone and reusable: a scene's omnidata baseline is run once, and a later
comparison including it finds it rather than repeating it. `out_root` is
outputs/test/end2end/<scene>; End2EndConfig holds knobs only, never a location.

report.py's two functions are pure formatting over evaluate()'s dicts, so they can be re-run
against results.json files already on disk without a GPU:

    from adaslam.end2end.report import compare, print_report

gaussian.utils.loss_utils and midas.omnidata live in hislam2/, which adaslam/__init__.py put on
sys.path before this file could run.
"""
from .config import End2EndConfig
from .prior import VggtPrior
from .stage import run_end2end_test

__all__ = ['End2EndConfig', 'VggtPrior', 'run_end2end_test']
