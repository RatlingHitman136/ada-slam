"""PriorTestConfig, and where the comparison's seen/unseen boundary comes from.

An arm here is the same thing it is in end2end - a depth-prior generator, named by the same
rule. That rule is IMPORTED, not restated: one edit to end2end/config.py:arm_name changes both
test kinds' directory names, and `test/end2end/<scene>/omni` and `test/prior/<scene>/omni` stay the
same generator measured two ways.
"""
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..adapt import LoRAConfig
from ..end2end.config import SENTINELS, adapter_path, arm_name

__all__ = ['PriorTestConfig', 'arm_split_at', 'resolve_split']


def arm_split_at(spec):
    """The frame this arm's adapter stopped training at, or None if it has no such frame.

    Sentinels have none. For an adapter, in order:
      1. config.json['split_at'], written by adapt/trainer.py
      2. the last frame index in its extract run's traj_full.txt, + 1 - exact, because the extract
         run streams exactly `extract_length` frames and traj_full covers all of them (a
         poses_slam.txt keyframe would land a few frames short)
      3. None, with a warning, when that extract directory is gone
    """
    if spec in SENTINELS:
        return None
    cfg_path = os.path.join(str(spec).rstrip('/'), 'config.json')
    if not os.path.exists(cfg_path):
        print(f'WARNING: {cfg_path} missing - {arm_name(spec)} scored without a seen/unseen split')
        return None
    recorded = json.load(open(cfg_path))
    if recorded.get('split_at') is not None:
        return int(recorded['split_at'])

    scene = recorded.get('scene')
    traj = os.path.join(scene, 'traj_full.txt') if scene else None
    if not traj or not os.path.exists(traj):
        print(f'WARNING: {arm_name(spec)} records no split_at and its extract run '
              f'({scene}) is gone - scored without a seen/unseen split')
        return None
    return int(np.loadtxt(traj)[-1, 0]) + 1


def resolve_split(priors):
    """(split_at for the whole table, {spec: its own split}) - the table's comes from the FIRST
    adapter in `priors`.

    One boundary for every arm, sentinels included, because a sentinel's seen/unseen rows are the
    CONTROL: "the adapter is worse on unseen frames" means nothing until you know whether every
    prior is worse there. If omnidata's unseen L1 is up too, the back half of the sequence is simply
    harder and the adapter has not degraded at all.

    With no adapter in the list there is no boundary to speak of and every arm is scored on `all`.
    """
    own = {spec: arm_split_at(spec) for spec in priors}
    table = next((own[s] for s in priors if own[s] is not None), None)
    return table, own


@dataclass(frozen=True)
class PriorTestConfig:
    """What to score and how. No location: the stage receives out_root.

    priors[0] is the report's baseline column, as in End2EndConfig.
    """
    priors: Tuple[str, ...]          # sentinels and/or adapt handoff dirs, in the table's order
    gt_depths: str                   # REQUIRED here - without GT there is nothing to score
    depth_png_scale: float           # 16-bit depth PNG scale used across the repo
    # ---------------------------------------------------------------- masking
    eval_min_depth: float            # metres; GT below this is sensor noise, not geometry
    eval_max_depth: float            # metres; TUM's Kinect is unreliable far out
    # Pixels kept per frame for the metrics, sampled with a fixed seed. Every metric - per-frame AND
    # the one global scale - is computed from this one sample, so their RATIO (the consistency
    # index) is internally consistent rather than mixing an exact numerator with a sampled
    # denominator. 0 = keep every valid pixel, which needs the whole sequence in RAM.
    eval_samples_per_frame: int
    seed: int
    # ---------------------------------------------------------------- the VGGT arms
    lora: LoRAConfig                 # the 'vggt_base' arm has no adapter to read a structure from
    omni_normal_ckpt: str            # VggtPrior's normal branch; the prior test discards its output
    omni_normal_hw: Tuple[int, int]

    def __post_init__(self):
        for name in ('priors', 'omni_normal_hw'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.priors:
            raise ValueError('priors is empty: nothing to test')
        names = [arm_name(s) for s in self.priors]
        clash = {n for n in names if names.count(n) > 1}
        if clash:
            raise ValueError(f'these priors infer the same arm directory {sorted(clash)}: '
                             f'{[s for s, n in zip(self.priors, names) if n in clash]}')
        if not 0 <= self.eval_min_depth < self.eval_max_depth:
            raise ValueError(f'need 0 <= eval_min_depth ({self.eval_min_depth}) < eval_max_depth '
                             f'({self.eval_max_depth})')
        if self.eval_samples_per_frame < 0:
            raise ValueError('eval_samples_per_frame must be >= 0 (0 = every valid pixel)')

    def check_priors_exist(self):
        """Same contract as End2EndConfig's, and called by the stage for the same reasons."""
        if not os.path.isdir(self.gt_depths):
            raise SystemExit(f'gt_depths={self.gt_depths!r} is not a directory; the prior test '
                             f'scores against ground-truth depth and cannot run without it')
        for spec in self.priors:
            if spec in SENTINELS:
                continue
            if not os.path.isdir(spec):
                raise SystemExit(f'{spec!r} is neither {tuple(SENTINELS)} nor a directory; an '
                                 f"adapter arm is the adapt stage's handoff directory")
            if not os.path.exists(adapter_path(spec)):
                raise SystemExit(f'{adapter_path(spec)} not found - {spec} is not an adapter '
                                 f'directory (run the adapt stage, or point at a checkpoint)')

    def eval_spec(self):
        """What a cached frames.csv must have been built with to be reusable.

        Only the things that change the PER-FRAME numbers. split_at is deliberately absent: it
        changes no per-frame value, only which rows get averaged together (metrics.py:aggregate).
        """
        return {'gt_depths': self.gt_depths, 'depth_png_scale': self.depth_png_scale,
                'eval_min_depth': self.eval_min_depth, 'eval_max_depth': self.eval_max_depth,
                'eval_samples_per_frame': self.eval_samples_per_frame, 'seed': self.seed}

    def arm_dirs(self, out_root):
        """{spec: directory} - what main() prints before the run."""
        return {spec: f'{out_root}/{arm_name(spec)}' for spec in self.priors}
