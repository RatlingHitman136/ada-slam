"""PriorTestConfig, and where the comparison's seen/unseen boundary comes from.

arm_name is IMPORTED from end2end, not restated, so both test kinds name a scene's arms identically.
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
    """The frame this arm's adapter stopped training at; None for a sentinel.

    In order: config.json['split_at'] | its extract run's last traj_full.txt frame + 1 | None.
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
    """(the table's split_at, {spec: its own}) - the table's comes from the FIRST adapter.

    One boundary for every arm, sentinels included: they are the CONTROL. If omnidata degrades
    past the split too, the back of the sequence is simply harder (9.2.2).
    """
    own = {spec: arm_split_at(spec) for spec in priors}
    table = next((own[s] for s in priors if own[s] is not None), None)
    return table, own


@dataclass(frozen=True)
class PriorTestConfig:
    """What to score and how. No location: the stage receives out_root."""
    priors: Tuple[str, ...]          # sentinels and/or adapt handoff dirs; [0] is the baseline
    gt_depths: str                   # REQUIRED here - without GT there is nothing to score
    depth_png_scale: float           # 16-bit depth PNG scale used across the repo
    # ---------------------------------------------------------------- masking
    eval_min_depth: float            # m; GT below this is sensor noise, not geometry
    eval_max_depth: float            # m; TUM's Kinect is unreliable far out
    # every metric comes from this one sample, so the consistency index is internally consistent
    eval_samples_per_frame: int      # pixels kept per frame; 0 = all valid (needs the RAM)
    seed: int
    # ---------------------------------------------------------------- the VGGT arms
    lora: LoRAConfig                 # the 'vggt_base' arm has no adapter to read a structure from
    omni_normal_ckpt: str            # VggtPrior's normal branch; this test discards its output
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

        Only what changes the PER-FRAME numbers - split_at changes none, just which rows average.
        """
        return {'gt_depths': self.gt_depths, 'depth_png_scale': self.depth_png_scale,
                'eval_min_depth': self.eval_min_depth, 'eval_max_depth': self.eval_max_depth,
                'eval_samples_per_frame': self.eval_samples_per_frame, 'seed': self.seed}

    def arm_dirs(self, out_root):
        """{spec: directory} - what main() prints before the run."""
        return {spec: f'{out_root}/{arm_name(spec)}' for spec in self.priors}
