"""TestConfig - one A/B experiment.

No field carries a default, as everywhere in ada-slam/. Fields SlamConfig already declares are not
repeated here: run_ab_test receives both, so the stream, the calibration and the resolution have
exactly one description and the arms cannot disagree about them.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from adapt import LoRAConfig

ARM_DIRS = {'omnidata': 'omnidata', 'vggt_lora': 'vggt', 'vggt_base': 'vggt_base'}
# short names for the comparison table's column headers; the full label goes in ab_results.json
ARM_NAMES = {'omnidata': 'Omnidata', 'vggt_lora': 'VGGT+LoRA', 'vggt_base': 'VGGT-base'}


@dataclass(frozen=True)
class TestConfig:
    """The arms and everything scoring them needs. arms[0] is compare()'s baseline column."""
    arms: Tuple[str, ...]            # subset of ARM_DIRS, in the order the table prints them
    out_root: str                    # per-arm dir: out_root/<scene>_<ARM_DIRS[arm]>
    scene: str                       # names those directories, nothing else
    length: int                      # 100000 = whole sequence
    buffer: int
    # ---------------------------------------------------------------- ground truth
    gt_traj: str                     # evo_ape's reference trajectory
    gt_mesh: Optional[str]           # None -> skip TSDF + eval_recon (TUM ships no GT mesh)
    gt_depths: Optional[str]         # Hi2's gtdepthdir AND split_render_metrics' reference. Here
                                     # the masking is correct: depth cannot be scored without GT
    depth_png_scale: float           # 16-bit depth PNG scale used across the repo
    # ---------------------------------------------------------------- mesh metric
    voxel_size: float                # pinned for ALL arms - differing sizes are not comparable
    voxel_fallbacks: Tuple[float, ...]   # marching cubes OOMs on a busy shared GPU
    mesh_weight: float
    # ---------------------------------------------------------------- the VGGT arms
    # Declared here because the 'vggt_base' arm has no adapter to read a structure back from and
    # so silently takes these values. Making that a field of the TEST config rather than a global
    # is the point: it is the only place the un-adapted arm's input resolution is stated.
    lora: LoRAConfig
    omni_normal_ckpt: str            # normals stay Omnidata, so depth is the only variable
    omni_normal_hw: Tuple[int, int]

    def __post_init__(self):
        # normalise, so a config rebuilt from JSON (lists) compares equal to a hand-written one
        for name in ('arms', 'voxel_fallbacks', 'omni_normal_hw'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.arms:
            raise ValueError('arms is empty: nothing to test')
        unknown = [a for a in self.arms if a not in ARM_DIRS]
        if unknown:
            raise ValueError(f'unknown arm(s) {unknown}; choose from {tuple(ARM_DIRS)}')
        if len(set(self.arms)) != len(self.arms):
            raise ValueError(f'duplicate arm in {self.arms}: each writes to one directory')
        if any(v <= 0 for v in (self.voxel_size, *self.voxel_fallbacks)):
            raise ValueError('voxel sizes must be > 0')

    def out_dir(self, arm):
        return f'{self.out_root}/{self.scene}_{ARM_DIRS[arm]}'
