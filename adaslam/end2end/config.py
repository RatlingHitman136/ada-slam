"""End2EndConfig, and the prior-generator specs an end2end comparison is made of.

An arm is a DEPTH-PRIOR GENERATOR, not a fixed name. Three kinds exist and all are given the same
way, as one entry in `priors`:

    'omnidata'                                            upstream's prior, the baseline
    'vggt_base'                                           stock VGGT-1B, no adapter
    outputs/adapt/<scene>/<aname>                          an adapt stage's handoff directory
    outputs/adapt/<scene>/<aname>/checkpoints/epoch_005     one of its checkpoints

That is the point of the refactor. The stage used to enumerate three arms and take ONE adapter
path, so "which adapter is the vggt_lora arm" was a fact about the run rather than about the arm,
and comparing two adapters - or an adapter against its own checkpoints - meant re-running the set
and moving directories by hand.

No field carries a default, as everywhere in adaslam/. Fields SlamConfig already declares are not
repeated here: run_end2end_test receives both, so the stream, the calibration and the resolution
have exactly one description and the arms cannot disagree about them.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from ..adapt import LoRAConfig
from ..common import ADAPT_CKPT_SUBDIR

OMNIDATA, VGGT_BASE = 'omnidata', 'vggt_base'
# The two priors that are not adapters, and the directory each is scored into.
SENTINELS = {OMNIDATA: 'omni', VGGT_BASE: 'base'}

ADAPTER_FILE = 'adapter.safetensors'
_CKPT_PREFIX = 'epoch_'


def arm_name(spec):
    """The directory one prior generator is scored into - INFERRED, never typed.

        'omnidata'                                        -> omni
        'vggt_base'                                       -> base
        outputs/adapt/<scene>/lr1e4                        -> lr1e4
        outputs/adapt/<scene>/lr1e4/checkpoints/epoch_005  -> lr1e4_chkp_005

    Inferring rather than naming is what makes an arm reusable: one adapter always scores into one
    directory, so a scene's omnidata baseline is run once and every later comparison finds it. The
    scene is already the parent directory, so the name only has to separate adapters within a scene.
    """
    if spec in SENTINELS:
        return SENTINELS[spec]
    head, tail = os.path.split(str(spec).rstrip('/'))
    if os.path.basename(head) == ADAPT_CKPT_SUBDIR:
        # a checkpoint means nothing without the adapter it belongs to, so the name carries both
        adapter = os.path.basename(os.path.dirname(head))
        epoch = tail[len(_CKPT_PREFIX):] if tail.startswith(_CKPT_PREFIX) else tail
        return f'{adapter}_chkp_{epoch}'
    return tail


def adapter_path(spec):
    """The .safetensors inside a spec's directory, or None for a sentinel.

    LoRAVGGT.from_adapter takes the FILE and reads config.json beside it (adapt/model.py:71), while
    a spec names the DIRECTORY - "the adapt stage's handoff artifacts". Resolving that asymmetry
    here keeps it from spreading through the stage.
    """
    return None if spec in SENTINELS else f'{str(spec).rstrip("/")}/{ADAPTER_FILE}'


@dataclass(frozen=True)
class End2EndConfig:
    """The prior generators to compare and everything scoring them needs.

    priors[0] is compare()'s baseline column - put 'omnidata' there unless you mean otherwise.
    """
    priors: Tuple[str, ...]          # sentinels and/or adapt handoff dirs, in the table's order
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
    # so silently takes these values. Making that a field of this config rather than a global is
    # the point: it is the only place the un-adapted arm's input resolution is stated.
    lora: LoRAConfig
    omni_normal_ckpt: str            # normals stay Omnidata, so depth is the only variable
    omni_normal_hw: Tuple[int, int]

    def __post_init__(self):
        # normalise, so a config rebuilt from JSON (lists) compares equal to a hand-written one
        for name in ('priors', 'voxel_fallbacks', 'omni_normal_hw'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.priors:
            raise ValueError('priors is empty: nothing to test')

        # Two specs inferring one name would write into one directory: the second arm's SLAM run
        # would be skipped as already done, and the comparison would score the first against
        # itself. Cheap to check here, invisible in the output.
        names = [arm_name(s) for s in self.priors]
        clash = {n for n in names if names.count(n) > 1}
        if clash:
            raise ValueError(f'these priors infer the same arm directory {sorted(clash)}: '
                             f'{[s for s, n in zip(self.priors, names) if n in clash]}')

        if any(v <= 0 for v in (self.voxel_size, *self.voxel_fallbacks)):
            raise ValueError('voxel sizes must be > 0')

    def check_priors_exist(self):
        """Every adapter spec is a real handoff directory. Called by the STAGE, not __post_init__.

        Deliberately not a construction-time check, for two reasons. This config is built in
        run_pipeline.py's PARAMETERS block, which is re-executed in every spawned reader child and
        must not touch the filesystem; and it is built before main()'s os.chdir, so its relative
        paths would resolve against whatever directory the script was invoked from. On top of that,
        an adapter listed here legitimately does not exist yet when the whole pipeline runs - the
        adapt stage is about to create it.
        """
        for spec in self.priors:
            if spec in SENTINELS:
                continue
            if not os.path.isdir(spec):
                raise SystemExit(f'{spec!r} is neither {tuple(SENTINELS)} nor a directory; an '
                                 f"adapter arm is the adapt stage's handoff directory")
            if not os.path.exists(adapter_path(spec)):
                raise SystemExit(f'{adapter_path(spec)} not found - {spec} is not an adapter '
                                 f'directory (run the adapt stage, or point at a checkpoint)')

    def arm_dirs(self, out_root):
        """{spec: directory} for this comparison - what main() prints before a two-hour run."""
        return {spec: f'{out_root}/{arm_name(spec)}' for spec in self.priors}
