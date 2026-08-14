"""End2EndConfig, and the prior-generator specs a comparison is made of.

An arm is a DEPTH-PRIOR GENERATOR, not a fixed name - every kind is one entry in `priors`:

    'omnidata'                                          upstream's prior, the baseline
    'vggt_base'                                         stock VGGT-1B, no adapter
    'omnidata_dense'                                    upstream's prior at DENSIFIED keyframing
    outputs/adapt/<scene>/<aname>                       an adapt stage's handoff directory
    outputs/adapt/<scene>/<aname>/checkpoints/epoch_005  one of its checkpoints

`omnidata_dense` is the odd one and the docstring has to say so: it is not a different prior at all,
it is the SAME Omnidata prior tracked at the extract's keyframe density. Density is a property of
the tracking config, and run_end2end_test hands every arm ONE arm_config (stage.py) - so this stage
cannot produce that arm, and listing it here only ever REUSES one already on disk. It is produced by
scripts/dense_kf_arm.py; stage.py:make_prior refuses to build it rather than silently running a
stock-density arm into a directory named for a dense one. Reuse needs SKIP_EXISTING, since that is
what reaches the cache before make_prior is called.
"""
import os
from dataclasses import dataclass
from typing import Tuple

from ..adapt import LoRAConfig
from ..common import ADAPT_CKPT_SUBDIR, ADAPTER_FILE

OMNIDATA, VGGT_BASE, OMNIDATA_DENSE = 'omnidata', 'vggt_base', 'omnidata_dense'
# The priors that are not adapters, and the directory each is scored into. Everything that reads
# this vocabulary - arm_name, adapter_path, check_priors_exist, arm_dirs, priortest's arm_split_at
# and export_end2end_results.py's SENTINEL_ARMS - is generic over the dict, so a sentinel is one
# entry. make_prior is the deliberate exception: see the docstring above on OMNIDATA_DENSE.
SENTINELS = {OMNIDATA: 'omni', VGGT_BASE: 'base', OMNIDATA_DENSE: 'omni_dense'}

# ADAPTER_FILE lives in common.py: adapt/stage.py needs the same name to warm-start from an
# adapter, and it cannot import this module - end2end imports adapt, so that would be a cycle.
_CKPT_PREFIX = 'epoch_'


def arm_name(spec):
    """The directory one prior generator is scored into - INFERRED, never typed (7.1).

        'omnidata' -> omni    'vggt_base' -> base    .../lr1e4 -> lr1e4
        .../lr1e4/checkpoints/epoch_005 -> lr1e4_chkp_005

    Inferring is what makes an arm reusable: one adapter always scores into one directory.
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

    from_adapter takes the FILE, a spec names the DIRECTORY; resolved here, once.
    """
    return None if spec in SENTINELS else f'{str(spec).rstrip("/")}/{ADAPTER_FILE}'


@dataclass(frozen=True)
class End2EndConfig:
    """The prior generators to compare and everything scoring them needs."""
    priors: Tuple[str, ...]          # sentinels and/or adapt handoff dirs; [0] is the baseline
    length: int                      # 100000 = whole sequence
    buffer: int
    gt_traj: str                     # evo_ape's reference - ATE is the only metric (11)
    lora: LoRAConfig                 # the 'vggt_base' arm has no adapter to read a structure from
    omni_normal_ckpt: str            # normals stay Omnidata, so depth is the only variable
    omni_normal_hw: Tuple[int, int]

    def __post_init__(self):
        # normalise, so a config rebuilt from JSON (lists) compares equal to a hand-written one
        for name in ('priors', 'omni_normal_hw'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.priors:
            raise ValueError('priors is empty: nothing to test')

        # two specs inferring one name would share a directory, and the comparison would score the
        # first arm against itself
        names = [arm_name(s) for s in self.priors]
        clash = {n for n in names if names.count(n) > 1}
        if clash:
            raise ValueError(f'these priors infer the same arm directory {sorted(clash)}: '
                             f'{[s for s, n in zip(self.priors, names) if n in clash]}')

    def check_priors_exist(self):
        """Every adapter spec is a real handoff directory. Called by the STAGE, not __post_init__.

        The PARAMETERS block builds this before chdir and in every spawned child, so it must not
        touch the filesystem - and these adapters may not exist yet when the pipeline starts.
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
        """{spec: directory} - what main() prints before a two-hour run."""
        return {spec: f'{out_root}/{arm_name(spec)}' for spec in self.priors}
