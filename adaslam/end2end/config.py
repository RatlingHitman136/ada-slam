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
import re
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

# The spec modifiers (14, 14.9): any base spec above may carry a far-field transform, e.g.
# 'omnidata@ceil2', 'vggt_base@ceil1p5', 'outputs/adapt/<scene>/<name>@ceil2', 'vggt_base@ped1p3'.
# A modifier is part of the SPEC, never a config field, because the arm directory is inferred from
# the spec - a transformed arm must name a different directory than its untransformed parent or
# the two would silently overwrite each other, which is the trap 9.3 exists to prevent.
#
#   @ceil<tag>   depth <- min(depth, tag x frame median)          prior.py:ceil_clamp
#   @soft<tag>   depth <- 1/hypot(1/depth, median(1/depth)/tag)   prior.py:soft_saturate
#   @ped<tag>    depth <- 1/(1/depth + median(1/depth)/tag)       prior.py:pedestal_shift
#   @mask<tag>   depth <- 0 where depth > tag x frame median      prior.py:mask_far
#
# THE TAGS ARE NOT ALL THE SAME UNIT and must not be read against each other - see
# pedestal_shift's docstring. @ceil, @soft and @mask all bound the served depth at `tag` x the
# frame's own median depth and are directly comparable at one tag; @ped's tag is in the same
# units only in the disparity OFFSET it adds, and realises its bound at `tag + 1` post-shift
# medians. Any may appear on one spec, written in MOD_ORDER and APPLIED in that order (left to
# right, so the rightmost is outermost): 'vggt_base@ceil1p5@ped2'.
#
# @ceil, @soft and @ped are three members of ONE family - served disparity
# q' = (q^k + b^k)^(1/k) with b = median(q)/tag - at k = inf (the clamp), k = 2 (quadrature) and
# k = 1 (the pedestal). They differ only in how much of the NEAR field they pay for the same far
# bound: at tag 1.45 a pixel at half the frame's median depth is moved 0% by @ceil, 6% by @soft
# and 26% by @ped. @mask is the fourth option and the odd one - it asserts NOTHING beyond the
# bound (disps_prior 0, so JDSA's m gates the pixel out and eta damps it) rather than pinning it.
_MOD_RES = {'ceil': re.compile(r'^ceil(\d+(?:p\d+)?)$'),
            'soft': re.compile(r'^soft(\d+(?:p\d+)?)$'),
            'ped': re.compile(r'^ped(\d+(?:p\d+)?)$'),
            'mask': re.compile(r'^mask(\d+(?:p\d+)?)$')}
# ceil/soft/ped compress the tail, so they go first and in decreasing sharpness; mask is last
# because it deletes pixels and must threshold whatever the chain before it produced.
MOD_ORDER = ('ceil', 'soft', 'ped', 'mask')


def ceil_tag(ratio):
    """The directory-safe spelling of a modifier ratio: 2.0 -> '2', 1.5 -> '1p5'."""
    return f'{ratio:g}'.replace('.', 'p')


def split_mods(spec):
    """(base_spec, {kind: ratio}) - strip every '@<kind><tag>' modifier off a prior spec.

    Split from the RIGHT, repeatedly, and only while the tail matches a known modifier exactly, so
    a path that happens to contain '@' still reads as a plain spec. Tags are canonicalised through
    ceil_tag on the way back out ('ceil1p50' names the same arm as 'ceil1p5').

    The written order is checked against MOD_ORDER rather than accepted in any order, because the
    modifiers do NOT commute - clamping then shifting is not shifting then clamping - so one
    spelling per composition keeps a spec and an arm directory one-to-one.
    """
    s, stripped, found = str(spec), [], {}
    while True:
        head, sep, tail = s.rpartition('@')
        if not sep:
            break
        hit = next(((k, rx.match(tail)) for k, rx in _MOD_RES.items() if rx.match(tail)), None)
        if hit is None:
            break
        kind, m = hit
        if kind in found:
            raise ValueError(f'{spec!r} carries two @{kind} modifiers; one arm has one ratio')
        ratio = float(m.group(1).replace('p', '.'))
        # THE FLOORS DIFFER BECAUSE THE TAGS ARE NOT ALL IN THE SAME UNITS (pedestal_shift's
        # docstring). A ceiling at or below 1.0 clamps at the median itself and is degenerate,
        # and a mask at or below 1.0 would delete HALF the frame or more, which divides by zero
        # in track_frontend.py's dscale init (it takes disps_prior.median() over ALL pixels).
        # A pedestal's ratio is in PRE-shift median units while the bound it realises sits at
        # `ratio + 1` POST-shift medians, so @ped0p5 bounds the frame at 1.5x its own median -
        # exactly as gently as @ceil1p5, and without flattening a pixel. Sub-1 ratios are the
        # only region where the pedestal ALONE reaches the tail the clamp reaches, so refusing
        # them made the clamp look necessary when it was the validator doing the work. Only
        # ratio <= 0 is meaningless: it is a negative disparity offset, not a bound. @soft's
        # tag bounds the depth like a ceiling's but nothing is clipped at it, so a sub-1 ratio
        # is meaningful there too.
        FLOORS = {'ceil': 1.0, 'mask': 1.0, 'soft': 0.0, 'ped': 0.0}
        WHY = {'ceil': 'a no-op and the arm would be a misleading duplicate of {head!r}',
               'mask': 'at or below the median, which deletes half the frame and divides by '
                       'zero in the dscale init',
               'soft': 'a negative disparity offset, which is not a bound at all',
               'ped': 'a negative disparity offset, which is not a bound at all'}
        floor = FLOORS[kind]
        if ratio <= floor:
            raise ValueError(f'{spec!r}: a @{kind} ratio must exceed {floor:g} - at {ratio:g} '
                             f'it is ' + WHY[kind].format(head=head))
        found[kind] = ratio
        stripped.append(kind)
        s = head
    written = list(reversed(stripped))
    expect = [k for k in MOD_ORDER if k in found]
    if written != expect:
        raise ValueError(f'{spec!r}: modifiers do not commute, so they have one spelling - write '
                         f'them as {"@".join([""] + expect)[1:]!r} order, i.e. '
                         f'{s + "".join("@" + k + ceil_tag(found[k]) for k in expect)!r}')
    return s, found


def parse_ceil(spec):
    """(base_spec, ceil_ratio | None) - the ceiling half of split_mods, kept for the call sites
    that only ever cared about the base (adapter_path, check_priors_exist). `base_spec` has EVERY
    modifier stripped, not just the ceiling.
    """
    base, mods = split_mods(spec)
    return base, mods.get('ceil')


def arm_name(spec):
    """The directory one prior generator is scored into - INFERRED, never typed (7.1).

        'omnidata' -> omni    'vggt_base' -> base    .../lr1e4 -> lr1e4
        .../lr1e4/checkpoints/epoch_005 -> lr1e4_chkp_005
        'omnidata@ceil2' -> omni_ceil2   .../lr1e4@ceil1p5 -> lr1e4_ceil1p5
        'vggt_base@ped1p3' -> base_ped1p3   'vggt_base@ceil1p5@ped2' -> base_ceil1p5_ped2

    Inferring is what makes an arm reusable: one adapter always scores into one directory.
    """
    spec, mods = split_mods(spec)
    if spec in SENTINELS:
        name = SENTINELS[spec]
    else:
        head, tail = os.path.split(str(spec).rstrip('/'))
        if os.path.basename(head) == ADAPT_CKPT_SUBDIR:
            # a checkpoint means nothing without the adapter it belongs to, so the name carries both
            adapter = os.path.basename(os.path.dirname(head))
            epoch = tail[len(_CKPT_PREFIX):] if tail.startswith(_CKPT_PREFIX) else tail
            name = f'{adapter}_chkp_{epoch}'
        else:
            name = tail
    return name + ''.join(f'_{k}{ceil_tag(mods[k])}' for k in MOD_ORDER if k in mods)


def adapter_path(spec):
    """The .safetensors inside a spec's directory, or None for a sentinel.

    from_adapter takes the FILE, a spec names the DIRECTORY; resolved here, once.
    """
    spec, _ = parse_ceil(spec)
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
            base, _ = parse_ceil(spec)             # a modifier changes the arm, not what must exist
            if base in SENTINELS:
                continue
            if not os.path.isdir(base):
                raise SystemExit(f'{base!r} is neither {tuple(SENTINELS)} nor a directory; an '
                                 f"adapter arm is the adapt stage's handoff directory")
            if not os.path.exists(adapter_path(base)):
                raise SystemExit(f'{adapter_path(base)} not found - {base} is not an adapter '
                                 f'directory (run the adapt stage, or point at a checkpoint)')

    def arm_dirs(self, out_root):
        """{spec: directory} - what main() prints before a two-hour run."""
        return {spec: f'{out_root}/{arm_name(spec)}' for spec in self.priors}
