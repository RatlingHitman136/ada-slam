"""OnlineConfig - one continuous-adaptation run (13).

A config of its own rather than AdaptConfig: that one carries a dozen fields this run never reads
(kf_fraction, val_source, train_frac, eval_*, keep_best) whose __post_init__ would force
meaningless choices. Field names that mean the same thing as AdaptConfig's are deliberately spelled
the same; no field carries a default (9.5).
"""
from dataclasses import dataclass
from typing import Optional

# The two adapt/trainer.py:schedule styles that are meaningful live. 'normal' is not: an epoch over
# a fixed train set does not exist while the set is still arriving.
ONLINE_STYLES = ('online', 'wonline')
WARMUP_PRIORS = ('omnidata',   # upstream's own prior - a genuinely different model
                 'self')       # the same VGGT this run adapts, frozen until handover_kf

# What the loss gate thresholds are read against. 'rel' is loss / median target depth
# (adapt/losses.py:relative_loss); 'raw' is the depth loss as depth_loss returns it, in the
# tracker's own units. 'raw' is offered because it is the obvious thing to try, not because it is
# the sounder one - see gate_metric below.
GATE_METRICS = ('rel', 'raw')


@dataclass(frozen=True)
class OnlineConfig:
    """Adapting the depth prior DURING the SLAM run that supervises it."""
    # ---------------------------------------------------------------- warm-up
    # TWO GATES, deliberately separate: warmup_kf is when the adapter starts LEARNING, handover_kf
    # is when it starts SERVING. They were one field, and that made the knob untunable - raising it
    # bought a longer fallback-served phase and paid for it with an equally delayed adaptation
    # start, so the two effects cancelled (rellis_00000: 10 -> 26.38, 12 -> 26.34, 13 -> 26.01,
    # 15 -> 27.28, no trend). Split, the fallback keeps driving while the adapter trains in the
    # background on what the tracker has already settled, and that costs nothing: the optimiser
    # steps run either way. handover_kf == warmup_kf reproduces the old single-gate behaviour
    # exactly, which is what every adapter written before this field did.
    warmup_kf: int           # keyframes before the first optimiser step: it lands at warmup_kf + 1
    handover_kf: int         # keyframes served by the FALLBACK prior; VGGT serves from here on.
                             # Must be >= warmup_kf. The frame it lands on is recorded as
                             # `warmup_end_frame` - that key name predates the split and is kept,
                             # because adapters already on disk are read through it (9.5).
    warmup_prior: str        # 'omnidata' | 'self' - see WARMUP_PRIORS above. Note the split is
                             # INERT at 'self': both branches are then the same model object, so
                             # below handover_kf it serves weights that are already adapting.

    # ---------------------------------------------------------------- schedule
    # Same vocabulary as adapt/trainer.py:schedule, so 12.1's adapt_cost table still applies.
    adapt_style: str         # 'online' = the arriving keyframe alone | 'wonline' = sliding window
    steps_per_kf: int        # 'online': optimiser steps on the arrival, one keyframe per step.
                             # 'wonline': shuffled batched passes over the window.
    window_size: int         # 'wonline' ONLY: the arrival + the window_size-1 keyframes before it
    batch_size: int          # 'wonline' ONLY - a keyframe arrives alone in 'online'
    lag: int                 # keyframes back from counter.value the target is taken. 2 matches
                             # track_frontend.py:65, the repo's own "settled enough to hand
                             # downstream" line: __update returns arange(ii.min(), t1-1).

    # ---------------------------------------------------------------- serving
    # The far-field ceiling (14): depth <- min(depth, ceil_ratio * frame median) on everything
    # this prior SERVES - both branches, warm-up fallback included, so the arm's serving is
    # "prior + ceiling" throughout. 1.0 = off, and off is exactly the pre-knob behaviour:
    # ceil_clamp returns before any tensor op, so every live run recorded before this field exists
    # stays comparable. Frozen/reference arms spell the same thing as a spec modifier instead
    # ('vggt_base@ceil2'), because their arm directory is inferred from the spec; this arm is
    # named by ONLINE_NAME, so it alone carries a knob.
    ceil_ratio: float
    # And the TRAINING side of the same ceiling (14.6): False = the target (SLAM depth) is never
    # clamped, the original behaviour; True = the target is clamped at the same ceil_ratio over
    # its VALID pixels (target.py:kf_target - zeros stay zero, min() cannot lift them), so the
    # adapter is TAUGHT "never assert past the ceiling" instead of only being served through it.
    # True at ceil_ratio 1.0 is refused: it would silently be a no-op, and a stated instruction
    # that does nothing is the failure mode rule 1 of 9.5 exists to prevent.
    ceil_target: bool
    # The far-field PEDESTAL (14.9): depth <- 1/(1/depth + median(1/depth)/ped_ratio) on the same
    # served depth, both branches, applied AFTER the ceiling (the MOD_ORDER a spec is written in).
    # None = off, and off is exactly the pre-knob behaviour - pedestal_shift returns before any
    # tensor op, so every live run recorded before this field exists stays comparable.
    #
    # NOTE THE OFF SENTINEL DIFFERS FROM ceil_ratio's, and it has to: a ceiling at 1.0 is
    # degenerate so 1.0 can mean off there, while a pedestal at 1.0 is a real (very strong)
    # transform. Off is the absence of a pedestal, which is None, not a ratio.
    # ANY POSITIVE ratio is legal, sub-1 included, and sub-1 is where the transform earns its
    # keep: the bound it realises is `ratio + 1` POST-shift medians (pedestal_shift's docstring),
    # so 0.5 bounds the frame at 1.5x its own median - the same tail @ceil1p5 serves, reached
    # without flattening a pixel.
    # Frozen/reference arms spell this as a spec modifier instead ('vggt_base@ped1p3'), because
    # their arm directory is inferred from the spec; this arm is named by ONLINE_NAME, so it alone
    # carries a knob. There is deliberately no ped_target twin - 14.6 retired ceil_target's
    # premise, and an unexercised lever is worse than none.
    ped_ratio: Optional[float]

    # ---------------------------------------------------------------- sample construction
    context_kf: int          # previous KEYFRAMES appended after the target. 0 = monocular and
                             # depth-only; >0 also supervises poses from video.poses. Not
                             # non-keyframes: those images are on Hi2, which the extractor cannot
                             # reach.
    stream_res: int          # must equal SlamConfig.stream_res - the tracking pixel budget

    # ---------------------------------------------------------------- optimisation
    lr: float
    weight_decay: float
    grad_clip: float
    lambda_pose: float       # unread at context_kf=0, where pose_loss returns zeros
    coupled_scale: bool      # True = the pose scale is reused by the depth loss
    min_mask_pixels: int     # below this a sample contributes no depth gradient
    seed: int
    log_every: int           # steps between log lines; 1 = every step

    # ---------------------------------------------------------------- supervision mask
    # Same names and meanings as ExtractConfig's - extract/export.py:confidence_mask reads them.
    mask_filter_thresh: float    # depth_filter disparity agreement
    mask_min_count: int          # min agreeing neighbours out of 6
    mask_min_disp_ratio: float   # drop pixels below this fraction of the frame's mean disparity

    # ---------------------------------------------------------------- the loss gate
    # SKIP an arrival whose newest keyframe already fits, or whose target is broken. Both bounds
    # are on the RELATIVE loss (adapt/losses.py:relative_loss), never the raw one: the raw loss
    # carries the tracker's shrinking depth unit, so a raw threshold silently becomes an
    # early-stopping schedule instead of a fit test.
    #
    # An UPPER bound is the half the evidence supports. The catastrophic units in a run are its
    # HIGHEST-loss ones - rellis_00000 `more_chkp` carries two at 490x and 1902x the median
    # relative loss, and the ATE degrades 24.704 -> 27.013 across exactly the interval containing
    # them. A gate with only a floor would train on those PREFERENTIALLY, which is backwards.
    # Reference distribution for that scene: median 0.023-0.029, p90 0.044-0.050, p98 0.056.
    gate_metric: str             # 'rel' | 'raw' - which quantity the two bounds are read against.
                                 # BOTH are always measured and logged; this only picks the one
                                 # that decides. Their scales are NOT interchangeable, so the
                                 # thresholds must be re-derived when this changes:
                                 #   rel  median 0.023-0.029, p98 ~0.056, outliers 0.9-55
                                 #   raw  median 0.015-0.026, p98 ~0.083, outliers 0.56-11
                                 # measured over five live runs on rellis_00000.
    gate_lo: float               # 0 = off; skip below this. Already-fit frames.
    gate_hi: float               # 0 = off; skip above this. Broken/degenerate targets.

    # ---------------------------------------------------------------- output
    checkpoint_every_kf: int     # 0 = off; N = a full loadable adapter dir every N adapted units

    def __post_init__(self):
        if self.adapt_style not in ONLINE_STYLES:
            raise ValueError(f'adapt_style={self.adapt_style!r} is not one of {ONLINE_STYLES}. '
                             f"'normal' has no meaning live - there is no fixed train set to make "
                             f'an epoch of.')
        if self.warmup_prior not in WARMUP_PRIORS:
            raise ValueError(f'warmup_prior={self.warmup_prior!r} is not one of {WARMUP_PRIORS}')
        if self.warmup_kf < 1:
            raise ValueError(f'warmup_kf={self.warmup_kf} must be >= 1: keyframe 0 has no settled '
                             f'predecessor to adapt on, so something must serve it')
        if self.handover_kf < self.warmup_kf:
            raise ValueError(f'handover_kf={self.handover_kf} is below warmup_kf='
                             f'{self.warmup_kf}: the adapter cannot serve before it has taken a '
                             f'step. Set them equal for the old single-gate behaviour, or raise '
                             f'handover_kf to let the fallback drive while the adapter trains')
        if self.adapt_style == 'wonline' and self.window_size < 1:
            raise ValueError(f'window_size={self.window_size} must be >= 1 in the wonline style')
        if self.lag < 1:
            raise ValueError(f'lag={self.lag} must be >= 1: the arriving keyframe has not been '
                             f'through BA yet when its prior is extracted')
        if self.ceil_ratio < 1.0:
            raise ValueError(f'ceil_ratio={self.ceil_ratio} must be >= 1.0 (1.0 = off; above it, '
                             f'served depth is clamped at ceil_ratio x the frame median)')
        if self.ceil_target and self.ceil_ratio <= 1.0:
            raise ValueError(f'ceil_target=True at ceil_ratio={self.ceil_ratio} clamps nothing - '
                             f'the target ceiling reuses ceil_ratio, so raise it above 1.0 or '
                             f'set ceil_target=False')
        # explicit, because runconfig._checked waves a value through whenever the literal default
        # is None - so a YAML `ped_ratio: 1e-4`-style string would otherwise reach the comparison
        if self.ped_ratio is not None and (isinstance(self.ped_ratio, bool)
                                           or not isinstance(self.ped_ratio, (int, float))):
            raise ValueError(f'ped_ratio={self.ped_ratio!r} must be a number or null, not '
                             f'{type(self.ped_ratio).__name__} (YAML reads 1e-4 as a string - '
                             f'write 1.3, not "1.3")')
        if self.ped_ratio is not None and self.ped_ratio <= 0.0:
            raise ValueError(f'ped_ratio={self.ped_ratio} must be positive, or null for off. It '
                             f'is the depth the served prior saturates at in units of the '
                             f"frame's PRE-shift median, so at or below 0 it is a negative "
                             f'disparity offset rather than a bound. Ratios BELOW 1 are legal and '
                             f'are the interesting ones - the realised bound is ratio + 1 '
                             f"POST-shift medians, so 0.5 bounds at 1.5x the frame's own median, "
                             f'as gently as ceil_ratio 1.5 and without flattening a pixel. Note '
                             f'the off sentinel is null, NOT 1.0 as ceil_ratio uses - a pedestal '
                             f'at 1.0 is a real transform (14.9)')
        if self.context_kf < 0:
            raise ValueError(f'context_kf={self.context_kf} must be >= 0')
        if self.steps_per_kf < 0:
            raise ValueError(f'steps_per_kf={self.steps_per_kf} must be >= 0 (0 = never step, the '
                             f'null-op arm)')
        if self.checkpoint_every_kf < 0:
            raise ValueError(f'checkpoint_every_kf={self.checkpoint_every_kf} must be >= 0 '
                             f'(0 = off)')
        if self.gate_metric not in GATE_METRICS:
            raise ValueError(f'gate_metric={self.gate_metric!r} is not one of {GATE_METRICS}')
        for name in ('gate_lo', 'gate_hi'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name}={getattr(self, name)} must be >= 0 (0 = off)')
        if 0 < self.gate_hi <= self.gate_lo:
            raise ValueError(f'gate_hi={self.gate_hi} must exceed gate_lo={self.gate_lo}: with '
                             f'both set the gate keeps the BAND between them, so this would skip '
                             f'every arrival and no optimiser step would ever run')

    def served_mods(self):
        """The spec-modifier dict this arm's SERVING is equivalent to (14, 14.9).

        The one place the two serving knobs become the vocabulary end2end/config.py:split_mods
        produces, so a live arm and a frozen '@ceil<tag>@ped<tag>' replay of it are spelled - and
        applied - identically rather than by two hand-kept-in-sync code paths. Off values simply
        do not appear, which is why the two different off sentinels (ceil_ratio 1.0, ped_ratio
        None) stop being visible past this point.
        """
        mods = {}
        if self.ceil_ratio > 1.0:
            mods['ceil'] = self.ceil_ratio
        if self.ped_ratio is not None:
            mods['ped'] = self.ped_ratio
        return mods
