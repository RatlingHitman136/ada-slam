"""OnlineConfig - one continuous-adaptation run (13).

A config of its own rather than AdaptConfig: that one carries a dozen fields this run never reads
(kf_fraction, val_source, train_frac, eval_*, keep_best) whose __post_init__ would force
meaningless choices. Field names that mean the same thing as AdaptConfig's are deliberately spelled
the same; no field carries a default (9.5).
"""
from dataclasses import dataclass

# The two adapt/trainer.py:schedule styles that are meaningful live. 'normal' is not: an epoch over
# a fixed train set does not exist while the set is still arriving.
ONLINE_STYLES = ('online', 'wonline')
WARMUP_PRIORS = ('omnidata',   # upstream's own prior - a genuinely different model
                 'self')       # the same VGGT this run adapts, frozen until handover_kf


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
        if self.context_kf < 0:
            raise ValueError(f'context_kf={self.context_kf} must be >= 0')
        if self.steps_per_kf < 0:
            raise ValueError(f'steps_per_kf={self.steps_per_kf} must be >= 0 (0 = never step, the '
                             f'null-op arm)')
        if self.checkpoint_every_kf < 0:
            raise ValueError(f'checkpoint_every_kf={self.checkpoint_every_kf} must be >= 0 '
                             f'(0 = off)')
