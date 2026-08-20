"""OnlineConfig - one continuous-adaptation run (13).

A config of its own rather than AdaptConfig: that one carries a dozen fields this run never reads
(kf_fraction, val_source, train_frac, eval_*, keep_best) whose __post_init__ would force
meaningless choices. Field names that mean the same thing as AdaptConfig's are deliberately spelled
the same; no field carries a default (9.5).
"""
from dataclasses import dataclass

# imported, not redeclared: adapt/trainer.py:coupling_fit reads cfg.coupling_axis against THIS
# vocabulary, so a second copy could drift out of agreement with the code that consumes it
from ..adapt.config import COUPLING_AXES, DEPTH_LOSSES, JDSA_LATTICES, JDSA_NORMS

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

    # ---------------------------------------------------------------- depth->scale coupling (E3)
    # The same term adapt/config.py documents, ported live. depth_loss aligns scale PER SAMPLE, so
    # L(c*p) = L(p) exactly and the per-frame output scale has identically zero gradient; this puts
    # a gradient on its DEPTH-COUPLED part - s_i varying with how far the scene is - and on nothing
    # else. Fitted over the arrival's WINDOW by adapt/trainer.py:coupling_fit, which is called
    # unchanged: it reads only the four field names below plus min_mask_pixels.
    #
    # 0.0 IS THE OLD LOSS, bit for bit - the statistics pass is skipped entirely.
    #
    # Offline this term is measured, and the measurement carries a warning worth repeating here:
    # its per-sample coefficients sum to zero only over the WHOLE window, so at batch_size <<
    # window_size each optimiser step applies a large uncancelled fragment (30x the depth term at
    # lambda=1, batch_size=2) which grad_clip then renormalises - the depth signal is effectively
    # erased and the scale is pushed around at random. Set batch_size = window_size here.
    coupling_lambda: float   # 0.0 = OFF
    coupling_axis: str       # 'target' = median TARGET depth (recommended: the network does not
                             #            control the axis, so it cannot game the fit)
                             # 'pred'   = median PREDICTED depth; invites range collapse
    coupling_min_var: float  # skip the term when the window has no depth spread (SUM(x~^2) below
                             # this), where the slope would be an ill-conditioned division
    # ---------------------------------------------------------------- WHICH OBJECTIVE (the knob)
    # The same three AdaptConfig documents, with the same meanings: 'normal' is every run before
    # this field existed, 'coupled' adds E3's slope penalty (measured null offline and destructive
    # live), 'jdsa' penalises the residual the solver's own 2x2 disparity grid cannot absorb.
    # The jdsa_* fields are unread outside 'jdsa', exactly as window_size is unread outside
    # 'wonline'.
    depth_loss: str
    jdsa_norm: str           # 'l1' | 'l2'
    jdsa_ridge: float        # ridge on the 4x4 normal equations, relative to their mean diagonal
    jdsa_lattice: str        # 'full' = fit at vggt_hw | 'ba' = the [3::8,3::8] grid BA reads
    coupling_shuffle: bool   # PLACEBO CONTROL, as AdaptConfig documents. Present here because
                             # coupling_fit is duck-typed on the config and reads this field, so a
                             # live run without it would fail inside the training loop rather than
                             # at construction. Leave False unless the run IS a control arm.

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
        if self.gate_metric not in GATE_METRICS:
            raise ValueError(f'gate_metric={self.gate_metric!r} is not one of {GATE_METRICS}')
        for name in ('gate_lo', 'gate_hi'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name}={getattr(self, name)} must be >= 0 (0 = off)')
        if self.depth_loss not in DEPTH_LOSSES:
            raise ValueError(f'depth_loss={self.depth_loss!r} is not one of {DEPTH_LOSSES}')
        if self.jdsa_norm not in JDSA_NORMS:
            raise ValueError(f'jdsa_norm={self.jdsa_norm!r} is not one of {JDSA_NORMS}')
        if self.jdsa_lattice not in JDSA_LATTICES:
            raise ValueError(f'jdsa_lattice={self.jdsa_lattice!r} is not one of {JDSA_LATTICES}')
        if self.jdsa_ridge < 0:
            raise ValueError(f'jdsa_ridge={self.jdsa_ridge} must be >= 0')
        # the objective is ONE choice; the coupling knobs are subordinate to it (adapt/config.py)
        if self.depth_loss == 'coupled' and self.coupling_lambda <= 0:
            raise ValueError(f"depth_loss='coupled' needs coupling_lambda > 0, not "
                             f'{self.coupling_lambda}')
        if self.depth_loss != 'coupled' and self.coupling_lambda > 0:
            raise ValueError(f'coupling_lambda={self.coupling_lambda} only has meaning under '
                             f"depth_loss='coupled', not {self.depth_loss!r}")
        if self.coupling_lambda < 0:
            raise ValueError(f'coupling_lambda={self.coupling_lambda} must be >= 0 (0 = off)')
        if self.coupling_axis not in COUPLING_AXES:
            raise ValueError(f'coupling_axis={self.coupling_axis!r} is not one of {COUPLING_AXES}')
        if self.coupling_min_var < 0:
            raise ValueError(f'coupling_min_var={self.coupling_min_var} must be >= 0')
        if self.coupling_shuffle and self.coupling_lambda <= 0:
            raise ValueError(
                f'coupling_shuffle=True needs coupling_lambda > 0, not {self.coupling_lambda}: it '
                f'permutes the coefficients the coupling term produces, and with the term off '
                f'there are none.')
        # Refused rather than ignored, for the reason adapt/config.py gives: a silently-inert lambda
        # would look like "the term did nothing" in the results instead of like a misconfiguration.
        if self.coupling_lambda > 0 and self.adapt_style != 'wonline':
            raise ValueError(
                f"coupling_lambda={self.coupling_lambda} needs adapt_style='wonline', not "
                f"{self.adapt_style!r}: the slope is fitted over the arrival's window, and 'online' "
                f'trains on the arrival alone - one point carries no slope.')
        # coupling_fit needs three points to return anything, so a shorter window is inert as well.
        # Note the window is CLIPPED at the start of the sequence (target.py:unit_keyframes), so
        # the first arrivals are short regardless; those units simply get no coupling term.
        if self.coupling_lambda > 0 and self.window_size < 3:
            raise ValueError(
                f'coupling_lambda={self.coupling_lambda} needs window_size >= 3, not '
                f'{self.window_size}: adapt/trainer.py:coupling_fit returns no coefficients below '
                f'three points, so the term would never fire.')
        # WARNED, not refused: batch_size < window_size is a legitimate thing to measure, and the
        # offline sweep did exactly that. But it is also the configuration that broke the term
        # there, and the failure is silent - the run trains, the loss looks plausible, and only the
        # prior's scale statistics show the damage. Printing costs nothing and a config is built
        # once per run.
        if self.coupling_lambda > 0 and self.batch_size < self.window_size:
            print(f'note: coupling_lambda={self.coupling_lambda} with batch_size='
                  f'{self.batch_size} < window_size={self.window_size}. The per-sample '
                  f'coefficients sum to zero only over the WHOLE window, so each step applies an '
                  f'uncancelled ~{self.window_size // max(self.batch_size, 1)}x fragment that '
                  f'grad_clip then renormalises. Set batch_size={self.window_size} unless the '
                  f'point of the run IS this ratio.')
        if 0 < self.gate_hi <= self.gate_lo:
            raise ValueError(f'gate_hi={self.gate_hi} must exceed gate_lo={self.gate_lo}: with '
                             f'both set the gate keeps the BAND between them, so this would skip '
                             f'every arrival and no optimiser step would ever run')
