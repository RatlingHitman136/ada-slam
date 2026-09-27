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

# WHERE the band (gate_lo, gate_hi) is read. 'arrival' reads the newest keyframe once, before its unit -
# the only behaviour before this field existed; a refusal skips the whole unit. 'sample' reads EVERY
# training sample inside every step, off the forward the step already runs; a refused sample gets no
# gradient and the rest of its batch still trains. The difference matters under 'wonline', where a
# keyframe is re-trained for window_size arrivals and its target can go bad AFTER it arrived: over
# M2DGR street_02's five full-route gauge runs, 58-83% of the keyframes whose loss later exceeded 1.0
# were below it on arrival, so an arrival gate at that threshold would have blocked only 15-30% of the
# corrupted exposures. At 'sample' the floor also changes meaning: it stops re-fitting window samples
# that already fit (a keyframe's loss falls to ~0.44x its arrival value over its 10 units there).
GATE_SCOPES = ('arrival', 'sample')

# What the tracker is served while the unit breaker is tripped - see OnlineConfig.breaker_serve.
BREAKER_SERVES = ('adapted', 'base')


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
    # THE frames-per-forward knob, and it governs BOTH ends: S = 1 + context_kf for the
    # training sample (target.py:sample) AND for the depth the prior serves
    # (end2end/prior.py:context_stack). That is the whole point of the field - it used to control
    # training alone while serving stayed monocular, so raising it trained the adapter in a regime
    # it was never asked to predict in, which is why every run on disk carries 0.
    # NOT non-keyframes: those images live on Hi2, which the extractor cannot reach.
    # >0 also turns on the pose loss - lambda_pose below is unread at 0, where pose_loss returns
    # zeros - so raising it changes two things at once unless lambda_pose is set to 0.0.
    # MEASURED on a 4090 at vggt_hw 168x518 (KITTI), base VGGT-1B: the cost is nearly FLAT in
    # S. Inference 93/92/103/90/111 ms at S=1..5, peak 4.56->4.62 GiB; one training step
    # (forward+backward+AdamW, gradient checkpointing on) 392/387/398/401 ms at S=1..4,
    # peak 6.19->6.48 GiB. A frame is only 12x37 = 444 patch tokens here, so the 48-block
    # aggregator is launch-bound rather than attention-bound and the S^2 term never bites.
    # Re-measure before assuming this holds at a larger vggt_hw.
    context_kf: int          # 0 = monocular and depth-only, the pre-knob behaviour exactly
    # Keyframes BETWEEN consecutive context frames (common.py:context_keyframes). 1 = the
    # immediately preceding keyframes, which is what every run before this field did.
    #
    # WHY IT EXISTS. The ATE on this track is ~100% cumulative scale drift - corr(local-scale sd,
    # ATE) = 0.997 at ~66 m per unit of sd - and that drift accrues at only -0.136% per keyframe.
    # A stride-1 sample of 3 keyframes therefore contains -0.41% of scale change, and BOTH loss
    # terms then remove it: depth_loss re-fits the scale inside the sample and pose_loss
    # normalises each translation set by its own mean norm. So the objective is blind to the very
    # quantity the arm is scored on. Striding widens the sample's baseline k-fold - stride 5
    # carries ~1.4% and stride 10 ~2.7% - which is the cheapest way to put some of the drift where
    # a loss can see it.
    #
    # The limit is visual overlap, not compute: at ~1.65 m per keyframe on KITTI 00, stride 5 is a
    # ~16 m baseline and stride 10 ~33 m. Far past that the frames stop overlapping, VGGT's
    # cross-frame attention has nothing to match and the pose target stops meaning anything.
    context_stride: int      # 1 = consecutive, the pre-knob behaviour exactly
    # TARGET NORMALIZATION (common.py:gauge_scale). True divides the target by the sample's own
    # gauge - the mean point distance, clamped at gauge_clamp x its median, over every valid-depth
    # pixel of every frame of the sample, in the target camera's frame - and makes depth_loss stop
    # re-fitting a scale (scale=1.0). Two consequences, and the second is the point:
    #   * the target lands in the gauge VGGT was PRETRAINED to emit, rather than in the tracker's
    #     map units, which inflate ~1.97x across KITTI 00;
    #   * the loss stops being scale-invariant, so for the first time the depth term carries
    #     gradient about ABSOLUTE scale. Every previous attempt at the drift added a penalty on
    #     top of the invariance instead of removing it, and all four landed at baseline.
    # g/gauge(g) is exactly invariant to a global rescale of g, so the map's drift cancels by
    # construction - no anchor, no window, no history.
    normalize_target: bool
    gauge_clamp: float       # clamp distances at this x the frame's MEDIAN distance before the
                             # mean. Must be > 1. RELATIVE, never absolute: an absolute cut would
                             # select a different physical set of pixels as the map inflated and
                             # the rescale invariance above would break.
    gauge_min_pixels: int    # below this many gauge pixels the sample is skipped rather than
                             # divided by a statistic computed on nothing
    # VGGT's own camera translation term (losses.py:pose_loss). True puts the ground-truth
    # TRANSLATIONS in the same gauge as the depth target and compares them directly, instead of
    # dividing each side by its own mean norm. That is what VGGT's training does - one avg_scale
    # divides camera translations AND depth maps, and camera_loss_single then compares translations
    # absolutely - and it is why VGGT needs no coupled_scale: scale agreement between the two heads
    # is a property of the data, not a term in the loss. Requires normalize_target (there is no
    # gauge otherwise) and lambda_pose > 0 (otherwise it computes something nothing reads).
    gauge_pose: bool
    stream_res: int          # must equal SlamConfig.stream_res - the tracking pixel budget

    # ---------------------------------------------------------------- optimisation
    lr: float
    weight_decay: float
    grad_clip: float
    lambda_pose: float       # unread at context_kf=0, where pose_loss returns zeros
    coupled_scale: bool      # True = the pose scale is reused by the depth loss
    # Hold that pose scale FIXED for the whole unit instead of recomputing it every forward.
    #
    # THE FAULT IT FIXES. coupled_scale is the only mechanism that ever moved the ATE, because
    # s_pose = |t_gt| / |t_pred| is built from camera MOTION rather than from the depths, so
    # depth_loss stops being invariant to rescaling the prediction and is finally pinned to a
    # metric that carries the tracker's drift. But recomputing it every forward makes the
    # objective non-stationary: within a unit the target window is FIXED, yet
    #
    #     L_k(theta) = mean| g - s_k * p(theta) |,   s_k = s_pose(theta_k)
    #
    # minimises a DIFFERENT function at every step. That is a self-consistent-field iteration,
    # not descent on one objective, and it converges only if theta -> s -> theta is contractive.
    # Measured, it is not: last/BEST 1.114-1.167, i.e. the iterate ends worse than its own best
    # point inside the unit, while the median_scale arms reach their minimum at the last step.
    #
    # Frozen, each keyframe's gauge is captured at its FIRST use in the unit and reused for the
    # rest, so the within-unit problem is stationary and can actually be descended into - while
    # the gauge still tracks the drift from unit to unit, which is where it has to move anyway
    # because the targets move too. Costs nothing: the value is taken from the forward that was
    # happening regardless, not from an extra pass.
    #
    # WHAT IT DOES NOT FIX. The coupling stays ONE-WAY. pose_loss returns s_pose detached
    # (losses.py:56), so dL/dtheta drops the p * ds/dtheta term - the depth residual can chase the
    # pose head's scale but never correct it. And l_trans normalises both translation sets by
    # their own mean norms, so |t_pred| - whose reciprocal IS s_pose - is unsupervised. Undetaching
    # without first anchoring that magnitude would let the depth loss shrink the translations to
    # cut its residual, the degenerate direction median_scale's docstring warns about.
    freeze_gauge: bool       # False = recompute every forward, the behaviour of every run so far
    # ONE depth scale fitted over the WHOLE batch, instead of one per sample.
    #
    # WHY. depth_loss aligns scale INSIDE each sample before measuring the residual - verified
    # bit-identical under a global rescale of the prediction - so disagreement BETWEEN keyframes
    # costs the objective nothing. But the ATE is ~100% cumulative scale drift (corr(local-scale
    # sd, ATE) = 0.997 within a seed triplet), i.e. exactly that disagreement. The objective has
    # been blind to the quantity it is scored on.
    #
    # Fitting one scale across the batch puts the disagreement in the residual. At batch_size 10
    # the batch IS the wonline window, so it spans 10 keyframes (~16 m on KITTI 00) carrying ~1.4%
    # of the sequence's drift. Below batch_size 2 it does nothing and is ignored.
    #
    # The scale is DETACHED and fixed for the step, like coupled_scale's pose_scale - but unlike
    # that one it does not move under the optimiser, which is what broke convergence there
    # (measured: coupled_scale runs sit 6-7x higher and never reach their minimum at the last
    # step, while median_scale runs do).
    #
    # COSTS ONE EXTRA FORWARD PER SAMPLE: the scale has to be known before the graph is built, so
    # a no-grad pass over the batch precedes the gradient pass. Roughly 2x the step time.
    #
    # Mutually exclusive with coupled_scale - they are two different answers to "which gauge", and
    # silently letting one win would be the kind of unstated choice 9.5's rule 1 exists to stop.
    batch_scale: bool
    # SCALE-CONSISTENCY penalty: lambda_cons * (log s_i - mean_j log s_j)^2 per sample, where s_i
    # is that sample's own undetached median_scale. 0 = off.
    #
    # WHY THIS RATHER THAN batch_scale. Both aim at the same thing - making cross-keyframe scale
    # disagreement cost something - but batch_scale does it by REPLACING the gauge with a detached
    # pooled one, and that removes the exact cancellation that pins the prediction's overall
    # scale. Measured consequence: the gauge walked smoothly over ~7 orders of magnitude
    # (0.0004 .. 3281, lag-1 autocorrelation of its log +0.999), i.e. the network's own output
    # scale wandered while the gauge absorbed it. Since the served prior is predict_depth's RAW
    # output - nothing rescales it at serve time - that is manufactured scale drift, and scale
    # drift is ~100% of the ATE here.
    #
    # This term leaves depth_loss alone. Each sample keeps its own undetached median_scale, so the
    # invariance and its exactly-zero gradient along "make everything bigger" survive, and the
    # penalty is a SEPARATE bounded term. Var of LOG scales is itself invariant to a global
    # rescale (shifting every log s_i equally leaves it unchanged), so it charges only RELATIVE
    # disagreement between keyframes - the drift - and cannot be reduced by moving the overall
    # scale. It cannot blow up the effective step size either, unlike a gauge that multiplies the
    # residual.
    #
    # The anchor mean_j log s_j is DETACHED and comes from a no-grad pass over the batch, so the
    # gradient pulls each scale toward the batch consensus rather than chasing a moving mean. That
    # costs one extra forward per sample, as batch_scale did.
    #
    # Needs batch_size >= 2 (one sample has nothing to be consistent WITH), and is refused
    # together with batch_scale - running both would be two answers to one question again.
    lambda_cons: float
    # Supervise depth on EVERY frame of the sample, under ONE shared scale, instead of on the
    # target frame alone.
    #
    # WHY. LoRAVGGT.forward runs the DPT head on frame 0 only, so a sample has always supervised
    # one keyframe however much context it carried - which is why widening the context stride
    # bought nothing: the depth term never saw more than one frame, and the only channel that
    # widened was one scalar in the pose term. With this on, the head runs on all S frames,
    # kf_target returns a target per frame, and depth_loss receives stacked (S,H,W) tensors.
    #
    # THE POINT IS THE SHARED SCALE, and it comes for free. median_scale pools the medians over
    # whatever tensor it is given, so a stacked sample gets ONE scale for the whole sequence -
    # meaning a keyframe whose scale has walked away from the others now carries a residual. That
    # is the drift the ATE measures (corr(local-scale sd, ATE) = 0.977) and the exact quantity a
    # per-sample gauge is blind to.
    #
    # AND IT NEEDS NO DETACHING, which is what killed batch_scale. median_scale stays a function
    # of the prediction, so the loss remains exactly invariant to rescaling ALL frames together -
    # only RELATIVE disagreement between them is charged. The output scale stays anchored.
    #
    # ONLINE ONLY. The offline stage cannot do this: SceneData draws its context from NON-keyframe
    # neighbours, which have no depth_slam/ or mask_slam/ entry at all. Serving is unchanged too -
    # predict_depth stays frame-0, which is all an arriving keyframe needs.
    #
    # COSTS the depth head on S frames instead of 1, forward and backward. The aggregator, which
    # dominates, is unchanged.
    depth_all_frames: bool
    # Keyframes from the START of the sequence that lambda_cons measures against, instead of the
    # current batch. 0 = the batch-derived reference, which is what every run so far used.
    #
    # WHY. Every attempt to put drift into the loss failed for one quantitative reason - the
    # horizon was too short. A 2-keyframe sample spans 0.27% of scale change and a 10-keyframe
    # window 1.36%, against a depth residual of ~1.9%: the signal sat BELOW the floor of the thing
    # measuring it. Meanwhile dscale - JDSA's own prior-vs-tracker ratio - falls 0.74 -> 0.32 over
    # the run, a log spread of ~0.83, some 60x larger than a window can see. Comparing against
    # keyframes from the beginning finally spans the accumulated drift.
    #
    # NO CO-VISIBILITY IS NEEDED, which is what makes this different from putting an anchor in
    # VGGT's input. Each keyframe gets its own forward with its own context; only the resulting
    # SCALES are compared. An anchor 400 keyframes back is ~660 m away on this scene, and the
    # measured context influence already collapses past ~33 m - so an anchor VGGT had to LOOK at
    # would be two unrelated images, while an anchor it is merely scored against is fine.
    #
    # The set is chosen ONCE, the first unit with enough history, and held for the run - a literal
    # anchor to the original gauge. It is stored as frame TIMESTAMPS, not slot indices:
    # track_frontend.py:52 prunes a keyframe and decrements counter, shifting every index after it,
    # so cached slots would silently re-point at different frames. Same reasoning as the unit
    # de-dup.
    #
    # Costs anchor_kf extra no-grad forwards per UNIT (not per step) - about 8% at anchor_kf 4
    # against 50 gradient forwards.
    anchor_kf: int
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
                                 # Under normalize_target the loss is in the target's own gauge,
                                 # so NONE of those carry over. M2DGR gauge runs (gate_01 and
                                 # street_02; per-keyframe losses recovered from train_log.json),
                                 # 'raw':
                                 #   hi  newest keyframe per arrival median 0.013-0.019, p99
                                 #       0.06-0.13; 6 of 54334 clean training samples exceed 1.0,
                                 #       the corrupted street_02 stretch (full routes past
                                 #       ~900 m) reaches 1-559 - a clean gap, gate_hi 1.0
                                 #   lo  per training sample p5/p25/p50 street_02 0.0045/0.0074/
                                 #       0.0101, gate_01 0.0027/0.0051/0.0073 - no gap, and the
                                 #       scale is scene-dependent: derive a floor per scene
                                 # 'rel' separates worse there (clean tail to ~8).
    gate_lo: float               # 0 = off; skip below this. Already-fit frames.
    gate_hi: float               # 0 = off; skip above this. Broken/degenerate targets.
    gate_scope: str              # 'arrival' | 'sample' - where the band is read, see GATE_SCOPES

    # ---------------------------------------------------------------- the unit breaker
    # SKIP an arrival's whole unit when its window has gone bad AS A WHOLE: the median loss over the
    # window's keyframes, measured before the unit trains (gate_value - eval mode, no grad, one forward
    # per keyframe), exceeds breaker_k x the median of the last breaker_window units that did train.
    # gate_metric picks the quantity. Not latching: a later window that measures clean again trains,
    # so adaptation can resume if the map recovers.
    #
    # Why a UNIT breaker and not a sample gate. On M2DGR street_02's full route the tracker fails at
    # ~850 m and the whole window goes moderately wrong (unit median losses 0.08-25 against a settled
    # 0.010), while depth_loss's L1 caps what any single sample can pull. gate_hi 1.0 per sample
    # refused 941 extreme samples and the run collapsed exactly as without it (ATE 121.5 vs 121.4 m),
    # the adapter still ending at 4.3x raw VGGT's lidar AbsRel. Replayed on the four full-route train
    # logs (window 100, warm-up 40): settled unit medians p50 0.010, p99 0.014-0.017; k=2 tripped
    # falsely once in ~1600 clean units and first at 872-967 m, ahead of the harm at 1161 m; k=3 never
    # falsely, first at 877-977 m.
    breaker_k: float             # 0 = off; else > 1 - trip when the window median > k x reference
    breaker_window: int          # untripped units the reference median spans
    breaker_warmup: int          # units that only FEED the reference - the first ones carry the
                                 # untrained adapter's loss (~0.3 on street_02, not ~0.01)
    # WHAT THE TRACKER IS SERVED while the breaker's latest check has tripped. 'adapted' keeps serving the
    # adapter - the only behaviour before this field existed. 'base' serves VGGT-base instead: every
    # LoRALinear's scaling is zeroed for that one served forward and restored straight after, so training,
    # the gates and the breaker itself keep measuring the adapter, and serving returns to it if the
    # breaker re-opens. Why: on street_02's full route three frozen VGGT runs recover from the ~850 m
    # collapse at 1261-1283 m, while the k=2 breaker runs - which stopped training at 833-922 m and kept
    # the adapter within 19-32% of raw VGGT's lidar AbsRel - recover at 1315-1443 m and score
    # 136.8 +/- 5.7 m on the last 792 m against frozen VGGT's 70.1 +/- 8.9. With training already stopped,
    # the model SERVED through the collapse is what is left between those arms.
    breaker_serve: str           # 'adapted' | 'base'

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
        if self.gauge_clamp <= 1.0:
            raise ValueError(f'gauge_clamp={self.gauge_clamp} must be > 1: it clamps distances at '
                             f'that multiple of the frame MEDIAN, so <= 1 would clip at least '
                             f'half of every frame and the statistic would stop tracking the '
                             f'scene. 2.0 is the measured choice.')
        if self.gauge_min_pixels < 1:
            raise ValueError(f'gauge_min_pixels={self.gauge_min_pixels} must be >= 1')
        if self.gauge_pose and not self.normalize_target:
            raise ValueError('gauge_pose=True needs normalize_target=True: it puts the ground-truth '
                             'translations in the DEPTH TARGET\'s gauge, and without '
                             'normalize_target there is no such gauge to put them in.')
        if self.gauge_pose and self.lambda_pose <= 0.0:
            raise ValueError(f'gauge_pose=True with lambda_pose={self.lambda_pose} changes how a '
                             f'term that is weighted to zero is computed. Set lambda_pose > 0, or '
                             f'gauge_pose False.')
        # NOT GUARDED, deliberately: lambda_pose > 0 alongside normalize_target. An earlier
        # version refused it, reasoning that the pose term carries a different gauge. Re-reading
        # pose_loss, that is only half true - l_t divides EACH side by its own mean translation
        # norm and l_r is quaternions, so both are scale-free and neither conflicts with a
        # normalized depth target. The only scale-carrying output is pose_scale, and that is
        # consumed solely under coupled_scale, which IS refused below. Forbidding lambda_pose as
        # well cost 2.35 m at ctx2 (_ctx2 12.881 vs _ctx2_cscale 15.231) for no principled reason.
        #
        # WHAT DOES CHANGE IS THE RELATIVE WEIGHT, and it is worth knowing before reusing a value
        # from the pre-gauge sweep. Measured on the ctx2 arms, normalize_target leaves l_depth
        # ~7.2x smaller in VALUE (0.0098 vs 0.0712) and roughly 2.8x smaller in SUBGRADIENT, since
        # the old loss multiplied the prediction by a fitted s ~ 2.8 that this one does not. So a
        # given lambda_pose bites harder here; the sweep's old optimum needs re-deriving rather
        # than inheriting. (And per the campaign's own lambda_pose lesson, the LOSS ratio is a poor
        # proxy for the gradient ratio - a term 192x smaller in loss was only 8.3x smaller in
        # gradient - so neither number above should be used as a rescaling factor on its own.)
        if self.normalize_target and self.coupled_scale:
            raise ValueError(
                'normalize_target=True with coupled_scale=True asks for two gauges at once: the '
                'target already carries one, and coupled_scale would have depth_loss re-fit the '
                "pose head's instead, dividing out the quantity being supervised. This is the "
                'real conflict between the gauge and the pose head; lambda_pose itself is fine, '
                'because pose_loss normalises each translation set by its own mean norm and is '
                'scale-free. Set coupled_scale False.')
        # Everything below is another answer to "which gauge", and normalize_target has already
        # answered it. Running either together would be two answers to one question - the same
        # class of silent conflict batch_scale/coupled_scale is already refused for.
        for name in ('batch_scale', 'freeze_gauge'):
            if self.normalize_target and getattr(self, name):
                raise ValueError(f'normalize_target=True with {name}=True sets the depth gauge '
                                 f'twice. The target already carries it; turn {name} off.')
        for name in ('lambda_cons', 'anchor_kf'):
            if self.normalize_target and getattr(self, name) > 0:
                raise ValueError(
                    f'normalize_target=True with {name}={getattr(self, name)}: {name} penalises '
                    f'scale INCONSISTENCY on top of a scale-invariant loss, which normalize_target '
                    f'removes outright. Set {name} to 0.')
        # gate_lo/gate_hi are thresholds on the depth loss itself, and normalize_target changes what
        # that loss measures, so a threshold tuned in map units gates on nothing meaningful here. Both
        # bounds are allowed under it because normalized references now exist (gate_metric above) -
        # read those, not the RELLIS ones, when choosing a band for a normalized run.
        if self.freeze_gauge and not self.coupled_scale:
            raise ValueError('freeze_gauge=True with coupled_scale=False does nothing: there is '
                             'no pose gauge to hold fixed, because depth_loss fits its own '
                             'median_scale per sample and that is already stationary within a '
                             'unit. Turn coupled_scale on, or set freeze_gauge False.')
        if self.depth_all_frames and self.context_kf < 1:
            raise ValueError(f'depth_all_frames=True at context_kf={self.context_kf} does '
                             f'nothing: the sample is one frame, so "all frames" is that frame '
                             f'and the shared scale is the per-sample scale. Raise context_kf.')
        if self.depth_all_frames and self.batch_scale:
            raise ValueError('depth_all_frames and batch_scale are the same idea at two '
                             'granularities - one pooled gauge within a sample, versus one across '
                             'the batch. batch_scale reaches it by DETACHING a pooled scale, '
                             'which let the served output walk over seven orders of magnitude; '
                             'depth_all_frames gets it from median_scale undetached. Pick one.')
        if self.depth_all_frames and self.lambda_cons > 0:
            raise ValueError('lambda_cons penalises the spread of PER-SAMPLE scales, and under '
                             'depth_all_frames a sample no longer has one - it has a single '
                             'pooled scale over its S frames. The two cannot both be defined on '
                             'the same quantity; set lambda_cons to 0.')
        if self.lambda_cons < 0:
            raise ValueError(f'lambda_cons={self.lambda_cons} must be >= 0 (0 = off)')
        if self.lambda_cons > 0 and self.batch_scale:
            raise ValueError('lambda_cons and batch_scale are two different attacks on the same '
                             'problem - a separate bounded penalty, versus replacing the gauge. '
                             'batch_scale is the one with evidence against it (the served scale '
                             'walked over 7 orders of magnitude). Pick one.')
        if self.lambda_cons > 0 and self.coupled_scale:
            raise ValueError('lambda_cons>0 needs coupled_scale=False. The penalty is on the '
                             "sample's OWN median_scale, which depth_loss only returns when it "
                             'fitted one; under coupled_scale it returns the pose scale, which is '
                             'DETACHED (losses.py:56), so the term would be a constant with no '
                             'gradient - a knob that silently does nothing.')
        if self.anchor_kf < 0:
            raise ValueError(f'anchor_kf={self.anchor_kf} must be >= 0 (0 = the reference comes '
                             f'from the batch, as it did before this field existed)')
        if self.anchor_kf > 0 and self.lambda_cons <= 0:
            raise ValueError(f'anchor_kf={self.anchor_kf} with lambda_cons=0 computes a reference '
                             f'scale that nothing then uses. Set lambda_cons above 0, or '
                             f'anchor_kf to 0.')
        # the reference needs SOMETHING to be consistent with - either two samples in the batch,
        # or two anchors from history. Before anchor_kf existed only the first was possible.
        if self.lambda_cons > 0 and self.batch_size < 2 and self.anchor_kf < 2:
            raise ValueError(f'lambda_cons>0 at batch_size={self.batch_size} and '
                             f'anchor_kf={self.anchor_kf}: one sample per step and no history '
                             f'anchor means there is no other scale to be consistent with. Raise '
                             f'batch_size, or set anchor_kf >= 2.')
        if self.batch_scale and self.coupled_scale:
            raise ValueError('batch_scale and coupled_scale are both True, and they are two '
                             'different gauges for the same residual: batch_scale fits one scale '
                             'over the batch, coupled_scale takes the pose head\'s. Choose one - '
                             'coupled_scale=False is the one with evidence behind it.')
        if self.batch_scale and self.batch_size < 2:
            raise ValueError(f'batch_scale=True at batch_size={self.batch_size} does nothing: one '
                             f'sample per step means the shared scale IS the per-sample scale. '
                             f'Raise batch_size (10 = the whole wonline window) or set it False.')
        if self.context_stride < 1:
            raise ValueError(f'context_stride={self.context_stride} must be >= 1 (1 = consecutive '
                             f'keyframes, the behaviour before this field existed)')
        if self.steps_per_kf < 0:
            raise ValueError(f'steps_per_kf={self.steps_per_kf} must be >= 0 (0 = never step, the '
                             f'null-op arm)')
        if self.checkpoint_every_kf < 0:
            raise ValueError(f'checkpoint_every_kf={self.checkpoint_every_kf} must be >= 0 '
                             f'(0 = off)')
        if self.gate_metric not in GATE_METRICS:
            raise ValueError(f'gate_metric={self.gate_metric!r} is not one of {GATE_METRICS}')
        if self.gate_scope not in GATE_SCOPES:
            raise ValueError(f'gate_scope={self.gate_scope!r} is not one of {GATE_SCOPES}')
        if self.gate_scope == 'sample' and self.gate_lo <= 0 and self.gate_hi <= 0:
            raise ValueError('gate_scope=sample with gate_lo=0 and gate_hi=0 gates nothing - a '
                             'silent no-op arm. Set a bound, or gate_scope arrival (the gates-off '
                             'default).')
        if self.breaker_k < 0 or 0 < self.breaker_k <= 1:
            raise ValueError(f'breaker_k={self.breaker_k} must be 0 (off) or > 1: at k <= 1 about half '
                             f'of all ordinary units sit above their own running median and would trip')
        if self.breaker_window < 1 or self.breaker_warmup < 1:
            raise ValueError(f'breaker_window={self.breaker_window} and breaker_warmup='
                             f'{self.breaker_warmup} must both be >= 1: the reference median needs at '
                             f'least one untripped unit before anything can trip')
        if self.breaker_serve not in BREAKER_SERVES:
            raise ValueError(f'breaker_serve={self.breaker_serve!r} is not one of {BREAKER_SERVES}')
        if self.breaker_serve == 'base' and self.breaker_k <= 0:
            raise ValueError("breaker_serve='base' with breaker_k=0 does nothing: the breaker never trips, "
                             "so VGGT-base is never served. Set breaker_k, or breaker_serve 'adapted'.")
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
