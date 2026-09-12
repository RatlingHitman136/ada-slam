"""The two configs: LoRAConfig is the adapter STRUCTURE, AdaptConfig the training RUN.

No field carries a default. Only the structure is recorded into the adapter's config.json and read
back by LoRAVGGT.from_adapter.
"""
from dataclasses import dataclass, replace
from typing import Optional, Tuple

ADAPT_STYLES = ('normal', 'online', 'wonline')
VAL_SOURCES = ('tail', 'rest')

# VGGT trained with width pinned to exactly 518 and height a multiple of 14, landscape or square
# (training/config/default.yaml:5, training/data/base_dataset.py:95-113).
VGGT_PATCH = 14
VGGT_LONG_SIDE = 518
# ...and never narrower than this. `aspects: [0.33, 1.0]` is H/W and get_target_shape floors
# int(518 * 0.33) = 170 to a multiple of 14, so (168, 518) is the SMALLEST shape VGGT ever saw
# (training/config/default_dataset.yaml:37, training/data/base_dataset.py:105-108).
VGGT_MIN_ASPECT = 0.33
VGGT_MIN_SIDE = VGGT_PATCH * (int(VGGT_LONG_SIDE * VGGT_MIN_ASPECT) // VGGT_PATCH)   # 168


def vggt_hw_for(stream_hw):
    """The VGGT input size for a stream: its aspect, CLAMPED to VGGT's trained band (9.6).

    Nothing letterboxes anywhere, so the size chosen here is the only thing keeping the image on
    VGGT's training distribution. For a stream inside the band that means matching its aspect
    exactly. For one WIDER than the band - KITTI's 848x256 is 0.302, below the 0.324 floor -
    matching it would keep the model out of distribution in the one way it cannot recover from, so
    the height is clamped and the image is squashed INTO the band instead: 1.074x vertically on
    KITTI, with the field of view kept whole. The alternative, cropping the stream's sides, would
    spend the tracker's peripheral parallax - which under forward motion is most of the parallax
    there is - on the prior's comfort.

    It is an aspect knob, not a quality one: the prior reaches BA at 1/8 of the tracking resolution
    through a point subsample.
    """
    h, w = stream_hw
    if h <= 0 or w <= 0:
        raise ValueError(f'stream_hw {stream_hw} must be positive')
    if h > w:
        raise ValueError(
            f'stream_hw {stream_hw} is portrait (aspect {w/h:.3f}). VGGT trained only on '
            f'landscape-or-square inputs (aspect {VGGT_MIN_ASPECT}-1.0 with width pinned at '
            f'{VGGT_LONG_SIDE}); there is no in-distribution size for this stream.')
    vh = VGGT_PATCH * round(VGGT_LONG_SIDE * h / w / VGGT_PATCH)
    return (min(max(vh, VGGT_MIN_SIDE), VGGT_LONG_SIDE), VGGT_LONG_SIDE)


def aspect_lines(stream_hw, vggt_hw, who):
    """Report the stream -> VGGT resize, warning above 5% distortion. Used on both paths."""
    h, w = stream_hw
    vh, vw = vggt_hw
    skew = (vw / vh) / (w / h)
    lines = [f'stream {w}x{h} (aspect {w/h:.3f}) -> VGGT {vw}x{vh} '
             f'(aspect {vw/vh:.3f}), squash {skew:.3f}x']
    if 0.95 < skew < 1.05:
        return lines
    if tuple(vggt_hw) == vggt_hw_for(stream_hw):
        # the clamp fired: this stream is wider than VGGT's trained band, so the squash is
        # DELIBERATE and there is no less-distorted size that is still in distribution
        lines.append(f'  note: stream aspect {h/w:.3f} is outside VGGT\'s trained '
                     f'{VGGT_MIN_ASPECT}-1.0, so {vh} is the clamp, not a mismatch. The image is '
                     f'stretched {1/skew:.3f}x vertically; the FoV is kept whole.')
    else:
        lines.append(f'  WARNING: aspect ratios differ by {abs(1-skew)*100:.0f}%. '
                     f'{who} resizes without letterboxing, so VGGT sees a distorted image. '
                     f'The matching size for this stream is {vggt_hw_for(stream_hw)}')
    return lines


@dataclass(frozen=True)
class LoRAConfig:
    """Model + adapter structure - what must be identical between training and inference."""
    weights: str                  # local VGGT-1B snapshot, e.g. pretrained_models/vggt
    vggt_hw: Optional[Tuple[int, int]]   # both dims %14; None = derive from the stream (9.3)
    rank: int
    alpha: int
    targets: Tuple[str, ...]      # Linear leaves to wrap inside each aggregator block
    patch_embed: bool             # False = adapt only the alternating-attention stack

    def __post_init__(self):
        # normalise, so a config rebuilt from JSON (lists) compares equal to a hand-written one
        object.__setattr__(self, 'targets', tuple(self.targets))
        if self.vggt_hw is None:
            return
        object.__setattr__(self, 'vggt_hw', tuple(self.vggt_hw))
        h, w = self.vggt_hw
        if h % VGGT_PATCH or w % VGGT_PATCH:
            raise ValueError(f'vggt_hw ({h}, {w}): both dims must be divisible by {VGGT_PATCH}')

    def resolved(self, stream_hw):
        """This config with vggt_hw derived, if it was left None. Call after chdir, before spawn."""
        return self if self.vggt_hw is not None else replace(self, vggt_hw=vggt_hw_for(stream_hw))


@dataclass(frozen=True)
class AdaptConfig:
    """One training run. Neither the target nor the objective is a knob: the export writes one
    depth directory, and the loss is always losses.py's median-aligned masked depth L1 plus
    lambda_pose * the pose loss.
    """
    # ---------------------------------------------------------------- data
    stream_res: int          # tracking resolution budget the export was produced at
    p_single_view: float     # 0 = always multi-view, 1 = always monocular
    max_left: int            # neighbour counts, drawn per sample
    max_right: int
    radius: int              # neighbour search radius, in frames
    # CONTEXT is the online path's mechanism (common.py:context_keyframes), not the neighbour
    # sampler above: the `context_kf` KEYFRAMES before the target, `context_stride` apart, placed
    # after it so the target stays frame 0. It is one variable for BOTH ends - the trainer builds
    # the sample with it and end2end/prior.py:context_stack builds the served sequence with it, so
    # the adapter is fitted in the regime it is asked to predict in. 0 = monocular, bit-identical
    # to every run recorded before this field existed.
    context_kf: int
    context_stride: int
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
    # ---------------------------------------------------------------- optimisation
    # The styles differ ONLY in the order batches reach the loop (trainer.py:schedule). A UNIT is
    # an epoch in 'normal', one arriving keyframe in 'online' and one window in 'wonline'; the
    # cadences below count units.
    adapt_style: str         # 'normal' | 'online' | 'wonline'
    epochs: int              # 'normal': passes over the train set | 'online': steps per keyframe |
                             # 'wonline': passes over the window
    batch_size: int          # not read in 'online' - a keyframe arrives alone
    window_size: int         # 'wonline' ONLY: keyframes per window (the arrival + the
                             # window_size-1 before it). Unread by the other two styles.
    lr: float
    weight_decay: float
    grad_clip: float
    lambda_pose: float
    coupled_scale: bool      # True = the pose scale is reused by the depth loss
    min_mask_pixels: int     # below this a sample contributes no depth gradient
    seed: int
    log_every: int
    # ---------------------------------------------------------------- split + eval
    # SELECT, then split: kf_fraction picks which of the exported keyframes are trained on at all,
    # and val_source says where the rest of the export goes.
    kf_fraction: float       # of the exported keyframes, TRAIN on this fraction, taken
                             # equidistant over the keyframe LIST (keyframes are unevenly spaced
                             # in time, so this is every Nth keyframe). 1.0 = every one.
    val_source: str          # 'tail' = the contiguous last (1 - train_frac) of the selection, so
                             #          val measures generalising FORWARD and the trained region
                             #          is a strict prefix. train_frac is read only in this mode.
                             # 'rest' = every exported keyframe the selection SKIPPED, interleaved
                             #          through the whole sequence - "the keyframes it never
                             #          trained on". Needs kf_fraction < 1 to leave anything over.
    train_frac: float        # 'tail' ONLY: val = the contiguous TAIL; 1.0 = no val set
    eval_on_train: bool      # report on the train subset too, so the train/val gap is visible
    eval_on_val: bool
    eval_every_epoch: bool   # False = base + final only; True in 'online' = one eval per keyframe
    eval_max_kf: int         # cap per eval subset, evenly subsampled; 0 = no cap
    keep_best: bool          # True = save the best-val unit instead of the last
    checkpoint_every: int    # full adapter snapshot every N units; 0 = off. The CADENCE only -
                             # the location is LoRAVGGT.train(ckpt_dir=...)

    def __post_init__(self):
        if self.adapt_style not in ADAPT_STYLES:
            raise ValueError(f'adapt_style={self.adapt_style!r} is not one of {ADAPT_STYLES}')
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
        if self.context_kf < 0:
            raise ValueError(f'context_kf={self.context_kf} must be >= 0 (0 = monocular)')
        if self.context_stride < 1:
            raise ValueError(f'context_stride={self.context_stride} must be >= 1')
        # TWO context mechanisms would silently mix: neighbours() draws a RANDOM number of
        # arbitrary frames within `radius` on both sides, context_keyframes() takes a FIXED set of
        # preceding keyframes - and only the second one is what the extractor serves. Refuse
        # rather than let a run average over both.
        if self.context_kf and self.p_single_view != 1.0:
            raise ValueError(
                f'context_kf={self.context_kf} builds the sequence from the preceding KEYFRAMES, '
                f'but p_single_view={self.p_single_view} still lets the random-neighbour sampler '
                f'fire on {100*(1-self.p_single_view):.0f}% of samples. Set p_single_view=1 to '
                f'turn that mechanism off, or context_kf=0 to use it instead.')
        if self.context_stride != 1 and not self.context_kf:
            raise ValueError(f'context_stride={self.context_stride} is unread at context_kf=0 - '
                             f'set context_kf > 0, or leave context_stride at 1')
        # only where it is read; whether it fits the keyframe count is data, checked in the trainer
        if self.adapt_style == 'wonline' and self.window_size < 1:
            raise ValueError(f'window_size={self.window_size} must be >= 1 in the wonline style')
        if not 0.0 < self.kf_fraction <= 1.0:
            raise ValueError(f'kf_fraction={self.kf_fraction} must be in (0, 1]')
        if self.val_source not in VAL_SOURCES:
            raise ValueError(f'val_source={self.val_source!r} is not one of {VAL_SOURCES}')
        # not a data question like window_size: at kf_fraction 1.0 the selection IS the export, so
        # 'rest' is empty for every possible keyframe count
        if self.val_source == 'rest' and self.kf_fraction >= 1.0:
            raise ValueError("val_source='rest' validates on the keyframes kf_fraction skipped, "
                             'but kf_fraction=1.0 selects every one and leaves none over. Lower '
                             "kf_fraction, or use val_source='tail'.")
        if not 0.0 < self.train_frac <= 1.0:
            raise ValueError(f'train_frac={self.train_frac} must be in (0, 1]')
        if self.checkpoint_every < 0:
            raise ValueError(f'checkpoint_every={self.checkpoint_every} must be >= 0 (0 = off)')
