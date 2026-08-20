"""The two configs: LoRAConfig is the adapter STRUCTURE, AdaptConfig the training RUN.

No field carries a default. Only the structure is recorded into the adapter's config.json and read
back by LoRAVGGT.from_adapter.
"""
from dataclasses import dataclass, replace
from typing import Optional, Tuple

ADAPT_STYLES = ('normal', 'online', 'wonline')
VAL_SOURCES = ('tail', 'rest')
# The x-axis the depth->scale slope is regressed against - see AdaptConfig.coupling_axis.
COUPLING_AXES = ('target', 'pred')
# WHICH OBJECTIVE a run trains on. Stated, not implied: before this existed the objective was
# implied by coupling_lambda > 0, which is how a placebo arm and a real one could look alike on
# disk. See AdaptConfig.depth_loss.
DEPTH_LOSSES = ('normal', 'coupled', 'jdsa')
JDSA_NORMS = ('l1', 'l2')
JDSA_LATTICES = ('full', 'ba')

# VGGT trained with width pinned to exactly 518 and height a multiple of 14, landscape or square
# (training/config/default.yaml:5, training/data/base_dataset.py:95-113).
VGGT_PATCH = 14
VGGT_LONG_SIDE = 518


def vggt_hw_for(stream_hw):
    """The VGGT input size matching a stream's aspect ratio - THE single definition (9.6).

    Nothing letterboxes anywhere, so matching the aspect here is the only thing keeping the image
    on VGGT's training distribution. It is an aspect knob, not a quality one: the prior reaches BA
    at 1/8 of the tracking resolution through a point subsample.
    """
    h, w = stream_hw
    if h <= 0 or w <= 0:
        raise ValueError(f'stream_hw {stream_hw} must be positive')
    if h > w:
        raise ValueError(
            f'stream_hw {stream_hw} is portrait (aspect {w/h:.3f}). VGGT trained only on '
            f'landscape-or-square inputs (aspect 0.33-1.0 with width pinned at '
            f'{VGGT_LONG_SIDE}); there is no in-distribution size for this stream.')
    vh = VGGT_PATCH * round(VGGT_LONG_SIDE * h / w / VGGT_PATCH)
    return (min(max(vh, VGGT_PATCH), VGGT_LONG_SIDE), VGGT_LONG_SIDE)


def aspect_lines(stream_hw, vggt_hw, who):
    """Report the stream -> VGGT resize, warning above 5% distortion. Used on both paths."""
    h, w = stream_hw
    vh, vw = vggt_hw
    skew = (vw / vh) / (w / h)
    lines = [f'stream {w}x{h} (aspect {w/h:.3f}) -> VGGT {vw}x{vh} '
             f'(aspect {vw/vh:.3f}), squash {skew:.3f}x']
    if not 0.95 < skew < 1.05:
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
    """One training run. The supervision target is not a knob - the export writes one directory."""
    # ---------------------------------------------------------------- data
    stream_res: int          # tracking resolution budget the export was produced at
    p_single_view: float     # 0 = always multi-view, 1 = always monocular
    max_left: int            # neighbour counts, drawn per sample
    max_right: int
    radius: int              # neighbour search radius, in frames
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
    # ---------------------------------------------------------------- depth->scale coupling (E3)
    # depth_loss aligns scale PER SAMPLE, so L(c*p) = L(p) exactly for any c > 0: the per-frame
    # output scale has identically zero gradient and the network is free to make s_i anything.
    # The harmful part of that freedom is the DEPTH-COUPLED part - s_i varying as a function of
    # how far the scene is, i.e. range compression - which correlates +0.95/+0.90 with ATE across
    # matched offline sets. This term puts a gradient on exactly that, and on nothing else:
    #
    #   x_i = log(median target depth)   DETACHED      y_i = log(s_i)
    #   b   = SUM(x~ y~) / SUM(x~^2)                   penalty = coupling_lambda * b^2
    #
    # b^2 rather than Var(y) because Var(y) = b^2 Var(x) + Var(resid) and the residual half is
    # neutral-to-beneficial (-0.72). Fitted over the WHOLE WINDOW, not the batch: batch_size is 2
    # and a slope through two points is exactly determined, so it carries no information.
    coupling_lambda: float   # 0.0 = OFF, and the extra pass is skipped entirely, so a run is
                             # byte-identical to one from before this field existed
    coupling_axis: str       # 'target' = median TARGET depth (recommended: the network does not
                             #            control the axis, so it cannot game the fit)
                             # 'pred'   = median PREDICTED depth. Available for comparison, but it
                             #            invites the degenerate escape of collapsing every
                             #            predicted median to one value - which IS range collapse.
    coupling_min_var: float  # skip the term when SUM(x~^2) is below this - the window has no
                             # depth spread and b would be an ill-conditioned division
    # ---------------------------------------------------------------- WHICH OBJECTIVE (the knob)
    # 'normal'  masked L1 in DEPTH after a per-sample median scale. Every run before this field
    #           existed, bit for bit, and what an absent depth_loss in a config.json means.
    # 'coupled' 'normal' plus E3's lambda*b^2 slope penalty. MEASURED AND NULL: over three seeds
    #           lambda=0 spans 23.680-26.958 (sd 1.75) against lambda=1's 23.782 - +1.18 +/- 1.43 m,
    #           t=0.83 - and a placebo that reassigns the coefficients to random keyframes scores
    #           the same as lambda=0. Live it is destructive (ATE 39-48 vs 24.7). Kept runnable, not
    #           recommended.
    # 'jdsa'    the residual the SOLVER'S own alignment cannot absorb. depth_loss aligns with one
    #           median scalar in depth; JDSA aligns with a 4-DOF bilinear field in DISPARITY, refit
    #           every BA iteration (geom/ba.py:161-196). This fits that same family and penalises
    #           what survives it, so the objective stops paying for errors the solver discards.
    depth_loss: str
    jdsa_norm: str           # 'jdsa' ONLY: 'l1' | 'l2' residual norm
    jdsa_ridge: float        # 'jdsa' ONLY: ridge on the 4x4 normal equations, RELATIVE to their
                             # mean diagonal, so it is unit-free. Guards the case where the mask
                             # sits in one image region and the four corners are not determined.
    jdsa_lattice: str        # 'jdsa' ONLY: 'full' = fit at vggt_hw (more points, better
                             # conditioned) | 'ba' = interpolate to the tracking resolution and take
                             # [3::8, 3::8], the 1/64 point subsample BA actually reads
    coupling_shuffle: bool   # PLACEBO CONTROL, not a training option. Reassigns the fitted
                             # coefficients to the window's keyframes at RANDOM, so the perturbation
                             # keeps its magnitude and its zero sum but loses every connection to
                             # depth. The point is to find out whether the measured ATE gain comes
                             # from the slope constraint or merely from the size of the per-sample
                             # pull; an arm with this on is a control arm and nothing else.

    def __post_init__(self):
        if self.depth_loss not in DEPTH_LOSSES:
            raise ValueError(f'depth_loss={self.depth_loss!r} is not one of {DEPTH_LOSSES}')
        if self.jdsa_norm not in JDSA_NORMS:
            raise ValueError(f'jdsa_norm={self.jdsa_norm!r} is not one of {JDSA_NORMS}')
        if self.jdsa_lattice not in JDSA_LATTICES:
            raise ValueError(f'jdsa_lattice={self.jdsa_lattice!r} is not one of {JDSA_LATTICES}')
        if self.jdsa_ridge < 0:
            raise ValueError(f'jdsa_ridge={self.jdsa_ridge} must be >= 0')
        # The objective is ONE choice, and the coupling knobs are subordinate to it. Refused rather
        # than resolved by precedence: a run that silently mixed two objectives would be
        # indistinguishable on disk from one that did not, which is the failure that made an
        # earlier replicate pair unreadable.
        if self.depth_loss == 'coupled' and self.coupling_lambda <= 0:
            raise ValueError(
                f"depth_loss='coupled' needs coupling_lambda > 0, not {self.coupling_lambda}: the "
                f"coupling term IS what 'coupled' adds, so this would be 'normal' under another "
                f"name. Set the lambda, or set depth_loss='normal'.")
        if self.depth_loss != 'coupled' and self.coupling_lambda > 0:
            raise ValueError(
                f"coupling_lambda={self.coupling_lambda} only has meaning under "
                f"depth_loss='coupled', not {self.depth_loss!r}. Two objectives at once is never "
                f'what is wanted, and the run\'s config.json would not show it.')
        if self.adapt_style not in ADAPT_STYLES:
            raise ValueError(f'adapt_style={self.adapt_style!r} is not one of {ADAPT_STYLES}')
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
        if self.coupling_lambda < 0:
            raise ValueError(f'coupling_lambda={self.coupling_lambda} must be >= 0 (0 = off)')
        if self.coupling_axis not in COUPLING_AXES:
            raise ValueError(f'coupling_axis={self.coupling_axis!r} is not one of {COUPLING_AXES}')
        if self.coupling_min_var < 0:
            raise ValueError(f'coupling_min_var={self.coupling_min_var} must be >= 0')
        # refused rather than ignored, like coupling_lambda below: a placebo flag that quietly did
        # nothing would put a control arm on disk that is really an ordinary lambda=0 run
        if self.coupling_shuffle and self.coupling_lambda <= 0:
            raise ValueError(
                f'coupling_shuffle=True needs coupling_lambda > 0, not {self.coupling_lambda}: it '
                f'permutes the coefficients the coupling term produces, and with the term off '
                f'there are none. The control for a shuffled arm is the UNSHUFFLED arm at the same '
                f'lambda, not this.')
        # the slope is fitted over a WINDOW, which only the wonline style has. Refused rather than
        # ignored: a silently-inert lambda would look like "the term did nothing" in the results.
        if self.coupling_lambda > 0 and self.adapt_style != 'wonline':
            raise ValueError(
                f"coupling_lambda={self.coupling_lambda} needs adapt_style='wonline', not "
                f"{self.adapt_style!r}: the slope is fitted over the window, and 'online' has a "
                f"window of 1 (no slope) while 'normal' has no window at all. Fitting it over a "
                f"batch instead is not an option - batch_size is typically 2 and a line through "
                f"two points is exactly determined.")
