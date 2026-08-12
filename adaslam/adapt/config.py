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

    def __post_init__(self):
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
