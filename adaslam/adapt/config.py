"""The two configs.

No field carries a default. Whoever runs an adaptation states every value, so there is exactly
one place per entry point where a hyperparameter is written down and nothing can be inherited
silently from this package.

The split is by lifetime, not by topic:

  LoRAConfig   the STRUCTURE - what must be identical between training and inference. It is
               recorded into the adapter's config.json and read back by LoRAVGGT.from_adapter.
  AdaptConfig  the RUN - what only training cares about.
"""
from dataclasses import dataclass, replace
from typing import Optional, Tuple

from ..common import DEPTH_SOURCES      # extract writes depth_<src>/, this reads it: one tuple

DEPTH_SPACES = ('depth', 'disparity')
SPLIT_MODES = ('stride', 'contiguous', 'random')

# VGGT's patch grid, and the one axis it was always trained at. training/config/default.yaml:5
# sets img_size 518 and training/data/base_dataset.py:95-113 builds every batch as
# [H = 518*aspect rounded to %14, W = 518] with aspect in [0.33, 1.0] - so WIDTH IS ALWAYS
# EXACTLY 518, height is a multiple of 14 at most 518, and portrait was never seen at any aspect.
VGGT_PATCH = 14
VGGT_LONG_SIDE = 518


def vggt_hw_for(stream_hw):
    """The VGGT input size matching a tracking stream's aspect ratio. (H, W) in, (H, W) out.

    THE single definition of this formula - SceneData.aspect_report() and LoRAConfig.resolved()
    both go through it. Pins W to 518 and takes H to the nearest multiple of 14, which is exactly
    the shape VGGT trained on. Nothing letterboxes anywhere in this repo, so matching the aspect
    here is the only thing keeping the image on VGGT's training distribution.

    Note what this deliberately does NOT try to do: raise resolution. The prior reaches BA at 1/8
    of the tracking resolution through a point subsample (depth_video.py:70-73), so vggt_hw is an
    aspect knob, not a quality one.
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
    """Report the stream -> VGGT resize, warning when it distorts by more than 5%.

    Used on BOTH paths - the adapt stage through SceneData.aspect_report() and every A/B arm
    through VggtPrior - because an adapter's recorded vggt_hw can disagree with the stream it is
    being run on, and the un-adapted 'vggt_base' arm has no adapter to read a size back from at
    all. `who` names the code that will do the resizing, so the message says where to look.
    """
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
    """Model + adapter structure.

    vggt_hw MUST match the tracking stream's aspect ratio: both SceneData.frame() and the prior
    extractor resize straight to it with no letterboxing, so a mismatched aspect squashes the
    image off VGGT's training distribution. (294, 518) suits Replica's 344x616, (378, 518) suits
    TUM's 400x544.

    Leave it None and resolved() derives it - that is the recommended setting, because the right
    value is a pure function of the stream and getting it wrong is silent. None is a stated
    instruction, not an omitted field: this config still has no defaults.
    """
    weights: str                  # local VGGT-1B snapshot, e.g. pretrained_models/vggt
    vggt_hw: Optional[Tuple[int, int]]   # both dims %14; None = derive from the stream
    rank: int
    alpha: int
    targets: Tuple[str, ...]      # Linear leaves to wrap inside each aggregator block
    patch_embed: bool             # False = adapt only the alternating-attention stack

    def __post_init__(self):
        # normalise, so a config rebuilt from JSON (lists) compares equal to a hand-written one
        object.__setattr__(self, 'targets', tuple(self.targets))
        if self.vggt_hw is None:
            return                       # unresolved; resolved() validates what it derives
        object.__setattr__(self, 'vggt_hw', tuple(self.vggt_hw))
        h, w = self.vggt_hw
        if h % VGGT_PATCH or w % VGGT_PATCH:
            raise ValueError(f'vggt_hw ({h}, {w}): both dims must be divisible by {VGGT_PATCH}')

    def resolved(self, stream_hw):
        """This config with vggt_hw derived from the stream, if it was left None.

        Call once, after chdir and before any Process is spawned - deriving reads a frame, which
        a module-level config literal must not do (the spawned reader re-executes that module).
        An explicitly stated vggt_hw is returned untouched, so pinning a value still works.
        """
        return self if self.vggt_hw is not None else replace(self, vggt_hw=vggt_hw_for(stream_hw))


@dataclass(frozen=True)
class AdaptConfig:
    """One training run."""
    # ---------------------------------------------------------------- data
    depth_source: str        # which export target supervises: depth_<src>/ + mask_<src>/
    stream_res: int          # tracking resolution budget the export was produced at
    p_single_view: float     # 0 = always multi-view, 1 = always monocular (how the prior is used)
    max_left: int            # neighbour counts, drawn per sample
    max_right: int
    radius: int              # neighbour search radius, in frames
    # ---------------------------------------------------------------- optimisation
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    lambda_pose: float
    depth_space: str         # 'disparity' (as HI-SLAM2 consumes it) | 'depth'
    coupled_scale: bool      # True = the pose scale is reused by the depth loss
    min_mask_pixels: int     # below this a sample contributes no depth gradient
    seed: int
    log_every: int
    # ---------------------------------------------------------------- split + eval
    train_frac: float        # 1.0 = train on every keyframe, no val set
    split_mode: str
    eval_on_train: bool      # report on the train subset too, so the train/val gap is visible
    eval_on_val: bool
    eval_every_epoch: bool   # False = only before training and after the last epoch
    eval_max_kf: int         # evenly subsample each eval subset to at most this many; 0 = no cap
    keep_best: bool          # True = snapshot on val improvement and save that, not the last epoch
    checkpoint_every: int    # full adapter snapshot every N epochs; 0 = off. The CADENCE only -
                             # where they land is the ckpt_dir argument of LoRAVGGT.train()

    def __post_init__(self):
        for name, allowed in (('depth_source', DEPTH_SOURCES), ('depth_space', DEPTH_SPACES),
                              ('split_mode', SPLIT_MODES)):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f'{name}={value!r} is not one of {allowed}')
        if self.checkpoint_every < 0:
            raise ValueError(f'checkpoint_every={self.checkpoint_every} must be >= 0 (0 = off)')
