"""The two configs.

No field carries a default. Whoever runs an adaptation states every value, so there is exactly
one place per entry point where a hyperparameter is written down and nothing can be inherited
silently from this package.

The split is by lifetime, not by topic:

  LoRAConfig   the STRUCTURE - what must be identical between training and inference. It is
               recorded into the adapter's config.json and read back by LoRAVGGT.from_adapter.
  AdaptConfig  the RUN - what only training cares about.
"""
from dataclasses import dataclass
from typing import Tuple

DEPTH_SOURCES = ('slam', 'rendered')
DEPTH_SPACES = ('depth', 'disparity')
SPLIT_MODES = ('stride', 'contiguous', 'random')


@dataclass(frozen=True)
class LoRAConfig:
    """Model + adapter structure.

    vggt_hw MUST match the tracking stream's aspect ratio: both SceneData.frame() and the prior
    extractor resize straight to it with no letterboxing, so a mismatched aspect squashes the
    image off VGGT's training distribution. (294, 518) suits Replica's 344x616, (378, 518) suits
    TUM's 400x544. SceneData.aspect_report() checks this and suggests a value.
    """
    weights: str                  # local VGGT-1B snapshot, e.g. pretrained_models/vggt
    vggt_hw: Tuple[int, int]      # both dims %14, VGGT's patch grid
    rank: int
    alpha: int
    targets: Tuple[str, ...]      # Linear leaves to wrap inside each aggregator block
    patch_embed: bool             # False = adapt only the alternating-attention stack

    def __post_init__(self):
        # normalise, so a config rebuilt from JSON (lists) compares equal to a hand-written one
        object.__setattr__(self, 'vggt_hw', tuple(self.vggt_hw))
        object.__setattr__(self, 'targets', tuple(self.targets))
        h, w = self.vggt_hw
        if h % 14 or w % 14:
            raise ValueError(f'vggt_hw ({h}, {w}): both dims must be divisible by 14')


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

    def __post_init__(self):
        for name, allowed in (('depth_source', DEPTH_SOURCES), ('depth_space', DEPTH_SPACES),
                              ('split_mode', SPLIT_MODES)):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f'{name}={value!r} is not one of {allowed}')
