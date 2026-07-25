"""LoRA adaptation of VGGT on HI-SLAM2's own SLAM depth (ARCHITECTURE.md §9, stage 2).

    from adapt import AdaptConfig, LoRAConfig, LoRAVGGT

    lora = LoRAVGGT(LoRAConfig(weights='pretrained_models/vggt', vggt_hw=(378, 518), ...),
                    seed=cfg.seed)
    lora.train(scene_dir, image_dir, out_dir, AdaptConfig(epochs=10, ...))
    lora.release()

and, on the inference side, the same class rebuilt from what an adapter recorded:

    lora = LoRAVGGT.from_adapter('.../lora-vggt/adapter.safetensors', fallback_cfg)
    depth = lora.predict_depth(rgb)

Nothing here carries a hyperparameter of its own: every value arrives through the two config
dataclasses, so the caller's constant block stays the only place a knob is written.

Import cost is deliberate. torch/cv2/numpy arrive at import time - every caller already has them -
but vggt, safetensors and scipy are imported inside the functions that need them, because
scripts/run_pipeline.py spawns its image reader with 'spawn', which re-imports the main module and
everything it pulled in.
"""
import os
import sys

# The irreducible four lines: `paths` is itself a sibling module, so ada-slam/ has to reach
# sys.path before it can be imported. Everything after this goes through paths.bootstrap.
_ADA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # <repo>/ada-slam
if _ADA not in sys.path:
    sys.path.insert(0, _ADA)

from paths import VGGT, bootstrap                                # noqa: E402

bootstrap(VGGT)      # `common` is a sibling module; vggt is vendored, not installed

from .config import AdaptConfig, LoRAConfig, aspect_lines, vggt_hw_for   # noqa: E402
from .data import SceneData, split_keyframes, tum_to_c2w         # noqa: E402
from .losses import depth_loss, median_scale, pose_loss          # noqa: E402
from .lora import LoRALinear, inject_lora, lora_state_dict       # noqa: E402
from .model import LoRAVGGT                                      # noqa: E402
from .trainer import eval_depth, run_training                    # noqa: E402

__all__ = ['AdaptConfig', 'LoRAConfig', 'LoRALinear', 'LoRAVGGT', 'SceneData', 'aspect_lines',
           'depth_loss', 'eval_depth', 'inject_lora', 'lora_state_dict', 'median_scale',
           'pose_loss', 'run_training', 'split_keyframes', 'tum_to_c2w', 'vggt_hw_for']
