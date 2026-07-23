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

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/ada-slam/adapt
_ROOT = os.path.dirname(os.path.dirname(_HERE))             # <repo>

# `common` is a sibling module and thirdparty/vggt is vendored rather than installed, so both need
# to be on sys.path. The caller has usually done it already; do it anyway so `import adapt` works
# from anywhere.
for _p in (os.path.dirname(_HERE), os.path.join(_ROOT, 'thirdparty/vggt')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .config import AdaptConfig, LoRAConfig                      # noqa: E402
from .data import SceneData, split_keyframes, tum_to_c2w         # noqa: E402
from .losses import depth_loss, median_scale, pose_loss          # noqa: E402
from .lora import LoRALinear, inject_lora, lora_state_dict       # noqa: E402
from .model import LoRAVGGT                                      # noqa: E402
from .trainer import eval_depth, run_training                    # noqa: E402

__all__ = ['AdaptConfig', 'LoRAConfig', 'LoRALinear', 'LoRAVGGT', 'SceneData', 'depth_loss',
           'eval_depth', 'inject_lora', 'lora_state_dict', 'median_scale', 'pose_loss',
           'run_training', 'split_keyframes', 'tum_to_c2w']
