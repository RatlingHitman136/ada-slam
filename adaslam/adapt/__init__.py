"""LoRA adaptation of VGGT on HI-SLAM2's own SLAM depth (ARCHITECTURE.md §9, stage 2).

    from adaslam.adapt import AdaptConfig, LoRAConfig, LoRAVGGT

    lora = LoRAVGGT(LoRAConfig(weights='pretrained_models/vggt', vggt_hw=(378, 518), ...),
                    seed=cfg.seed)
    summary = lora.train(scene_dir, image_dir, out_dir, AdaptConfig(epochs=10, ...),
                         ckpt_dir=ckpt_dir)          # ckpt_dir holds checkpoint_every's snapshots
    lora.save(out_dir, state=summary['state'], extra=summary['run'])   # training does not save
    lora.release()                                                     # save() first: this kills it

and, on the inference side, the same class rebuilt from what an adapter recorded:

    lora = LoRAVGGT.from_adapter('.../lora-vggt/adapter.safetensors', fallback_cfg)
    depth = lora.predict_depth(rgb)

Nothing here carries a hyperparameter of its own: every value arrives through the two config
dataclasses, so the caller's constant block stays the only place a knob is written.

Import cost is deliberate. torch/cv2/numpy arrive at import time - every caller already has them -
but vggt, safetensors and scipy are imported inside the functions that need them, because
scripts/run_pipeline.py spawns its image reader with 'spawn', which re-imports the main module and
everything it pulled in.

vggt is vendored, not installed; adaslam/__init__.py put thirdparty/vggt on sys.path before this
file could run.
"""
from .config import AdaptConfig, LoRAConfig, aspect_lines
from .model import LoRAVGGT

__all__ = ['AdaptConfig', 'LoRAConfig', 'LoRAVGGT', 'aspect_lines']
