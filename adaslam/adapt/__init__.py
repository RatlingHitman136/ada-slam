"""Stage 2: LoRA adaptation of VGGT on HI-SLAM2's own SLAM depth (9.2.1).

    lora = LoRAVGGT(LoRAConfig(...), seed=cfg.seed)      # seed here: A is init'd at injection
    summary = lora.train(scene_dir, image_dir, out_dir, AdaptConfig(...), ckpt_dir=ckpt_dir)
    lora.save(out_dir, state=summary['state'], extra=summary['run'])   # training does not save
    lora.release()                                                     # save() first: this kills it

vggt, safetensors and scipy are imported inside the functions that need them - run_pipeline.py
spawns its reader, which re-imports the main module and everything it pulled in.
"""
from .config import AdaptConfig, LoRAConfig, aspect_lines
from .model import LoRAVGGT

__all__ = ['AdaptConfig', 'LoRAConfig', 'LoRAVGGT', 'aspect_lines']
