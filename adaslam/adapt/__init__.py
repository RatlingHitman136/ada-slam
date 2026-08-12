"""Stage 2: LoRA adaptation of VGGT on HI-SLAM2's own SLAM depth (9.2.1).

`run_adapt` is the whole stage; what it wraps, for a REPL that wants the pieces:

    lora = LoRAVGGT.from_adapter(None, LoRAConfig(...), seed=cfg.seed)  # None = stock VGGT-1B
    summary = lora.train(scene_dir, image_dir, out_dir, AdaptConfig(...), ckpt_dir=ckpt_dir)
    lora.save(out_dir, state=summary['state'], extra=summary['run'])   # training does not save
    lora.release()                                                     # save() first: this kills it

vggt, safetensors and scipy are imported inside the functions that need them - a driver spawns its
reader, which re-imports the main module and everything it pulled in. data/losses/trainer are
likewise absent from this __init__, arriving only when train() defers to run_training.
"""
from .config import AdaptConfig, LoRAConfig, aspect_lines
from .model import LoRAVGGT
from .stage import run_adapt

__all__ = ['AdaptConfig', 'LoRAConfig', 'LoRAVGGT', 'aspect_lines', 'run_adapt']
