"""LoRAVGGT - VGGT-1B carrying a LoRA adapter.

The one object callers talk to. Both the adapt stage (which trains it) and the A/B arms (which
run it as MotionFilter's depth prior) go through this class, so there is a single definition of
what "VGGT with our adapter" means.
"""
import json
import os
from dataclasses import replace

import torch

from .config import LoRAConfig
from .lora import inject_lora, lora_state_dict

# structural keys in the adapter's config.json -> LoRAConfig fields. These names predate this
# package and MUST NOT change: adapters already on disk are read back through them.
_RECORDED = {'rank': 'rank', 'alpha': 'alpha', 'targets': 'targets',
             'lora_patch_embed': 'patch_embed', 'vggt_hw': 'vggt_hw'}


class LoRAVGGT:
    """VGGT-1B with LoRA injected into the aggregator, optionally loaded from an adapter.

    `seed`, when given, seeds torch immediately before the model is built. It lives here rather
    than in the trainer because LoRALinear's A matrix is kaiming-initialised at injection time:
    seeding afterwards is too late, and A's values steer the whole training trajectory even though
    the adapter is identity at step 0 (B starts at zero).
    """

    def __init__(self, cfg: LoRAConfig, adapter=None, seed=None):
        from safetensors.torch import load_file
        from vggt.models.vggt import VGGT

        if seed is not None:
            torch.manual_seed(seed)

        self.cfg, self.adapter = cfg, adapter
        model = VGGT.from_pretrained(cfg.weights)
        model.point_head, model.track_head = None, None       # not supervised
        for p in model.parameters():
            p.requires_grad_(False)
        self.n_wrapped = inject_lora(model, cfg)
        if adapter is not None:
            missing = model.load_state_dict(load_file(adapter), strict=False)
            assert not missing.unexpected_keys, missing.unexpected_keys
        self.model = model.cuda()

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_adapter(cls, adapter, cfg: LoRAConfig):
        """Build the model the adapter was TRAINED with, not the one `cfg` asks for.

        An adapter only means anything inside the structure it was trained in: a different rank or
        target set is a different set of tensors, and a different vggt_hw is an input distribution
        it never saw. So its config.json wins over `cfg`, which supplies only what the file cannot
        - where the VGGT-1B snapshot lives on this machine.
        """
        return cls(cls.recorded_config(adapter, cfg), adapter=adapter)

    @staticmethod
    def recorded_config(adapter, cfg: LoRAConfig):
        """`cfg` with every structural field the adapter recorded substituted in."""
        path = os.path.join(os.path.dirname(adapter or ''), 'config.json')
        if not adapter or not os.path.exists(path):
            return cfg
        recorded = json.load(open(path))
        merged = replace(cfg, **{field: recorded[key] for key, field in _RECORDED.items()
                                 if key in recorded})
        for field in _RECORDED.values():
            if getattr(merged, field) != getattr(cfg, field):
                print(f'note: adapter recorded {field}={getattr(merged, field)}, '
                      f'ignoring the configured {getattr(cfg, field)}')
        return merged

    # ---------------------------------------------------------------- inference

    def forward(self, images):
        """Aggregator once; depth head on frame 0 only; camera head on everything."""
        tok, ps_idx = self.model.aggregator(images[None])
        # this build caches only layers 4/11/17/23 and leaves the rest None to save memory
        # (aggregator.py:196) - the frame slice must preserve those Nones
        tok0 = [t[:, :1] if t is not None else None for t in tok]
        depth, _ = self.model.depth_head(tok0, images[None][:, :1], ps_idx)
        pose_enc = self.model.camera_head(tok)[-1]
        return depth[0, 0, :, :, 0], pose_enc[0]

    @torch.no_grad()
    def predict_depth(self, images):
        """Depth for a single frame. Skips camera_head, and runs the DPT head on frame 0 only."""
        tok, ps_idx = self.model.aggregator(images[None])
        tok0 = [t[:, :1] if t is not None else None for t in tok]
        depth, _ = self.model.depth_head(tok0, images[None][:, :1], ps_idx)
        return depth[0, 0, :, :, 0]

    # ---------------------------------------------------------------- training

    def train(self, scene_dir, image_dir, out_dir, cfg):
        """LoRA-adapt on an extract stage's export. Returns the run summary."""
        from .trainer import run_training
        return run_training(self, scene_dir, image_dir, out_dir, cfg)

    def train_mode(self):
        self.model.train()          # also enables the aggregator's gradient checkpointing
        return self

    def eval_mode(self):
        self.model.eval()
        return self

    # ---------------------------------------------------------------- bookkeeping

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def n_trainable(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def summary(self):
        n_train = self.n_trainable()
        n_total = sum(p.numel() for p in self.model.parameters())
        return (f'LoRA r={self.cfg.rank} on {self.n_wrapped} Linears -> {n_train/1e6:.2f}M '
                f'trainable / {n_total/1e9:.2f}B ({100*n_train/n_total:.2f}%)')

    def state_dict(self):
        return lora_state_dict(self.model)

    def save(self, out_dir, state=None, extra=None):
        """adapter.safetensors + config.json.

        The structural keys are this class's to write - they are what from_adapter reads back.
        `extra` is whatever the caller wants recorded alongside; the trainer puts the run there.
        `state` overrides the live weights, for keep_best's snapshot.
        """
        from safetensors.torch import save_file
        os.makedirs(out_dir, exist_ok=True)
        save_file(self.state_dict() if state is None else state, f'{out_dir}/adapter.safetensors')

        cfg = {'rank': self.cfg.rank, 'alpha': self.cfg.alpha, 'targets': list(self.cfg.targets),
               'lora_patch_embed': self.cfg.patch_embed, 'vggt_hw': list(self.cfg.vggt_hw),
               'weights': self.cfg.weights, 'trainable_params': self.n_trainable()}
        cfg.update(extra or {})
        json.dump(cfg, open(f'{out_dir}/config.json', 'w'), indent=2)
        return f'{out_dir}/adapter.safetensors'

    def release(self):
        """Drop the model so the caller's cache-emptying can actually reclaim the VRAM."""
        self.model = None
