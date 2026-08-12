"""LoRAVGGT - VGGT-1B carrying a LoRA adapter. Both the adapt stage and the arms go through it."""
import gc
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

    `seed` belongs HERE, not in the trainer: A is kaiming-initialised at injection, so seeding
    afterwards is too late and the run is not reproducible.
    """

    def __init__(self, cfg: LoRAConfig, adapter=None, seed=None):
        from safetensors.torch import load_file
        from vggt.models.vggt import VGGT

        self.released = False    # set first, so it exists even if construction raises below
        if cfg.vggt_hw is None:
            raise SystemExit('LoRAConfig.vggt_hw is None (unresolved). Derive it first with '
                             'cfg.resolved(probe_stream_hw(colors, stream_res)) - the model needs '
                             'a concrete input size, and it must match the tracking stream.')
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
    def from_adapter(cls, adapter, cfg: LoRAConfig, seed=None):
        """Build the model the adapter was TRAINED with: its config.json wins over `cfg` (9.5).

        `adapter=None` falls through to a plain stock-VGGT build - recorded_config returns `cfg`
        untouched and nothing is loaded - which is what lets a warm and a cold start be one call.
        """
        return cls(cls.recorded_config(adapter, cfg), adapter=adapter, seed=seed)

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

    def _ensure_live(self):
        if self.released:
            raise RuntimeError('this LoRAVGGT was release()d; construct a new instance')

    def forward(self, images):
        """Aggregator once; depth head on frame 0 only; camera head on everything."""
        self._ensure_live()
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
        self._ensure_live()
        tok, ps_idx = self.model.aggregator(images[None])
        tok0 = [t[:, :1] if t is not None else None for t in tok]
        depth, _ = self.model.depth_head(tok0, images[None][:, :1], ps_idx)
        return depth[0, 0, :, :, 0]

    # ---------------------------------------------------------------- training

    def train(self, scene_dir, image_dir, out_dir, cfg, ckpt_dir=None):
        """LoRA-adapt on an extract stage's export. Returns the run summary; does NOT save.

        The caller passes summary['state'] / ['run'] to save(). `ckpt_dir` takes
        cfg.checkpoint_every's snapshots, each a full adapter directory.
        """
        self._ensure_live()
        from .trainer import run_training
        return run_training(self, scene_dir, image_dir, out_dir, cfg, ckpt_dir)

    def train_mode(self):
        self._ensure_live()
        self.model.train()          # also enables the aggregator's gradient checkpointing
        return self

    def eval_mode(self):
        self._ensure_live()
        self.model.eval()
        return self

    # ---------------------------------------------------------------- bookkeeping

    def trainable_parameters(self):
        self._ensure_live()
        return [p for p in self.model.parameters() if p.requires_grad]

    def n_trainable(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def summary(self):
        self._ensure_live()
        n_train = self.n_trainable()
        n_total = sum(p.numel() for p in self.model.parameters())
        return (f'LoRA r={self.cfg.rank} on {self.n_wrapped} Linears -> {n_train/1e6:.2f}M '
                f'trainable / {n_total/1e9:.2f}B ({100*n_train/n_total:.2f}%)')

    def state_dict(self):
        self._ensure_live()
        return lora_state_dict(self.model)

    def save(self, out_dir, state=None, extra=None):
        """adapter.safetensors + config.json. `state` overrides the live weights (keep_best)."""
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
        """Free the model. The instance is unusable afterwards - save() first."""
        if self.released:
            return
        self.released = True
        self.model = None
        gc.collect()                 # break any hook/optimiser cycles so the module is truly freed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
