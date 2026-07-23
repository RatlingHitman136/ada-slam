"""The LoRA mechanism itself - hand-rolled, no peft dependency."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """y = W0 x + (B A) x * alpha/r, with B zero-initialised so the adapter starts as identity."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scaling


def inject_lora(model, cfg):
    """Wrap the targeted Linears inside the aggregator's frame/global blocks.

    Heads and patch_embed stay frozen; gradients still reach the aggregator *through* them.
    Returns the number of Linears wrapped.
    """
    blocks = list(model.aggregator.frame_blocks) + list(model.aggregator.global_blocks)
    if cfg.patch_embed and hasattr(model.aggregator.patch_embed, 'blocks'):
        blocks += list(model.aggregator.patch_embed.blocks)

    n = 0
    for blk in blocks:
        for tgt in cfg.targets:
            parent_path, _, leaf = tgt.rpartition('.')
            parent = blk.get_submodule(parent_path) if parent_path else blk
            child = getattr(parent, leaf, None)
            if isinstance(child, nn.Linear):
                setattr(parent, leaf, LoRALinear(child, cfg.rank, cfg.alpha))
                n += 1
    return n


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith('.A') or k.endswith('.B')}
