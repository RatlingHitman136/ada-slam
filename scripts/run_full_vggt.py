"""HI-SLAM2 on a full sequence with VGGT supplying the depth prior.

Normals still come from Omnidata, so exactly ONE thing differs from run_full_omnidata.py:

    omnidata run : depth = Omnidata   normal = Omnidata
    vggt run     : depth = VGGT       normal = Omnidata

  python scripts/run_full_vggt.py --output outputs/ab/room0_vggt \
      --adapter outputs/replica/room0_p40/lora-vggt/adapter.safetensors

With `--adapter none` the same script runs stock VGGT-1B. That third arm is what makes a null
result readable: omnidata->base separates "is VGGT the better prior" from base->adapted's "does
adapting on our own SLAM depth add anything".

The swap is a monkey-patch of MotionFilter.prior_extractor, so hislam2/ is untouched and the
baseline arm runs the completely stock code path.
"""
import os    # nopep8
import sys   # nopep8
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # nopep8

import json

import torch
import torch.nn.functional as F
from torchvision import transforms

from _full_run_common import main
from motion_filter import MotionFilter
from midas.omnidata import OmnidataModel
import lora_adapt_vggt
from lora_adapt_vggt import load_model, parse_hw

_MODEL = None
_HW = lora_adapt_vggt.VGGT_HW


@torch.no_grad()
def _vggt_depth(images):
    """Depth for a single frame. Skips camera_head, and runs the DPT head on frame 0 only."""
    tok, ps_idx = _MODEL.aggregator(images[None])
    # this VGGT build leaves uncached layers as None so indices stay stable (aggregator.py:196)
    tok0 = [t[:, :1] if t is not None else None for t in tok]
    depth, _ = _MODEL.depth_head(tok0, images[None][:, :1], ps_idx)
    return depth[0, 0, :, :, 0]


@torch.amp.autocast('cuda', enabled=True)   # matches upstream prior_extractor's decorator
@torch.no_grad()
def vggt_prior_extractor(self, im_tensor):
    """Drop-in for MotionFilter.prior_extractor: VGGT depth, Omnidata normals."""
    input_size = im_tensor.shape[-2:]

    # --- normals: unchanged from upstream (motion_filter.py:70-72), minus the depth model ---
    if getattr(self, 'omni_normal', None) is None:
        self.omni_normal = OmnidataModel('normal', 'pretrained_models/omnidata_dpt_normal_v2.ckpt',
                                         device='cuda:0')
    resized = transforms.Resize((512, 512), antialias=True)(im_tensor).cuda()
    normal = self.omni_normal(resized) * 2.0 - 1.0
    normal = F.interpolate(normal, input_size, mode='bicubic').float().squeeze()

    # --- depth: adapted VGGT ---
    # motion_filter.py:88-89 hands us an ImageNet-NORMALISED tensor, but VGGT expects [0,1] and
    # normalises internally (aggregator.py:205). Undo it, or VGGT sees doubly-normalised input.
    rgb = (im_tensor * self.STDV + self.MEAN).clamp(0, 1)
    rgb = F.interpolate(rgb, _HW, mode='bilinear', align_corners=False)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        depth = _vggt_depth(rgb.cuda())
    # bilinear, not bicubic: bicubic can overshoot to negative depth at edges
    depth = F.interpolate(depth.float()[None, None], input_size, mode='bilinear',
                          align_corners=False).squeeze().clamp(min=1e-3)
    return depth, normal


def install(args):
    global _MODEL, _HW
    adapter = None if args.adapter.lower() in ('', 'none') else args.adapter

    # The adapter was trained at one input size and must be run at that same size, so the value
    # written into its config.json wins over anything on the command line. Only the un-adapted arm
    # is free to take --vggt_hw.
    cfg = os.path.join(os.path.dirname(adapter or ''), 'config.json')
    if adapter and os.path.exists(cfg):
        _HW = tuple(json.load(open(cfg))['vggt_hw'])
        if args.vggt_hw and parse_hw(args.vggt_hw) != _HW:
            print(f'note: ignoring --vggt_hw, adapter was trained at {_HW[0]},{_HW[1]}')
    elif args.vggt_hw:
        _HW = parse_hw(args.vggt_hw)

    _MODEL = load_model(adapter=adapter)[0].eval()
    MotionFilter.prior_extractor = vggt_prior_extractor
    which = f'LoRA-adapted VGGT ({adapter})' if adapter else 'base VGGT-1B (no adapter)'
    print(f'depth prior: {which} at {_HW[1]}x{_HW[0]}')
    print('normals    : Omnidata (unchanged, so depth is the only variable)')
    return f'{"VGGT+LoRA" if adapter else "base VGGT"} depth / Omnidata normals'


def add_args(p):
    p.add_argument('--adapter', default='outputs/replica/room0_p40/lora-vggt/adapter.safetensors',
                   help="path to adapter.safetensors, or 'none' to run stock VGGT-1B")
    p.add_argument('--vggt_hw', default=None, metavar='H,W',
                   help='VGGT input size; ignored when the adapter records its own')


if __name__ == '__main__':
    main('VGGT depth / Omnidata normals', patch=install, extra_args=add_args)
