"""HI-SLAM2 on a full sequence with LoRA-adapted VGGT supplying the depth prior.

Normals still come from Omnidata, so exactly ONE thing differs from run_full_omnidata.py:

    omnidata run : depth = Omnidata   normal = Omnidata
    vggt run     : depth = VGGT+LoRA  normal = Omnidata

  python scripts/run_full_vggt.py --output outputs/ab/room0_vggt \
      --adapter outputs/replica/room0_p40/lora-vggt/adapter.safetensors

The swap is a monkey-patch of MotionFilter.prior_extractor, so hislam2/ is untouched and the
baseline arm runs the completely stock code path.
"""
import os    # nopep8
import sys   # nopep8
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # nopep8

import torch
import torch.nn.functional as F
from torchvision import transforms

from _full_run_common import main
from motion_filter import MotionFilter
from midas.omnidata import OmnidataModel
from lora_adapt_vggt import load_model, VGGT_HW

_MODEL = None


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
    rgb = F.interpolate(rgb, VGGT_HW, mode='bilinear', align_corners=False)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        depth = _vggt_depth(rgb.cuda())
    # bilinear, not bicubic: bicubic can overshoot to negative depth at edges
    depth = F.interpolate(depth.float()[None, None], input_size, mode='bilinear',
                          align_corners=False).squeeze().clamp(min=1e-3)
    return depth, normal


def install(args):
    global _MODEL
    _MODEL = load_model(adapter=args.adapter)[0].eval()
    MotionFilter.prior_extractor = vggt_prior_extractor
    print(f'depth prior: LoRA-adapted VGGT ({args.adapter})')
    print('normals    : Omnidata (unchanged, so depth is the only variable)')


def add_args(p):
    p.add_argument('--adapter', default='outputs/replica/room0_p40/lora-vggt/adapter.safetensors')


if __name__ == '__main__':
    main('VGGT+LoRA depth / Omnidata normals', patch=install, extra_args=add_args)
