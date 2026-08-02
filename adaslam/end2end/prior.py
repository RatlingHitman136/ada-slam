"""VggtPrior - the depth prior an end2end arm swaps in.

A drop-in for MotionFilter.prior_extractor; installing and restoring it is SlamRunner's job, so
nothing here can leak a patch into a later arm. Normals stay Omnidata: depth is the only variable.
"""
import torch
import torch.nn.functional as F


class VggtPrior:
    """VGGT depth + Omnidata normals. `adapter=None` is the un-adapted 'vggt_base' arm."""

    def __init__(self, cfg, adapter=None, stream_hw=None):
        from ..adapt import LoRAVGGT, aspect_lines

        # from_adapter rebuilds the structure the adapter was trained in; only the un-adapted arm
        # has nothing to read back and takes cfg.lora as written
        self.model = (LoRAVGGT.from_adapter(adapter, cfg.lora) if adapter
                      else LoRAVGGT(cfg.lora)).eval_mode()
        self.cfg = cfg
        self.hw = self.model.cfg.vggt_hw         # from_adapter may have overridden cfg.lora's
        self.label = f'{"VGGT+LoRA" if adapter else "base VGGT"} depth / Omnidata normals'

        which = f'LoRA-adapted VGGT ({adapter})' if adapter else 'base VGGT-1B (no adapter)'
        print(f'depth prior: {which} at {self.hw[1]}x{self.hw[0]}')
        print('normals    : Omnidata (unchanged, so depth is the only variable)')

        # covers the two cases the adapt stage's report cannot: an adapter trained on another
        # stream, and 'vggt_base', which has no adapter to read a size from
        if stream_hw is not None:
            for line in aspect_lines(stream_hw, self.hw, 'VggtPrior'):
                print(f'  {line}')

    def extractor(self):
        """A plain FUNCTION to install as MotionFilter.prior_extractor - never a bound method.

        Functions are descriptors, so `mf` binds as arg 0 while this VggtPrior arrives through the
        closure. A bound method or partial is not, and mf.MEAN / mf.STDV would be lost (9.3).
        """
        prior = self
        cfg = self.cfg

        @torch.amp.autocast('cuda', enabled=True)   # matches upstream prior_extractor's decorator
        @torch.no_grad()
        def prior_extractor(mf, im_tensor):
            from midas.omnidata import OmnidataModel
            from torchvision import transforms
            input_size = im_tensor.shape[-2:]

            # normals: upstream's own code. Cached on the MotionFilter, NOT on the prior - here it
            # would hold ~1 GB alive across arms and change the VRAM profile.
            if getattr(mf, 'omni_normal', None) is None:
                mf.omni_normal = OmnidataModel('normal', cfg.omni_normal_ckpt, device='cuda:0')
            resized = transforms.Resize(cfg.omni_normal_hw, antialias=True)(im_tensor).cuda()
            normal = mf.omni_normal(resized) * 2.0 - 1.0
            normal = F.interpolate(normal, input_size, mode='bicubic').float().squeeze()

            # depth: motion_filter hands us an ImageNet-NORMALISED tensor; VGGT wants [0,1] and
            # normalises internally, so undo it or it sees doubly normalised input (9.3)
            rgb = (im_tensor * mf.STDV + mf.MEAN).clamp(0, 1)
            rgb = F.interpolate(rgb, prior.hw, mode='bilinear', align_corners=False)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                depth = prior.model.predict_depth(rgb.cuda())
            # bilinear, not bicubic: bicubic can overshoot to negative depth at edges
            depth = F.interpolate(depth.float()[None, None], input_size, mode='bilinear',
                                  align_corners=False).squeeze().clamp(min=1e-3)
            return depth, normal

        return prior_extractor

    def release(self):
        """~2.5 GB, not needed by the evaluation that follows the arm's SLAM run."""
        self.model.release()
