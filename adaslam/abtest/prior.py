"""VggtPrior - the depth prior an A/B arm swaps in.

It produces a drop-in for MotionFilter.prior_extractor and owns the model behind it; installing
and restoring is SlamRunner.run(prior=...)'s job, so this module never touches MotionFilter and
nothing here can leak a patch into a later arm.

Normals stay Omnidata, unchanged from upstream - depth is the only variable between the arms.
"""
import torch
import torch.nn.functional as F


class VggtPrior:
    """VGGT depth + Omnidata normals. `adapter=None` is the un-adapted 'vggt_base' arm."""

    def __init__(self, cfg, adapter=None, stream_hw=None):
        from ..adapt import LoRAVGGT, aspect_lines

        # from_adapter rebuilds cfg.lora from what the adapter recorded - rank, targets and above
        # all the input size it was trained at - and says so when that differs. Only the
        # un-adapted arm has nothing to read back and is free to take cfg.lora as written.
        self.model = (LoRAVGGT.from_adapter(adapter, cfg.lora) if adapter
                      else LoRAVGGT(cfg.lora)).eval_mode()
        self.cfg = cfg
        self.hw = self.model.cfg.vggt_hw         # from_adapter may have overridden cfg.lora's
        self.label = f'{"VGGT+LoRA" if adapter else "base VGGT"} depth / Omnidata normals'

        which = f'LoRA-adapted VGGT ({adapter})' if adapter else 'base VGGT-1B (no adapter)'
        print(f'depth prior: {which} at {self.hw[1]}x{self.hw[0]}')
        print('normals    : Omnidata (unchanged, so depth is the only variable)')

        # The check the inference path never had. It matters most for exactly the two cases the
        # adapt stage's report cannot cover: an adapter whose recorded size was trained on a
        # different stream, and the 'vggt_base' arm, which has no adapter to read a size from.
        if stream_hw is not None:
            for line in aspect_lines(stream_hw, self.hw, 'VggtPrior'):
                print(f'  {line}')

    def extractor(self):
        """A plain function to install as MotionFilter.prior_extractor.

        It MUST be a function, not a bound method. Functions are descriptors, so reaching this
        through `self.prior_extractor(...)` binds the MotionFilter as the first argument while
        this VggtPrior arrives through the closure cell. A bound method - or functools.partial -
        is not a descriptor: instance access hands back the same object still bound to the
        VggtPrior, `mf` is never passed, and mf.MEAN / mf.STDV / the cached normal model are lost.
        """
        prior = self
        cfg = self.cfg

        @torch.amp.autocast('cuda', enabled=True)   # matches upstream prior_extractor's decorator
        @torch.no_grad()
        def prior_extractor(mf, im_tensor):
            from midas.omnidata import OmnidataModel
            from torchvision import transforms
            input_size = im_tensor.shape[-2:]

            # --- normals: unchanged from upstream (motion_filter.py:70-72), minus the depth model
            # cached on the MotionFilter, NOT on the prior: keeping it here would hold ~1 GB alive
            # across arms and change the VRAM profile
            if getattr(mf, 'omni_normal', None) is None:
                mf.omni_normal = OmnidataModel('normal', cfg.omni_normal_ckpt, device='cuda:0')
            resized = transforms.Resize(cfg.omni_normal_hw, antialias=True)(im_tensor).cuda()
            normal = mf.omni_normal(resized) * 2.0 - 1.0
            normal = F.interpolate(normal, input_size, mode='bicubic').float().squeeze()

            # --- depth: VGGT ---
            # motion_filter.py:88-89 hands us an ImageNet-NORMALISED tensor, but VGGT expects
            # [0,1] and normalises internally (aggregator.py:205). Undo it, or VGGT sees doubly
            # normalised input.
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
