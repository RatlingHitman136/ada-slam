"""OnlineVggtPrior - the depth prior that adapts itself while the map is being built (13).

Same contract as end2end/prior.py:VggtPrior - `.label`, `.extractor()`, `.release()` - so
SlamRunner.run installs and restores it with no change at all, including the finally that stops one
arm's prior leaking into the next.

What it adds is two branches around the parent's extractor:

  * before warmup_kf keyframes exist, SLAM is served by the FALLBACK prior, so the map has
    something to bootstrap from that this run has not yet influenced;
  * from warmup_kf + 1 on, every call first takes a burst of optimiser steps on the keyframes the
    tracker has already settled, then predicts with the weights that produced.

Normals stay Omnidata on BOTH branches (the parent's job), so depth remains the only variable
between this arm and the baselines.
"""
import torch

from ..end2end.prior import VggtPrior

from .trainer import LiveTrainer


class OnlineVggtPrior(VggtPrior):
    """VGGT adapted live on the SLAM depth of the run it is serving."""

    def __init__(self, cfg, online_cfg, adapter=None, stream_hw=None, ckpt_dir=None, record=None):
        # Seed BEFORE super(), which builds the model: LoRALinear.A is kaiming-initialised when
        # LoRA is injected, so seeding afterwards is too late and the run is not reproducible
        # (9.5). The parent has no seed argument because an arm only ever runs a frozen adapter.
        #
        # RESTORED IMMEDIATELY AFTERWARDS, and that is not tidiness. manual_seed resets the GLOBAL
        # stream, and the Gaussian backend draws from it all run long - the mapping window's random
        # earlier keyframes, densification, pcd_downsample. A vggt_base arm never seeds, so leaving
        # it reset builds a different map, and gs.finalize()'s pose deltas are written back into
        # video.poses (11.2), so the whole trajectory moves. Measured on rellis_00000/500: 2.2e-2
        # max pose difference against the base arm, 13x the 1.7e-3 non-reproducibility floor.
        # Snapshotting keeps A seeded AND leaves the SLAM run the exact stream base would have had.
        rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        torch.manual_seed(online_cfg.seed)
        try:
            super().__init__(cfg, adapter, stream_hw)
        finally:
            torch.set_rng_state(rng_cpu)
            torch.cuda.set_rng_state_all(rng_cuda)
        self.online = online_cfg
        self.trainer = LiveTrainer(self.model, online_cfg, ckpt_dir=ckpt_dir, record=record)

        # CAPTURED HERE, NOT AT CALL TIME. SlamRunner.run overwrites
        # MotionFilter.prior_extractor with ours (runner.py:81-83) before the first frame, so
        # fetching the stock one lazily from inside our own extractor would fetch itself and
        # recurse forever. This runs while the class attribute is still upstream's.
        self._fallback = None
        if online_cfg.warmup_prior == 'omnidata':
            from ..slam import stock_prior_extractor
            self._fallback = stock_prior_extractor()

        which = 'Omnidata' if self._fallback else 'this same VGGT, frozen'
        self.label = (f'VGGT adapted ONLINE ({online_cfg.adapt_style}, '
                      f'{online_cfg.steps_per_kf} steps/kf) / Omnidata normals')
        print(f'online     : first {online_cfg.warmup_kf} keyframes served by {which}; '
              f'adaptation starts at keyframe {online_cfg.warmup_kf + 1}')
        print(f'             target = 1/disps_up of keyframe counter-1-{online_cfg.lag} '
              f'(local BA, not global), context {online_cfg.context_kf} keyframes')

    def extractor(self):
        """The parent's extractor with the warm-up branch and the adaptation step around it.

        Still a plain FUNCTION, never a bound method - 9.3's descriptor reasoning is unchanged: mf
        binds as arg 0 and everything else arrives through the closure.
        """
        vggt_fn = super().extractor()             # normals + VGGT depth, reused verbatim
        fallback = self._fallback or vggt_fn      # 'self' = the same model, just not adapting yet
        cfg, trainer = self.online, self.trainer

        @torch.no_grad()
        def prior_extractor(mf, im_tensor):
            video = mf.video                      # the tracker's own state - see target.py
            n = video.counter.value               # keyframes already in the map

            # ready is set ONLY at hi2.py:107, the first line of terminate(). Past that point the
            # extractor is still called - for the keyframes terminate() inserts into
            # low-covisibility gaps (hi2.py:143) - while video.shift is moving every index, so
            # adapting there would train on a map that is being rewritten underneath.
            if video.ready.value == 0 and n > cfg.warmup_kf:
                with torch.enable_grad():
                    trainer.on_keyframe(video)

            if n < cfg.warmup_kf:
                return fallback(mf, im_tensor)
            # recorded on the TRAINER, which is what writes the adapter's config.json. It is the
            # one frame index that separates fallback-served tracking from VGGT-served tracking,
            # so it is the split any later re-scoring would want (12.3).
            if trainer.warmup_end_frame is None and n:
                trainer.warmup_end_frame = int(video.tstamp[n - 1].item()) + 1
                print(f'  [online] handover at keyframe {n}, frame '
                      f'{trainer.warmup_end_frame}: VGGT is the depth prior from here')
            return vggt_fn(mf, im_tensor)

        return prior_extractor

    def save(self, out_dir, extra=None):
        """The adapter this run produced, in the normal handoff shape (adapt/stage.py's).

        Before release(), always: LoRAVGGT.save goes through _ensure_live().
        """
        return self.model.save(out_dir, extra=extra)

    def release(self):
        self.trainer.release()
        super().release()
