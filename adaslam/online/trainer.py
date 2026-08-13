"""LiveTrainer - one optimiser step burst per arriving keyframe, inside the SLAM run.

The offline trainer walks a finished export; this one is called from the depth prior itself and
sees the map as it grows. What it shares with adapt/trainer.py is deliberate: batches_of, the two
losses, and the log record shape, so train_log.json reads the same either way.

ONE AdamW is built for the whole run. That is what makes this continual rather than a sequence of
independent fits - the moments carry from the first keyframe to the last, exactly as the offline
'online' style already does within its loop.
"""
import time

import numpy as np
import torch

from ..adapt.losses import depth_loss, pose_loss
from ..adapt.trainer import batches_of

from .target import LiveSampler, unit_keyframes


class LiveTrainer:
    """Adapts `lora` on keyframes the tracker has already settled. Writes nothing but checkpoints.

    `record` is `f(trainer, unit) -> dict`, supplied by the stage: a checkpoint has to carry the
    whole run's configuration, which this class does not know.
    """

    def __init__(self, lora, cfg, ckpt_dir=None, record=None):
        self.lora, self.cfg = lora, cfg
        self.sampler = LiveSampler(cfg, lora.cfg.vggt_hw)
        self.trainable = lora.trainable_parameters()
        self.opt = torch.optim.AdamW(self.trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.rng = np.random.default_rng(cfg.seed)
        self.ckpt_dir, self._record = ckpt_dir, record

        self.log = []
        self.units = 0             # arriving keyframes adapted on
        self.visits = 0            # keyframes pushed through VGGT - 12.1's adapt_cost
        self.trained_kf = set()    # distinct FRAME indices ever trained on
        self.first_kf = None       # the keyframe index of the first step
        self.last_tstamp = None    # the target's FRAME index, so a pruned-and-refilled keyframe
                                   # slot is not mistaken for a new arrival - see on_keyframe
        self.warmup_end_frame = None   # set by the prior at handover; recorded, never read here
        self.t0 = time.time()

    # ---------------------------------------------------------------- schedule

    def batches(self, kfs):
        """The batches one arrival trains on - the ONLY place the two live styles differ.

        online   the arrival alone, steps_per_kf consecutive single-keyframe steps. batch_size is
                 not read: a keyframe arrives alone.
        wonline  steps_per_kf shuffled passes over the window, batch_size at a time - so a keyframe
                 is revisited for window_size arrivals instead of being seen once and dropped.
        """
        if self.cfg.adapt_style == 'wonline':
            return [b for _ in range(self.cfg.steps_per_kf)
                    for b in batches_of(self.rng.permutation(kfs), self.cfg.batch_size)]
        return [[int(kfs[-1])] for _ in range(self.cfg.steps_per_kf)]

    # ---------------------------------------------------------------- the step

    def on_keyframe(self, video):
        """One unit of adaptation for the keyframe that just arrived. Returns the unit index.

        Called from inside the depth prior, i.e. under MotionFilter.track's no_grad AND its fp16
        autocast. Both are undone here: the caller opens enable_grad, and this disables the
        ambient autocast so only the explicit bfloat16 block around the forward is in effect -
        the conditions adapt/trainer.py trains under.
        """
        kfs = unit_keyframes(video, self.cfg)
        if not kfs or self.cfg.steps_per_kf < 1:
            return None

        # ONE UNIT PER DISTINCT ARRIVAL. The extractor runs for every keyframe the motion filter
        # accepts, but track_frontend.py:52 prunes a redundant one and DECREMENTS counter, so the
        # next acceptance lands on the same index and would re-train the same target - measured on
        # rellis_00000, 500 frames: 88 calls collapse to 29 units. Identity is the frame TIMESTAMP,
        # not the index: indices shift under that same pruning, timestamps do not.
        tstamp = float(video.tstamp[kfs[-1]].item())
        if tstamp == self.last_tstamp:
            return None
        self.last_tstamp = tstamp

        batches = self.batches(kfs)
        if not batches:
            return None
        if self.first_kf is None:
            self.first_kf = int(tstamp)      # a FRAME index, like trained_kf and the log's 'kfs'

        unit = self.units
        self.lora.train_mode()          # also enables the aggregator's gradient checkpointing
        try:
            with torch.amp.autocast('cuda', enabled=False):
                for step, batch in enumerate(batches):
                    self._step(video, unit, step, len(batches), batch)
        finally:
            self.lora.eval_mode()       # the extractor predicts next

        self.units += 1
        # by FRAME index, for the same reason as last_tstamp: keyframe slot 31 before a pruning and
        # slot 31 after it are different frames, and n_train_kf feeds 12.1's adapt_cost
        self.trained_kf.update(int(video.tstamp[t].item()) for t in kfs)
        self._checkpoint()
        return unit

    def _step(self, video, unit, step, n_steps, batch):
        """One optimiser step over `batch`, as adapt/trainer.py:199-243. Returns the mean loss."""
        cfg = self.cfg
        self.opt.zero_grad(set_to_none=True)
        acc = {'loss': [], 'l_depth': [], 'l_trans': [], 'l_rot': [], 'scale_ratio': []}
        seq_lens = []

        for t in batch:
            images, gt, mask, gt_enc, seq = self.sampler.sample(video, t)
            images, gt, mask, gt_enc = images.cuda(), gt.cuda(), mask.cuda(), gt_enc.cuda()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_depth, pred_enc = self.lora.forward(images)
            pred_depth, pred_enc = pred_depth.float(), pred_enc.float()

            l_t, l_r, pose_scale = pose_loss(pred_enc, gt_enc)
            l_d, depth_scale = depth_loss(pred_depth, gt, mask, cfg,
                                          scale=pose_scale if cfg.coupled_scale else None)
            loss = l_d + cfg.lambda_pose * (l_t + l_r)

            # the MEAN over the batch, so grad magnitude is independent of batch_size
            (loss / len(batch)).backward()
            self.visits += 1

            seq_lens.append(len(seq))
            acc['loss'].append(loss.item())
            acc['l_depth'].append(l_d.item())
            acc['l_trans'].append(l_t.item())
            acc['l_rot'].append(l_r.item())
            if pose_scale is not None and depth_scale is not None:
                acc['scale_ratio'].append((depth_scale / pose_scale).item())

        torch.nn.utils.clip_grad_norm_(self.trainable, cfg.grad_clip)
        self.opt.step()

        # same shape as adapt/trainer.py:233, so one reader serves both logs - and 'kfs' means the
        # same thing in both, FRAME indices: offline they come from poses_slam.txt, live they must
        # be translated off video.tstamp, because a keyframe slot is not stable across a pruning
        rec = {'epoch': unit, 'step': step, 'S': seq_lens,
               'kfs': [int(video.tstamp[t].item()) for t in batch],
               **{k: float(np.mean(v)) for k, v in acc.items() if v}}
        self.log.append(rec)

        if step % cfg.log_every == 0:
            print(f'  [adapt] kf{unit} s{step}/{n_steps}  loss {rec["loss"]:.4f} '
                  f'(d {rec["l_depth"]:.4f} t {rec["l_trans"]:.4f} r {rec["l_rot"]:.4f})  '
                  f'kfs={rec["kfs"]}  {torch.cuda.max_memory_allocated()/2**30:.1f}GiB')
        return rec['loss']

    # ---------------------------------------------------------------- bookkeeping

    def _checkpoint(self):
        """A full adapter dir every checkpoint_every_kf units, so any of them can be run as an arm.

        epoch_NNN is a CONTRACT: end2end/config.py:arm_name parses arm names off that prefix, which
        is what makes a mid-run snapshot testable as <NAME>_chkp_NNN.
        """
        if not (self.ckpt_dir and self.cfg.checkpoint_every_kf):
            return
        if self.units % self.cfg.checkpoint_every_kf:
            return
        unit = self.units - 1
        extra = {**(self._record(self, unit) if self._record else {}), 'checkpoint': True}
        print(f'  [adapt] checkpoint -> '
              f'{self.lora.save(f"{self.ckpt_dir}/epoch_{unit:03d}", extra=extra)}')

    def stats(self):
        """What the adapter's config.json records about the training that happened.

        The key NAMES are the offline ones wherever they mean the same thing, so
        scripts/export_end2end_results.py computes adapt_cost with no change (12.1): in both live
        styles a UNIT is an arriving keyframe and `epochs` is the steps taken on it, which is
        exactly what that table's 'online'/'wonline' rows already assume.
        """
        cfg = self.cfg
        return {'online': True,
                'adapt_style': cfg.adapt_style,
                'epochs': cfg.steps_per_kf,           # 'epochs' IS steps-per-unit in both styles
                'batch_size': cfg.batch_size, 'window_size': cfg.window_size,
                'n_units': self.units, 'n_train_kf': len(self.trained_kf),
                'kf_visits': self.visits, 'first_adapted_kf': self.first_kf,
                'steps': len(self.log), 'lr': cfg.lr,
                'weight_decay': cfg.weight_decay, 'grad_clip': cfg.grad_clip,
                'lambda_pose': cfg.lambda_pose, 'coupled_scale': cfg.coupled_scale,
                'context_kf': cfg.context_kf, 'lag': cfg.lag, 'seed': cfg.seed,
                'stream_res': cfg.stream_res,
                'warmup_kf': cfg.warmup_kf, 'warmup_prior': cfg.warmup_prior,
                'warmup_end_frame': self.warmup_end_frame,
                'checkpoint_every_kf': cfg.checkpoint_every_kf,
                # lineage as data, read off the model rather than passed in - so checkpoints carry
                # it too, exactly as adapt/trainer.py:180 does
                'init_adapter': self.lora.adapter,
                'train_seconds': round(time.time() - self.t0, 1)}

    def summary(self):
        """The one-paragraph read on a finished run, printed by the stage."""
        if not self.log:
            return ('  NO optimiser step ran - either steps_per_kf is 0 (the null-op arm) or the '
                    'run never got past warmup_kf keyframes')
        losses = [r['loss'] for r in self.log]
        head, tail = losses[:max(1, len(losses) // 10)], losses[-max(1, len(losses) // 10):]
        return (f'  {self.units} units / {len(self.log)} steps / {self.visits} keyframe visits '
                f'over {len(self.trained_kf)} distinct keyframes, from frame {self.first_kf}\n'
                f'  loss first 10% {np.mean(head):.4f} -> last 10% {np.mean(tail):.4f} '
                f'({time.time()-self.t0:.0f}s) - NOT a learning curve: every step has a different\n'
                f'  target, so this tracks how hard the scene got as much as how well it fits')

    def release(self):
        """Drop the optimiser state before the model goes; the arm's evaluation needs neither."""
        self.opt = None
        self.trainable = None
