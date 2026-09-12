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

from ..adapt.losses import depth_loss, mean_scale, median_scale, pose_loss, relative_loss
from ..adapt.trainer import batches_of

from .target import LiveSampler, unit_keyframes


class LiveTrainer:
    """Adapts `lora` on keyframes the tracker has already settled. Writes nothing but checkpoints.

    `record` is `f(trainer, unit) -> dict`, supplied by the stage: a checkpoint has to carry the
    whole run's configuration, which this class does not know.
    """

    def __init__(self, lora, cfg, ckpt_dir=None, record=None, frame_offset=0):
        self.lora, self.cfg = lora, cfg
        # SlamConfig.start, handed down by the stage rather than mirrored into OnlineConfig - there
        # is then one source for it and nothing to keep in sync. It matters because video.tstamp is
        # the index WITHIN the run (mono_stream yields t = 0..len-1) while traj_full.txt, GT and
        # evo/timestamps.npy are all absolute frame numbers. Every index this class records goes
        # through frame() so the two agree; at start=0 they coincide, which is why they could
        # disagree unnoticed until a window was runnable.
        self.frame_offset = int(frame_offset)
        self.sampler = LiveSampler(cfg, lora.cfg.vggt_hw)
        self.trainable = lora.trainable_parameters()
        self.opt = torch.optim.AdamW(self.trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.rng = np.random.default_rng(cfg.seed)
        self.ckpt_dir, self._record = ckpt_dir, record

        self.log = []
        self._unit_gauge = {}      # keyframe -> frozen pose gauge, reset every unit
        # the history anchor (config.py:anchor_kf). Stored as frame TIMESTAMPS because keyframe
        # SLOTS are not stable - track_frontend.py:52 prunes one and decrements counter, shifting
        # every index after it, so a cached slot would silently become a different frame.
        self._anchor_ts = []       # chosen once, held for the run
        self._unit_anchor = None   # the detached reference log-scale, recomputed once per unit
        self.units = 0             # arriving keyframes adapted on
        self.visits = 0            # keyframes pushed through VGGT - 12.1's adapt_cost
        self.trained_kf = set()    # distinct FRAME indices ever trained on
        self.first_kf = None       # the keyframe index of the first step
        self.last_tstamp = None    # the target's FRAME index, so a pruned-and-refilled keyframe
                                   # slot is not mistaken for a new arrival - see on_keyframe
        self.warmup_end_frame = None   # set by the prior at handover; recorded, never read here
        # the loss gate. gate_log holds EVERY arrival the gate saw, trained or not, so a threshold
        # can be re-chosen from one run instead of re-running per candidate value.
        self.gate_log = []
        self.skipped = {'low': 0, 'high': 0, 'empty': 0}
        self.t0 = time.time()

    # ---------------------------------------------------------------- frames

    def frame(self, video, i):
        """Keyframe slot `i` as an ABSOLUTE frame index - what traj_full.txt and GT are keyed by."""
        return self.frame_offset + int(video.tstamp[i].item())

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

    # ---------------------------------------------------------------- the loss gate

    def gate_value(self, video, t):
        """Keyframe `t` under the current weights, as (relative loss, raw loss). (None, None)
        when the mask is too thin to measure.

        BOTH are returned whichever one gate_metric selects, and both go into gate_log, so one run
        answers the threshold question for either metric instead of needing a run per candidate.

        One extra no-grad forward per arrival, against the steps_per_kf * window_size the burst it
        may skip would cost - 80 for the e8 configuration, so ~1%. Deliberately NOT the first
        training step's loss: reading that would mean one optimiser step had already landed on the
        very target the gate exists to reject, and a 1902x-median target does its damage in one
        step.

        The model is in eval_mode here (on_keyframe enters train_mode after the gate), which is
        also the mode it serves in - so the gate measures the weights as the tracker will see them.
        """
        images, gt, mask, _, _, _ = self.sampler.sample(video, t)
        # THE GATE READS FRAME 0, always, even under depth_all_frames. gate_lo/gate_hi are
        # calibrated against this exact quantity (online/config.py's reference distributions) and
        # every gate_log.json on disk records it, so pooling over S here would silently
        # reinterpret every recorded threshold rather than measure a new thing.
        if gt.dim() == 3:
            gt, mask = gt[0], mask[0]
        images, gt, mask = images.cuda(), gt.cuda(), mask.cuda()
        if mask.sum() < self.cfg.min_mask_pixels:
            return None, None                # depth_loss would return a zero with no gradient
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            # cache_enabled=False IS LOAD-BEARING, and silently so. torch.autocast caches the
            # bf16 casts of every weight it touches, and that cache lives until the OUTERMOST
            # autocast region exits - which here is motion_filter.track's
            # @torch.cuda.amp.autocast(enabled=True), i.e. not before this whole keyframe is done.
            # Casting the LoRA weights under no_grad would therefore leave DETACHED bf16 copies in
            # that cache, and _step's forward would reuse them a few lines later: its output comes
            # back with requires_grad=False and backward() dies on "element 0 of tensors does not
            # require grad". Nothing about the gate looks wrong when that happens - grad IS
            # enabled, the model IS in train mode, the parameters DO require grad - so it is worth
            # the sentence. Only this forward needs the opt-out: _step's own casts are made under
            # grad and are correct to cache.
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                pred_depth, _ = self.lora.forward(images)
            # gate_lo/gate_hi are calibrated against THIS quantity (online/config.py's
            # reference distributions), so the gate must keep measuring the training loss itself.
            l_d, _ = depth_loss(pred_depth.float(), gt, mask, self.cfg)
        return relative_loss(l_d, gt, mask), float(l_d)

    def gate(self, video, kfs, frame):
        """Should this arrival be trained on? Records the verdict either way.

        The gate keeps the BAND (gate_lo, gate_hi) of whichever metric gate_metric names: too low
        means the frame already fits and the update is not worth its cost, too high means the
        target is broken rather than informative. See online/config.py for why the upper bound is
        the half with evidence behind it, and why 'rel' is the sounder of the two metrics.
        """
        cfg = self.cfg
        lo, hi = cfg.gate_lo, cfg.gate_hi
        if lo <= 0 and hi <= 0:
            return True                      # gate off - do not spend the forward
        rel, raw = self.gate_value(video, kfs[-1])
        val = rel if cfg.gate_metric == 'rel' else raw
        if val is None:
            verdict = 'empty'
        elif lo > 0 and val < lo:
            verdict = 'low'
        elif hi > 0 and val > hi:
            verdict = 'high'
        else:
            verdict = 'train'
        # both metrics are recorded whichever one decided, so gate_log.json can be re-thresholded
        # on either axis afterwards without another run
        self.gate_log.append({'frame': frame, 'rel': rel, 'raw': raw, 'metric': cfg.gate_metric,
                              'verdict': verdict, 'unit': self.units})
        if verdict == 'train':
            return True
        self.skipped[verdict] += 1
        print(f'  [adapt] SKIP kf frame {frame}: {cfg.gate_metric} '
              f'{"n/a" if val is None else f"{val:.4f}"} ({verdict})')
        return False

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
        tstamp = float(self.frame(video, kfs[-1]))
        if tstamp == self.last_tstamp:
            return None
        self.last_tstamp = tstamp

        # AFTER the de-dup gate, so a skipped arrival is not retried on the next extractor call,
        # and BEFORE first_kf is claimed, so first_adapted_kf stays "the first frame actually
        # trained on" rather than the first one merely looked at.
        if not self.gate(video, kfs, int(tstamp)):
            return None

        batches = self.batches(kfs)
        if not batches:
            return None

        if self.first_kf is None:
            self.first_kf = int(tstamp)      # a FRAME index, like trained_kf and the log's 'kfs'

        unit = self.units
        # one gauge per keyframe per UNIT (config.py:freeze_gauge). Cleared here, filled on each
        # keyframe's first step, reused for the rest - so the window's loss surface stops moving
        # under the optimiser while the window itself is fixed.
        self._unit_gauge = {}
        # the history anchor, recomputed ONCE per unit rather than per step: per step would cost
        # anchor_kf x len(batches) extra forwards, per unit it is anchor_kf against 50.
        self._unit_anchor = None
        if self.cfg.anchor_kf > 0 and self.cfg.lambda_cons > 0:
            if not self._anchor_ts:
                self._pick_anchors(video, kfs[-1])
            if self._anchor_ts:
                self._unit_anchor = self._history_anchor(video)
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
        self.trained_kf.update(self.frame(video, t) for t in kfs)
        self._checkpoint()
        return unit

    def _batch_scale(self, samples):
        """ONE depth scale over the whole batch, from a no-grad pass (online/config.py).

        Per-sample alignment is what makes the objective blind to cross-frame scale inconsistency;
        pooling the masked pixels and taking one median ratio puts that inconsistency into the
        residual. Returns None when there is not enough valid depth to fit on, so the caller falls
        back to the per-sample gauge rather than to a garbage constant.

        cache_enabled=False IS LOAD-BEARING here for the same reason it is in gate_value: an
        autocast cache filled under no_grad survives until the OUTERMOST autocast region exits -
        motion_filter.track's - and _step's forward a few lines later would reuse those DETACHED
        bf16 weight copies, so backward() would die on "element 0 does not require grad".
        """
        gts, prs = [], []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                for images, gt, mask, _, _ in samples:
                    if not bool(mask.any()):
                        continue
                    pd, _ = self.lora.forward(images)
                    gts.append(gt[mask])
                    prs.append(pd.float().clamp(min=1e-3)[mask])
        if not gts:
            return None
        g, p = torch.cat(gts), torch.cat(prs)
        if g.numel() < self.cfg.min_mask_pixels:
            return None
        return (g.median() / p.median().clamp(min=1e-6)).detach()

    def _scale_anchor(self, samples):
        """mean of log(per-sample median scale) over the batch, DETACHED (online/config.py).

        The consensus each sample's own scale is pulled toward by lambda_cons. Detached and taken
        from a no-grad pass so it is a fixed target for the step rather than a mean that moves as
        the samples move - the failure batch_scale had was a gauge that chased its own input.

        None when fewer than two samples yield a scale: consistency needs something to be
        consistent with, and the caller then adds no penalty at all.

        cache_enabled=False is load-bearing for the same reason as in gate_value and _batch_scale.
        """
        ls = []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                for images, gt, mask, _, _ in samples:
                    if mask.sum() < self.cfg.min_mask_pixels:
                        continue
                    pd, _ = self.lora.forward(images)
                    s = median_scale(pd.float().clamp(min=1e-3), gt.clamp(min=1e-3), mask)
                    ls.append(torch.log(s.clamp(min=1e-9)))
        if len(ls) < 2:
            return None
        return torch.stack(ls).mean().detach()

    def _pick_anchors(self, video, hi):
        """Choose the anchor keyframes ONCE, as far back as the map reaches, and keep them.

        Eligible slots are [0, hi - window_size] - everything the current unit's window does not
        already cover - and `evenly` spreads anchor_kf of them across it, endpoints included
        (adapt/data.py:28). Nothing is chosen until that range holds at least anchor_kf keyframes,
        so the set is not built out of two barely-settled frames at the very start of the run.

        Converted to timestamps immediately: slots move under pruning, frames do not.
        """
        from ..adapt.data import evenly
        span = hi - self.cfg.window_size
        if span + 1 < self.cfg.anchor_kf:
            return
        self._anchor_ts = [float(video.tstamp[i].item())
                           for i in evenly(range(0, span + 1), self.cfg.anchor_kf)]
        print(f'  [adapt] anchor set fixed at frames '
              f'{[self.frame_offset + int(t) for t in self._anchor_ts]} - lambda_cons now measures '
              f'against the start of the sequence, not the current window')

    def _history_anchor(self, video):
        """Reference log-scale from the anchor keyframes, DETACHED (config.py:anchor_kf).

        MEAN ratio, not median. median_scale routes its gradient to the single selected element, so
        a penalty built on it reaches the model through ONE pixel per sample - which is very likely
        why lambda_cons did nothing even where its magnitude should have been felt. The mean ratio
        has dense gradient over every masked pixel. depth_loss keeps median_scale untouched; the
        two terms ask different questions and only this one needs to move the whole prediction.

        Timestamps are re-resolved to slots every call; an anchor whose frame has been pruned out
        of the map simply drops. None when fewer than two survive.

        cache_enabled=False is load-bearing for the same reason as in gate_value and _batch_scale.
        """
        n = video.counter.value
        ts = video.tstamp[:n]
        ls = []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                for want in self._anchor_ts:
                    hit = (ts == want).nonzero()
                    if not len(hit):
                        continue                       # pruned away since it was chosen
                    images, gt, mask, _, _, _ = self.sampler.sample(video,
                                                                   int(hit[0].item()))
                    images, gt, mask = images.cuda(), gt.cuda(), mask.cuda()
                    if mask.sum() < self.cfg.min_mask_pixels:
                        continue
                    pd, _ = self.lora.forward(images)
                    s = mean_scale(pd.float().clamp(min=1e-3), gt.clamp(min=1e-3), mask)
                    ls.append(torch.log(s.clamp(min=1e-9)))
        if len(ls) < 2:
            return None
        return torch.stack(ls).mean().detach()

    def _step(self, video, unit, step, n_steps, batch):
        """One optimiser step over `batch`, the live twin of adapt/trainer.py's. Returns its loss."""
        cfg = self.cfg
        self.opt.zero_grad(set_to_none=True)
        acc = {'loss': [], 'l_depth': [], 'l_trans': [], 'l_rot': [], 'scale_ratio': [],
               'l_cons': [], 'pred_med': [], 'gt_med': [], 'dscale': [], 'gauge': []}
        seq_lens = []

        # built once, up front: the shared gauge needs every sample of the batch before any graph
        # is built, and re-sampling for the second pass would redo the resizes and depth_filter
        samples = [self.sampler.sample(video, t) for t in batch]
        samples = [(im.cuda(), gt.cuda(), m.cuda(), e.cuda(), s, g)
                   for im, gt, m, e, s, g in samples]
        shared = self._batch_scale(samples) if cfg.batch_scale and len(samples) > 1 else None
        # the history anchor wins when it exists: it spans the whole sequence, where the
        # batch-derived one spans only the current window
        anchor = self._unit_anchor
        if anchor is None and cfg.lambda_cons > 0 and len(samples) > 1 and not cfg.anchor_kf:
            anchor = self._scale_anchor(samples)

        for images, gt, mask, gt_enc, seq, norm in samples:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_depth, pred_enc = self.lora.forward(
                    images, all_frames=cfg.depth_all_frames)
            pred_depth, pred_enc = pred_depth.float(), pred_enc.float()

            l_t, l_r, pose_scale = pose_loss(pred_enc, gt_enc, absolute=cfg.gauge_pose)
            # the gauge, in precedence order: the batch-wide one, else the pose head's, else the
            # sample's own median. config.py refuses the first two together. Under freeze_gauge
            # the pose gauge is the one captured at this keyframe's FIRST step of the unit, so the
            # loss surface stays put while the window does; l_t and l_r still use the CURRENT
            # pred_enc, because only the gauge is frozen, not the pose loss.
            # normalize_target puts the gauge in the TARGET itself (target.py:sample), so the
            # loss must NOT re-fit one: scale=1.0 is exactly what stops depth_loss being
            # scale-invariant, which is the whole point of the change. config.py refuses every
            # other gauge source alongside it, so `shared` is None and coupled_scale is off here.
            gauge = 1.0 if cfg.normalize_target else shared
            if gauge is None and cfg.coupled_scale:
                gauge = pose_scale
                if cfg.freeze_gauge and gauge is not None:
                    gauge = self._unit_gauge.setdefault(int(seq[0]), gauge)
            l_d, depth_scale = depth_loss(pred_depth, gt, mask, cfg, scale=gauge)
            loss = l_d + cfg.lambda_pose * (l_t + l_r)

            # the scale-consistency penalty. depth_scale here is the sample's OWN undetached
            # median_scale (gauge is None when lambda_cons is on - config.py refuses the pair), so
            # the gradient reaches the model through it; the anchor is a detached constant.
            l_c = None
            if anchor is not None and bool(mask.any()):
                # the SAME mean-ratio estimator the anchor is built from - a penalty comparing a
                # mean ratio against a median-derived reference would be measuring two things.
                # Undetached, so the gradient reaches the model densely.
                s_i = mean_scale(pred_depth.clamp(min=1e-3), gt.clamp(min=1e-3), mask)
                l_c = (torch.log(s_i.clamp(min=1e-9)) - anchor) ** 2
                loss = loss + cfg.lambda_cons * l_c

            # the diagnostic that would have identified the batch_scale pathology in one look:
            # the ratio alone cannot say whether the PREDICTION or the TARGET moved
            if bool(mask.any()):
                # the quantity under test: normalize_target's premise is that it does NOT
                # ramp across the run. The anchor arm's reference was never logged and had
                # to be reconstructed afterwards - not again.
                if norm is not None:
                    acc['gauge'].append(float(norm))
                acc['pred_med'].append(float(pred_depth.detach()[mask].median()))
                acc['gt_med'].append(float(gt[mask].median()))
            # JDSA's OWN measurement of prior-vs-tracker disagreement: the 2x2 scale grid it must
            # multiply this keyframe's prior by (geom/ba.py:249 updates it every BA iteration).
            # Its spread ACROSS keyframes is the drift as the solver actually experiences it, and
            # nothing in this track has ever looked at it. Free - the tensor is already on the GPU.
            acc['dscale'].append(float(video.dscales[int(seq[0])].mean()))
            if l_c is not None:
                acc['l_cons'].append(float(l_c))

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
               'kfs': [self.frame(video, t) for t in batch],
               **({'batch_scale': float(shared)} if shared is not None else {}),
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
                # one depth scale over the whole batch instead of one per sample. Absent on every
                # adapter written before it existed, which the export reads as blank, not False.
                'batch_scale': cfg.batch_scale,
                # the scale-consistency penalty's weight; 0 = off. Absent on adapters written
                # before it existed, which the export reads as blank rather than as 0.
                'lambda_cons': cfg.lambda_cons,
                # depth supervised on every frame of the sample under one shared scale. Absent on
                # every adapter written before it existed, which the export reads as blank.
                'depth_all_frames': cfg.depth_all_frames,
                # keyframes from the start of the sequence lambda_cons measures against; 0 = the
                # batch-derived reference. Absent on adapters written before it existed.
                'anchor_kf': cfg.anchor_kf,
                # the pose gauge held fixed for the unit rather than recomputed every forward
                'freeze_gauge': cfg.freeze_gauge,
                # the target carries its own gauge and depth_loss stops re-fitting a scale, which
                # is the only setting under which the depth term sees ABSOLUTE scale
                # (common.py:gauge_scale). Absent on every adapter written before it existed,
                # which the export reads as blank rather than as False.
                'normalize_target': cfg.normalize_target,
                'gauge_clamp': cfg.gauge_clamp,
                'gauge_min_pixels': cfg.gauge_min_pixels,
                # frames per VGGT forward minus one, for the training sample AND for the
                # served prediction - one number, both ends (online/config.py)
                'context_kf': cfg.context_kf,
                # keyframes between context frames; 1 = consecutive. Absent on every adapter
                # written before striding existed, which the export reads as blank, not as 1.
                'context_stride': cfg.context_stride,
                'lag': cfg.lag, 'seed': cfg.seed,
                'stream_res': cfg.stream_res,
                # the far-field ceiling on the SERVED depth (14); 1.0 = off. Pre-knob adapters
                # have no such key, which the export reads as blank rather than as 1.0. Same
                # for ceil_target (14.6), the TRAINING side of the same ceiling.
                'ceil_ratio': cfg.ceil_ratio,
                'ceil_target': cfg.ceil_target,
                # 14.9's pedestal. null here is OFF, not "not measured" - the export column
                # distinguishes the two by key presence, as it does for ceil_target.
                'ped_ratio': cfg.ped_ratio,
                # SlamConfig.start, i.e. the frame every index above is offset by. Recorded so a
                # windowed adapter's first_adapted_kf / warmup_end_frame can be read without
                # knowing which driver produced it.
                'start': self.frame_offset,
                # two gates, not one (online/config.py): warmup_kf is when learning starts,
                # handover_kf when serving does. warmup_end_frame is the FRAME the second landed
                # on - the key name predates the split and is kept, adapters on disk use it.
                'warmup_kf': cfg.warmup_kf, 'handover_kf': cfg.handover_kf,
                'warmup_prior': cfg.warmup_prior,
                'warmup_end_frame': self.warmup_end_frame,
                'checkpoint_every_kf': cfg.checkpoint_every_kf,
                # the loss gate, and what it actually did. n_gate_checks counts arrivals that
                # reached the gate, so n_gate_checks - sum(skipped) is what n_units should equal.
                'gate_metric': cfg.gate_metric,
                'gate_lo': cfg.gate_lo, 'gate_hi': cfg.gate_hi,
                'n_gate_checks': len(self.gate_log),
                'n_skipped_low': self.skipped['low'],
                'n_skipped_high': self.skipped['high'],
                'n_skipped_empty': self.skipped['empty'],
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
        out = (f'  {self.units} units / {len(self.log)} steps / {self.visits} keyframe visits '
               f'over {len(self.trained_kf)} distinct keyframes, from frame {self.first_kf}\n'
               f'  loss first 10% {np.mean(head):.4f} -> last 10% {np.mean(tail):.4f} '
               f'({time.time()-self.t0:.0f}s) - NOT a learning curve: every step has a different\n'
               f'  target, so this tracks how hard the scene got as much as how well it fits')
        if self.gate_log:
            n = sum(self.skipped.values())
            out += (f'\n  gate on {self.cfg.gate_metric} ({self.cfg.gate_lo}, {self.cfg.gate_hi}): '
                    f'{len(self.gate_log)} arrivals checked, {n} skipped '
                    f'(low {self.skipped["low"]}, high {self.skipped["high"]}, '
                    f'empty {self.skipped["empty"]})')
            # BOTH metrics, so the run also reports what the OTHER threshold should have been
            for key in ('rel', 'raw'):
                v = [g[key] for g in self.gate_log if g[key] is not None]
                if v:
                    q = np.percentile(v, [25, 50, 90, 98, 100])
                    out += (f'\n    {key:<3} p25 {q[0]:.4f}  median {q[1]:.4f}  p90 {q[2]:.4f}  '
                            f'p98 {q[3]:.4f}  max {q[4]:.4f}')
            out += '\n  retune either axis off gate_log.json - it needs no second run'
        return out

    def release(self):
        """Drop the optimiser state before the model goes; the arm's evaluation needs neither."""
        self.opt = None
        self.trainable = None
