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

from ..adapt.losses import depth_loss, jdsa_loss, median_scale, pose_loss, relative_loss
from ..adapt.trainer import batches_of, coupling_fit

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
        images, gt, mask, _, _ = self.sampler.sample(video, t)
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
            # ALWAYS depth_loss, never the run's objective: gate_lo/gate_hi are calibrated
            # against this quantity (online/config.py's reference distributions), so measuring the
            # gate in JDSA-residual units would silently invalidate every threshold ever recorded.
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

        # E3: one statistics pass per WINDOW, before any of its batches - the live counterpart of
        # adapt/trainer.py:249-263. Placed AFTER the gate so the slope is only ever fitted over
        # windows that are actually going to be stepped on.
        #
        # No sample caching is needed here, unlike offline: LiveSampler.sample takes no rng and is
        # a pure function of (video, t), so this pass leaves self.rng - and therefore self.batches'
        # permutation - untouched. A lambda=0 run is byte-identical to one from before this term.
        #
        # The model is still in eval_mode (train_mode is entered below), where offline it is in
        # train mode. coupling_fit runs under no_grad, so the only thing that differs is the
        # aggregator's gradient checkpointing - a no-op without a backward - and the statistics
        # come out the same. Measuring in the serving mode also matches gate_value's reasoning.
        coup_coef = {}
        if self.cfg.coupling_lambda > 0:
            samples = {int(t): self.sampler.sample(video, t) for t in kfs}
            samples = {t: tuple(v.cuda() if torch.is_tensor(v) else v for v in s)
                       for t, s in samples.items()}
            b_hat, coup_coef, sum_x2 = coupling_fit(self.lora, self.cfg, samples)
            if self.units % self.cfg.log_every == 0:
                print(f'  [adapt] kf{self.units} coupling: '
                      f'b={"n/a" if b_hat is None else f"{b_hat:+.4f}"}  sum_x2={sum_x2:.4f}  '
                      f'over {len(samples)} kf')
            del samples                      # the cuda tensors are re-drawn per step below

        if self.first_kf is None:
            self.first_kf = int(tstamp)      # a FRAME index, like trained_kf and the log's 'kfs'

        unit = self.units
        self.lora.train_mode()          # also enables the aggregator's gradient checkpointing
        try:
            with torch.amp.autocast('cuda', enabled=False):
                for step, batch in enumerate(batches):
                    self._step(video, unit, step, len(batches), batch, coup_coef)
        finally:
            self.lora.eval_mode()       # the extractor predicts next

        self.units += 1
        # by FRAME index, for the same reason as last_tstamp: keyframe slot 31 before a pruning and
        # slot 31 after it are different frames, and n_train_kf feeds 12.1's adapt_cost
        self.trained_kf.update(self.frame(video, t) for t in kfs)
        self._checkpoint()
        return unit

    def _step(self, video, unit, step, n_steps, batch, coup_coef=None):
        """One optimiser step over `batch`, as adapt/trainer.py:199-243. Returns the mean loss.

        `coup_coef` is {keyframe: coefficient} from the window's coupling_fit, empty when the term
        is off.
        """
        coup_coef = coup_coef or {}
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
            if cfg.depth_loss == 'jdsa':
                # the residual the solver's own 2x2 disparity grid cannot absorb; the grid subsumes
                # any single scale, so pose_scale is deliberately not handed to it as well
                l_d, depth_scale = jdsa_loss(pred_depth, gt, mask, cfg, self.sampler.stream_hw)
                with torch.no_grad():   # keep the train log comparable across objectives
                    acc.setdefault('l_depth_aligned', []).append(
                        depth_loss(pred_depth, gt, mask, cfg)[0].item())
            else:
                l_d, depth_scale = depth_loss(pred_depth, gt, mask, cfg,
                                              scale=pose_scale if cfg.coupled_scale else None)
            loss = l_d + cfg.lambda_pose * (l_t + l_r)

            # E3: the window's slope penalty, entered as its exact per-sample gradient. coef is a
            # constant from the statistics pass, so this is linear in log s_i and the window's
            # backward passes sum to grad(coupling_lambda * b^2) - PROVIDED they all land in one
            # optimiser step, which is why batch_size should equal window_size (online/config.py).
            # Recomputed rather than reused from depth_loss because `scale` may have come from
            # pose_loss.
            coef = coup_coef.get(int(t))
            if coef:
                l_coup = coef * median_scale(pred_depth.clamp(min=1e-3),
                                             gt.clamp(min=1e-3), mask).log()
                loss = loss + l_coup
                acc.setdefault('l_coup', []).append(l_coup.item())

            # the MEAN over the batch, so grad magnitude is independent of batch_size
            (loss / len(batch)).backward()
            self.visits += 1

            seq_lens.append(len(seq))
            acc['loss'].append(loss.item())
            acc['l_depth'].append(l_d.item())
            acc['l_trans'].append(l_t.item())
            acc['l_rot'].append(l_r.item())
            if (pose_scale is not None and depth_scale is not None
                    and cfg.depth_loss != 'jdsa'):     # jdsa returns the 4-vector, not a scale
                acc['scale_ratio'].append((depth_scale / pose_scale).item())

        torch.nn.utils.clip_grad_norm_(self.trainable, cfg.grad_clip)
        self.opt.step()

        # same shape as adapt/trainer.py:233, so one reader serves both logs - and 'kfs' means the
        # same thing in both, FRAME indices: offline they come from poses_slam.txt, live they must
        # be translated off video.tstamp, because a keyframe slot is not stable across a pruning
        rec = {'epoch': unit, 'step': step, 'S': seq_lens,
               'kfs': [self.frame(video, t) for t in batch],
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
                # E3. Recorded always, so an adapter says which objective produced it - a run made
                # before these fields existed reads back as absent, which is the same as 0.0.
                # WHICH OBJECTIVE produced this adapter - stated, so two arms cannot look alike
                'depth_loss': cfg.depth_loss, 'jdsa_norm': cfg.jdsa_norm,
                'jdsa_ridge': cfg.jdsa_ridge, 'jdsa_lattice': cfg.jdsa_lattice,
                'coupling_lambda': cfg.coupling_lambda, 'coupling_axis': cfg.coupling_axis,
                'coupling_min_var': cfg.coupling_min_var,
                'coupling_shuffle': cfg.coupling_shuffle,
                'context_kf': cfg.context_kf, 'lag': cfg.lag, 'seed': cfg.seed,
                'stream_res': cfg.stream_res,
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
