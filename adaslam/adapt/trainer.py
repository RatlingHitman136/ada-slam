"""The training loop, reached through LoRAVGGT.train().

One keyframe = one sample; depth supervises frame 0, poses every frame. It trains and reports but
does NOT write the final adapter - only cfg.checkpoint_every's snapshots.
"""
import json
import math
import os
import time

import numpy as np
import torch

from .data import SceneData
from .losses import depth_loss, pose_loss


@torch.no_grad()
def eval_depth(lora, data, kfs, cfg):
    """Scale-aligned masked depth L1 over a keyframe subset - the export table's metric."""
    if not kfs:
        return None
    if cfg.eval_max_kf and len(kfs) > cfg.eval_max_kf:
        pick = np.linspace(0, len(kfs) - 1, cfg.eval_max_kf).round().astype(int)
        kfs = [kfs[i] for i in sorted(set(pick.tolist()))]
    was_training = lora.model.training
    lora.model.eval()
    rng = np.random.default_rng(cfg.seed)
    errs = []
    for t in kfs:
        images, gt, mask, _, _ = data.sample(rng, t=t, single=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred, _ = lora.forward(images.cuda())
        l, _ = depth_loss(pred.float(), gt.cuda(), mask.cuda(), cfg)
        errs.append(l.item())
    lora.model.train(was_training)
    return float(np.mean(errs))


def schedule(data, cfg, rng):
    """(unit, its batches) - the ONLY place the two adaptation styles differ.

    normal  a unit is an EPOCH: every train keyframe once, shuffled, batch_size at a time.
    online  a unit is ONE ARRIVING KEYFRAME: ascending frame order, cfg.epochs steps on each
            before the next. batch_size is not read - a keyframe arrives alone.
    """
    if cfg.adapt_style == 'online':
        for i, t in enumerate(data.train_kf):          # ascending frame order = arrival order
            yield i, [[int(t)] for _ in range(cfg.epochs)]
    else:
        for epoch in range(cfg.epochs):
            order = [int(t) for t in rng.permutation(data.train_kf)]
            yield epoch, [order[s * cfg.batch_size:(s + 1) * cfg.batch_size]   # tail batch is short
                          for s in range(math.ceil(len(order) / cfg.batch_size))]


def run_training(lora, scene_dir, image_dir, out_dir, cfg, ckpt_dir=None):
    """LoRA-adapt `lora` on the exported depth + poses, reporting train/val depth L1.

    torch is NOT seeded here - LoRAVGGT(seed=...) had to do it before injection. cfg.seed still
    drives the data order below.
    """
    if cfg.checkpoint_every and not ckpt_dir:
        raise SystemExit(f'checkpoint_every={cfg.checkpoint_every} asks for a snapshot every '
                         f'{cfg.checkpoint_every} epochs, but no ckpt_dir was given. The cadence '
                         f'is AdaptConfig.checkpoint_every, the location is train(ckpt_dir=...); '
                         f'set both, or set checkpoint_every=0.')
    rng = np.random.default_rng(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)

    data = SceneData(scene_dir, image_dir, lora.cfg, cfg)
    print(f'scene {scene_dir}: {len(data.kf)} keyframes, frames {data.t_min}..{data.t_max}, '
          f'supervised on {data.ddir}/')
    tail = f', frames {data.val_kf[0]}..{data.val_kf[-1]}' if data.val_kf else ''
    print(f'split @ {cfg.train_frac}: {len(data.train_kf)} train / {len(data.val_kf)} val '
          f'keyframes (val = the contiguous tail{tail})')
    if not data.val_kf:
        print('  note: empty val set - val eval and keep_best are disabled')
    for line in data.aspect_report():
        print(line)

    trainable = lora.trainable_parameters()
    n_train = lora.n_trainable()
    print(lora.summary())

    def evaluate_subsets(tag):
        row = {'tag': tag}
        if cfg.eval_on_train:
            row['train_l1'] = eval_depth(lora, data, data.train_kf, cfg)
        if cfg.eval_on_val:
            row['val_l1'] = eval_depth(lora, data, data.val_kf, cfg)
        cells = '  '.join(f'{k.split("_")[0]} {v:.4f}' for k, v in row.items()
                          if k != 'tag' and v is not None)
        if cells:
            print(f'  depth L1 [{tag:>5}]  {cells}')
        return row

    print('evaluating base VGGT (masked, scale-aligned):')
    history = [evaluate_subsets('base')]

    lora.train_mode()
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    log, t0 = [], time.time()
    best = {'val_l1': float('inf'), 'epoch': None, 'state': None}

    if not data.train_kf:
        raise SystemExit('no training keyframes - raise train_frac or check the export')

    # a UNIT is an epoch, or one arriving keyframe in 'online' - see schedule()
    online = cfg.adapt_style == 'online'
    n_units = len(data.train_kf) if online else cfg.epochs
    steps_per_unit = cfg.epochs if online else math.ceil(len(data.train_kf) / cfg.batch_size)
    unit_word, unit_tag = ('keyframe', 'k') if online else ('epoch', 'e')
    if online:
        print(f'online: {n_units} keyframes in frame order x {steps_per_unit} steps each = '
              f'{n_units * steps_per_unit} optimiser steps, 1 keyframe per step '
              f'(batch_size={cfg.batch_size} is not used in this style)')
        if cfg.eval_every_epoch:
            print(f'  WARNING: eval_every_epoch evaluates after EVERY keyframe - {n_units} '
                  f'evaluations. Set it False for base + final only.')
    else:
        print(f'{len(data.train_kf)} train keyframes / batch {cfg.batch_size} = {steps_per_unit} '
              f'optimiser steps per epoch, {n_units * steps_per_unit} in total')

    # What a saved adapter records about its run. Built before the loop: checkpoints need it too.
    run_cfg = {'adapt_style': cfg.adapt_style, 'epochs': cfg.epochs, 'batch_size': cfg.batch_size,
               'n_units': n_units,          # epochs, or arriving keyframes in 'online'
               'steps_per_epoch': steps_per_unit, 'samples_per_epoch': len(data.train_kf),
               'lr': cfg.lr, 'weight_decay': cfg.weight_decay, 'grad_clip': cfg.grad_clip,
               'lambda_pose': cfg.lambda_pose, 'coupled_scale': cfg.coupled_scale,
               'p_single_view': cfg.p_single_view, 'max_left': cfg.max_left,
               'max_right': cfg.max_right, 'radius': cfg.radius, 'scene': scene_dir,
               # the frame this adapter stopped seeing; priortest reads it HERE, not from the
               # extract dir, which may be deleted long before the adapter is
               'split_at': int(data.t_max) + 1,
               'seed': cfg.seed, 'train_frac': cfg.train_frac,
               'n_train_kf': len(data.train_kf), 'n_val_kf': len(data.val_kf),
               'val_kf': data.val_kf, 'keep_best': cfg.keep_best,
               'checkpoint_every': cfg.checkpoint_every}

    def record(unit):
        """run_cfg for a save of `unit`, with the evaluation known by then."""
        return {**run_cfg, 'saved_epoch': unit, 'eval_history': history}

    if ckpt_dir and cfg.checkpoint_every:
        print(f'checkpointing every {cfg.checkpoint_every} {unit_word}s -> {ckpt_dir}/epoch_NNN/')

    for unit, batches in schedule(data, cfg, rng):
        run = []
        for step, batch in enumerate(batches):
            opt.zero_grad(set_to_none=True)
            acc = {'loss': [], 'l_depth': [], 'l_trans': [], 'l_rot': [], 'scale_ratio': [],
                   'S': []}

            for t in batch:
                images, gt, mask, gt_enc, seq = data.sample(rng, t=t)
                images, gt, mask, gt_enc = images.cuda(), gt.cuda(), mask.cuda(), gt_enc.cuda()

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred_depth, pred_enc = lora.forward(images)
                pred_depth, pred_enc = pred_depth.float(), pred_enc.float()

                l_t, l_r, pose_scale = pose_loss(pred_enc, gt_enc)
                l_d, depth_scale = depth_loss(pred_depth, gt, mask, cfg,
                                              scale=pose_scale if cfg.coupled_scale else None)
                loss = l_d + cfg.lambda_pose * (l_t + l_r)

                # the MEAN over the batch, so grad magnitude is independent of batch_size
                (loss / len(batch)).backward()

                acc['loss'].append(loss.item())
                acc['l_depth'].append(l_d.item())
                acc['l_trans'].append(l_t.item())
                acc['l_rot'].append(l_r.item())
                acc['S'].append(len(seq))
                # they agreed to 1% on the pretrained model; divergence = broken depth/pose
                # consistency
                if pose_scale is not None and depth_scale is not None:
                    acc['scale_ratio'].append((depth_scale / pose_scale).item())

            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            opt.step()

            rec = {'epoch': unit, 'step': step, 'kfs': batch, 'S': acc['S'],
                   **{k: float(np.mean(v)) for k, v in acc.items()
                      if k not in ('S',) and v}}
            log.append(rec)
            run.append(rec['loss'])

            if step % cfg.log_every == 0:
                print(f'  {unit_tag}{unit} s{step:4d}/{len(batches)}  '
                      f'loss {np.mean(run[-cfg.log_every:]):.4f}  (d {rec["l_depth"]:.4f} '
                      f't {rec["l_trans"]:.4f} r {rec["l_rot"]:.4f})  B={len(batch)} '
                      f'S={acc["S"]}  {torch.cuda.max_memory_allocated()/2**30:.1f}GiB')
        print(f'{unit_word} {unit}: mean loss {np.mean(run):.4f} over {len(batches)} steps  '
              f'({time.time()-t0:.0f}s elapsed)')

        if cfg.eval_every_epoch or unit == n_units - 1:
            row = evaluate_subsets(f'{unit_tag}{unit}')
            history.append(row)
            v = row.get('val_l1')
            if cfg.keep_best and v is not None and v < best['val_l1']:
                best = {'val_l1': v, 'epoch': unit, 'state': lora.state_dict()}

        # A full adapter dir, so any unit can be run as an arm. epoch_NNN is a CONTRACT:
        # end2end/config.py:arm_name parses arm names off that prefix.
        if cfg.checkpoint_every and (unit + 1) % cfg.checkpoint_every == 0:
            extra = {**record(unit), 'checkpoint': True}
            print(f'  checkpoint -> {lora.save(f"{ckpt_dir}/epoch_{unit:03d}", extra=extra)}')

    keep = cfg.keep_best and best['state'] is not None
    json.dump(log, open(f'{out_dir}/train_log.json', 'w'))

    # ---- summary: the val row is the one that means something ----
    print('\ndepth L1 (masked, scale-aligned):')
    print(f'  {"":<8}' + ''.join(f'{r["tag"]:>10}' for r in history))
    for key, name in (('train_l1', 'train'), ('val_l1', 'val')):
        if any(r.get(key) is not None for r in history):
            print(f'  {name:<8}' + ''.join(
                f'{r[key]:>10.4f}' if r.get(key) is not None else f'{"n/a":>10}' for r in history))
    if cfg.keep_best and best['epoch'] is not None:
        print(f'  {unit_word} {best["epoch"]} kept (best val L1 {best["val_l1"]:.4f})')
    print(f'trained {n_train/1e6:.1f}M adapter params; log in {out_dir}/train_log.json. '
          f'The caller writes the adapter.')

    # 'state' and 'run' are save()'s state= and extra=
    return {'state': best['state'] if keep else None,
            'run': record(best['epoch'] if keep else n_units - 1),
            'history': history, 'train_kf': data.train_kf, 'val_kf': data.val_kf}
