"""The training loop.

Reached through LoRAVGGT.train(); split out only because it is long. One keyframe = one sample,
depth supervises frame 0, poses supervise every frame in the sample.
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
    """Scale-aligned masked depth L1 over a keyframe subset, same metric as the export table."""
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


def run_training(lora, scene_dir, image_dir, out_dir, cfg):
    """LoRA-adapt `lora` on the exported depth + poses, reporting train/val depth L1.

    torch is NOT seeded here: LoRAVGGT(seed=...) had to do it before injection, because the
    adapter's A matrices are initialised then. cfg.seed still drives the data order below.
    """
    rng = np.random.default_rng(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)

    data = SceneData(scene_dir, image_dir, lora.cfg, cfg)
    print(f'scene {scene_dir}: {len(data.kf)} keyframes, frames {data.t_min}..{data.t_max}, '
          f'supervised on {data.ddir}/')
    print(f'split {cfg.split_mode} @ {cfg.train_frac}: {len(data.train_kf)} train / '
          f'{len(data.val_kf)} val keyframes')
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

    print(f'evaluating base VGGT ({cfg.depth_space} space, masked, scale-aligned):')
    history = [evaluate_subsets('base')]

    lora.train_mode()
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    log, t0 = [], time.time()
    best = {'val_l1': float('inf'), 'epoch': None, 'state': None}

    if not data.train_kf:
        raise SystemExit('no training keyframes - lower train_frac or check the export')
    steps_per_epoch = math.ceil(len(data.train_kf) / cfg.batch_size)
    print(f'{len(data.train_kf)} train keyframes / batch {cfg.batch_size} = {steps_per_epoch} '
          f'optimiser steps per epoch, {cfg.epochs * steps_per_epoch} in total')

    for epoch in range(cfg.epochs):
        # every training keyframe exactly once per epoch, in a fresh order each time
        order = [int(t) for t in rng.permutation(data.train_kf)]
        run = []
        for step in range(steps_per_epoch):
            batch = order[step * cfg.batch_size:(step + 1) * cfg.batch_size]  # tail batch is short
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

                # divide before backward: accumulating the MEAN over the batch is what makes this
                # equivalent to one wider batch, and keeps the gradient magnitude (and so grad_clip
                # and lr) independent of batch_size
                (loss / len(batch)).backward()

                acc['loss'].append(loss.item())
                acc['l_depth'].append(l_d.item())
                acc['l_trans'].append(l_t.item())
                acc['l_rot'].append(l_r.item())
                acc['S'].append(len(seq))
                # depth and pose scale agreed to 1% on the pretrained model; if they diverge during
                # training the adapter is breaking depth/pose consistency
                if pose_scale is not None and depth_scale is not None:
                    acc['scale_ratio'].append((depth_scale / pose_scale).item())

            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            opt.step()

            rec = {'epoch': epoch, 'step': step, 'kfs': batch, 'S': acc['S'],
                   **{k: float(np.mean(v)) for k, v in acc.items()
                      if k not in ('S',) and v}}
            log.append(rec)
            run.append(rec['loss'])

            if step % cfg.log_every == 0:
                print(f'  e{epoch} s{step:4d}/{steps_per_epoch}  '
                      f'loss {np.mean(run[-cfg.log_every:]):.4f}  (d {rec["l_depth"]:.4f} '
                      f't {rec["l_trans"]:.4f} r {rec["l_rot"]:.4f})  B={len(batch)} '
                      f'S={acc["S"]}  {torch.cuda.max_memory_allocated()/2**30:.1f}GiB')
        print(f'epoch {epoch}: mean loss {np.mean(run):.4f} over {len(order)} keyframes  '
              f'({time.time()-t0:.0f}s elapsed)')

        if cfg.eval_every_epoch or epoch == cfg.epochs - 1:
            row = evaluate_subsets(f'e{epoch}')
            history.append(row)
            v = row.get('val_l1')
            if cfg.keep_best and v is not None and v < best['val_l1']:
                best = {'val_l1': v, 'epoch': epoch, 'state': lora.state_dict()}

    keep = cfg.keep_best and best['state'] is not None
    run_cfg = {'epochs': cfg.epochs, 'batch_size': cfg.batch_size,
               'steps_per_epoch': steps_per_epoch, 'samples_per_epoch': len(data.train_kf),
               'lr': cfg.lr, 'weight_decay': cfg.weight_decay, 'grad_clip': cfg.grad_clip,
               'lambda_pose': cfg.lambda_pose, 'depth_space': cfg.depth_space,
               'depth_source': cfg.depth_source, 'coupled_scale': cfg.coupled_scale,
               'p_single_view': cfg.p_single_view, 'max_left': cfg.max_left,
               'max_right': cfg.max_right, 'radius': cfg.radius, 'scene': scene_dir,
               'seed': cfg.seed, 'split_mode': cfg.split_mode, 'train_frac': cfg.train_frac,
               'n_train_kf': len(data.train_kf), 'n_val_kf': len(data.val_kf),
               'val_kf': data.val_kf, 'keep_best': cfg.keep_best,
               'saved_epoch': best['epoch'] if keep else cfg.epochs - 1,
               'eval_history': history}
    adapter = lora.save(out_dir, state=best['state'] if keep else None, extra=run_cfg)
    json.dump(log, open(f'{out_dir}/train_log.json', 'w'))

    # ---- summary: the val row is the one that means something ----
    print(f'\ndepth L1 (masked, scale-aligned, {cfg.depth_space} space):')
    print(f'  {"":<8}' + ''.join(f'{r["tag"]:>10}' for r in history))
    for key, name in (('train_l1', 'train'), ('val_l1', 'val')):
        if any(r.get(key) is not None for r in history):
            print(f'  {name:<8}' + ''.join(
                f'{r[key]:>10.4f}' if r.get(key) is not None else f'{"n/a":>10}' for r in history))
    if cfg.keep_best and best['epoch'] is not None:
        print(f'  saved epoch {best["epoch"]} (best val L1 {best["val_l1"]:.4f})')
    print(f'saved adapter ({n_train/1e6:.1f}M params) to {out_dir}')

    return {'adapter': adapter, 'history': history, 'run': run_cfg,
            'train_kf': data.train_kf, 'val_kf': data.val_kf}
