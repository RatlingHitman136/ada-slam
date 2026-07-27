"""Scoring one arm: ATE, mesh, and render metrics split seen/unseen.

Everything here reads an arm's finished output directory; nothing re-runs SLAM. The split is the
point of the whole harness - HI-SLAM2's own eval_rendering reports sequence means only, but it
writes every render named by original frame index, so the seen/unseen breakdown can be recovered
without re-rendering.
"""
import json
import os

import cv2
import numpy as np
import torch

from ..common import stream_resize
from ..paths import ROOT
from ..runtime import free_vram, sh

# One arm's scores. It records the split_at it was computed at, because arms are reused across
# comparisons and a score split at the wrong frame is worse than no score (stage.py:cached_results).
RESULTS = 'results.json'


def run_ate(out, gt_traj):
    """evo with Sim3 alignment. Returns (overall rmse, per-frame errors, timestamps)."""
    sh(f'cd {out} && evo_ape tum {os.path.abspath(gt_traj)} traj_full.txt -vas '
       f'--save_results evo.zip --no_warnings > ape.txt 2>&1')
    sh(f'rm -rf {out}/evo && unzip -q {out}/evo.zip -d {out}/evo')
    err = np.load(f'{out}/evo/error_array.npy')
    ts = np.load(f'{out}/evo/timestamps.npy')
    return float(np.sqrt((err ** 2).mean())), err, ts


def run_mesh(out, cfg):
    """TSDF fuse, Sim3-align (mandatory: SLAM scale is arbitrary), then score vs the GT mesh."""
    import open3d as o3d
    w = cfg.mesh_weight

    # tsdf_integrate builds an Open3D VoxelBlockGrid on cuda:0; without releasing what the just
    # finished SLAM run is still holding, that subprocess OOMs and dies silently
    free_vram()

    # Open3D's marching-cubes allocates a large assistance structure on the GPU and fails at fine
    # voxel sizes when the shared card is busy. Retry coarser rather than losing the metric - but
    # record which size won, because the two arms MUST be compared at the same voxel size.
    raw = f'{out}/tsdf_mesh_w{w:.1f}.ply'
    voxel_used, r = None, None
    for vs in (cfg.voxel_size, *cfg.voxel_fallbacks):
        if os.path.exists(raw):
            os.remove(raw)
        r = sh(f'cd {ROOT} && python tsdf_integrate.py --result {out} '
               f'--voxel_size {vs} --weight {w}')
        if os.path.exists(raw):
            voxel_used = vs
            break
        print(f'  [mesh] tsdf_integrate failed at voxel_size={vs} (rc={r.returncode})')
    if voxel_used is None:
        print('   ', (r.stderr or r.stdout).strip().splitlines()[-2:])
        return None
    if voxel_used != cfg.voxel_size:
        print(f'  [mesh] fell back to voxel_size={voxel_used}; the other arm must match')

    # without this the mesh sits in SLAM units and every number is ~50x off; ICP inside
    # eval_recon is rigid-only and cannot recover scale
    mesh = o3d.io.read_triangle_mesh(raw)
    mesh.transform(np.load(f'{out}/evo/alignment_transformation_sim3.npy'))
    aligned = f'{out}/tsdf_mesh_w{w:.1f}_aligned.ply'
    o3d.io.write_triangle_mesh(aligned, mesh)

    res = f'{out}/eval_recon.txt'
    r = sh(f'cd {ROOT} && python scripts/eval_recon.py {aligned} '
           f'{os.path.abspath(cfg.gt_mesh)} --eval_3d --save {res}')
    if not os.path.exists(res):
        print(f'  [mesh] eval_recon failed (rc={r.returncode}):')
        print('   ', (r.stderr or r.stdout).strip().splitlines()[-3:])
        return None
    out_d = eval(open(res).read(), {'np': np, 'array': np.array})
    out_d['voxel_size'] = voxel_used
    return out_d


def split_render_metrics(out, split_at, slam_cfg, cfg):
    """Recompute PSNR/SSIM and depth L1 per frame from the saved renders, then split.

    One global depth scale is fitted across all frames (SLAM units are arbitrary) and the errors
    are then split - fitting per half would hide exactly the drift we are looking for.
    """
    from gaussian.utils.loss_utils import psnr, ssim
    files = sorted(os.listdir(slam_cfg.colors))
    gtd = sorted(os.listdir(cfg.gt_depths)) if cfg.gt_depths else None
    rows = []

    for f in sorted(os.listdir(f'{out}/renders/image_after_opt')):
        idx = int(f[:-4])
        render = cv2.imread(f'{out}/renders/image_after_opt/{f}')
        gt = stream_resize(cv2.imread(os.path.join(slam_cfg.colors, files[idx])),
                           slam_cfg.stream_res)
        if render is None or gt is None or render.shape != gt.shape:
            continue
        r = torch.from_numpy(render[..., ::-1].copy()).permute(2, 0, 1).float().cuda() / 255.
        g = torch.from_numpy(gt[..., ::-1].copy()).permute(2, 0, 1).float().cuda() / 255.
        m = g > 0
        row = {'idx': idx,
               'psnr': psnr(r[m].unsqueeze(0), g[m].unsqueeze(0)).mean().item(),
               'ssim': ssim(r.unsqueeze(0), g.unsqueeze(0)).item()}

        dp = f'{out}/renders/depth_after_opt/{idx:06d}.png'
        if gtd and os.path.exists(dp):
            pred = cv2.imread(dp, cv2.IMREAD_ANYDEPTH) / cfg.depth_png_scale
            gd = cv2.imread(os.path.join(cfg.gt_depths, gtd[idx]),
                            cv2.IMREAD_ANYDEPTH) / cfg.depth_png_scale
            gd = cv2.resize(gd, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            v = (gd > 0) & (pred > 0)
            if v.sum() > 0:
                row['_d'] = (gd[v], pred[v])
        rows.append(row)

    # one global median-ratio scale over every frame, then split the per-frame errors
    dv = [r['_d'] for r in rows if '_d' in r]
    if dv:
        s = np.median(np.concatenate([g for g, _ in dv])) / \
            np.median(np.concatenate([p for _, p in dv]))
        for r in rows:
            if '_d' in r:
                g, p = r.pop('_d')
                r['depth_l1'] = float(np.abs(g - s * p).mean())

    def agg(sel):
        sub = [r for r in rows if sel(r['idx'])]
        if not sub:
            return None
        o = {'n': len(sub), 'psnr': float(np.mean([r['psnr'] for r in sub])),
             'ssim': float(np.mean([r['ssim'] for r in sub]))}
        d = [r['depth_l1'] for r in sub if 'depth_l1' in r]
        if d:
            o['depth_l1'] = float(np.mean(d))
        return o

    return {'all': agg(lambda i: True), 'seen': agg(lambda i: i < split_at),
            'unseen': agg(lambda i: i >= split_at)}


def evaluate(out, label, split_at, slam_cfg, cfg):
    """Every metric for one arm, written to out/results.json and returned.

    Reads a finished arm directory and re-runs nothing expensive, so re-scoring an existing arm at
    a different split_at is cheap - which is what makes arms reusable across comparisons.
    """
    res = {'label': label, 'output': out, 'split_at': split_at}

    ate, err, ts = run_ate(out, cfg.gt_traj)
    res['ate_all'] = ate
    for name, sel in (('seen', ts < split_at), ('unseen', ts >= split_at)):
        res[f'ate_{name}'] = float(np.sqrt((err[sel] ** 2).mean())) if sel.sum() else None

    pj = f'{out}/psnr/after_opt/final_result.json'
    if os.path.exists(pj):
        res['hislam2_eval'] = json.load(open(pj))

    res['render'] = split_render_metrics(out, split_at, slam_cfg, cfg)
    res['mesh'] = run_mesh(out, cfg) if cfg.gt_mesh else None

    json.dump(res, open(f'{out}/{RESULTS}', 'w'), indent=2, default=float)
    return res
