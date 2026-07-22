"""Shared harness for the full-sequence depth-prior A/B (Omnidata vs adapted VGGT).

Both entry points - run_full_omnidata.py and run_full_vggt.py - use this module so the two runs
cannot drift apart. The ONLY thing they do differently is what MotionFilter.prior_extractor
returns for depth; everything else, including normals, is identical.

Metrics are reported split at a frame index (default 800, the end of the 40% the adapter was
trained on) so the seen and unseen halves can be compared separately.
"""
import os    # nopep8
import sys   # nopep8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # nopep8
sys.path.insert(0, _ROOT)                                             # nopep8
sys.path.insert(0, os.path.join(_ROOT, 'hislam2'))                    # nopep8
import argparse
import json
import subprocess

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.multiprocessing import Process, Queue

from demo import mono_stream, save_trajectory          # reuse, do not duplicate
from hi2 import Hi2
from gaussian.utils.loss_utils import psnr, ssim       # same metrics eval_rendering uses


def stream_resize(img):
    """The resize demo.py's mono_stream applies (demo.py:49-55), for re-deriving GT frames."""
    RES = 341 * 640
    h0, w0, _ = img.shape
    h1 = int(h0 * np.sqrt(RES / (h0 * w0)))
    w1 = int(w0 * np.sqrt(RES / (h0 * w0)))
    return cv2.resize(img, (w1 - w1 % 8, h1 - h1 % 8))


def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--imagedir', default='data/Replica/room0/colors')
    p.add_argument('--gtdepthdir', default='data/Replica/room0/depths')
    p.add_argument('--gttraj', default='data/Replica/room0/traj_tum.txt')
    p.add_argument('--gtmesh', default='data/Replica/gt_mesh_culled/room0.ply')
    p.add_argument('--config', default='config/replica_config.yaml')
    p.add_argument('--calib', default='calib/replica.txt')
    p.add_argument('--output', required=True)
    p.add_argument('--split_at', type=int, default=800,
                   help='frame index separating the adapter-trained region from the unseen one')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--length', type=int, default=100000)
    p.add_argument('--voxel_size', type=float, default=0.006)
    p.add_argument('--mesh_weight', type=float, default=2.0)
    p.add_argument('--skip_mesh', action='store_true',
                   help='no GT mesh for this dataset (TUM RGB-D); skip TSDF + eval_recon')
    p.add_argument('--weights', default=os.path.join(_ROOT, 'pretrained_models/droid.pth'))
    p.add_argument('--buffer', type=int, default=-1)
    p.add_argument('--undistort', action='store_true')
    p.add_argument('--cropborder', type=int, default=0)
    p.add_argument('--skip_slam', action='store_true', help='reuse an existing run, eval only')
    return p


# ------------------------------------------------------------------ SLAM

def run_slam(args):
    """demo.py's main loop. Any prior monkey-patch must already be installed on MotionFilter."""
    os.makedirs(args.output, exist_ok=True)
    args.droidvis = args.gsvis = False

    queue = Queue(maxsize=8)
    reader = Process(target=mono_stream, args=(queue, args.imagedir, args.calib,
                                               args.undistort, args.cropborder,
                                               args.start, args.length))
    reader.start()

    N = len(os.listdir(args.imagedir))
    args.buffer = min(1000, N // 10 + 150) if args.buffer < 0 else args.buffer

    hi2 = None
    pbar = tqdm(range(min(N, args.length)), desc='Processing keyframes')
    while True:
        t, image, intrinsics, is_last = queue.get()
        pbar.update()
        if hi2 is None:
            args.image_size = [image.shape[2], image.shape[3]]
            hi2 = Hi2(args)
        hi2.track(t, image, intrinsics=intrinsics, is_last=is_last)
        pbar.set_description(f'keyframe {hi2.video.counter.value} gs {hi2.gs.gaussians._xyz.shape[0]}')
        if is_last:
            pbar.close()
            break
    reader.join()

    traj = hi2.terminate()
    save_trajectory(hi2, traj, args.imagedir, args.output, start=args.start)
    return hi2.video.counter.value


# ------------------------------------------------------------------ eval

def _sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def run_ate(args):
    """evo with Sim3 alignment. Returns (overall rmse, per-frame errors, timestamps)."""
    out = args.output
    _sh(f'cd {out} && evo_ape tum {os.path.abspath(args.gttraj)} traj_full.txt -vas '
        f'--save_results evo.zip --no_warnings > ape.txt 2>&1')
    _sh(f'rm -rf {out}/evo && unzip -q {out}/evo.zip -d {out}/evo')
    err = np.load(f'{out}/evo/error_array.npy')
    ts = np.load(f'{out}/evo/timestamps.npy')
    return float(np.sqrt((err ** 2).mean())), err, ts


def run_mesh(args):
    """TSDF fuse, Sim3-align (mandatory: SLAM scale is arbitrary), then score against the GT mesh."""
    import gc
    import open3d as o3d
    out, w = args.output, args.mesh_weight

    # tsdf_integrate builds an Open3D VoxelBlockGrid on cuda:0; without releasing what the just
    # finished SLAM run is still holding, that subprocess OOMs and dies silently
    gc.collect()
    torch.cuda.empty_cache()

    # Open3D's marching-cubes allocates a large assistance structure on the GPU and fails at fine
    # voxel sizes when the shared card is busy. Retry coarser rather than losing the metric - but
    # record which size won, because the two arms MUST be compared at the same voxel size.
    raw = f'{out}/tsdf_mesh_w{w:.1f}.ply'
    voxel_used = None
    for vs in [args.voxel_size, 0.01, 0.02]:
        if os.path.exists(raw):
            os.remove(raw)
        r = _sh(f'cd {_ROOT} && python tsdf_integrate.py --result {out} '
                f'--voxel_size {vs} --weight {w}')
        if os.path.exists(raw):
            voxel_used = vs
            break
        print(f'  [mesh] tsdf_integrate failed at voxel_size={vs} (rc={r.returncode})')
    if voxel_used is None:
        print('   ', (r.stderr or r.stdout).strip().splitlines()[-2:])
        return None
    if voxel_used != args.voxel_size:
        print(f'  [mesh] fell back to voxel_size={voxel_used}; the other arm must match')
    # without this the mesh sits in SLAM units and every number is ~50x off; ICP inside
    # eval_recon is rigid-only and cannot recover scale
    mesh = o3d.io.read_triangle_mesh(raw)
    mesh.transform(np.load(f'{out}/evo/alignment_transformation_sim3.npy'))
    aligned = f'{out}/tsdf_mesh_w{w:.1f}_aligned.ply'
    o3d.io.write_triangle_mesh(aligned, mesh)

    res = f'{out}/eval_recon.txt'
    r = _sh(f'cd {_ROOT} && python scripts/eval_recon.py {aligned} {os.path.abspath(args.gtmesh)} '
            f'--eval_3d --save {res}')
    if not os.path.exists(res):
        print(f'  [mesh] eval_recon failed (rc={r.returncode}):')
        print('   ', (r.stderr or r.stdout).strip().splitlines()[-3:])
        return None
    out_d = eval(open(res).read(), {'np': np, 'array': np.array})
    out_d['voxel_size'] = voxel_used
    return out_d


def split_render_metrics(args):
    """Recompute PSNR/SSIM and depth L1 per frame from the saved renders, then split.

    eval_rendering only reports sequence means, but it writes every render named by original frame
    index, so the seen/unseen breakdown can be recovered without re-rendering. One global depth
    scale is fitted across all frames (SLAM units are arbitrary) and the errors are then split -
    fitting per half would hide exactly the drift we are looking for.
    """
    out = args.output
    files = sorted(os.listdir(args.imagedir))
    gtd = sorted(os.listdir(args.gtdepthdir)) if args.gtdepthdir else None
    rows = []

    for f in sorted(os.listdir(f'{out}/renders/image_after_opt')):
        idx = int(f[:-4])
        render = cv2.imread(f'{out}/renders/image_after_opt/{f}')
        gt = stream_resize(cv2.imread(os.path.join(args.imagedir, files[idx])))
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
            pred = cv2.imread(dp, cv2.IMREAD_ANYDEPTH) / 6553.5
            gd = cv2.imread(os.path.join(args.gtdepthdir, gtd[idx]), cv2.IMREAD_ANYDEPTH) / 6553.5
            gd = cv2.resize(gd, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            v = (gd > 0) & (pred > 0)
            if v.sum() > 0:
                row['_d'] = (gd[v], pred[v])
        rows.append(row)

    # one global median-ratio scale over every frame, then split the per-frame errors
    dv = [r['_d'] for r in rows if '_d' in r]
    if dv:
        s = np.median(np.concatenate([g for g, _ in dv])) / np.median(np.concatenate([p for _, p in dv]))
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

    k = args.split_at
    return {'all': agg(lambda i: True), 'seen': agg(lambda i: i < k), 'unseen': agg(lambda i: i >= k)}


def evaluate(args, label):
    res = {'label': label, 'output': args.output, 'split_at': args.split_at}

    ate, err, ts = run_ate(args)
    res['ate_all'] = ate
    k = args.split_at
    for name, sel in (('seen', ts < k), ('unseen', ts >= k)):
        res[f'ate_{name}'] = float(np.sqrt((err[sel] ** 2).mean())) if sel.sum() else None

    pj = f'{args.output}/psnr/after_opt/final_result.json'
    if os.path.exists(pj):
        res['hislam2_eval'] = json.load(open(pj))

    res['render'] = split_render_metrics(args)
    res['mesh'] = None if args.skip_mesh else run_mesh(args)

    json.dump(res, open(f'{args.output}/ab_results.json', 'w'), indent=2, default=float)
    return res


def print_report(res):
    k = res['split_at']
    print(f"\n{'='*66}\n  {res['label']}  ->  {res['output']}\n{'='*66}")
    print(f"  {'metric':<22}{'all':>13}{f'seen <{k}':>13}{f'unseen >={k}':>15}")
    print(f"  {'-'*63}")
    print(f"  {'ATE RMSE (m)':<22}{res['ate_all']:>13.4f}"
          f"{res.get('ate_seen') or float('nan'):>13.4f}{res.get('ate_unseen') or float('nan'):>15.4f}")
    r = res['render']
    for key, name in (('psnr', 'PSNR (dB)'), ('ssim', 'SSIM'), ('depth_l1', 'depth L1 (m)')):
        vals = [(r[s] or {}).get(key) for s in ('all', 'seen', 'unseen')]
        if any(v is not None for v in vals):
            cells = ''.join(f'{v:>13.4f}' if i < 2 else f'{v:>15.4f}'
                            if v is not None else f'{"n/a":>13}' for i, v in enumerate(vals))
            print(f"  {name:<22}{cells}")
    print(f"  {'frames evaluated':<22}{(r['all'] or {}).get('n', 0):>13}"
          f"{(r['seen'] or {}).get('n', 0):>13}{(r['unseen'] or {}).get('n', 0):>15}")
    if res.get('mesh'):
        m = res['mesh']
        print(f"\n  mesh (whole sequence, Sim3-aligned, voxel {m['voxel_size']}): "
              f"acc {m['mean precision']:.4f} m  comp {m['mean recall']:.4f} m  "
              f"comp-ratio {100*m['recall']:.1f}%  F {m['f-score']:.3f}")
    print()


def main(label, patch=None, extra_args=None):
    parser = build_parser(label)
    if extra_args is not None:
        extra_args(parser)
    args = parser.parse_args()
    if patch is not None:
        # installs the depth prior before Hi2 is built; may refine the label now that it knows
        # which variant of the prior it actually loaded
        label = patch(args) or label
    if not args.skip_slam:
        torch.multiprocessing.set_start_method('spawn')
        n_kf = run_slam(args)
        print(f'{label}: SLAM done, {n_kf} keyframes')
    print_report(evaluate(args, label))
