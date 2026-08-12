"""Preprocess a RELLIS-3D sequence into the layout the rest of this repo expects.

    python scripts/preprocess_rellis3d.py \
        --src   data/RELLIS-3D/seq/Rellis-3D/00000 \
        --calib data/RELLIS-3D/_raw/calib \
        --dst   data/RELLIS/00000 --jobs 16

Produces preprocess_tum.py's shape:

    colors/%06d.png   half resolution, 960x600
    depths/%06d.png   same index and size, uint16 * --depth_png_scale, 0 = no lidar return
    traj_tum.txt      "<index> tx ty tz qx qy qz qw", camera-to-world
    calib.txt         "fx fy cx cy"   (RELLIS publishes no distortion coefficients)

GT depth comes from the 64-beam Ouster sweep: ~7k points land in the camera's 38x24 deg frame, and
densify() interpolates them over a Delaunay triangulation, dropping triangles that would bridge a
gap or a depth discontinuity. Those pixels stay 0 and every consumer masks them out - the depth
never reaches the tracker (gtdepthdir is None on every run, 9.3), only the prior test and the
extract accuracy table.

Conventions are MEASURED, not assumed - the wrong choice fails --selftest's warp check:
transforms.yaml holds the camera's pose in the lidar frame (so lidar->camera is its inverse), and
poses.txt is body->world, so the camera trajectory is poses @ A.
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

IMAGE_DIR = 'pylon_camera_node'
CLOUD_DIR = 'os1_cloud_node_kitti_bin'
MIN_Z = 0.5                  # m in front of the camera; also culls the rear half of the 360 sweep


# ---------------------------------------------------------------- calibration

def read_calib(calib_root, seq):
    """(fx, fy, cx, cy), A - A being the camera's pose in the lidar frame.

    The two files live in differently spelled directories (Rellis-3D vs Rellis_3D) and the
    extrinsic differs per sequence, so both are resolved per sequence rather than hardcoded.
    """
    import yaml

    def find(name):
        for spelling in ('Rellis-3D', 'Rellis_3D'):
            p = os.path.join(calib_root, spelling, seq, name)
            if os.path.exists(p):
                return p
        raise SystemExit(f'{name} for sequence {seq} not found under {calib_root} '
                         f'(looked in Rellis-3D/ and Rellis_3D/)')

    fx, fy, cx, cy = np.loadtxt(find('camera_info.txt'))
    raw = yaml.safe_load(open(find('transforms.yaml')))
    entry = next(v for v in raw.values() if isinstance(v, dict) and 'q' in v and 't' in v)
    q, t = entry['q'], entry['t']
    A = np.eye(4)
    A[:3, :3] = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
    A[:3, 3] = [t['x'], t['y'], t['z']]
    return (fx, fy, cx, cy), A


def load_poses(path):
    """poses.txt (N,12) -> (N,4,4) body-to-world."""
    p = np.loadtxt(path).reshape(-1, 3, 4)
    return np.concatenate([p, np.tile([[[0, 0, 0, 1.0]]], (len(p), 1, 1))], axis=1)


def to_tum(T):
    """One 4x4 camera-to-world -> tx ty tz qx qy qz qw, as preprocess_replica.py writes it."""
    return np.hstack((T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()))


# ---------------------------------------------------------------- lidar -> depth image

def project(pts, E, K, hw):
    """Lidar points -> (u, v, z) inside the frame, nearest kept per pixel.

    The z-buffer matters: the sweep is 360 deg and sees past occluders the camera cannot see
    around, so two points can land on one pixel with very different depths.
    """
    h, w = hw
    P = pts @ E[:3, :3].T + E[:3, 3]
    P = P[P[:, 2] > MIN_Z]
    uv = (K @ P.T).T
    u, v, z = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2], P[:, 2]
    m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[m], v[m], z[m]
    order = np.argsort(z)                                  # nearest first
    _, first = np.unique((v.astype(int) * w + u.astype(int))[order], return_index=True)
    keep = order[first]
    return u[keep], v[keep], z[keep]


def densify(u, v, z, hw, max_edge, max_ratio):
    """Linear interpolation of the projected beams over their Delaunay triangulation.

    Triangles that bridge a gap (longest edge > max_edge px) or straddle a depth discontinuity
    (max(z)/min(z) > max_ratio) are dropped, so sky and silhouettes stay 0 rather than being filled
    with fictitious surface. The ratio test is relative because lidar noise and real scene
    gradients both scale with range.
    """
    h, w = hw
    if len(z) < 4:
        return np.zeros(hw, np.float64)
    tri = Delaunay(np.c_[u, v])
    yy, xx = np.mgrid[0:h, 0:w]
    pix = np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]
    s = tri.find_simplex(pix)
    out = np.zeros(h * w)
    ok = s >= 0                                            # outside the hull stays invalid
    if not ok.any():
        return out.reshape(hw)

    T = tri.transform[s[ok]]
    b = np.einsum('ijk,ik->ij', T[:, :2], pix[ok] - T[:, 2])
    out[ok] = (np.c_[b, 1 - b.sum(1)] * z[tri.simplices[s[ok]]]).sum(1)

    corners = np.c_[u, v][tri.simplices]
    edge = np.max([np.linalg.norm(corners[:, i] - corners[:, j], axis=1)
                   for i, j in ((0, 1), (1, 2), (2, 0))], axis=0)
    zs = z[tri.simplices]
    bad = (edge > max_edge) | (zs.max(1) / np.maximum(zs.min(1), 1e-6) > max_ratio)
    out[ok] = np.where(bad[s[ok]], 0.0, out[ok])
    return out.reshape(hw)


def sparse_image(u, v, z, hw):
    """The raw projection, one pixel per beam return - what --fill none writes."""
    d = np.zeros(hw)
    d[v.astype(int), u.astype(int)] = z
    return d


# ---------------------------------------------------------------- per-frame worker

def process(job, cfg):
    """One frame: colour -> png, lidar -> depth png. Returns the stats the summary aggregates."""
    cv2.setNumThreads(1)                                   # one thread per process, not per op
    n, impath, binpath = job
    h, w = cfg['hw']

    img = cv2.imread(impath)
    if img is None:
        raise SystemExit(f'could not read {impath}')
    cv2.imwrite(f'{cfg["dst"]}/colors/{n:06d}.png',
                cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))

    pts = np.fromfile(binpath, np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    u, v, z = project(pts, cfg['E'], cfg['K'], cfg['hw'])
    d = (sparse_image(u, v, z, cfg['hw']) if cfg['fill'] == 'none'
         else densify(u, v, z, cfg['hw'], cfg['max_edge'], cfg['max_ratio']))

    raw = np.clip(d * cfg['png_scale'], 0, 65535).astype(np.uint16)
    cv2.imwrite(f'{cfg["dst"]}/depths/{n:06d}.png', raw)
    valid = d > 0
    return (len(z), float(valid.mean()), float(d[valid].min()) if valid.any() else 0.0,
            float(d[valid].max()) if valid.any() else 0.0, int(raw.max()))


# ---------------------------------------------------------------- checks

def selftest(frames, cfg, rng):
    """Leave-one-out: interpolate from 80% of the beams, score at the held-out 20%."""
    print('\nself-test - interpolation accuracy against held-out beams:')
    print(f'  {"frame":>7}{"covered":>10}{"MAE m":>9}{"median m":>11}{"AbsRel":>9}{"d<1.25":>9}')
    for n, _, binpath in frames:
        pts = np.fromfile(binpath, np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        u, v, z = project(pts, cfg['E'], cfg['K'], cfg['hw'])
        hold = rng.random(len(z)) < 0.2
        if hold.sum() < 32:
            continue
        d = densify(u[~hold], v[~hold], z[~hold], cfg['hw'], cfg['max_edge'], cfg['max_ratio'])
        pred = d[v[hold].astype(int), u[hold].astype(int)]
        m = pred > 0
        if m.sum() < 32:
            continue
        gt = z[hold][m]
        err = np.abs(pred[m] - gt)
        d125 = np.mean(np.maximum(pred[m] / gt, gt / pred[m]) < 1.25)
        print(f'  {n:>7}{f"{m.sum()}/{hold.sum()}":>10}{err.mean():>9.2f}{np.median(err):>11.2f}'
              f'{np.mean(err / gt):>9.3f}{d125:>9.3f}')


def warp_test(frames, T_wc, cfg):
    """Warp a frame into a later one with the written calib, poses and lidar depth.

    Covers intrinsics, extrinsics, pose convention and depth scale at once: the ratio must exceed
    1.0 or one of them is wrong. Expect ~1.1-2.6 here, not TUM's ~7x - see the script's caveats.

    Only frames that actually moved are informative: with the camera static the warp is a no-op that
    can only add resampling blur, so the ratio drops below 1 while nothing is wrong. RELLIS 00000
    opens with ~200 near-stationary frames, hence the baseline column and the skip.
    """
    h, w = cfg['hw']
    print('\nwarp check - unwarped/warped photometric MAE, must be > 1.0:')
    print(f'  {"frame":>7}{"moved":>9}' + ''.join(f'{f"+{n}":>9}' for n in (5, 10, 20)))
    for n, impath, binpath in frames:
        if n + 20 >= len(T_wc):
            continue
        moved = np.linalg.norm(T_wc[n + 20][:3, 3] - T_wc[n][:3, 3])
        if moved < 0.2:
            print(f'  {n:>7}{moved:>8.2f}m' + f'{"static, skipped":>27}')
            continue
        pts = np.fromfile(binpath, np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        u, v, z = project(pts, cfg['E'], cfg['K'], cfg['hw'])
        near = z < 30                                      # distant points barely move
        u, v, z = u[near], v[near], z[near]
        P = np.c_[(u - cfg['K'][0, 2]) * z / cfg['K'][0, 0],
                  (v - cfg['K'][1, 2]) * z / cfg['K'][1, 1], z]
        def grey(p):
            img = cv2.resize(cv2.imread(p), (w, h), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        def smp(I, a, b):
            return cv2.remap(I, a.astype(np.float32).reshape(1, -1),
                             b.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
        I0 = grey(impath)
        s0 = smp(I0, u, v)
        cells = []
        for k in (5, 10, 20):
            T = np.linalg.inv(T_wc[n + k]) @ T_wc[n]
            Q = P @ T[:3, :3].T + T[:3, 3]
            f = Q[:, 2] > MIN_Z
            uv = (cfg['K'] @ Q[f].T).T
            un, vn = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
            g = (un > 1) & (un < w - 2) & (vn > 1) & (vn < h - 2)
            if g.sum() < 100:
                cells.append(f'{"-":>9}')
                continue
            IN = grey(frames_path(cfg, n + k))
            warped = np.abs(s0[f][g] - smp(IN, un[g], vn[g])).mean()
            still = np.abs(s0[f][g] - smp(IN, u[f][g], v[f][g])).mean()
            cells.append(f'{still / warped:>9.2f}')
        print(f'  {n:>7}{moved:>8.2f}m' + ''.join(cells))


def frames_path(cfg, n):
    return cfg['images'][n]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='a RELLIS-3D sequence directory, e.g. .../00000')
    ap.add_argument('--calib', required=True, help='the calib root holding Rellis[-_]3D/<seq>/')
    ap.add_argument('--dst', required=True, help='output sequence directory')
    ap.add_argument('--seq', default=None, help='sequence id; default: the --src basename')
    ap.add_argument('--scale', type=float, default=0.5, help='image downscale factor')
    ap.add_argument('--fill', default='linear', choices=['linear', 'none'],
                    help="'linear' interpolates the beams; 'none' writes the raw projection")
    ap.add_argument('--max_edge', type=float, default=32.0,
                    help='px, at the OUTPUT resolution; longer triangle edges are dropped')
    ap.add_argument('--max_ratio', type=float, default=1.3,
                    help='drop a triangle whose max/min depth exceeds this')
    ap.add_argument('--depth_png_scale', type=float, default=256.0,
                    help='metres = px / this. 256 saturates at 256 m; the repo default 6553.5 '
                         'would clip RELLIS at 10 m')
    ap.add_argument('--stride', type=int, default=1, help='keep every Nth frame')
    ap.add_argument('--limit', type=int, default=0, help='stop after N frames (0 = all)')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument('--selftest', action='store_true',
                    help='leave-one-out interpolation accuracy and the warp check, then exit')
    ap.add_argument('--force', action='store_true', help='overwrite a non-empty --dst')
    args = ap.parse_args()

    seq = args.seq or os.path.basename(args.src.rstrip('/'))
    (fx, fy, cx, cy), A = read_calib(args.calib, seq)
    E = np.linalg.inv(A)                                   # lidar -> camera
    s = args.scale
    K = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1]])

    images = sorted(os.listdir(f'{args.src}/{IMAGE_DIR}'))
    clouds = sorted(os.listdir(f'{args.src}/{CLOUD_DIR}'))
    T_wl = load_poses(f'{args.src}/poses.txt')
    if not len(images) == len(clouds) == len(T_wl):
        raise SystemExit(f'{len(images)} images, {len(clouds)} scans, {len(T_wl)} poses - RELLIS '
                         f'pairs them by index, so all three must match')
    T_wc = T_wl @ A                                        # camera-to-world

    idx = list(range(0, len(images), args.stride))[:args.limit or None]
    frames = [(n, f'{args.src}/{IMAGE_DIR}/{images[i]}', f'{args.src}/{CLOUD_DIR}/{clouds[i]}')
              for n, i in enumerate(idx)]

    h0, w0 = cv2.imread(frames[0][1]).shape[:2]
    hw = (int(round(h0 * s)), int(round(w0 * s)))
    cfg = {'dst': args.dst, 'hw': hw, 'K': K, 'E': E, 'fill': args.fill,
           'max_edge': args.max_edge, 'max_ratio': args.max_ratio,
           'png_scale': args.depth_png_scale, 'images': [f[1] for f in frames]}

    fov = (2 * np.degrees(np.arctan(w0 / 2 / fx)), 2 * np.degrees(np.arctan(h0 / 2 / fy)))
    print(f'source   : {args.src}   sequence {seq}')
    print(f'calib    : fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}  '
          f'FoV {fov[0]:.1f} x {fov[1]:.1f} deg')
    print(f'images   : {len(images)} at {w0}x{h0} -> {len(frames)} at {hw[1]}x{hw[0]} '
          f'(stride {args.stride})')
    print(f'depth    : {args.fill}, max_edge {args.max_edge} px, max_ratio {args.max_ratio}, '
          f'scale {args.depth_png_scale} (saturates at {65535/args.depth_png_scale:.1f} m)')
    step = np.linalg.norm(np.diff(T_wc[idx][:, :3, 3], axis=0), axis=1)
    net = np.linalg.norm(T_wc[idx][-1, :3, 3] - T_wc[idx][0, :3, 3])
    print(f'motion   : {step.sum():.0f} m path, {net:.0f} m net, median '
          f'{100*np.median(step):.1f} cm/frame, {(step < 0.01).sum()} near-static frames')

    if args.selftest:
        probe = [frames[i] for i in np.linspace(0, len(frames) - 1, 5).astype(int)]
        selftest(probe, cfg, np.random.default_rng(0))
        warp_test(probe[:-1], T_wc[idx], cfg)
        return

    for sub in ('colors', 'depths'):
        d = f'{args.dst}/{sub}'
        if os.path.isdir(d) and os.listdir(d) and not args.force:
            raise SystemExit(f'{d} is not empty; pass --force to overwrite it')
        os.makedirs(d, exist_ok=True)

    print(f'\nwriting with {args.jobs} processes...')
    from tqdm import tqdm
    with Pool(args.jobs) as pool:
        stats = list(tqdm(pool.imap(partial(process, cfg=cfg), frames, chunksize=8),
                          total=len(frames), desc='frames'))
    stats = np.array(stats, dtype=np.float64)

    np.savetxt(f'{args.dst}/traj_tum.txt',
               [np.hstack(([n], to_tum(T_wc[i]))) for n, i in enumerate(idx)])
    with open(f'{args.dst}/calib.txt', 'w') as f:
        f.write(f'{fx*s} {fy*s} {cx*s} {cy*s}')
    with open(f'{args.dst}/preprocess_info.txt', 'w') as f:
        f.write(f'source {args.src}\nsequence {seq}\nscale {s}\nfill {args.fill}\n'
                f'max_edge {args.max_edge}\nmax_ratio {args.max_ratio}\n'
                f'depth_png_scale {args.depth_png_scale}\nstride {args.stride}\n'
                f'frames {len(frames)}\nresolution {hw[1]}x{hw[0]}\n')

    beams = stats[:, 0].mean()
    print(f'\nwrote {len(frames)} frames to {args.dst}')
    print(f'  beams in frame : {beams:.0f} avg  ({100*beams/(hw[0]*hw[1]):.2f}% of pixels)')
    print(f'  valid depth    : {100*stats[:,1].mean():.1f}% avg, '
          f'{100*stats[:,1].min():.1f}% worst, {100*stats[:,1].max():.1f}% best')
    print(f'  depth range    : {stats[stats[:,2]>0,2].min():.1f} .. {stats[:,3].max():.1f} m')
    sat = '   WARNING: SATURATED, raise --depth_png_scale' if stats[:, 4].max() >= 65535 else ''
    print(f'  max png value  : {stats[:,4].max():.0f} of 65535{sat}')
    print(f'  calib.txt      : {fx*s} {fy*s} {cx*s} {cy*s}   (no distortion - do NOT pass '
          f'--undistort/--cropborder)')
    print(f'  set DEPTH_PNG_SCALE = {args.depth_png_scale} in init_adapt_pipeline.py PARAMETERS')


if __name__ == '__main__':
    main()
