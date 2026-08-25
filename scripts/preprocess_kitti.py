"""Preprocess a KITTI odometry (colour) sequence into the layout the rest of this repo expects.

    python scripts/preprocess_kitti.py --seq 00 --dst data/KITTI/00

Produces preprocess_rellis3d.py's shape, minus the depth:

    colors -> SYMLINK to the mirror's sequences/<seq>/image_2   (0 bytes; --copy to duplicate)
    traj_tum.txt      "<index> tx ty tz qx qy qz qw", camera-to-world
    calib.txt         "fx fy cx cy"   (KITTI's colour images are RECTIFIED - no distortion, so do
                      NOT pass --undistort / --cropborder)

    depths/%06d.png   OPTIONAL, --with-depth; same index and size as colors, uint16 *
                      --depth_png_scale, 0 = no lidar return

The odometry benchmark itself ships no depth, but the velodyne for these exact sequences is on the
mirror under SemanticKitti/sequences/<seq>/, with `Tr` (velo->cam0) in its calib.txt - and its
P0..P3 are bit-identical to the odometry calib, which is what makes the two safe to combine (this
script asserts it). `--with-depth` projects that sweep into cam2 and reuses preprocess_rellis3d.py's
Delaunay densify, so KITTI gets the same depths/ every other scene has: the extract accuracy table
and the whole `prior` stage (9.2.2) start working, which is the only way to tell "the adapter got
closer to reality" from "the adapter got closer to the tracker's own errors".

Without it, run with DEPTHS = None - every consumer handles that: check_sequence skips it,
extract/accuracy.py's table no-ops, and the driver refuses the `prior` stage.

Two properties of the lidar GT are worth carrying with the numbers. It is a SINGLE sweep, not the
KITTI depth benchmark's 11-scan accumulation (that is not on this mirror - kitti.squashfs holds the
raw drives but its proj_depth/ and velodyne_points/ are empty), so it is ~4 % dense before densify
and carries the usual occlusion artifact: the lidar sits above and behind the camera, so background
points it can see past an occluder still project into the image. And the HDL-64E's upper FoV ends
about a quarter of the way down the frame, so the sky and the tops of buildings have no GT at all
and every metric is over the lower ~75 %.

Two conventions are worth stating, because both are silent when wrong:

  * the images are SYMLINKED, not renamed. KITTI already names them %06d, which is what
    slam/runner.py:save_trajectory needs - it reads the timestamp out of the filename, so
    traj_tum.txt's index column and the frame number are the same thing. This script asserts that
    the two agree rather than assuming it.
  * GT poses are for cam0 (the left GREY camera) while image_2 is cam2 (the left COLOUR one), 6 cm
    apart. The poses are moved onto cam2 here. It is small against KITTI-scale ATE and evo's Sim(3)
    would not absorb it (the offset is a right-multiplication, the alignment a left one), so it is
    corrected at the source instead of argued about later.
"""
import argparse
import os
import shutil
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# the lidar->image half of this script is RELLIS's, unchanged: same 64-beam geometry, same (N, 4)
# float32 bin layout (its cloud dir is literally named os1_cloud_node_kitti_bin), same output
# contract. Importing beats restating it - a second densify is a second set of thresholds to drift.
from preprocess_rellis3d import densify, project, sparse_image

MIRROR = '/storage/group/dataset_mirrors/01_incoming/kitti_odom_color'
VELO_MIRROR = '/storage/group/dataset_mirrors/01_incoming/SemanticKitti'   # the only lidar source
IMAGE_DIR = 'image_2'          # the left COLOUR camera; image_3 is the right one, unused (mono)
POSE_SEQS = 11                 # sequences 00..10 have ground-truth poses; 11..21 do not


def read_calib(path):
    """(fx, fy, cx, cy), and cam2's origin in the cam0 frame.

    KITTI's calib.txt holds the 3x4 PROJECTION matrices of the rectified rig: x = P_i [X; 1] for a
    point X in cam0 coordinates. P2 is therefore K [I | t], so K is its left 3x3 and the camera
    centre is -K^-1 P2[:, 3].
    """
    rows = {}
    for line in open(path):
        key, _, values = line.partition(':')
        if values.strip():
            rows[key.strip()] = np.array(values.split(), dtype=np.float64)
    if 'P2' not in rows:
        raise SystemExit(f'{path} has no P2 row - is this a KITTI odometry calib.txt?')

    P2 = rows['P2'].reshape(3, 4)
    K = P2[:, :3]
    return (K[0, 0], K[1, 1], K[0, 2], K[1, 2]), -np.linalg.solve(K, P2[:, 3])


def calib_rows(path):
    """Every `KEY: values` row of a KITTI calib.txt as a float array."""
    rows = {}
    for line in open(path):
        key, _, values = line.partition(':')
        if values.strip():
            rows[key.strip()] = np.array(values.split(), dtype=np.float64)
    return rows


def velo_to_cam2(velo_calib, odom_calib, c2):
    """The velodyne -> cam2 4x4, and the guard that the two calib files describe one rig.

    `Tr` lives only in the SemanticKitti copy; P0..P3 live in both and must agree bit for bit, or
    the two directories are not the same sequence of the same recording and every projected depth
    would be silently wrong rather than absent.
    """
    velo, odom = calib_rows(velo_calib), calib_rows(odom_calib)
    if 'Tr' not in velo:
        raise SystemExit(f'{velo_calib} has no Tr row - that is the velodyne->cam0 extrinsic, and '
                         f'without it the sweep cannot be projected into the image')
    for k in ('P0', 'P1', 'P2', 'P3'):
        if k in velo and k in odom and not np.allclose(velo[k], odom[k]):
            raise SystemExit(f'{k} differs between {odom_calib} and {velo_calib}; they are not '
                             f'the same rig, so the lidar must not be projected with this calib')

    E = np.vstack([velo['Tr'].reshape(3, 4), [0, 0, 0, 1.0]])   # velodyne -> cam0 (rectified)
    E[:3, 3] -= c2                       # ...and on to cam2, the camera the images come from
    return E


def depth_job(job, cfg):
    """One frame: velodyne bin -> depths/%06d.png. Returns the stats the summary aggregates."""
    cv2.setNumThreads(1)
    n, binpath = job
    pts = np.fromfile(binpath, np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    u, v, z = project(pts, cfg['E'], cfg['K'], cfg['hw'])
    d = (sparse_image(u, v, z, cfg['hw']) if cfg['fill'] == 'none'
         else densify(u, v, z, cfg['hw'], cfg['max_edge'], cfg['max_ratio']))
    raw = np.clip(d * cfg['png_scale'], 0, 65535).astype(np.uint16)
    cv2.imwrite(f'{cfg["dst"]}/depths/{n:06d}.png', raw)
    valid = d > 0
    return (len(z), float(valid.mean()), float(d[valid].min()) if valid.any() else 0.0,
            float(d[valid].max()) if valid.any() else 0.0, int(raw.max()))


def write_depths(args, seq, K, c2, n_images):
    """depths/ for the whole sequence, 1:1 by index with colors/. Returns the summary line."""
    seq_dir = f'{args.velodyne}/sequences/{seq}'
    velo_dir = f'{seq_dir}/velodyne'
    for f in (velo_dir, f'{seq_dir}/calib.txt'):
        if not os.path.exists(f):
            raise SystemExit(f'missing lidar input: {f}  (pass --velodyne, or drop --with-depth)')

    scans = sorted(os.listdir(velo_dir))
    if len(scans) != n_images:
        raise SystemExit(f'{len(scans)} velodyne scans but {n_images} images - every consumer '
                         f'indexes GT depth by RGB frame number, so they must be 1:1')

    E = velo_to_cam2(f'{seq_dir}/calib.txt', f'{args.src}/sequences/{seq}/calib.txt', c2)
    h0, w0 = cv2.imread(f'{args.src}/sequences/{seq}/{IMAGE_DIR}/{scans[0][:-4]}.jpg').shape[:2]
    cfg = {'dst': args.dst, 'hw': (h0, w0), 'K': K, 'E': E, 'fill': args.fill,
           'max_edge': args.max_edge, 'max_ratio': args.max_ratio,
           'png_scale': args.depth_png_scale}

    os.makedirs(f'{args.dst}/depths', exist_ok=True)
    jobs = [(n, f'{velo_dir}/{f}') for n, f in enumerate(scans)]
    print(f'\ndepths   : {len(jobs)} frames from {velo_dir}')
    print(f'           fill={args.fill} max_edge={args.max_edge} max_ratio={args.max_ratio} '
          f'scale={args.depth_png_scale} (saturates at {65535/args.depth_png_scale:.1f} m)')
    with Pool(args.jobs) as pool:
        stats = np.array(pool.map(partial(depth_job, cfg=cfg), jobs, chunksize=8))

    sat = '   WARNING: SATURATED, raise --depth_png_scale' if stats[:, 4].max() >= 65535 else ''
    print(f'           {stats[:, 0].mean():.0f} beams/frame in frame, '
          f'{100*stats[:, 1].mean():.1f}% of pixels valid, '
          f'depth {stats[:, 2].min():.2f}-{stats[:, 3].max():.1f} m{sat}')
    return (f'depths {len(jobs)} frames, fill {args.fill}, max_edge {args.max_edge}, '
            f'max_ratio {args.max_ratio}, scale {args.depth_png_scale}, '
            f'{100*stats[:, 1].mean():.1f}% valid\nvelodyne {velo_dir}\n')


def load_poses(path):
    """poses/<seq>.txt (N, 12) -> (N, 4, 4) cam0-to-world."""
    p = np.loadtxt(path).reshape(-1, 3, 4)
    return np.concatenate([p, np.tile([[[0, 0, 0, 1.0]]], (len(p), 1, 1))], axis=1)


def to_tum(T):
    """One 4x4 camera-to-world -> tx ty tz qx qy qz qw, as preprocess_rellis3d.py writes it."""
    return np.hstack((T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()))


def link_colors(src, dst, copy, force):
    """`dst` as a symlink to the mirror's image directory, or a copy of it.

    Already pointing at `src` is success, not a conflict: re-running to add --with-depth must not
    be gated behind --force, which means "replace colours" and is a different intent.
    """
    if os.path.islink(dst) and os.path.realpath(dst) == os.path.abspath(src):
        return
    if os.path.islink(dst) or os.path.exists(dst):
        if not force:
            raise SystemExit(f'{dst} already exists; pass --force to replace it')
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            shutil.rmtree(dst)
    if copy:
        shutil.copytree(src, dst)
    else:
        # absolute, because the mirror is outside the repo and data/ is itself a symlink
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=MIRROR, help=f'KITTI odometry root (default {MIRROR})')
    ap.add_argument('--seq', default='00', help='sequence id, 00..10 (11..21 have no GT poses)')
    ap.add_argument('--dst', default=None, help='output dir (default data/KITTI/<seq>)')
    ap.add_argument('--copy', action='store_true',
                    help='copy the images instead of symlinking them (~600 MB for seq 00)')
    ap.add_argument('--force', action='store_true', help='replace an existing colors/')
    ap.add_argument('--with-depth', action='store_true', dest='with_depth',
                    help='also write depths/ by projecting the SemanticKitti velodyne into cam2')
    ap.add_argument('--velodyne', default=VELO_MIRROR,
                    help=f'root holding sequences/<seq>/velodyne + calib.txt with Tr '
                         f'(default {VELO_MIRROR})')
    ap.add_argument('--fill', default='linear', choices=['linear', 'none'],
                    help="'linear' interpolates the beams over their Delaunay triangulation; "
                         "'none' writes the raw ~4%% projection")
    ap.add_argument('--max_edge', type=float, default=24.0,
                    help='px; triangles with a longer edge are dropped, so sky and silhouettes '
                         'stay 0 rather than being filled with fictitious surface')
    ap.add_argument('--max_ratio', type=float, default=1.3,
                    help='drop a triangle whose max/min depth exceeds this')
    ap.add_argument('--depth_png_scale', type=float, default=256.0,
                    help='metres = px / this. 256 is the KITTI convention and saturates at 256 m; '
                         "the repo's 6553.5 would clip at 10 m")
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 8) // 2))
    args = ap.parse_args()

    seq = args.seq.zfill(2)            # the mirror's directories are two-digit
    dst = args.dst or f'data/KITTI/{seq}'
    seq_dir = f'{args.src}/sequences/{seq}'
    pose_file = f'{args.src}/poses/{seq}.txt'
    image_dir = f'{seq_dir}/{IMAGE_DIR}'

    if int(seq) >= POSE_SEQS:
        raise SystemExit(f'sequence {seq} has no ground-truth poses (only 00..{POSE_SEQS-1} do), '
                         f'and every arm is scored by ATE against them')
    for f in (image_dir, f'{seq_dir}/calib.txt', pose_file):
        if not os.path.exists(f):
            raise SystemExit(f'missing input: {f}')

    (fx, fy, cx, cy), c2 = read_calib(f'{seq_dir}/calib.txt')
    T_wc = load_poses(pose_file)
    images = sorted(os.listdir(image_dir))
    if len(images) != len(T_wc):
        raise SystemExit(f'{len(images)} images but {len(T_wc)} poses - KITTI pairs them by index, '
                         f'so they must match')

    # the filename IS the timestamp (save_trajectory reads it back out), so a gap in the numbering
    # would silently offset every pose from the frame it belongs to
    numbers = [int(os.path.splitext(f)[0]) for f in images]
    if numbers != list(range(len(images))):
        raise SystemExit(f'{image_dir} is not numbered 0..{len(images)-1} without gaps; '
                         f'traj_tum.txt indexes poses by frame number and they would not line up')

    # cam0 -> cam2: rectified, so the rotation is identity and only the 6 cm baseline moves
    T_02 = np.eye(4)
    T_02[:3, 3] = c2
    T_wc = T_wc @ T_02

    h0, w0 = cv2.imread(f'{image_dir}/{images[0]}').shape[:2]
    fov = (2 * np.degrees(np.arctan(w0 / 2 / fx)), 2 * np.degrees(np.arctan(h0 / 2 / fy)))
    step = np.linalg.norm(np.diff(T_wc[:, :3, 3], axis=0), axis=1)
    net = np.linalg.norm(T_wc[-1, :3, 3] - T_wc[0, :3, 3])

    print(f'source   : {seq_dir}')
    print(f'calib    : fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}  '
          f'FoV {fov[0]:.1f} x {fov[1]:.1f} deg   (rectified, no distortion)')
    print(f'cam0->cam2: {100*np.linalg.norm(c2):.1f} cm applied to every GT pose')
    print(f'images   : {len(images)} at {w0}x{h0}')
    print(f'motion   : {step.sum():.0f} m path, {net:.0f} m net, median '
          f'{100*np.median(step):.1f} cm/frame, {(step < 0.01).sum()} near-static frames')

    args.dst = dst          # write_depths writes beside the rest of the layout
    os.makedirs(dst, exist_ok=True)
    link_colors(image_dir, f'{dst}/colors', args.copy, args.force)
    np.savetxt(f'{dst}/traj_tum.txt',
               [np.hstack(([n], to_tum(T))) for n, T in enumerate(T_wc)])
    with open(f'{dst}/calib.txt', 'w') as f:
        f.write(f'{fx} {fy} {cx} {cy}')

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    depth_note = (write_depths(args, seq, K, c2, len(images)) if args.with_depth
                  else 'no depths: --with-depth not passed\n')
    with open(f'{dst}/preprocess_info.txt', 'w') as f:
        f.write(f'source {seq_dir}\nsequence {seq}\nframes {len(images)}\n'
                f'resolution {w0}x{h0}\ncolors {"copied" if args.copy else "symlinked"}\n'
                f'cam0_to_cam2 {c2.tolist()}\n{depth_note}')

    print(f'\nwrote {dst}')
    print(f'  colors     : {"copy" if args.copy else "symlink"} -> {image_dir}')
    print(f'  calib.txt  : {fx} {fy} {cx} {cy}   (do NOT pass --undistort/--cropborder)')
    print(f'  traj_tum.txt: {len(T_wc)} poses, camera-to-world, indexed by frame number')
    print("\nin the driver PARAMETERS block: CONFIG = 'config/kitti_config.yaml', and")
    if args.with_depth:
        print(f"  DEPTHS = f'{{DATA}}/depths', DEPTH_PNG_SCALE = {args.depth_png_scale}  "
              f"- 'prior' can now be in STAGES")
    else:
        print("  DEPTHS = None (no GT depth without --with-depth); drop 'prior' from STAGES, "
              'that stage scores against GT depth')


if __name__ == '__main__':
    main()
