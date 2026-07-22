"""Preprocess a TUM RGB-D sequence into the layout the rest of this repo expects.

    python scripts/preprocess_tum.py \
        --src /storage/group/dataset_mirrors/01_incoming/TUM_RGBD_Dataset/rgbd_dataset_freiburg1_room \
        --dst data/TUM/rgbd_dataset_freiburg1_room

Produces the same shape as preprocess_replica.py, i.e. colors/ depths/ traj_tum.txt, plus a
calib.txt the way preprocess_scannet.py does:

    colors/%06d.png   undistorted, cropped
    depths/%06d.png   same index, 16-bit, rescaled from TUM's 5000 to HI-SLAM2's 6553.5
    traj_tum.txt      "<index> tx ty tz qx qy qz qw"
    calib.txt         "fx fy cx cy"   (no distortion terms - see below)

Two things here are load-bearing rather than cosmetic.

**Sequential %06d names.** demo.py:72 derives the trajectory timestamp by pulling the last number
out of the filename, so index names make the timestamps frame indices - matching Replica and
letting evo_ape associate exactly. Real TUM names would break more than that:
lora_adapt_vggt.py:131 keys poses by int(timestamp), and 1305031910.765238 truncates to the same
integer for ~30 consecutive frames, silently collapsing the pose dict. colors/ and depths/ must
also stay 1:1 by index, because eval_utils.py:47, export_slam_depth.py:122 and
_full_run_common.py:193 all index GT depth by RGB frame number.

**Undistortion happens here, not at runtime.** demo.py has --undistort/--cropborder, but
_full_run_common.py:31 stream_resize re-derives the GT frame with resize only, so those flags
would compare undistorted renders against distorted GT. Doing it offline and shipping a
distortion-free calib.txt keeps every consumer correct with no changes. Do NOT pass --undistort
or --cropborder for data produced by this script.
"""
import argparse
import os

import cv2
import numpy as np

# fx, fy, cx, cy, (k1, k2, p1, p2, k3)
# https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
CAMERAS = {
    'fr1': (517.306408, 516.469215, 318.643040, 255.313989,
            (0.262383, -0.953104, -0.005358, 0.002628, 1.163314)),
    'fr2': (520.908620, 521.007327, 325.141442, 249.701764,
            (0.231222, -0.784899, -0.003257, -0.000105, 0.917205)),
    'fr3': (535.4, 539.2, 320.1, 247.6, (0.0, 0.0, 0.0, 0.0, 0.0)),
}

TUM_DEPTH_SCALE = 5000.0      # TUM: metres = px / 5000
HI2_DEPTH_SCALE = 6553.5      # this repo: metres = px / 6553.5 (eval_utils.py:47)


def read_index(path):
    """Parse one of TUM's '<timestamp> <payload...>' listings, skipping # comments."""
    rows = [ln.split() for ln in open(path) if not ln.startswith('#') and ln.strip()]
    return np.array([float(r[0]) for r in rows]), [r[1:] for r in rows]


def nearest(src_t, dst_t):
    """For each src timestamp, the index of the nearest dst timestamp and the gap."""
    i = np.clip(np.searchsorted(dst_t, src_t), 1, len(dst_t) - 1)
    j = np.where(np.abs(dst_t[i] - src_t) < np.abs(dst_t[i - 1] - src_t), i, i - 1)
    return j, np.abs(dst_t[j] - src_t)


def undistort_maps(K, dist, shape):
    """Precomputed remap maps, so interpolation can be chosen per image type.

    cv2.undistort is bilinear-only. That is fine for colour but wrong for depth: interpolating
    across a depth discontinuity invents surfaces that were never measured, and blending with an
    invalid (0) pixel drags real depths toward zero. Same maps, two interpolation modes.
    """
    h, w = shape
    return cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)


def valid_border(maps, shape):
    """Largest symmetric crop containing no undistortion black border.

    Undistortion maps every output pixel back through the distortion model, so with fr1's strong
    barrel term (k1=0.26) the periphery samples outside the source image and comes back black.
    Measure the extent instead of guessing a constant - fr3 needs 0, fr1 needs 18. Probe with
    INTER_LINEAR because that is what the colour images use, and it bleeds the border one pixel
    further in than INTER_NEAREST would.
    """
    h, w = shape
    probe = cv2.remap(np.full((h, w), 255, np.uint8), *maps, cv2.INTER_LINEAR)
    valid = probe > 0
    lo, hi = 0, min(h, w) // 2
    while lo < hi:
        b = (lo + hi) // 2
        if valid[b:h - b, b:w - b].all():
            hi = b
        else:
            lo = b + 1
    return lo


def stream_shape(h, w):
    """The resolution demo.py:49-55 will resize this to, for the sanity print."""
    RES = 341 * 640
    h1 = int(h * np.sqrt(RES / (h * w)))
    w1 = int(w * np.sqrt(RES / (h * w)))
    return h1 - h1 % 8, w1 - w1 % 8


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='an unpacked rgbd_dataset_freiburgN_* directory')
    ap.add_argument('--dst', required=True, help='output sequence directory')
    ap.add_argument('--camera', default='auto', choices=['auto'] + list(CAMERAS),
                    help="intrinsics set; 'auto' reads freiburgN out of the source path")
    ap.add_argument('--max_diff', type=float, default=0.02,
                    help='max |dt| in seconds when associating rgb->depth and rgb->groundtruth')
    ap.add_argument('--border', type=int, default=-1,
                    help='crop border in px; -1 measures the undistortion border automatically')
    args = ap.parse_args()

    cam = args.camera
    if cam == 'auto':
        cam = next((c for c in CAMERAS if f'freiburg{c[-1]}' in os.path.basename(args.src)), None)
        if cam is None:
            raise SystemExit(f'cannot infer camera from {args.src!r}; pass --camera fr1|fr2|fr3')
    fx, fy, cx, cy, dist = CAMERAS[cam]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array(dist)
    print(f'source  : {args.src}\ncamera  : {cam}  fx={fx} fy={fy} cx={cx} cy={cy}')

    # ---- associate rgb -> depth and rgb -> groundtruth ----
    t_rgb, f_rgb = read_index(f'{args.src}/rgb.txt')
    t_dep, f_dep = read_index(f'{args.src}/depth.txt')
    t_gt, p_gt = read_index(f'{args.src}/groundtruth.txt')
    gt = np.array([[float(x) for x in r] for r in p_gt])          # tx ty tz qx qy qz qw

    i_dep, dt_dep = nearest(t_rgb, t_dep)
    i_gt, dt_gt = nearest(t_rgb, t_gt)
    keep = np.where((dt_dep <= args.max_diff) & (dt_gt <= args.max_diff))[0]
    print(f'frames  : {len(t_rgb)} rgb, {len(t_dep)} depth, {len(t_gt)} gt poses -> '
          f'{len(keep)} kept, {len(t_rgb) - len(keep)} dropped (max_diff={args.max_diff}s)')
    print(f'          worst |dt|: rgb->depth {dt_dep[keep].max()*1000:.1f} ms, '
          f'rgb->gt {dt_gt[keep].max()*1000:.1f} ms')
    if len(keep) < 20:
        raise SystemExit('too few associated frames; check --max_diff and the source layout')

    # ---- geometry ----
    h0, w0 = cv2.imread(f'{args.src}/{f_rgb[0][0]}').shape[:2]
    maps = undistort_maps(K, dist, (h0, w0))
    b = valid_border(maps, (h0, w0)) if args.border < 0 else args.border
    h, w = h0 - 2 * b, w0 - 2 * b
    print(f'undistort: {w0}x{h0}, border {b} px -> {w}x{h}, tracked at '
          f'{stream_shape(h, w)[1]}x{stream_shape(h, w)[0]}')

    os.makedirs(f'{args.dst}/colors', exist_ok=True)
    os.makedirs(f'{args.dst}/depths', exist_ok=True)

    for n, k in enumerate(keep):
        rgb = cv2.remap(cv2.imread(f'{args.src}/{f_rgb[k][0]}'), *maps, cv2.INTER_LINEAR)
        cv2.imwrite(f'{args.dst}/colors/{n:06d}.png', rgb[b:h0 - b, b:w0 - b])

        # TUM depth is already registered to the colour frame, so it carries the same distortion
        # and takes the same maps - but nearest-neighbour, see undistort_maps().
        d = cv2.imread(f'{args.src}/{f_dep[i_dep[k]][0]}', cv2.IMREAD_ANYDEPTH)
        d = cv2.remap(d, *maps, cv2.INTER_NEAREST)
        d = d[b:h0 - b, b:w0 - b].astype(np.float64) * (HI2_DEPTH_SCALE / TUM_DEPTH_SCALE)
        cv2.imwrite(f'{args.dst}/depths/{n:06d}.png', np.clip(d, 0, 65535).astype(np.uint16))

        if n % 200 == 0:
            print(f'  {n}/{len(keep)}')

    # row index as timestamp, matching preprocess_replica.py:28
    np.savetxt(f'{args.dst}/traj_tum.txt',
               np.concatenate([np.arange(len(keep))[:, None], gt[i_gt[keep]]], axis=1))
    with open(f'{args.dst}/calib.txt', 'w') as f:
        f.write(f'{fx} {fy} {cx - b} {cy - b}')

    print(f'\nwrote {len(keep)} frames to {args.dst}')
    print(f'calib.txt: {fx} {fy} {cx - b} {cy - b}   (distortion already applied - do NOT pass '
          f'--undistort/--cropborder)')


if __name__ == '__main__':
    main()
