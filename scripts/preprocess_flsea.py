"""Preprocess an FLSea VI (visual-inertial) dive into the layout the rest of this repo expects.

    python scripts/preprocess_flsea.py --src <dive dir> --inspect        # ALWAYS do this first
    python scripts/preprocess_flsea.py --src <dive dir> --dst data/FLSea/<dive> [mapping flags]

Produces preprocess_tum.py's shape:

    colors/%06d.png   undistorted; TIFF -> PNG, RENUMBERED sequentially
    depths/%06d.png   same index and size as colors, uint16, metres = px / --depth-png-scale
    traj_tum.txt      "<index> tx ty tz qx qy qz qw", camera-to-world
    calib.txt         "fx fy cx cy"   (no distortion terms - undistorted here, not at runtime)
    preprocess_info.txt

WHY THIS SCRIPT LEADS WITH --inspect. FLSea VI ships 12 dives of forward-looking shallow-water
(<10 m) imagery with photogrammetry depth and pose, but the published papers do not document the
delivered file names, the pose file's columns, or the depth maps' units - and the Kaggle page is
JavaScript-rendered, so none of that could be read before writing this. Rather than guess a layout
and fail silently, --inspect walks the source, reports every extension, sample file names, the head
of every small text/CSV file, and the dtype/range of one image and one depth map, then PROPOSES a
mapping. Confirm or override it with the flags below and only then convert.

THREE INVARIANTS THIS SCRIPT ENFORCES, because each is silent when wrong (preprocess_tum.py:15):
  * colors/ is renumbered %06d. slam/runner.py:save_trajectory parses the timestamp out of the
    filename, so the index column of traj_tum.txt and the frame number must be the same integer.
    FLSea names files by timestamp, which would collapse that mapping.
  * colors/ and depths/ are 1:1 BY INDEX. Every consumer looks up GT depth by RGB frame number.
  * the images are UNDISTORTED HERE, not at runtime (10.1). Do NOT then pass UNDISTORT/CROP_BORDER
    in the run config: common.py:stream_resize re-derives a frame with a resize alone, so runtime
    undistortion would misalign it against the GT depth this script wrote.

DEPTH SCALE IS NOT 6553.5 BY DEFAULT, deliberately. That is TUM's convention and it caps at
65535/6553.5 = 9.9999 m - and FLSea is described as shallower than 10 m, i.e. right at the ceiling,
where any deeper pixel would wrap silently. The default here is 1000.0: a 65.5 m ceiling at 1 mm
resolution, which is far more headroom than photogrammetry depth needs. --inspect prints the actual
depth range it finds; set DEPTH_PNG_SCALE in the run config to whatever is used here.

IMU IS IGNORED. HI-SLAM2 is visual-only - there is no inertial term anywhere in the tracker - so
the VI dataset's IMU stream is not converted. It is reported by --inspect for completeness.
"""
import argparse
import os
import sys
import glob

import cv2
import numpy as np

IMG_EXT = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
TXT_EXT = ('.txt', '.csv', '.yaml', '.yml', '.json', '.md')
DEFAULT_DEPTH_SCALE = 1000.0


# ---------------------------------------------------------------- inspection

def walk(src, max_depth=3):
    """Every directory under src with its file count by extension."""
    rows = []
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        if rel.count(os.sep) >= max_depth:
            dirs[:] = []
        ext = {}
        for f in files:
            ext[os.path.splitext(f)[1].lower()] = ext.get(os.path.splitext(f)[1].lower(), 0) + 1
        if ext:
            rows.append((rel, ext, sorted(files)[:3]))
    return rows


def describe_image(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return f'{os.path.basename(path)}: UNREADABLE by cv2'
    fin = np.isfinite(im) if im.dtype.kind == 'f' else np.ones_like(im, bool)
    v = im[fin]
    return (f'{os.path.basename(path)}: shape {im.shape} dtype {im.dtype}  '
            f'range {v.min():.4g}..{v.max():.4g}  '
            f'finite {100*fin.mean():.1f}%  nonzero {100*(v != 0).mean():.1f}%')


def inspect(src):
    print(f'=== FLSea source: {src}\n')
    rows = walk(src)
    print(f'{"directory":<44} files')
    for rel, ext, sample in rows:
        e = '  '.join(f'{k or "(none)"}:{v}' for k, v in sorted(ext.items(), key=lambda x: -x[1]))
        print(f'  {rel:<42} {e}')
        print(f'{"":44}   e.g. {", ".join(sample)}')
    print('\n=== small text / config files, first 6 lines each')
    for root, _, files in os.walk(src):
        for f in sorted(files):
            p = os.path.join(root, f)
            if os.path.splitext(f)[1].lower() in TXT_EXT and os.path.getsize(p) < 3_000_000:
                print(f'\n--- {os.path.relpath(p, src)}  ({os.path.getsize(p)} B)')
                with open(p, errors='replace') as fh:
                    for i, line in enumerate(fh):
                        if i >= 6:
                            print('    ...')
                            break
                        print('   ', line.rstrip()[:160])
    print('\n=== one sample of each image-like extension')
    seen = set()
    for root, _, files in os.walk(src):
        for f in sorted(files):
            e = os.path.splitext(f)[1].lower()
            if e in IMG_EXT and (root, e) not in seen:
                seen.add((root, e))
                print(f'  {os.path.relpath(root, src)}/  {describe_image(os.path.join(root, f))}')
    print('\n=== what to do next')
    print('  Pass the directories and the pose file you see above via --images / --depths /')
    print('  --poses, and the intrinsics via --calib "fx fy cx cy k1 k2 p1 p2 [k3]".')
    print('  A float depth map in metres needs no --depth-in-scale; an integer one does.')


# ---------------------------------------------------------------- readers

def read_poses(path, cols, invert):
    """(timestamps, 4x4 camera-to-world). `cols` names the column order actually present."""
    raw = [l.split() if ',' not in l else l.split(',')
           for l in open(path) if l.strip() and not l.lstrip().startswith('#')]
    try:
        arr = np.array([[float(x) for x in r] for r in raw[1:] if len(r) >= 8], float)
    except ValueError:
        arr = np.array([[float(x) for x in r] for r in raw if len(r) >= 8], float)
    if not len(arr):
        raise SystemExit(f'{path}: no rows with >= 8 numeric columns - check --pose-cols')
    from scipy.spatial.transform import Rotation
    i = {c: k for k, c in enumerate(cols.split(','))}
    for need in ('t', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'):
        if need not in i:
            raise SystemExit(f'--pose-cols is missing {need!r}: got {cols}')
    ts = arr[:, i['t']]
    T = np.tile(np.eye(4), (len(arr), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(arr[:, [i['qx'], i['qy'], i['qz'], i['qw']]]).as_matrix()
    T[:, :3, 3] = arr[:, [i['tx'], i['ty'], i['tz']]]
    if invert:                       # the file gives world-to-camera; this repo wants c2w
        T = np.linalg.inv(T)
    return ts, T


def stamp_of(path):
    """The numeric stem of a file name, which FLSea uses as its timestamp."""
    s = os.path.splitext(os.path.basename(path))[0]
    s = ''.join(ch for ch in s if ch.isdigit() or ch == '.')
    return float(s) if s else None


# ---------------------------------------------------------------- conversion

def undistort_maps(K, dist, shape):
    h, w = shape
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    return cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_32FC1), newK


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='one FLSea VI dive directory')
    ap.add_argument('--dst', help='output, e.g. data/FLSea/<dive>')
    ap.add_argument('--inspect', action='store_true', help='report the layout and stop')
    ap.add_argument('--images', default='imgs', help='image subdirectory, relative to --src')
    ap.add_argument('--depths', default='depth', help='depth subdirectory; "" = no GT depth')
    ap.add_argument('--poses', default='', help='pose file, relative to --src')
    ap.add_argument('--pose-cols', default='t,tx,ty,tz,qx,qy,qz,qw',
                    help='column order actually present in the pose file')
    ap.add_argument('--pose-invert', action='store_true',
                    help='the file stores world-to-camera; this repo wants camera-to-world')
    ap.add_argument('--calib', default='',
                    help='"fx fy cx cy k1 k2 p1 p2 [k3]" for the RAW images')
    ap.add_argument('--depth-in-scale', type=float, default=1.0,
                    help='divide the source depth by this to get METRES (1.0 = already metres)')
    ap.add_argument('--depth-png-scale', type=float, default=DEFAULT_DEPTH_SCALE,
                    help=f'output metres = px / this (default {DEFAULT_DEPTH_SCALE}; NOT TUM\'s '
                         f'6553.5, which caps at 10 m)')
    ap.add_argument('--max-assoc', type=float, default=0.05,
                    help='max |image - pose| timestamp gap, seconds')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    if not os.path.isdir(a.src):
        raise SystemExit(f'--src {a.src} is not a directory')
    if a.inspect or not a.dst:
        inspect(a.src)
        if not a.dst:
            print('\n(no --dst given, so nothing was written)')
        return

    imgs = sorted(p for p in glob.glob(f'{a.src}/{a.images}/*')
                  if os.path.splitext(p)[1].lower() in IMG_EXT)
    if not imgs:
        raise SystemExit(f'no images under {a.src}/{a.images} - run --inspect and set --images')
    if not a.calib:
        raise SystemExit('--calib is required: "fx fy cx cy k1 k2 p1 p2 [k3]" for the RAW images. '
                         'Run --inspect to find the calibration file.')
    c = [float(x) for x in a.calib.replace(',', ' ').split()]
    K = np.array([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1]], np.float64)
    dist = np.array(c[4:], np.float64) if len(c) > 4 else np.zeros(5)

    probe = cv2.imread(imgs[0], cv2.IMREAD_UNCHANGED)
    if probe is None:
        raise SystemExit(f'cv2 cannot read {imgs[0]}')
    hw = probe.shape[:2]
    (mx, my), newK = undistort_maps(K, dist, hw)

    # ---- association: images to poses by timestamp parsed from the file name ----
    keep = list(range(len(imgs)))
    poses = None
    if a.poses:
        ts, T = read_poses(f'{a.src}/{a.poses}', a.pose_cols, a.pose_invert)
        istamp = np.array([stamp_of(p) if stamp_of(p) is not None else np.nan for p in imgs])
        if np.isnan(istamp).any():
            raise SystemExit('image names are not numeric timestamps - cannot associate poses')
        j = np.abs(istamp[:, None] - ts[None, :]).argmin(1)
        gap = np.abs(istamp - ts[j])
        keep = [k for k in range(len(imgs)) if gap[k] <= a.max_assoc]
        print(f'pose association: {len(keep)} of {len(imgs)} images within {a.max_assoc}s '
              f'(median gap {np.median(gap):.4f}s, worst kept {gap[keep].max():.4f}s)')
        poses = T[j]
    if not keep:
        raise SystemExit('no image matched a pose inside --max-assoc')

    os.makedirs(f'{a.dst}/colors', exist_ok=True)
    dep_dir = f'{a.src}/{a.depths}' if a.depths else None
    if dep_dir and os.path.isdir(dep_dir):
        os.makedirs(f'{a.dst}/depths', exist_ok=True)
    else:
        dep_dir = None
        print('no depth directory - writing colors + traj only (DEPTHS: null in the run config)')

    dstats = []
    for n, k in enumerate(keep):
        img = cv2.imread(imgs[k], cv2.IMREAD_UNCHANGED)
        if img.dtype != np.uint8:                     # FLSea ships TIFF; normalise to 8-bit
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f'{a.dst}/colors/{n:06d}.png', cv2.remap(img, mx, my, cv2.INTER_LINEAR))
        if dep_dir:
            stem = os.path.splitext(os.path.basename(imgs[k]))[0]
            cand = [p for e in IMG_EXT for p in (f'{dep_dir}/{stem}{e}',) if os.path.exists(p)]
            if not cand:
                raise SystemExit(f'no depth map for {stem} under {dep_dir} - names must match the '
                                 f'image stems, or pass --depths "" to skip GT depth')
            d = cv2.imread(cand[0], cv2.IMREAD_UNCHANGED).astype(np.float64) / a.depth_in_scale
            d[~np.isfinite(d)] = 0.0
            d = cv2.remap(d.astype(np.float32), mx, my, cv2.INTER_NEAREST)
            dstats.append((float(d[d > 0].min()) if (d > 0).any() else 0.0,
                           float(d.max()), float((d > 0).mean())))
            over = d * a.depth_png_scale
            if over.max() > 65535:
                raise SystemExit(
                    f'{stem}: depth {d.max():.2f} m x {a.depth_png_scale} exceeds uint16. '
                    f'Lower --depth-png-scale to below {65535/max(d.max(),1e-9):.1f}.')
            cv2.imwrite(f'{a.dst}/depths/{n:06d}.png', over.astype(np.uint16))
        if n % 250 == 0:
            print(f'  {n}/{len(keep)}', flush=True)

    if poses is not None:
        from scipy.spatial.transform import Rotation
        with open(f'{a.dst}/traj_tum.txt', 'w') as f:
            for n, k in enumerate(keep):
                T = poses[k]
                q = Rotation.from_matrix(T[:3, :3]).as_quat()
                f.write(f'{n} {T[0,3]:.9f} {T[1,3]:.9f} {T[2,3]:.9f} '
                        f'{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n')

    with open(f'{a.dst}/calib.txt', 'w') as f:
        f.write(f'{newK[0,0]:.6f} {newK[1,1]:.6f} {newK[0,2]:.6f} {newK[1,2]:.6f}\n')

    lines = [f'source        {a.src}',
             f'frames        {len(keep)} of {len(imgs)} (renumbered 000000..{len(keep)-1:06d})',
             f'image size    {hw[1]}x{hw[0]}',
             f'raw calib     {a.calib}',
             f'undistorted   fx {newK[0,0]:.4f} fy {newK[1,1]:.4f} '
             f'cx {newK[0,2]:.4f} cy {newK[1,2]:.4f}  -> calib.txt',
             f'poses         {a.poses or "NONE"}  cols={a.pose_cols}  invert={a.pose_invert}']
    if dstats:
        d = np.array(dstats)
        lines += [f'depth range   {d[:,0].min():.3f}..{d[:,1].max():.3f} m, '
                  f'valid {100*d[:,2].mean():.1f}% of pixels',
                  f'depth scale   metres = px / {a.depth_png_scale}  '
                  f'-> set DEPTH_PNG_SCALE: {a.depth_png_scale} in the run config']
    lines += ['undistorted offline: do NOT set UNDISTORT/CROP_BORDER in the run config (10.1)']
    open(f'{a.dst}/preprocess_info.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
