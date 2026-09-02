"""One keyframe's depth as a colour overlay on its own image - what the numbers cannot show.

    python scripts/viz_depth.py -e kitti_00_f0-1000/normal                 # list keyframes, exit
    python scripts/viz_depth.py -e kitti_00_f0-1000/normal --frame 300
    python scripts/viz_depth.py -e kitti_00_fg2a05_f0-1000/normal \
                                -e kitti_00_fg2a05_f0-1000/normal_ceil1p5 \
                                --frame 300 --source prior -o outputs/plots/ceil_vs_none.png

`-e` is an extract experiment: `<scene>/<name>`, or a path. Several stack into one figure on a
SHARED colour scale, which is the whole point when comparing a clamped prior against its parent.

WHY THE SCALE IS `depth / frame median` AND NOT METRES. Two reasons, both hard. Monocular SLAM
depth is in ARBITRARY units and every run picks its own, so two extracts have no common absolute
scale and a shared metre axis would be a lie. And raw depth_slam runs 0.75 to exactly 1000.0 - the
`disps.clamp(min=0.001)` floor in geom/ba.py showing through as 1/0.001 - against a median near
2.2, so a linear raw ramp puts the entire visible scene in the bottom 0.2% of the colours. Dividing
by the frame's own median fixes both and is the unit ARCHITECTURE 14 already states ceilings in, so
a `@ceil1p5` arm is readable directly off the picture: nothing may exceed 1.5.

WHY magma AND NOT jet/turbo. A rainbow ramp for magnitude is an anti-pattern - its lightness is not
monotonic, so it invents edges the data does not have and collapses in greyscale or under colour
blindness. magma is perceptually uniform, monotonic in lightness, and its dark end recedes against
a photograph while its bright end carries the near field. It is a multi-hue ramp, which is allowed
for "semantic heat" ONLY WITH A SCALE LEGEND - hence the colorbar is not optional here.

TWO SOURCES, and they answer different questions:

    --source slam    depth_slam/%06d.npy - 1/disps_up, what the TRACKER concluded (default)
    --source prior   disps_prior from full/slam_depth.npz - what the PRIOR asserted, at 1/8 res
                     upsampled for display. This is where a ceiling is applied, so it is the
                     direct view of the clamp; `slam` shows what BA made of it afterwards.

`-m` / `--mask` OUTLINES the confidence mask - the pixels extract keeps as supervision, i.e.
`droid_backends.depth_filter & depth > 0`, the multi-view agreement test. It is drawn as a CYAN
CONTOUR rather than a tint or a fill, for three reasons. Colour is already spent on depth, and
magma contains no cyan at any value, so the two encodings cannot be confused. The outline is
ADDITIVE - it marks the region without hiding the depth inside it, so one render answers both
"what is the depth" and "which of it is trusted". And the mask's shape suits it: it is computed at
1/8 resolution and nearest-upsampled, so it is a blocky grid of large clean-edged regions (on one
sampled keyframe: 69 components, ZERO single pixels, 77% of the area in the top ten blobs), which
outlines trace crisply where they would turn speckle into noise.

Coverage is printed per panel, and it is low - 5-7% typically, 11% on the frame sampled above.
That is the number export.txt reports as "% of pixels kept", and seeing WHERE the survivors sit is
the point: they cluster on near structure and the road, and thin out in exactly the far field 14
is about.

NOT A STAGE. Nothing under adaslam/ imports this. Read-only, no GPU, no torch.
"""
import os
import sys
import argparse
import glob

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adaslam.common import DEPTH_DIR, MASK_DIR, extract_run_dir

CMAP = 'magma'
OUT_DIR = 'outputs/plots'


def resolve(spec):
    """'<scene>/<name>' or a path -> the experiment directory, checked."""
    cand = spec if os.path.isabs(spec) or spec.startswith('outputs/') else \
        os.path.join('outputs', 'extract', spec)
    if not os.path.isdir(cand):
        raise SystemExit(f'no such extract: {cand}')
    return cand.rstrip('/')


def keyframes(exp):
    """The frame numbers this extract exported, ascending."""
    return sorted(int(os.path.basename(p)[:-4])
                  for p in glob.glob(f'{exp}/{DEPTH_DIR}/*.npy'))


def load_slam(exp, frame):
    d = np.load(f'{exp}/{DEPTH_DIR}/{frame:06d}.npy').astype(np.float64)
    return d


def load_prior(exp, frame, hw):
    """disps_prior for `frame`, inverted to depth and upsampled from 1/8 to `hw`.

    The npz stores one row per keyframe in tracking order, so the frame number has to be looked
    up through `tstamp` rather than used as an index - keyframe 300 is not row 300.
    """
    npz = f'{extract_run_dir(exp)}/slam_depth.npz'
    if not os.path.exists(npz):
        raise SystemExit(f'--source prior needs {npz}, which this extract does not have '
                         f'(full/ may have been deleted to reclaim space - 7.1)')
    z = np.load(npz)
    rows = np.where(np.rint(z['tstamp']).astype(int) == frame)[0]
    if not len(rows):
        raise SystemExit(f'frame {frame} is not a keyframe of {exp}')
    q = z['disps_prior'][rows[0]].astype(np.float64)
    depth = 1.0 / np.clip(q, 1e-6, None)
    # INTER_NEAREST: the prior reaching BA really is one sample per 8x8 block (9.3), and smoothing
    # it here would draw detail the solver never saw
    return cv2.resize(depth.astype(np.float32), (hw[1], hw[0]),
                      interpolation=cv2.INTER_NEAREST).astype(np.float64)


def panel_data(exp, frame, source, use_mask):
    """(base RGB uint8, depth/median float, mask or None, stats) for one extract at one frame.

    `stats` carries the raw extremes as well as the normalised ones, because the two answer
    different questions. The RATIO extremes compare across panels (each run's monocular scale is
    its own, so only ratios are shared); the RAW ones are the only place a clamp is visible as a
    number - a slam-source max of exactly 1000.0 is geom/ba.py's `disps.clamp(min=0.001)` floor,
    not a measurement, and a prior-source max landing on 1.50x the median is a @ceil1p5 arm
    working.
    """
    img = cv2.imread(f'{exp}/image/{frame:06d}.jpg')
    if img is None:
        raise SystemExit(f'no image for frame {frame} in {exp}/image/')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hw = img.shape[:2]

    depth = load_slam(exp, frame) if source == 'slam' else load_prior(exp, frame, hw)
    if depth.shape != hw:
        depth = cv2.resize(depth.astype(np.float32), (hw[1], hw[0]),
                           interpolation=cv2.INTER_NEAREST).astype(np.float64)

    finite = np.isfinite(depth) & (depth > 0)
    med = float(np.median(depth[finite])) if finite.any() else 1.0
    ratio = np.where(finite, depth / max(med, 1e-9), np.nan)

    v = depth[finite]
    stats = {'median': med,
             'raw_min': float(v.min()) if v.size else float('nan'),
             'raw_max': float(v.max()) if v.size else float('nan'),
             'ratio_min': float(v.min() / max(med, 1e-9)) if v.size else float('nan'),
             'ratio_max': float(v.max() / max(med, 1e-9)) if v.size else float('nan')}

    mask = None
    if use_mask:
        m = cv2.imread(f'{exp}/{MASK_DIR}/{frame:06d}.png', cv2.IMREAD_UNCHANGED)
        if m is None:
            print(f'  WARNING: no {MASK_DIR}/{frame:06d}.png in {exp} - drawing no outline')
        else:
            mask = m > 0
            stats['mask_frac'] = float(mask.mean())
    return img, ratio, mask, stats


def draw(ax, img, ratio, mask, vmax, alpha):
    """Greyscale photo + magma overlay. Grey base so the ramp is not fighting the scene's colour."""
    # Grey, and COMPRESSED into a mid band. A full-range base is mostly white on a sunlit street,
    # and blending a colour ramp over near-white returns pastel - the depth stops reading. Keeping
    # the photo in [0.10, 0.70] preserves every edge and lets the ramp carry the signal.
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    base = np.repeat((0.10 + 0.60 * grey)[:, :, None], 3, axis=2)

    norm = plt.Normalize(vmin=0.0, vmax=vmax)
    rgba = matplotlib.colormaps[CMAP](norm(np.nan_to_num(ratio, nan=0.0)))[:, :, :3]

    a = np.full(ratio.shape, alpha)
    a[~np.isfinite(ratio)] = 0.0                 # no depth -> show the photo untouched
    out = np.clip(base * (1 - a[:, :, None]) + rgba * a[:, :, None], 0, 1)

    if mask is not None:
        # Outline, not a tint: the depth underneath stays fully readable, and cyan cannot collide
        # with magma (which spans black-purple-orange-cream and never reaches cyan) or with the
        # greyscale base. Drawn LAST so it sits above the overlay.
        u8 = (out * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(u8, cnts, -1, (0, 245, 245), 1)      # RGB - the array is RGB, not BGR
        out = u8.astype(np.float64) / 255.0

    ax.imshow(out)
    ax.set_xticks([]); ax.set_yticks([])
    return norm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-e', '--extract', action='append', required=True,
                    help='<scene>/<name> or a path; repeat to stack panels on one shared scale')
    ap.add_argument('--frame', type=int, help='keyframe by its own frame number')
    ap.add_argument('--nth', type=int, help='keyframe by position in the exported list')
    ap.add_argument('--source', choices=('slam', 'prior'), default='slam')
    ap.add_argument('-m', '--mask', action='store_true',
                    help='outline the confidence mask (what extract keeps as supervision)')
    ap.add_argument('--vmax', default=3.0, type=lambda s: s if s == 'p99' else float(s),
                    help='top of the scale in units of the frame median (default 3.0); '
                         '"p99" to fit it to the data instead')
    ap.add_argument('--alpha', type=float, default=0.65)
    ap.add_argument('-o', '--out', default=None)
    args = ap.parse_args()

    os.chdir(_ROOT)
    exps = [resolve(e) for e in args.extract]
    kfs = keyframes(exps[0])
    if not kfs:
        raise SystemExit(f'{exps[0]}/{DEPTH_DIR}/ is empty - was the export interrupted?')

    if args.frame is None and args.nth is None:
        print(f'{len(kfs)} keyframes in {exps[0]}: {kfs[0]}..{kfs[-1]}')
        print('  ' + ' '.join(str(k) for k in kfs[:40]) + (' ...' if len(kfs) > 40 else ''))
        print('\npick one with --frame N (the numbers above) or --nth N (position in the list)')
        return
    frame = kfs[max(0, min(args.nth, len(kfs) - 1))] if args.nth is not None else args.frame
    if frame not in kfs:
        near = min(kfs, key=lambda k: abs(k - frame))
        raise SystemExit(f'frame {frame} is not a keyframe of {exps[0]}; nearest is {near}')

    panels = [panel_data(e, frame, args.source, args.mask) for e in exps]

    # A FIXED default, not a percentile, and the reason is the data: the sky carries near-zero
    # disparity, so depth there runs into the hundreds of medians (p99 reads 456x on a slam panel)
    # and any data-fitted top collapses the whole visible scene into the first 1% of the ramp.
    # 3.0 is the range 14's ceiling sweep actually explored, it keeps every render comparable
    # without thinking, and what exceeds it saturates - which IS the far-field story, so the
    # colorbar carries an arrow and each title prints the fraction over the top.
    if isinstance(args.vmax, str):        # --vmax p99
        allv = np.concatenate([r[np.isfinite(r)].ravel() for _, r, _, _ in panels])
        vmax = float(np.percentile(allv, 99))
        print(f'scale: 0 .. {vmax:.2f} x frame median (p99 over every panel)')
    else:
        vmax = float(args.vmax)
        print(f'scale: 0 .. {vmax:.2f} x frame median')

    h, w = panels[0][0].shape[:2]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 11 * h / w * len(panels) + 0.9),
                             squeeze=False, constrained_layout=True)
    norm = None
    for ax, exp, (img, ratio, mask, st) in zip(axes[:, 0], exps, panels):
        norm = draw(ax, img, ratio, mask, vmax, args.alpha)
        over = float(np.nanmean(ratio > vmax)) if np.isfinite(ratio).any() else 0.0
        ax.set_title(
            f'{os.path.relpath(exp, "outputs/extract")}   frame {frame}   {args.source} depth\n'
            f'min {st["ratio_min"]:.2f}x   max {st["ratio_max"]:.2f}x   '
            f'(raw {st["raw_min"]:.4g} .. {st["raw_max"]:.4g}, median {st["median"]:.4g})   '
            f'{over:.1%} above the scale top'
            + (f'   |  cyan = confidence mask, {st["mask_frac"]:.1%} of pixels'
               if 'mask_frac' in st else ''),
            fontsize=9, loc='left')
        print(f'  {os.path.relpath(exp, "outputs/extract")}: '
              f'min {st["raw_min"]:.4g} max {st["raw_max"]:.4g} median {st["median"]:.4g}  '
              f'-> {st["ratio_min"]:.2f}x .. {st["ratio_max"]:.2f}x'
              + (f'  mask {st["mask_frac"]:.1%}' if 'mask_frac' in st else ''))

    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP),
                      ax=axes[:, 0].tolist(), orientation='horizontal',
                      fraction=0.05, pad=0.01, extend='max')
    cb.set_label('depth ÷ the frame\'s own median depth  (unit-free: monocular scale is arbitrary)')

    out = args.out or f'{OUT_DIR}/depth_{args.source}_f{frame}.png'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
