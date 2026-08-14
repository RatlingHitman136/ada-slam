"""Where in the sequence an end2end arm's ATE lives - the per-pose series, as a vertical table.

    python scripts/ate_over_time.py -s rellis_00000 omni
    python scripts/ate_over_time.py -s rellis_00000 omni normal_r8_e20_p10 --bins 40

compare() reduces an arm to ate_all / ate_seen / ate_unseen, which says WHICH arm is better and
nothing about WHERE. This prints what evo already saved per pose, so nothing is recomputed and no
GPU is involved. Read-only over outputs/.

HOW TO READ IT. The number is APE - absolute position error against GT, in metres, after ONE
Sim(3) Umeyama alignment fitted over the WHOLE trajectory (run_ate passes evo -vas). Three
consequences, all visible in the real numbers:

  * it does NOT start at zero (omni frame 0 reads 37.6 m). The alignment is a least-squares
    compromise over every pose, not a pinning of the first one.
  * it is NOT monotonic (omni: 35 -> 2.5 at frame 504 -> 56 at frame 2742). A dip is where the
    estimated path happens to cross the globally-fitted GT path, not a recovery.
  * the absolute level is NOT attributable to a frame. A better-SHAPED trajectory gets a different
    alignment and its whole curve shifts, so read shape and relative change - never "arm B was
    6 m better at frame 0".

The columns:

  dist(m)   GT path length so far. Error rising while it rises = drift with motion; error rising
            while it is FLAT = the rig has stopped and the estimate is wandering.
  APE(m)    the per-pose residual above.
  cumRMSE   the running ATE over the rows so far - how the headline number builds. Its last value
            equals results.json:ate_all exactly, which is a free check on the whole table.
  delta     multi-arm only: this arm minus the first, '+' when it beats it.

Reading a comparison: SIGN CONSISTENCY across the sequence is the signal - an arm that wins
everywhere is a real win, one that wins in a single stretch and loses elsewhere is one good
segment. A WIDENING delta means the baseline drifts faster; a constant offset usually means the
two differ through the global alignment rather than through accumulated drift. Below ~1e-3 is
noise: ARCHITECTURE.md 11.2 measured the tracker as non-reproducible to ~1.7e-3 between two
identical runs.

--bins N is the better first look - each row is a window's RMSE rather than one pose's residual,
so a single outlier frame cannot be mistaken for a trend.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)          # repo root, so `adaslam` imports

from adaslam.common import test_dir                                      # noqa: E402
from adaslam.end2end.metrics import RESULTS, arm_dir, load_ape           # noqa: E402
from adaslam.print_utils import delta_header, delta_row                  # noqa: E402

BAR = '█'                     # the profile column; width scales to the largest APE shown


def rmse(x):
    return float(np.sqrt((np.asarray(x, float) ** 2).mean())) if len(x) else None


def load_arm(root, scene, arm):
    """One arm's per-frame APE, keyed by frame index, plus what results.json says about it."""
    out = arm_dir(root, scene, arm)
    err, ts, dist = load_ape(out)
    res = json.load(open(f'{out}/{RESULTS}')) if os.path.exists(f'{out}/{RESULTS}') else {}
    frames = ts.astype(int)
    return {'arm': arm, 'out': out, 'err': err, 'frames': frames, 'dist': dist,
            'by_frame': dict(zip(frames.tolist(), err.tolist())),
            'label': res.get('label', arm), 'ate_all': res.get('ate_all', rmse(err))}


def keyframe_rows(out):
    """The frames in an arm's traj_kf.txt - the poses the tracker actually optimised."""
    path = f'{out}/traj_kf.txt'
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found, so --keyframes has no frame list to index by')
    return np.loadtxt(path)[:, 0].astype(int)


def build_rows(arms, keyframes, bins):
    """[(label, member frames, dist)] - one entry per printed row, in one of the three modes.

    A row is a SET OF FRAMES, one in the per-frame and keyframe modes and a window in --bins, so
    every mode goes through one value computation downstream: the RMSE over the row's members.

    Rows are always indexed by arms[0]: under --keyframes the arms genuinely disagree about which
    frames are keyframes, and sampling every arm at the SAME frames is the only row-aligned
    comparison there is. Every arm has a value at every frame, so nothing is missing.
    """
    a0 = arms[0]
    dist_of = dict(zip(a0['frames'].tolist(), a0['dist'].tolist()))

    if bins:
        n = len(a0['frames'])
        edges = np.linspace(0, n, min(bins, n) + 1).round().astype(int)
        windows = [a0['frames'][lo:hi].tolist()
                   for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]
        return [(f'{w[0]:06d}-{w[-1]:06d}', w, dist_of.get(w[-1])) for w in windows]

    frames = keyframe_rows(a0['out']) if keyframes else a0['frames']
    return [(f'{f:06d}', [f], dist_of.get(f)) for f in frames.tolist()]


def row_value(arm, members):
    """One arm's number for one row: the RMSE over the row's frames (the frame itself, if one)."""
    vals = [arm['by_frame'][f] for f in members if f in arm['by_frame']]
    return rmse(vals)


def print_header(arms, rows, keyframes, bins):
    """What the numbers are, before any of them - the table is easy to misread without it."""
    a0 = arms[0]
    info = f'{a0["out"]}/evo/info.json'
    title = json.load(open(info))['title'].replace('\n', ' ') if os.path.exists(info) else 'APE'
    print(f'\n  {title}')
    print(f'  {len(a0["frames"])} poses, one per FRAME of traj_full.txt; '
          f'{a0["dist"][-1]:.1f} m of GT path')
    for a in arms:
        print(f'    {a["arm"]:<28} ATE {a["ate_all"]:>9.4f} m   {a["label"]}')

    if bins:
        print(f'\n  {len(rows)} windows of ~{len(a0["frames"]) // max(len(rows), 1)} frames; '
              f'each row is the RMSE WITHIN its window')
    elif keyframes:
        who = '; every arm is sampled at those same frames' if len(arms) > 1 else ''
        print(f'\n  rows are {a0["arm"]}\'s {len(rows)} KEYFRAMES{who}.')
        if len(arms) > 1:
            print('  The arms do not agree on which frames are keyframes - each is its own SLAM '
                  'run - so this')
            print('  is the only row-aligned comparison there is. Their own counts:')
            for a in arms:
                print(f'    {a["arm"]:<28} {len(keyframe_rows(a["out"]))} keyframes')
    else:
        print(f'\n  {len(rows)} rows, one per frame')

    print('\n  APE is a residual after ONE global Sim(3) fit, so it neither starts at zero nor')
    print('  rises monotonically, and its LEVEL is not attributable to a frame. Read shape and')
    print('  relative change. See this file\'s docstring.\n')


def print_single(arm, rows, name):
    """One arm: value, running RMSE, and a bar so a long scroll stays scannable.

    cumRMSE accumulates the row MEMBERS, not the row values - with unequal bin sizes an
    equal-weighted RMSE of RMSEs is not the RMSE, and the last row would miss ate_all.
    """
    values = [row_value(arm, m) for _, m, _ in rows]
    hi = max((v for v in values if v is not None), default=1.0) or 1.0
    print(f"  {name:<15}{'dist(m)':>9}{'APE(m)':>10}{'cumRMSE':>10}   profile")
    print('  ' + '-' * 62)
    seen = []
    for (label, members, dist), v in zip(rows, values):
        if v is None:
            print(f'  {label:<15}{"n/a":>9}{"n/a":>10}')
            continue
        seen += [arm['by_frame'][f] for f in members if f in arm['by_frame']]
        d = f'{dist:>9.1f}' if dist is not None else f'{"":>9}'
        print(f'  {label:<15}{d}{v:>10.3f}{rmse(seen):>10.3f}   {BAR * int(round(24 * v / hi))}')


def print_multi(arms, rows, name, name_width=23):
    """Two or more arms: print_utils' delta rows, indexed by frame instead of by metric."""
    labels = [a['arm'] for a in arms]
    # delta_header pads each label to `width`, so a longer arm name would collide with the delta
    # column beside it - the same derivation end2end/report.py:compare uses
    width = max(13, max(len(lbl) for lbl in labels) + 2)
    delta_header(labels, width=width, name_width=name_width,
                 name=f"{name:<14}{'dist(m)':>9}")
    for (label, members, dist) in rows:
        d = f'{dist:>9.1f}' if dist is not None else f'{"":>9}'
        delta_row(f'{label:<14}{d}', [row_value(a, members) for a in arms], True,
                  width=width, name_width=name_width)


def write_csv(path, arms, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row', 'n_frames', 'dist_m'] + [a['arm'] for a in arms])
        for (label, members, dist) in rows:
            vals = [row_value(a, members) for a in arms]
            w.writerow([label, len(members), '' if dist is None else round(dist, 4)]
                       + ['' if v is None else round(v, 6) for v in vals])
    print(f'\n  wrote {path}  ({len(rows)} rows x {len(arms)} arms)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arms', nargs='+',
                    help='end2end arm directory names; the FIRST is the baseline and supplies '
                         'the rows')
    ap.add_argument('-s', '--scene', required=True,
                    help='the scene; arm names are unique only within one')
    ap.add_argument('--root', default='outputs', help='the outputs tree (default: outputs)')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--keyframes', action='store_true',
                   help="only the frames in the first arm's traj_kf.txt")
    g.add_argument('--bins', type=int, metavar='N',
                   help='N equal frame windows instead, each row the RMSE within it')
    ap.add_argument('--csv', metavar='PATH', help='also write the printed rows as CSV')
    args = ap.parse_args()

    os.chdir(_ROOT)                              # --root is repo-root relative, however invoked
    if not os.path.isdir(test_dir(args.root, 'end2end', args.scene)):
        scenes = sorted(os.listdir(f'{args.root}/test/end2end')) \
            if os.path.isdir(f'{args.root}/test/end2end') else []
        raise SystemExit(f'no end2end results for scene {args.scene!r} under {args.root}/; '
                         f'available: {scenes or "(none)"}')
    if args.bins is not None and args.bins < 1:
        raise SystemExit(f'--bins {args.bins} must be >= 1')

    arms = [load_arm(args.root, args.scene, a) for a in args.arms]
    rows = build_rows(arms, args.keyframes, args.bins)
    print_header(arms, rows, args.keyframes, args.bins)

    name = 'frames' if args.bins else 'frame'
    if len(arms) == 1:
        print_single(arms[0], rows, name)
    else:
        print_multi(arms, rows, name)

    print()
    shown_frames = [f for _, members, _ in rows for f in members]
    for a in arms:
        line = f'  {a["arm"]:<28} RMSE over all {len(a["err"])} frames  {rmse(a["err"]):>9.4f}'
        if args.keyframes:
            # the two differ, and the difference is itself a result: the non-keyframe poses come
            # from PoseTrajectoryFiller's motion-only BA, not from the tracker's own optimisation
            v = row_value(a, shown_frames)
            line += f'   over the {len(shown_frames)} keyframes shown  {v:>9.4f}'
        print(line)
    if args.csv:
        write_csv(args.csv, arms, rows)


if __name__ == '__main__':
    main()
