"""Where in SPACE an end2end arm's ATE lives - the estimated paths from above, one PNG.

    python scripts/plot_trajectories.py -s rellis_00000 -o omni_vs_lora omni normal_r8_e20_p10
    python scripts/plot_trajectories.py -s rellis_00000 -o survey omni base normal_r8_e20_p10 \
                                        wonline_r8_e5_w20_p10 --plane xy
    python scripts/plot_trajectories.py -s rellis_00000 -o drift --align start omni base

compare() reduces an arm to one ATE and ate_over_time.py spreads that over the sequence; both
are a column of numbers, and a column of 2847 residuals cannot show that a run cut a corner,
over-rotated at a turn, or wandered while the rig was parked. This draws the path itself, seen
from above, every pose coloured by its distance to the GT pose at the same frame.

By default nothing is recomputed and no GPU is involved - evo already saved both halves of it
(12.3): `evo/error_array.npy` is the per-pose APE and `evo/alignment_transformation_sim3.npy` is
the Sim(3) that puts the estimate in GT coordinates. Applying the second reproduces the first to
~6e-14, so the geometry you see and the colour on it come from one transform, and that is the
same transform results.json's ate_all was measured under. Read-only over outputs/.

HOW TO READ IT. The colour is APE - absolute position error against GT, in metres - after ONE
Sim(3) Umeyama alignment fitted over the WHOLE trajectory. Four consequences:

  * green is not "correct". The alignment is a least-squares compromise over every pose, so
    even the best pose of the best arm on rellis_00000 sits ~1.9 m from GT. The ramp spans the
    arms you asked for; it is a comparison, not a grade.
  * EACH ARM HAS ITS OWN ALIGNMENT. A better-shaped trajectory earns a different fit and its
    whole path shifts. Two arms drawn here are each individually best-fitted to GT, which is
    what makes their SHAPES comparable and their absolute offsets not.
  * one colour scale spans the whole image, so a panel that never goes green is an arm that is
    worse everywhere - that comparison is the point of sharing it.
  * a stretch that reddens while the path barely moves is the rig standing still and the
    estimate drifting, not a manoeuvre gone wrong.

WHEN NOTHING OVERLAPS, READ THE DRIFT COLUMN FIRST. On rellis_00000 no arm's path lies on GT
anywhere, not even at frame 0 (omni's first pose is 37.6 m out), and that is a RESULT rather
than a bad fit: the estimate's scale is not constant along the sequence. Measured as GT metres
per estimate unit over the first and last tenth of the poses, omni runs 5.45 -> 37.71, a factor
of 6.9; the same measurement on the TUM scene reads 1.04. A Sim(3) has ONE scale, so on a
trajectory whose scale drifts there is no transform that overlays it - evo picks the
least-squares compromise (11.71) and every pose is wrong by the difference. The `drift` column
prints that factor per arm and the summary says so out loud above ~1.5x. It is the
cross-frame scale inconsistency this track targets (9.7), seen in the trajectory instead of in
a depth L1 column, and it is IN THE KEYFRAME POSES (traj_kf.txt: 5.80 -> 47.52), so it is the
tracker's, not the filler's.

--align is what that column is for. `evo` (the default) draws the saved transform, so the
picture stays the one ate_all was measured under. `start` fits a fresh Sim(3) on the first N
poses through evo's OWN aligner - evo_traj --align --correct_scale --n_to_align N, called as a
library - so the paths leave the same point together and the drift reads as divergence rather
than as a global offset. That view RECOMPUTES the colour: the APE under a different alignment is
a different number, and every label on the figure says which alignment it belongs to.

The plane is chosen by dropping the GT axis with the least spread, which is the vertical one
for a ground vehicle (rellis_00000: x 154 m, y 202 m, z 1.5 m -> x/y). --plane overrides it.

Green-to-red is the one pair red-green colourblind readers cannot separate. It is the default
because the ramp means good-to-bad and the colourbar states the scale; --cmap viridis is the
same picture in a ramp anyone can read.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)          # repo root, so `adaslam` imports

from adaslam.common import test_dir                                      # noqa: E402
from adaslam.end2end.metrics import (RESULTS, arm_dir, gt_traj_of,       # noqa: E402
                                     load_alignment, load_ape)

PLOT_SUBDIR = 'plots'              # outputs/plots/<name>.png, unless -o names a path itself
OVERLAY_MAX = 3                    # more arms than this and one panel is spaghetti - facet

# WHICH Sim(3) puts an arm in GT coordinates. 'evo' is the one run_ate fitted and results.json was
# measured under; 'start' is fitted here, on the first N poses only.
ALIGN_EVO, ALIGN_START = 'evo', 'start'
N_FRACTION = 0.10                  # --align start's default N, as a share of the DRAWN poses
N_MIN = 10                         # ...and its floor; 3 is the geometric minimum, 10 is a fit
DRIFT_NOTE = 1.5                   # scale drift worth interrupting the summary to explain

# The two plotted axes, by name. 'auto' drops the GT axis with the smallest spread.
PLANES = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
AXIS_NAME = 'xyz'

# Green -> amber -> red. A multi-hue sequential ramp is only legitimate as SEMANTIC HEAT and
# only with a scale legend, which the colourbar is. Stops are picked dark enough to read as a
# 2px line on white; a raw green->red interpolation would pass through muddy low-chroma greys.
RAMP = ['#1a7f37', '#69a223', '#d0a215', '#e0662b', '#a4161a']

TEXT_INK, MUTED_INK, GRID_INK, GT_INK, SURFACE = \
    '#1f2328', '#6e7781', '#e4e7eb', '#9aa0a6', '#ffffff'

# Arm identity is carried by MARKER SHAPE, never by colour - colour is the error and nothing
# else. Not by dash pattern: a path is one LineCollection of ~2850 segments each a few
# centimetres long, far shorter than any dash period, so a linestyle on it renders as solid.
# Waypoints survive that, and they double as a reading of speed - bunched where the rig slowed.
MARKERS = ['o', 's', 'D', '^']
WAYPOINTS = 9


def rmse(x):
    return float(np.sqrt((np.asarray(x, float) ** 2).mean())) if len(x) else None


def load_gt(path):
    """{frame: xyz} for the reference trajectory - TUM rows, cols 1:4 are the camera position."""
    if not os.path.exists(path):
        raise SystemExit(f'GT trajectory {path} not found. It is the reference evo recorded when '
                         f'the arm was scored; if the dataset has moved, pass --gt')
    gt = np.loadtxt(path)
    return dict(zip(gt[:, 0].astype(int).tolist(), gt[:, 1:4]))


def keyframe_frames(out):
    """The frames in an arm's traj_kf.txt - the poses the tracker actually optimised."""
    path = f'{out}/traj_kf.txt'
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found, so --keyframes has no frame list to restrict to')
    return set(np.loadtxt(path)[:, 0].astype(int).tolist())


def evo_sim3(est, ref, n=-1):
    """(4x4 with the scale baked in, the scale) mapping `est` onto `ref` - EVO'S OWN aligner.

    Exactly what `evo_traj --align --correct_scale --n_to_align N` runs, called as a library:
    PosePath3D.align is Umeyama over the POSITIONS and lie_algebra.sim3 packs the result into the
    same 4x4 layout run_ate saved. Hand-rolling Umeyama here would be a second definition of the
    one thing this whole file is about, and at n=-1 this returns
    evo/alignment_transformation_sim3.npy to the last bit - which is the check that says so.

    Orientations are required by PosePath3D and unread by the fit, so identity quaternions are
    the honest filler: nothing here reads a rotation, and passing the estimate's would imply the
    alignment used them. n greater than the pose count is the whole trajectory, as it is in evo.
    """
    from evo.core import lie_algebra
    from evo.core.trajectory import PosePath3D

    def path(xyz):                              # align() MUTATES its object, so build both here
        return PosePath3D(positions_xyz=xyz,
                          orientations_quat_wxyz=np.tile([1.0, 0, 0, 0], (len(xyz), 1)))

    r, t, s = path(est).align(path(ref), correct_scale=True, n=n)
    return lie_algebra.sim3(r, t, s), float(s)


def scale_drift(est, ref, n):
    """(scale over the first n poses, over the last n, the factor between them).

    The one number that says whether ANY Sim(3) could have overlaid this arm. A trajectory whose
    local scale is constant fits one global scale; one whose scale drifts does not, and the
    least-squares compromise is then wrong at both ends of the sequence rather than at neither.
    """
    n = max(min(n, len(est) // 2), 3)           # halves, so the two windows cannot overlap
    _, s_first = evo_sim3(est[:n], ref[:n])
    _, s_last = evo_sim3(est[-n:], ref[-n:])
    return s_first, s_last, max(s_last / s_first, s_first / s_last)


def load_arm(root, scene, arm, gt_xyz, keyframes, mode, n_align):
    """One arm's path in GT coordinates, plus the per-pose APE that colours it.

    Rows are indexed by FRAME rather than by position: every arm on disk happens to have one
    traj_full.txt row per error_array entry, but evo's association is what decides that, and a
    silent off-by-one here would draw a plausible picture of the wrong poses.

    Under `mode` ALIGN_START the transform is refitted on the first n poses and the APE recomputed
    under it; `ate_all` is still carried, because the console prints both and the distance between
    them is worth seeing.
    """
    out = arm_dir(root, scene, arm)
    err, ts, _ = load_ape(out)
    frames = ts.astype(int)

    est = np.loadtxt(f'{out}/traj_full.txt')
    row_of = {f: i for i, f in enumerate(est[:, 0].astype(int).tolist())}
    missing = [f for f in frames.tolist() if f not in row_of]
    if missing:
        raise SystemExit(f'{out}: {len(missing)} scored frames are absent from traj_full.txt '
                         f'(first {missing[:5]}) - evo/ and the trajectory are out of step, so '
                         f're-run the end2end stage for this arm')
    absent = [f for f in frames.tolist() if f not in gt_xyz]
    if absent:
        raise SystemExit(f'{out}: {len(absent)} scored frames are absent from the GT trajectory '
                         f'(first {absent[:5]}) - this arm was scored against a different GT')

    raw = est[[row_of[f] for f in frames.tolist()], 1:4]      # untransformed, in tracker units
    ref = np.array([gt_xyz[f] for f in frames.tolist()])
    T_evo = load_alignment(out)
    xyz_evo = raw @ T_evo[:3, :3].T + T_evo[:3, 3]

    # The whole picture rests on this identity, so check it rather than trust it. It runs in BOTH
    # modes: what it catches is an evo/ that no longer describes the traj_full.txt beside it, and
    # `frames` comes out of that same evo/ whichever alignment ends up being drawn.
    stale = float(np.abs(np.linalg.norm(xyz_evo - ref, axis=1) - err).max())
    if stale > 1e-6:
        raise SystemExit(f'{out}: applying evo\'s own Sim(3) to traj_full.txt disagrees with '
                         f'evo/error_array.npy by {stale:.3g} m. One of them is stale - delete '
                         f'the arm\'s evo/ and re-score it')

    if keyframes:
        keep = np.array([f in keyframe_frames(out) for f in frames.tolist()])
        if not keep.any():
            raise SystemExit(f'{out}: none of its keyframes are among the scored frames, so '
                             f'--keyframes leaves nothing to draw')
        raw, ref, xyz_evo, err, frames = raw[keep], ref[keep], xyz_evo[keep], err[keep], \
            frames[keep]

    # n is a share of the poses actually DRAWN, so --keyframes fits on a tenth of the keyframes
    # rather than on a tenth of a frame count that is no longer on screen
    n = n_align if n_align else max(int(len(raw) * N_FRACTION), N_MIN)
    if mode == ALIGN_EVO:
        xyz, scale = xyz_evo, float(np.linalg.det(T_evo[:3, :3]) ** (1 / 3))
    else:
        T, scale = evo_sim3(raw, ref, n)
        xyz = raw @ T[:3, :3].T + T[:3, 3]
        err = np.linalg.norm(xyz - ref, axis=1)

    s_first, s_last, drift = scale_drift(raw, ref, n)
    res = json.load(open(f'{out}/{RESULTS}')) if os.path.exists(f'{out}/{RESULTS}') else {}
    # `shown` is the RMSE over the poses actually DRAWN, which under --keyframes or --align start
    # is not ate_all - the non-keyframe poses come from the trajectory filler's motion-only BA,
    # not the tracker's own optimisation, and a refitted alignment is a different measurement
    # entirely; labelling either picture with ate_all would misreport it
    return {'arm': arm, 'out': out, 'xyz': xyz, 'err': err, 'frames': frames, 'n_align': n,
            'scale': scale, 's_first': s_first, 's_last': s_last, 'drift': drift,
            'ate_all': res.get('ate_all', rmse(err)), 'shown': rmse(err)}


def pick_plane(gt_pts, requested):
    """(key, (i, j), why) - the two axes to draw. 'auto' drops the least-varying GT axis."""
    if requested != 'auto':
        return requested, PLANES[requested], 'requested'
    spread = np.ptp(gt_pts, axis=0)
    up = int(np.argmin(spread))
    key = ''.join(AXIS_NAME[k] for k in range(3) if k != up)
    why = (f'GT spread {AXIS_NAME[0]} {spread[0]:.1f} m  {AXIS_NAME[1]} {spread[1]:.1f} m  '
           f'{AXIS_NAME[2]} {spread[2]:.1f} m, so {AXIS_NAME[up]} is the vertical axis')
    return key, PLANES[key], why


def ramp(name):
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    if name:
        return plt.get_cmap(name)
    return LinearSegmentedColormap.from_list('ate_gr', RAMP)


def draw_gt(ax, pts, ij, label):
    """The reference path, under everything, as a hairline - context, not a series."""
    ax.plot(pts[:, ij[0]], pts[:, ij[1]], color=GT_INK, lw=1.1, zorder=1,
            solid_capstyle='round', label=label)


def draw_arm(ax, a, ij, cmap, norm, marker, waypoints):
    """One arm's path as per-segment colour, with waypoints carrying which arm it is."""
    from matplotlib.collections import LineCollection
    xy = a['xyz'][:, list(ij)]
    if len(xy) > 1:
        seg = np.stack([xy[:-1], xy[1:]], axis=1)
        lc = LineCollection(seg, cmap=cmap, norm=norm, linewidth=2.0, capstyle='round', zorder=3)
        lc.set_array(0.5 * (a['err'][:-1] + a['err'][1:]))
        ax.add_collection(lc)

    at = np.linspace(0, len(xy) - 1, max(waypoints, 2)).round().astype(int)
    # surface-filled with an ink ring, so a waypoint stays legible where paths overlap; the last
    # one is filled solid, which is how the direction of travel is read
    ax.plot(xy[at, 0], xy[at, 1], linestyle='none', marker=marker, ms=6.5, mfc=SURFACE,
            mec=TEXT_INK, mew=1.3, zorder=4)
    ax.plot(*xy[-1], marker=marker, ms=6.5, mfc=TEXT_INK, mec=TEXT_INK, mew=1.3, zorder=5)
    return xy


def limits(pts_list, ij, margin=0.04):
    """The x/y range every panel shares - GT and every arm, so no panel crops a path."""
    allpts = np.concatenate(pts_list)
    out = []
    for k in ij:
        lo, hi = float(allpts[:, k].min()), float(allpts[:, k].max())
        pad = max((hi - lo) * margin, 1e-3)
        out.append((lo - pad, hi + pad))
    return out


def style_axes(ax, plane, adjustable, xlabel=True, ylabel=True):
    # 'datalim' expands the shorter axis instead of shrinking the box, which is what keeps a
    # single panel filling the figure; shared axes forbid it, so facets use 'box'.
    ax.set_aspect('equal', adjustable=adjustable)
    ax.grid(True, color=GRID_INK, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID_INK)
    ax.tick_params(colors=MUTED_INK, labelsize=8, length=3)
    # facets share the axes, so only the outer edge is labelled - repeating it in every panel
    # is chrome competing with the paths
    if xlabel:
        ax.set_xlabel(f'{plane[0]} (m)', color=MUTED_INK, fontsize=9)
    if ylabel:
        ax.set_ylabel(f'{plane[1]} (m)', color=MUTED_INK, fontsize=9)


def figure_size(pts_list, ij, ncols, nrows, panel_w):
    """Panels are equal-aspect, so the figure has to carry the data's aspect or it letterboxes."""
    allpts = np.concatenate(pts_list)
    dx = max(np.ptp(allpts[:, ij[0]]), 1e-6)
    dy = max(np.ptp(allpts[:, ij[1]]), 1e-6)
    # clamped: equal aspect with adjustable='datalim' pads the shorter axis rather than
    # letterboxing, so a 200x1 m sequence costs empty margin instead of a 3000px-tall PNG
    panel_h = min(max(panel_w * dy / dx, panel_w * 0.5), panel_w * 1.5)
    return ncols * panel_w + 1.9, nrows * panel_h + 1.7


def score_label(a, keyframes, mode):
    """The number under an arm's name on the figure - named for what it actually measures.

    ate_all is only the drawn number when the drawn poses and the drawn alignment are both the
    ones it was measured with. Under --keyframes it is not, and under a refitted alignment it is
    not; in either case the label says which measurement is on screen. 4 significant digits
    rather than 2 decimals: a room-scale scene runs at 0.06 m and two arms would print alike.
    """
    if mode != ALIGN_EVO:
        return f'{mode}{a["n_align"]} RMSE {a["shown"]:.4g} m'
    return f'kf RMSE {a["shown"]:.4g} m' if keyframes else f'ATE {a["ate_all"]:.4g} m'


def align_caption(arms, mode):
    """How the alignment is named on the colourbar and in the title, in that order."""
    if mode == ALIGN_EVO:
        return "after each arm's own Sim(3) alignment", 'evo Sim(3)'
    n = sorted({a['n_align'] for a in arms})
    which = n[0] if len(n) == 1 else f'{n[0]}-{n[-1]}'    # --keyframes: arms differ, so say so
    return (f"after a Sim(3) refitted on each arm's\nfirst {which} poses",
            f'Sim(3) on the first {which}')


def render(arms, gt_pts, ij, plane, cmap, norm, scene, path, dpi, keyframes, mode):
    import matplotlib
    matplotlib.use('Agg')                       # the venv backend is headless; savefig only
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    facet = len(arms) > OVERLAY_MAX
    ncols = min(4, math.ceil(math.sqrt(len(arms)))) if facet else 1
    nrows = math.ceil(len(arms) / ncols) if facet else 1
    fw, fh = figure_size([gt_pts] + [a['xyz'] for a in arms], ij, ncols, nrows,
                         3.5 if facet else 7.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False,
                             sharex=facet, sharey=facet, layout='constrained')
    fig.patch.set_facecolor(SURFACE)
    flat = axes.ravel()

    handles = [Line2D([], [], color=GT_INK, lw=1.1, label='ground truth')]
    cb_label, title_align = align_caption(arms, mode)

    def score(a):
        return score_label(a, keyframes, mode)
    if facet:
        for k, (ax, a) in enumerate(zip(flat, arms)):
            draw_gt(ax, gt_pts, ij, None)
            draw_arm(ax, a, ij, cmap, norm, MARKERS[0], 2)   # one panel is one arm; shape is free
            ax.set_title(f'{a["arm"]}\n{score(a)}', fontsize=9, color=TEXT_INK, pad=6)
            bottom = k + ncols >= len(arms)      # nothing below it, ragged last row included
            style_axes(ax, plane, 'box', xlabel=bottom, ylabel=k % ncols == 0)
            # sharex hides tick labels by GRID position, so a ragged row's last panel would
            # carry an axis label over unlabelled ticks
            ax.tick_params(labelbottom=bottom)
        for ax in flat[len(arms):]:
            ax.set_visible(False)
        # each panel draws ONE arm, so autoscale would give each its own frame; the comparison
        # needs every panel on the same range, over GT and all arms
        xlim, ylim = limits([gt_pts] + [a['xyz'] for a in arms], ij)
        flat[0].set_xlim(*xlim)          # shared, so one panel sets them all
        flat[0].set_ylim(*ylim)
        handles.append(Line2D([], [], linestyle='none', marker='o', ms=6.5, mfc=SURFACE,
                              mec=TEXT_INK, mew=1.3, label='start (filled marker = end)'))
    else:
        ax = flat[0]
        draw_gt(ax, gt_pts, ij, 'ground truth')
        for a, mk in zip(arms, MARKERS):
            draw_arm(ax, a, ij, cmap, norm, mk, WAYPOINTS)
            # an ink proxy: the legend must never imply that an arm owns a colour
            handles.append(Line2D([], [], linestyle='none', marker=mk, ms=6.5, mfc=SURFACE,
                                  mec=TEXT_INK, mew=1.3,
                                  label=f'{a["arm"]}   {score(a)}'))
        # no explicit limits here: 'datalim' expands an axis to satisfy the aspect and warns if
        # it has to override one that was set, and autoscale over the artists is already the
        # union of GT and every arm
        ax.margins(0.04)
        style_axes(ax, plane, 'datalim')

    fig.legend(handles=handles, loc='outside lower center', ncols=min(len(handles), 4),
               frameon=False, fontsize=8, labelcolor=TEXT_INK)

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=flat.tolist(),
                      fraction=0.04, shrink=0.6, pad=0.02)
    # the caveat rides on the thing it qualifies, where it cannot collide with the paths
    cb.set_label(f'APE to GT (m)\n{cb_label}', color=MUTED_INK, fontsize=9)
    cb.ax.tick_params(colors=MUTED_INK, labelsize=8, length=3)
    cb.outline.set_visible(False)

    # under --keyframes the arms genuinely disagree about which frames are keyframes, so list
    # every count rather than print one that is true of only the first
    counts = [len(a['err']) for a in arms]
    same = len(set(counts)) == 1
    poses = f'{counts[0]}{" each" if len(arms) > 1 else ""}' if same else ' / '.join(map(str,
                                                                                        counts))
    fig.suptitle(f'{scene} — top view ({plane[0]}/{plane[1]}), '
                 f'{len(arms)} arm{"s" if len(arms) > 1 else ""}, '
                 f'{"keyframes" if keyframes else "poses"}: {poses}, aligned by {title_align}',
                 fontsize=12, color=TEXT_INK)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def out_path(root, name):
    """`-o` is a NAME under outputs/plots/ unless it already looks like a path."""
    if '/' in name or name.endswith('.png'):
        return name if name.endswith('.png') else f'{name}.png'
    return f'{root}/{PLOT_SUBDIR}/{name}.png'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arms', nargs='+',
                    help='end2end arm directory names; the FIRST supplies the GT reference')
    ap.add_argument('-s', '--scene', required=True,
                    help='the scene; arm names are unique only within one')
    ap.add_argument('-o', '--out', required=True, metavar='NAME',
                    help=f'written to <root>/{PLOT_SUBDIR}/NAME.png, or verbatim if NAME is a '
                         f'path or ends in .png')
    ap.add_argument('--root', default='outputs', help='the outputs tree (default: outputs)')
    ap.add_argument('--plane', default='auto', choices=['auto'] + list(PLANES),
                    help='the two axes to draw (default: auto, dropping the flattest GT axis)')
    ap.add_argument('--gt', metavar='PATH',
                    help="the reference trajectory; default is the one evo recorded in the "
                         "first arm's evo/info.json")
    ap.add_argument('--keyframes', action='store_true',
                    help="only the frames in each arm's own traj_kf.txt")
    ap.add_argument('--align', default=ALIGN_EVO, choices=[ALIGN_EVO, ALIGN_START],
                    help=f'which Sim(3) puts an arm in GT coordinates: {ALIGN_EVO} (default) is '
                         f'the one evo fitted over the whole trajectory and results.json was '
                         f'measured under; {ALIGN_START} refits it on the first -n poses, so the '
                         f'paths start together and scale drift reads as divergence')
    ap.add_argument('-n', '--n-to-align', type=int, metavar='N',
                    # %% because argparse %-expands help strings, and a lone % raises there
                    help=f'--align {ALIGN_START} only: fit on the first N drawn poses '
                         f'(default: {round(N_FRACTION * 100)}%% of them, at least {N_MIN})')
    ap.add_argument('--vmin', type=float, help='low end of the shared colour scale, in metres')
    ap.add_argument('--vmax', type=float, help='high end of the shared colour scale, in metres')
    ap.add_argument('--cmap', metavar='NAME',
                    help='any matplotlib colormap instead of the green-to-red ramp, e.g. viridis')
    ap.add_argument('--dpi', type=int, default=200)
    args = ap.parse_args()

    # -n silently ignored under the default alignment would be a picture that is not the one that
    # was asked for, and nothing on it would say so
    if args.n_to_align is not None:
        if args.align == ALIGN_EVO:
            raise SystemExit(f'-n {args.n_to_align} sets how many poses a REFITTED alignment is '
                             f'fitted on, but --align is {ALIGN_EVO!r}, which draws the transform '
                             f'evo already saved. Pass --align {ALIGN_START} to use it.')
        if args.n_to_align < 3:
            raise SystemExit(f'-n {args.n_to_align} is below the 3 non-collinear poses a Sim(3) '
                             f'needs; {N_MIN} or more is a fit rather than an interpolation')

    os.chdir(_ROOT)                              # --root is repo-root relative, however invoked
    if not os.path.isdir(test_dir(args.root, 'end2end', args.scene)):
        scenes = sorted(os.listdir(f'{args.root}/test/end2end')) \
            if os.path.isdir(f'{args.root}/test/end2end') else []
        raise SystemExit(f'no end2end results for scene {args.scene!r} under {args.root}/; '
                         f'available: {scenes or "(none)"}')

    # One reference for the whole image. Arms scored against different GT are in different
    # frames, and overlaying them would draw a comparison that does not exist.
    if args.gt:
        gt_path = args.gt
    else:
        refs = {a: gt_traj_of(arm_dir(args.root, args.scene, a)) for a in args.arms}
        if len(set(refs.values())) > 1:
            raise SystemExit('these arms were scored against different GT trajectories, so they '
                             'are not comparable in one frame:\n' +
                             '\n'.join(f'  {a:<32} {p}' for a, p in refs.items()) +
                             '\n  Pass --gt to force one.')
        gt_path = refs[args.arms[0]]
    gt_xyz = load_gt(gt_path)

    arms = [load_arm(args.root, args.scene, a, gt_xyz, args.keyframes, args.align, args.n_to_align)
            for a in args.arms]
    gt_pts = np.array([gt_xyz[f] for f in sorted(gt_xyz)])

    plane, ij, why = pick_plane(gt_pts, args.plane)
    allerr = np.concatenate([a['err'] for a in arms])
    vmin = args.vmin if args.vmin is not None else float(allerr.min())
    vmax = args.vmax if args.vmax is not None else float(allerr.max())
    if not vmax > vmin:
        raise SystemExit(f'--vmax {vmax} must exceed --vmin {vmin}')

    from matplotlib.colors import Normalize
    path = out_path(args.root, args.out)
    render(arms, gt_pts, ij, plane, ramp(args.cmap), Normalize(vmin, vmax),
           args.scene, path, args.dpi, args.keyframes, args.align)

    print(f'\n  {args.scene}  —  top view {plane[0]}/{plane[1]} ({why})')
    print(f'  GT {gt_path}  ({len(gt_pts)} poses)')
    drawn = 'keyframes' if args.keyframes else 'poses'
    print(f'  {"arm":<32}{drawn:>8}{"ATE(m)":>10}{"drawn(m)":>10}{"APE min":>9}{"max":>9}'
          f'{"scale":>10}{"drift":>8}')
    print('  ' + '-' * 96)
    for a in arms:
        print(f'  {a["arm"]:<32}{len(a["err"]):>8}{a["ate_all"]:>10.4f}{a["shown"]:>10.4f}'
              f'{a["err"].min():>9.3f}{a["err"].max():>9.3f}{a["scale"]:>10.3f}'
              f'{a["drift"]:>7.2f}x')
    print(f'\n  colour scale {vmin:.3f} .. {vmax:.3f} m, shared by every panel'
          f'{" (--vmin/--vmax)" if args.vmin is not None or args.vmax is not None else ""}')
    if args.align == ALIGN_EVO:
        print('  each arm carries its OWN Sim(3) alignment, so compare SHAPES, not offsets')
    else:
        n = sorted({a['n_align'] for a in arms})
        print(f'  ALIGNMENT REFITTED on the first {n[0] if len(n) == 1 else f"{n[0]}-{n[-1]}"} '
              f'poses, so drawn(m) is NOT ate_all - the ATE column is the evo alignment, which '
              f'this picture is not')

    # the number that says whether one Sim(3) could ever have overlaid these paths - printed as a
    # column above, explained here when it is the reason the picture looks wrong
    drifted = [a for a in arms if a['drift'] > DRIFT_NOTE]
    if drifted:
        print(f'\n  NOTE: the estimate\'s SCALE DRIFTS along the sequence, in GT metres per '
              f'tracker unit\n        over the first vs the last {drifted[0]["n_align"]} poses:')
        for a in drifted:
            print(f'          {a["arm"]:<32}{a["s_first"]:>9.2f} -> {a["s_last"]:<9.2f}'
                  f'{a["drift"]:>6.1f}x')
        print('        A Sim(3) has ONE scale, so no alignment can overlay a path whose scale '
              'moves:\n        the offset you see is that drift, not a bad fit.')
        if args.align == ALIGN_EVO:
            print(f'        --align {ALIGN_START} fits the first poses only, showing where they '
                  f'diverge instead.')
    print(f'  wrote {path}\n')


if __name__ == '__main__':
    main()
