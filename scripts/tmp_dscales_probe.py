"""TMP DIAGNOSTIC - the scale grid the tracker CHOSE vs the one the prior WANTED.

    python scripts/tmp_dscales_probe.py                      # every extract on disk
    python scripts/tmp_dscales_probe.py kitti_00_f0-1000/normal

WHY THIS EXISTS. §14 established that KITTI's ATE is monotone scale drift, and §14.2 established
that JDSA's scale-grid Jacobian is `-q * w_c` (geom/ba.py:215) - proportional to the PRIOR's own
disparity - so far pixels have almost no leverage on the stretch factor that is nevertheless
applied to them. Both halves are measured. What was never measured is the LINK: does the far
field actually disagree about the scale, and does that disagreement track the trajectory's drift?
This reads that off `dscales`, which every extract already saved, and adds no GPU work.

WHAT IT MEASURES, per keyframe, all in DISPARITY where JDSA works (q = prior, d = tracker):

  chosen        `dscales`, the 2x2 grid the frontend fitted, bilinearly interpolated
  s_all         the single scale that best fits d ~ s*q over ALL pixels (least squares, which
                weights each pixel by q^2 - the same weighting Hs carries)
  s_near        the same fitted on q >= median(q)      - the near field's opinion
  s_far         the same fitted on q <  median(q)/2    - the far field's opinion
  far leverage  the far pixels' share of sum q^2, i.e. how much of a vote they get
  far cost      |s_far/s_all - 1|, the relative depth error the far field eats by being
                overruled: it is served 1/(s_all*q) where it wanted 1/(s_far*q)

THREE THINGS TO KNOW BEFORE READING THE NUMBERS.

  * The saved `disps` are POST-GLOBAL-BA and the saved `dscales` are NOT. factor_graph.py:278
    runs global BA with use_mono=False, so it moves the depths twice without ever re-fitting the
    grid. `chosen vs s_all` therefore measures how far the frontend's alignment has been left
    behind by later depth updates - it is NOT evidence that JDSA failed to converge.
  * Every extract on disk ran the OMNIDATA prior (run_extract does not take a prior), so this
    compares regimes and tracking configs, never omni against VGGT. A VGGT column needs a VGGT
    extract run, which is GPU work this script deliberately does not do.
  * `s_far` is fitted on the pixels with the least information by construction, so it is the
    noisiest column here. Read its MEDIAN over keyframes, not any single frame.

NOT A STAGE. Nothing under adaslam/ imports this; numpy only, no torch, no cv2, no GPU.
"""
import os
import sys
import glob

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GT_TRAJ = 'data/KITTI/00/traj_tum.txt'
FAR_RATIO = 2.0        # "far" = depth beyond this x the frame median, i.e. q < median(q)/FAR_RATIO


def grid_weights(ht, wd):
    """The 2x2 bilinear weights, exactly geom/ba.py:get_prior_depth_aligned's meshgrid."""
    fy = np.linspace(0, 1 - 1e-6, ht)[:, None]
    fx = np.linspace(0, 1 - 1e-6, wd)[None, :]
    return np.stack([(1 - fx) * (1 - fy), fx * (1 - fy),
                     (1 - fx) * fy, fx * fy]).reshape(4, -1)      # TL TR BL BR


def fit_scale(d, q, sel):
    """argmin sum (d - s*q)^2 over `sel` - the q^2-weighted least squares JDSA's Hs carries."""
    if sel.sum() < 16:
        return np.nan
    qq = (q[sel] ** 2).sum()
    return float((q[sel] * d[sel]).sum() / qq) if qq > 1e-12 else np.nan


def cam_centres(poses):
    """lietorch SE3 world->cam [t | qx qy qz qw] -> camera centres in world coordinates."""
    t, quat = poses[:, :3], poses[:, 3:]
    x, y, z, w = quat.T
    R = np.empty((len(poses), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return -np.einsum('nji,nj->ni', R, t)                    # -R^T t


def blockwise_scale(tstamp, poses, gt, n_blocks=5):
    """GT metres per estimated unit, over blocks of keyframes - §12.4's drift measure."""
    c = cam_centres(poses)
    keep = [i for i, t in enumerate(tstamp) if int(round(t)) in gt]
    if len(keep) < n_blocks * 4:
        return None
    c, g = c[keep], np.array([gt[int(round(tstamp[i]))] for i in keep])
    out = []
    for b in np.array_split(np.arange(len(c)), n_blocks):
        de = np.linalg.norm(np.diff(c[b], axis=0), axis=1).sum()
        dg = np.linalg.norm(np.diff(g[b], axis=0), axis=1).sum()
        out.append(dg / de if de > 1e-9 else np.nan)
    return out


def probe(path, gt):
    z = np.load(path)
    disps, prior, dsc, tstamp, poses = (z['disps'], z['disps_prior'], z['dscales'],
                                        z['tstamp'], z['poses'])
    K, ht, wd = disps.shape
    w4 = grid_weights(ht, wd)

    rows = {k: [] for k in ('chosen', 's_all', 's_near', 's_far', 'lev', 'cost')}
    degen = 0
    for i in range(K):
        d = disps[i].reshape(-1).astype(np.float64)
        q = prior[i].reshape(-1).astype(np.float64)
        ok = q > 0
        if ok.sum() < 64:
            continue
        med = np.median(q[ok])
        near = ok & (q >= med)
        far = ok & (q < med / FAR_RATIO)

        s_all = fit_scale(d, q, ok)
        s_near = fit_scale(d, q, near)
        s_far = fit_scale(d, q, far)
        rows['chosen'].append(float((w4 * dsc[i].reshape(4, 1)).sum(0).mean()))
        rows['s_all'].append(s_all)
        rows['s_near'].append(s_near)
        rows['s_far'].append(s_far)
        rows['lev'].append(float((q[far] ** 2).sum() / (q[ok] ** 2).sum()))
        rows['cost'].append(abs(s_far / s_all - 1.0) if s_all and np.isfinite(s_far) else np.nan)
        degen += int(dsc[i].min() <= 0)

    # the same three questions with the far-field CEILING applied to the prior (§14): clamping
    # depth at R x median floors q, which is the only thing that can buy the far field a vote
    # The same questions with the far-field CEILING applied (§14). NOTE the metric has to change:
    # clamping RAISES q, and s_far = sum(q*d)/sum(q^2) then falls by exactly the clamp factor, so
    # `s_far/s_near` is not comparable across clamps - it moves for arithmetic reasons alone. The
    # clamp-invariant question is what the far field is SERVED versus what the tracker believes:
    # served disparity is s_all*q, the tracker's is d, so |d/(s_all*q) - 1| is a relative depth
    # disagreement in one fixed unit however q was produced.
    ceil = {}
    for R in (None, 2.0, 1.5):
        acc = {'lev': [], 'err': []}
        for i in range(K):
            d = disps[i].reshape(-1).astype(np.float64)
            q0 = prior[i].reshape(-1).astype(np.float64)
            ok = q0 > 0
            if ok.sum() < 64:
                continue
            med0 = np.median(q0[ok])
            far = ok & (q0 < med0 / FAR_RATIO)                 # the SAME pixels in every row
            q = q0.copy()
            if R is not None:
                q[ok] = np.maximum(q0[ok], med0 / R)           # depth <= R*median <=> q >= qmed/R
            s_all = fit_scale(d, q, ok)
            if not s_all or not np.isfinite(s_all):
                continue
            served = s_all * q[far]
            acc['lev'].append(float((q[far] ** 2).sum() / (q[ok] ** 2).sum()))
            acc['err'].append(float(np.median(np.abs(d[far] / np.maximum(served, 1e-9) - 1.0))))
        ceil[R] = {k: float(np.nanmedian(v)) for k, v in acc.items()}

    med = {k: float(np.nanmedian(v)) for k, v in rows.items()}
    name = os.path.relpath(os.path.dirname(os.path.dirname(path)), 'outputs/extract')
    print(f'\n=== {name}   {K} keyframes ===')
    print(f'  chosen grid (mean of the 4 corners), median over kf : {med["chosen"]:.4f}')
    print(f'  least-squares s on the SAME disps, median           : {med["s_all"]:.4f}'
          f'   (ratio chosen/s_all {med["chosen"]/med["s_all"]:.3f} - see caveat 1)')
    print(f'  keyframes with a NEGATIVE corner                    : {degen} '
          f'({degen/max(K,1):.1%})')
    print(f'\n  who wants what scale (median over keyframes)')
    print(f'    near field (q >= median)   s_near = {med["s_near"]:.4f}')
    print(f'    far  field (q <  median/2) s_far  = {med["s_far"]:.4f}'
          f'    s_far/s_near = {med["s_far"]/med["s_near"]:.3f}')
    print(f'    far field\'s share of the vote (sum q^2) : {med["lev"]:.2%}')
    print(f'    relative depth error the far field eats : {med["cost"]:.1%}')

    print(f'\n  with the far-field CEILING on the prior (14) - same pixels in every row')
    print(f'    {"served":>12}{"far vote":>11}{"served vs tracker depth":>26}')
    for R, c in ceil.items():
        print(f'    {("unclamped" if R is None else "ceil " + format(R, "g")):>12}'
              f'{c["lev"]:>10.2%}{c["err"]:>25.1%}')

    # does the disagreement drift with the sequence?
    n = len(rows['s_all'])
    third = max(n // 3, 1)
    def trend(k):
        a, b = np.nanmedian(rows[k][:third]), np.nanmedian(rows[k][-third:])
        return a, b, (b / a if a else np.nan)
    print(f'\n  first third -> last third')
    for k, lab in (('chosen', 'chosen grid'), ('s_all', 's_all     '), ('s_far', 's_far     ')):
        a, b, r = trend(k)
        print(f'    {lab}  {a:8.4f} -> {b:8.4f}   x{r:.3f}')
    bs = blockwise_scale(tstamp, poses, gt)
    if bs:
        print('    trajectory scale (GT m per est unit, 5 blocks): '
              + ' -> '.join(f'{x:.2f}' for x in bs) + f'   x{bs[-1]/bs[0]:.3f}')
        print('    ^ if these two move together, the prior alignment and the drift are one story')


def main():
    os.chdir(_ROOT)
    gt = {}
    if os.path.exists(GT_TRAJ):
        for r in np.loadtxt(GT_TRAJ):
            gt[int(round(r[0]))] = r[1:4]
    want = sys.argv[1:]
    paths = sorted(glob.glob('outputs/extract/*/*/full/slam_depth.npz'))
    if want:
        paths = [p for p in paths if any(w in p for w in want)]
    if not paths:
        raise SystemExit('no slam_depth.npz found under outputs/extract/')
    print(__doc__.split('WHAT IT MEASURES')[0].strip())
    for p in paths:
        if 'kitti' not in p:
            continue                      # the GT trajectory above is KITTI 00's
        probe(p, gt)
    print('\nreading it')
    print('  * s_far/s_near far from 1.0 = the two regions want DIFFERENT scales, one 2x2 grid')
    print('    cannot serve both, and the vote share says which one it serves. The cost row is')
    print('    what the far field pays for losing, as a fraction of its own depth.')
    print('  * the ceiling block: the vote goes UP but served-vs-tracker disagreement goes UP too.')
    print('    So the clamp does NOT work by making the far field more agreeable - it is a BOUND')
    print('    on what the prior may assert (14.1), and it wins while being more wrong.')
    print('  * the fitted scale drifting in step with the trajectory scale, on the long runs but')
    print('    not the short one, is the drift link. The prior is stateless and does not drift;')
    print('    JDSA re-fits 4 DoF per KEYFRAME, so tracker drift is absorbed, never resisted.')


if __name__ == '__main__':
    main()
