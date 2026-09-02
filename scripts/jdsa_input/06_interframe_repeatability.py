"""Inter-frame REPEATABILITY of a depth source on the SAME 3D points.

Consecutive keyframes are linked by GT pose + GT depth, so every measurement below is independent
of the SLAM run's own geometry.  For a source q (prior, or the tracker's fused depth) define
L = log(q / gt).  A free per-frame scale is an additive constant in L and is exactly what JDSA's
dscales grid absorbs, so it is removed pair-wise; what is left is the part a trajectory can drift
on: how the SAME point's log-error changes as it approaches the camera.
"""
import sys, numpy as np
sys.path.insert(0, sys.path[0])
from geo import quat2R, load_gt_tum

def bilin2x2(g, ht, wd):
    y = np.linspace(0, 1-1e-6, ht)[:, None]; x = np.linspace(0, 1-1e-6, wd)[None, :]
    return (1-y)*(1-x)*g[0,0] + (1-y)*x*g[0,1] + y*(1-x)*g[1,0] + y*x*g[1,1]

gidx, gp = load_gt_tum('data/KITTI/00/traj_tum.txt')
RB = [(0, 10), (10, 20), (20, 40), (40, 80)]

for path in sys.argv[1:]:
    z = np.load(path)
    dp, db, gt, ds, ts = z['dp'], z['db'], z['gt'], z['dscales'], z['tstamp']
    fx, fy, cx, cy = z['intrinsics']
    K, ht, wd = dp.shape
    prior_d = np.where(dp > 0, 1.0/np.maximum(dp, 1e-9), 0.0)
    track_d = np.where(db > 0, 1.0/np.maximum(db, 1e-9), 0.0)
    jj, ii = np.meshgrid(np.arange(wd), np.arange(ht))
    xn = (8*jj + 3 - cx)/fx; yn = (8*ii + 3 - cy)/fy

    out = {'prior': [], 'track': []}
    slope = {'prior': [], 'track': []}
    for a in range(K-1):
        b = a+1
        Ra, ta = quat2R(gp[ts[a]][3:]), gp[ts[a]][:3]
        Rb, tb = quat2R(gp[ts[b]][3:]), gp[ts[b]][:3]
        gb = gt[b]
        v = gb > 0
        Xb = np.stack([xn*gb, yn*gb, gb], -1)[v]                 # cam b
        Xw = Xb @ Rb.T + tb
        Xa = (Xw - ta) @ Ra
        Za = Xa[:, 2]
        ok = Za > 1.0
        u = (fx*Xa[ok, 0]/Za[ok] + cx - 3)/8.0
        w_ = (fy*Xa[ok, 1]/Za[ok] + cy - 3)/8.0
        ja = np.round(u).astype(int); ia = np.round(w_).astype(int)
        inb = (ja >= 0) & (ja < wd) & (ia >= 0) & (ia < ht)
        idx_b = np.flatnonzero(v)[ok][inb]
        ja, ia, Za = ja[inb], ia[inb], Za[ok][inb]
        ga = gt[a][ia, ja]
        # occlusion / lidar-fill guard: the GT depth at the landing pixel must agree with the
        # depth the GT motion predicts for this point
        good = (ga > 0) & (np.abs(ga - Za) < 0.05*Za)
        idx_b, ia, ja, ga, Za = idx_b[good], ia[good], ja[good], ga[good], Za[good]
        if len(ga) < 100:
            continue
        rb = gt[b].ravel()[idx_b]
        for nm, q in (('prior', prior_d), ('track', track_d)):
            qb = q[b].ravel()[idx_b]; qa = q[a][ia, ja]
            m = (qb > 0) & (qa > 0)
            if m.sum() < 100:
                continue
            L = np.log(qb[m]/rb[m]) - np.log(qa[m]/ga[m])     # same point, b vs a
            L = L - np.median(L)                              # the free per-frame scale
            out[nm].append((np.percentile(L, 75)-np.percentile(L, 25), len(L)))
            # systematic part: how the residual depends on the point's range in frame a
            s = []
            for lo, hi in RB:
                mm = (ga[m] >= lo) & (ga[m] < hi)
                s.append(np.median(L[mm]) if mm.sum() > 30 else np.nan)
            slope[nm].append(s)
    nm_ = path.split('/')[-1][:-4]
    print(f'\n=== {nm_}   pairs used {len(out["prior"])}')
    for nm in ('prior', 'track'):
        iqr = np.array([o[0] for o in out[nm]]); n = np.array([o[1] for o in out[nm]])
        S = np.array(slope[nm])
        print(f'  {nm:6} point-wise log-ratio change, IQR: median {np.median(iqr):.4f} '
              f'p90 {np.percentile(iqr,90):.4f}   (mean {n.mean():.0f} points/pair)')
        print(f'  {nm:6} systematic median shift by range in frame a: ' +
              '  '.join(f'{lo}-{hi}m {np.nanmedian(S[:,i]):+.4f}' for i, (lo, hi) in enumerate(RB)))
