"""The gauge itself: the fused map's metric scale, frame by frame, measured on the SAME 3D points
(GT pose + GT depth link consecutive keyframes, so the SLAM geometry never enters)."""
import sys, numpy as np
sys.path.insert(0, sys.path[0])
from geo import quat2R, c2w_from_w2c, umeyama_sim3, load_gt_tum

gidx, gp = load_gt_tum('data/KITTI/00/traj_tum.txt')

def gauge_series(z):
    dp, db, gt, ts = z['dp'], z['db'], z['gt'], z['tstamp']
    fx, fy, cx, cy = z['intrinsics']; K, ht, wd = dp.shape
    jj, ii = np.meshgrid(np.arange(wd), np.arange(ht))
    xn = (8*jj+3-cx)/fx; yn = (8*ii+3-cy)/fy
    track = np.where(db > 0, 1.0/np.maximum(db, 1e-9), 0.0)
    prior = np.where(dp > 0, 1.0/np.maximum(dp, 1e-9), 0.0)
    gsh = np.full((K-1, 2), np.nan)
    for a in range(K-1):
        b = a+1
        Ra, ta = quat2R(gp[ts[a]][3:]), gp[ts[a]][:3]
        Rb, tb = quat2R(gp[ts[b]][3:]), gp[ts[b]][:3]
        gb = gt[b]; v = gb > 0
        Xb = np.stack([xn*gb, yn*gb, gb], -1)[v]
        Xa = ((Xb @ Rb.T + tb) - ta) @ Ra
        Za = Xa[:, 2]; ok = Za > 1.0
        ja = np.round((fx*Xa[ok,0]/Za[ok]+cx-3)/8).astype(int)
        ia = np.round((fy*Xa[ok,1]/Za[ok]+cy-3)/8).astype(int)
        inb = (ja >= 0) & (ja < wd) & (ia >= 0) & (ia < ht)
        idx_b = np.flatnonzero(v)[ok][inb]; ja, ia, Za = ja[inb], ia[inb], Za[ok][inb]
        ga = gt[a][ia, ja]
        good = (ga > 0) & (np.abs(ga-Za) < 0.05*Za)
        idx_b, ia, ja, ga = idx_b[good], ia[good], ja[good], ga[good]
        if len(ga) < 100: continue
        rb = gt[b].ravel()[idx_b]
        for c, q in enumerate((track, prior)):
            qb = q[b].ravel()[idx_b]; qa = q[a][ia, ja]
            m = (qb > 0) & (qa > 0)
            if m.sum() > 100:
                gsh[a, c] = np.median(np.log(qb[m]/rb[m]) - np.log(qa[m]/ga[m]))
    return gsh

for path in sys.argv[1:]:
    z = np.load(path)
    ts = z['tstamp']
    Xs = np.array([c2w_from_w2c(p)[1] for p in z['poses']]); Xg = gp[ts][:, :3]
    s, R, t = umeyama_sim3(Xs, Xg)
    err = np.linalg.norm((s*(R@Xs.T).T+t)-Xg, axis=1)
    g = gauge_series(z)
    tr = g[:, 0][np.isfinite(g[:, 0])]
    # cumulative gauge: the map's scale relative to frame 0, in log units
    cum = np.nancumsum(np.nan_to_num(g[:, 0]))
    ac = np.corrcoef(tr[:-1], tr[1:])[0, 1]
    print(f'\n=== {path.split("/")[-1][:-4]}  ATE {np.sqrt((err**2).mean()):.2f} m '
          f'(max {err.max():.1f} m)')
    print(f'  per-step gauge change of the FUSED map, on shared points:  std {tr.std():.4f} '
          f'mean {tr.mean():+.5f}  lag-1 autocorr {ac:+.3f}')
    print(f'  cumulative gauge over the run: min {cum.min():+.3f} max {cum.max():+.3f} '
          f'end {cum[-1]:+.3f}  (log units; e^x)')
    q = np.array_split(tr, 4)
    print('  per-step gauge std by quarter of the run: ' + '  '.join(f'{x.std():.4f}' for x in q))
    print('  ATE by quarter (m): ' + '  '.join(f'{x.mean():.2f}' for x in np.array_split(err, 4)))
