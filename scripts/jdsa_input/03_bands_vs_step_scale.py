"""Where does the trajectory's scale gauge live?  Per keyframe: local translation scale error vs
GT, against the SLAM depth error measured in separate RANGE BANDS.  Nothing here reads any
previous result - GT poses from data/KITTI/00/traj_tum.txt, GT depth from the lidar dumps."""
import sys, numpy as np
sys.path.insert(0, sys.path[0])
from geo import c2w_from_w2c, umeyama_sim3, load_gt_tum

BANDS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80)]

for path in sys.argv[1:]:
    z = np.load(path)
    ts, dp, db, gt = z['tstamp'], z['dp'], z['db'], z['gt']
    K = len(ts)
    gidx, gp = load_gt_tum('data/KITTI/00/traj_tum.txt')
    Xs = np.array([c2w_from_w2c(p)[1] for p in z['poses']])       # SLAM camera centres
    Xg = gp[ts][:, :3]                                            # GT camera centres
    s, R, t = umeyama_sim3(Xs, Xg)
    ate = np.linalg.norm((s * (R @ Xs.T).T + t) - Xg, axis=1)
    print(f'\n=== {path.split("/")[-1][:-4]}  K={K}  Sim3 scale slam->metric {s:.4f}  '
          f'ATE rmse {np.sqrt((ate**2).mean()):.3f} m')

    # local translation scale, keyframe k -> k+1
    dS = np.linalg.norm(np.diff(Xs, axis=0), axis=1) * s
    dG = np.linalg.norm(np.diff(Xg, axis=0), axis=1)
    ok = dG > 0.3
    r = np.full(K-1, np.nan); r[ok] = dS[ok]/dG[ok]
    print(f'  local KF-step scale |dt_slam|/|dt_gt| : mean {np.nanmean(r):.4f} '
          f'median {np.nanmedian(r):.4f}  std {np.nanstd(r):.4f}  '
          f'(steps used {ok.sum()}/{K-1}, mean step {dG[ok].mean():.2f} m)')

    # SLAM depth error per range band, per keyframe (one global scale s, so drift shows)
    dep = s / np.clip(db, 1e-9, None)
    rho = np.full((K, len(BANDS)), np.nan)
    npx = np.zeros((K, len(BANDS)))
    for k in range(K):
        g = gt[k].ravel(); d = dep[k].ravel()
        v = (g > 0) & np.isfinite(d) & (d > 0)
        for i, (lo, hi) in enumerate(BANDS):
            m = v & (g >= lo) & (g < hi)
            npx[k, i] = m.sum()
            if m.sum() > 30:
                rho[k, i] = np.median(d[m]/g[m])
    print('  SLAM depth / GT depth, median per band (1.0 = right); px per frame')
    for i, (lo, hi) in enumerate(BANDS):
        print(f'    {lo:>3}-{hi:<3} m : ratio {np.nanmedian(rho[:,i]):.4f} '
              f'(iqr {np.nanpercentile(rho[:,i],75)-np.nanpercentile(rho[:,i],25):.4f})'
              f'   px {npx[:,i].mean():6.0f}  frames {np.isfinite(rho[:,i]).sum()}')
    # which band's depth error tracks the local trajectory scale error?
    print('  corr(local step scale r_k , band depth ratio)  over keyframes')
    for i, (lo, hi) in enumerate(BANDS):
        a = rho[:-1, i]; m = np.isfinite(a) & np.isfinite(r)
        if m.sum() > 30:
            c = np.corrcoef(a[m], r[m])[0, 1]
            # slope of r on rho: 1.0 would mean the band's depth error IS the step scale error
            sl = np.polyfit(a[m], r[m], 1)[0]
            print(f'    {lo:>3}-{hi:<3} m : corr {c:+.3f}  slope {sl:+.3f}  n {m.sum()}')
