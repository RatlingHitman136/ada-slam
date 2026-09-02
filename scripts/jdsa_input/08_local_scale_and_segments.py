"""Is the ATE gap a LOCAL SCALE gap?  Local Sim(3) scale in a sliding window of keyframes, the
KITTI-style relative segment error, and the local scale against the map's own depth ratio."""
import sys, numpy as np
sys.path.insert(0, sys.path[0])
from geo import c2w_from_w2c, umeyama_sim3, load_gt_tum

gidx, gp = load_gt_tum('data/KITTI/00/traj_tum.txt')
W = 5     # +-W keyframes

for path in sys.argv[1:]:
    z = np.load(path)
    ts, db, gt = z['tstamp'], z['db'], z['gt']
    K = len(ts)
    Xs = np.array([c2w_from_w2c(p)[1] for p in z['poses']]); Xg = gp[ts][:, :3]
    s, R, t = umeyama_sim3(Xs, Xg)
    err = np.linalg.norm((s*(R@Xs.T).T+t)-Xg, axis=1)
    lam = np.full(K, np.nan)
    for k in range(K):
        a, b = max(0, k-W), min(K, k+W+1)
        if b-a >= 5:
            lam[k] = umeyama_sim3(Xs[a:b], Xg[a:b])[0]/s
    # per-frame map gauge: depth ratio in a fixed 8-20 m band (composition-controlled)
    rho = np.full(K, np.nan)
    for k in range(K):
        g = gt[k].ravel(); d = (s/np.maximum(db[k].ravel(), 1e-9))
        m = (g >= 8) & (g < 20)
        if m.sum() > 50:
            rho[k] = np.median(d[m]/g[m])
    # KITTI-style relative segment error
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Xg, axis=0), axis=1))])
    rel = {}
    for L in (100, 200, 300, 400):
        e = []
        for i in range(K):
            j = np.searchsorted(dist, dist[i]+L)
            if j >= K: break
            ds_ = np.linalg.norm(Xs[j]-Xs[i])*s; dg_ = np.linalg.norm(Xg[j]-Xg[i])
            e.append(ds_/dg_)
        if e: rel[L] = (np.mean(e), np.std(e))
    m = np.isfinite(lam) & np.isfinite(rho)
    print(f'\n=== {path.split("/")[-1][:-4]}  ATE {np.sqrt((err**2).mean()):.2f} m')
    print(f'  local Sim3 scale (window {2*W+1} kf): std {np.nanstd(lam):.4f} '
          f'p5 {np.nanpercentile(lam,5):.3f} p95 {np.nanpercentile(lam,95):.3f}')
    print(f'  map depth ratio in 8-20 m band     : std {np.nanstd(rho):.4f} '
          f'p5 {np.nanpercentile(rho,5):.3f} p95 {np.nanpercentile(rho,95):.3f}')
    print(f'  corr(local scale, map depth ratio) : {np.corrcoef(lam[m], rho[m])[0,1]:+.3f}')
    print('  chord/GT over segments: ' + '  '.join(
        f'{L}m {v[0]:.3f}+-{v[1]:.3f}' for L, v in rel.items()))
