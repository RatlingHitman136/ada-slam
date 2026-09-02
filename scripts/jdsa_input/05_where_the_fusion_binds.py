"""Where does the fusion BIND?  Uses the dumped dscales - JDSA's own realised alignment - and
compares the aligned prior disparity with the tracker's own, by GT range band.  Plus a fresh ATE
and gauge-jitter number per arm, so the two can be read against each other."""
import sys, numpy as np
sys.path.insert(0, sys.path[0])
from geo import c2w_from_w2c, umeyama_sim3, load_gt_tum

BANDS = [(2, 5), (5, 10), (10, 20), (20, 40), (40, 80)]

def bilin2x2(g, ht, wd):
    y = np.linspace(0, 1-1e-6, ht)[:, None]; x = np.linspace(0, 1-1e-6, wd)[None, :]
    return (1-y)*(1-x)*g[0,0] + (1-y)*x*g[0,1] + y*(1-x)*g[1,0] + y*x*g[1,1]

gidx, gp = load_gt_tum('data/KITTI/00/traj_tum.txt')
print(f'{"arm":<22}{"ATE":>7}{"step-scale":>12}{"gauge":>8}   aligned-prior disparity / tracker disparity, by GT range')
print(f'{"":<22}{"(m)":>7}{"mean/std":>12}{"iqr":>8}   ' + '  '.join(f'{lo}-{hi}m' for lo, hi in BANDS))
for path in sys.argv[1:]:
    z = np.load(path)
    dp, db, gt, ds, ts = z['dp'], z['db'], z['gt'], z['dscales'], z['tstamp']
    K, ht, wd = dp.shape
    Xs = np.array([c2w_from_w2c(p)[1] for p in z['poses']]); Xg = gp[ts][:, :3]
    s, R, t = umeyama_sim3(Xs, Xg)
    ate = np.sqrt((np.linalg.norm((s*(R@Xs.T).T+t)-Xg, axis=1)**2).mean())
    dS = np.linalg.norm(np.diff(Xs, axis=0), axis=1)*s; dG = np.linalg.norm(np.diff(Xg, axis=0), axis=1)
    r = dS/np.maximum(dG, 1e-9)
    ratio = np.full((K, len(BANDS)), np.nan)      # aligned prior disparity / BA disparity
    rho = np.full(K, np.nan)                      # frame gauge: BA depth / GT depth, all valid px
    for k in range(K):
        al = (dp[k]*bilin2x2(ds[k], ht, wd)).ravel()
        b = db[k].ravel(); g = gt[k].ravel()
        v = (g > 0) & (b > 0) & (dp[k].ravel() > 0)
        rho[k] = np.median((s/b[v])/g[v])
        for i, (lo, hi) in enumerate(BANDS):
            m = v & (g >= lo) & (g < hi)
            if m.sum() > 20:
                ratio[k, i] = np.median(al[m]/b[m])
    iqr = np.nanpercentile(rho, 75)-np.nanpercentile(rho, 25)
    nm = path.split('/')[-1][:-4]
    print(f'{nm:<22}{ate:7.2f}{r.mean():7.3f}/{r.std():.3f}{iqr:8.3f}   ' +
          '  '.join(f'{np.nanmedian(ratio[:,i]):6.3f}' for i in range(len(BANDS))))
