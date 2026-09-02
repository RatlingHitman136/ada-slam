"""Did the ceiling actually move the FUSED field, and where?  Row-band profile of the served
prior (after its own dscales) and of the tracker's disparity, in frame-median units."""
import sys, numpy as np

def bilin2x2(g, ht, wd):
    y = np.linspace(0, 1-1e-6, ht)[:, None]; x = np.linspace(0, 1-1e-6, wd)[None, :]
    return (1-y)*(1-x)*g[0,0] + (1-y)*x*g[0,1] + y*(1-x)*g[1,0] + y*x*g[1,1]

ROWS = [(0, 8), (8, 16), (16, 24), (24, 32)]
for path in sys.argv[1:]:
    z = np.load(path); dp, db, ds, gt = z['dp'], z['db'], z['dscales'], z['gt']
    K, ht, wd = dp.shape
    pr = np.zeros((K, len(ROWS))); tr = np.zeros((K, len(ROWS))); gg = np.full((K, len(ROWS)), np.nan)
    cov = np.zeros(len(ROWS))
    for k in range(K):
        al = dp[k]*bilin2x2(ds[k], ht, wd)
        mb = np.median(db[k][db[k] > 0])
        for i, (a, b) in enumerate(ROWS):
            pr[k, i] = np.median(al[a:b][dp[k][a:b] > 0])/mb
            tr[k, i] = np.median(db[k][a:b])/mb
            g = gt[k][a:b]
            if (g > 0).sum() > 20:
                gg[k, i] = np.median(g[g > 0])
            cov[i] += (g > 0).mean()/K
    print(f'\n=== {path.split("/")[-1][:-4]}   rows are top(0-8) .. bottom(24-32) of the 1/8 grid')
    print('   rows      aligned prior / frame med   tracker / frame med   GT depth (m)   GT cover')
    for i, (a, b) in enumerate(ROWS):
        print(f'   {a:2d}-{b:<3d}   {pr[:,i].mean():>18.3f} (sd {pr[:,i].std():.3f})'
              f'   {tr[:,i].mean():>10.3f} (sd {tr[:,i].std():.3f})'
              f'   {np.nanmean(gg[:,i]):>10.1f}   {cov[i]*100:5.1f}%')
