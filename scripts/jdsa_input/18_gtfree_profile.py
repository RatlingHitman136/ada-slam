"""A GT-FREE read of the same anatomy: what the served prior asserts against what the tracker
believes, binned by the prior's own depth in multiples of the frame median.

Everything here is computable ONLINE - disps_prior, disps and dscales are all in the video buffer -
so whatever separates a scene where bounding the tail pays from one where it does not has to be
visible in this table, or no automatic rule can exist.
"""
import sys, numpy as np

BINS = [0, .5, 1, 1.5, 2, 3, 5, 1e9]

def bilin2x2(g, ht, wd):
    y = np.linspace(0, 1-1e-6, ht)[:, None]; x = np.linspace(0, 1-1e-6, wd)[None, :]
    return (1-y)*(1-x)*g[0,0] + (1-y)*x*g[0,1] + y*(1-x)*g[1,0] + y*x*g[1,1]

for path in sys.argv[1:]:
    z = np.load(path)
    dp, db, ds = z['dp'], z['db'], z['dscales']
    K, ht, wd = dp.shape
    px = np.zeros(len(BINS)-1); rat = [[] for _ in BINS[:-1]]; inf = np.zeros(len(BINS)-1)
    tail = np.zeros((K, 3))
    for k in range(K):
        p = dp[k].ravel(); t = db[k].ravel()
        v = (p > 0) & (t > 0)
        al = (dp[k]*bilin2x2(ds[k], ht, wd)).ravel()[v]      # the realised alignment
        u = (1.0/p[v]) / np.median(1.0/p[v])                 # served depth / frame median
        idx = np.digitize(u, BINS) - 1
        w = (p[v]/np.median(p[v]))**2
        for i in range(len(BINS)-1):
            m = idx == i
            px[i] += m.mean()/K; inf[i] += w[m].sum()/w.sum()/K
            if m.sum() > 20:
                rat[i].append(np.median(al[m]/t[m]))
        q = np.percentile(1.0/p[v], [95, 99, 100])/np.median(1.0/p[v])
        tail[k] = q
    print(f'\n=== {path.split("/")[-1][:-4]}   K={K}')
    print(f'  served tail over the frame median: p95 {tail[:,0].mean():.2f}  '
          f'p99 {tail[:,1].mean():.2f}  max {tail[:,2].mean():.2f}')
    print(f'  {"served depth / frame median":<30}' +
          ''.join((f'{BINS[i]:g}-{BINS[i+1]:g}' if BINS[i+1] < 1e8 else f'>{BINS[i]:g}').rjust(9)
                  for i in range(len(BINS)-1)))
    print(f'  {"share of pixels":<30}' + ''.join(f'{100*v:8.1f}%' for v in px))
    print(f'  {"share of the scale fit":<30}' + ''.join(f'{100*v:8.1f}%' for v in inf))
    print(f'  {"aligned prior / tracker disp":<30}' +
          ''.join(f'{np.median(r) if r else np.nan:9.3f}' for r in rat))
