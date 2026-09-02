"""What does each served distribution ASSERT about far structure, in metres, after JDSA's own
near-anchored alignment - and where do the arms that actually track well sit on that axis?"""
import sys, os, numpy as np
sys.path.insert(0, sys.path[0])
from geo import umeyama_sim3, load_gt_tum

# the per-keyframe caches 01_build_cache.py writes (86 MB, storage partition; $JDSA_CACHE overrides)
CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
gidx, gp = load_gt_tum('data/KITTI/00/traj_tum.txt')
SCENE = 'outputs/test/end2end/kitti_00_fg2a05_f0-1000'
BANDS = [(5, 10), (10, 20), (20, 40), (40, 80)]

def wob_ate(arm, W=15):
    a = np.loadtxt(f'{SCENE}/{arm}/traj_full.txt')
    idx = a[:, 0].astype(int); X = a[:, 1:4]; G = gp[idx][:, :3]
    s, R, t = umeyama_sim3(X, G)
    e = np.linalg.norm((s*(R@X.T).T+t)-G, axis=1)
    lam = [umeyama_sim3(X[max(0,k-W):k+W+1], G[max(0,k-W):k+W+1])[0]/s for k in range(len(X))]
    return np.sqrt((e**2).mean()), np.nanstd(lam)

def ceil_clamp(d, r): return np.minimum(d, r*np.median(d))
def pedestal(d, r):
    q = 1.0/np.maximum(d, 1e-9); return 1.0/(q + np.median(q)/r)

def assertions(dp, gt, tf):
    K = dp.shape[0]; out = []
    for k in range(K):
        d0 = dp[k].ravel(); g = gt[k].ravel()
        v = (d0 > 0) & (g > 0)
        q = 1.0/tf(1.0/d0[v])                       # served disparity
        y = 1.0/g[v]                                # GT disparity
        s = (q*y).sum()/(q*q).sum()                 # JDSA's own weighted scale fit
        al = 1.0/(s*q)                              # asserted depth, metres
        row = [np.median(al[(g[v] >= lo) & (g[v] < hi)]/g[v][(g[v] >= lo) & (g[v] < hi)])
               if ((g[v] >= lo) & (g[v] < hi)).sum() > 20 else np.nan for lo, hi in BANDS]
        row.append(np.percentile(al, 99))           # the farthest thing the prior will assert (m)
        out.append(row)
    return np.nanmedian(out, 0)

rows = []
for pri, npz in (('omni', 'omni_fg2a05.npz'), ('base', 'vggt_fg2a05.npz')):
    z = np.load(f'{CACHE}/{npz}'); dp, gt = z['dp'], z['gt']
    specs = [(pri, lambda d: d)]
    specs += [(f'{pri}_ceil{r:g}'.replace('.', 'p'), (lambda r: lambda d: ceil_clamp(d, r))(r))
              for r in (1.25, 1.35, 1.4, 1.45, 1.5, 2, 3)]
    specs += [(f'{pri}_ped{r:g}'.replace('.', 'p'), (lambda r: lambda d: pedestal(d, r))(r))
              for r in (0.6, 0.7, 0.8, 0.85, 0.9, 1, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.8, 2, 3)]
    for arm, tf in specs:
        if not os.path.exists(f'{SCENE}/{arm}/traj_full.txt'):
            continue
        ate, wob = wob_ate(arm)
        rows.append((arm, ate, wob) + tuple(assertions(dp, gt, tf)))

print(f'  {"arm":<16}{"ATE":>7}{"wobble":>8}   asserted depth / true depth' +
      f'{"":18}{"p99 asserted":>14}')
print(f'  {"":<16}{"":7}{"":8}   ' + ''.join(f'{lo}-{hi}m'.rjust(10) for lo, hi in BANDS) + '     (m)')
for r in sorted(rows, key=lambda r: r[1]):
    print(f'  {r[0]:<16}{r[1]:7.2f}{r[2]:8.4f}   ' + ''.join(f'{v:10.3f}' for v in r[3:7]) +
          f'{r[7]:12.1f}')
