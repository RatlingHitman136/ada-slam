"""Which statistic of the SERVED disparity distribution predicts the trajectory?

The transform is applied offline to the same prior the arm was served, the resulting distribution
is summarised a few different ways, and each summary is regressed against the wobble that arm
actually realised (re-measured from its raw trajectory, an12).
"""
import sys, os, numpy as np
sys.path.insert(0, sys.path[0])
from geo import umeyama_sim3, load_gt_tum

# the per-keyframe caches 01_build_cache.py writes (86 MB, storage partition; $JDSA_CACHE overrides)
CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
gidx, gp = load_gt_tum('data/KITTI/00/traj_tum.txt')
SCENE = 'outputs/test/end2end/kitti_00_fg2a05_f0-1000'

def wobble_ate(arm, W=15):
    a = np.loadtxt(f'{SCENE}/{arm}/traj_full.txt')
    idx = a[:, 0].astype(int); X = a[:, 1:4]; G = gp[idx][:, :3]
    s, R, t = umeyama_sim3(X, G)
    err = np.linalg.norm((s*(R@X.T).T+t)-G, axis=1)
    lam = [umeyama_sim3(X[max(0,k-W):k+W+1], G[max(0,k-W):k+W+1])[0]/s for k in range(len(X))]
    return np.sqrt((err**2).mean()), np.nanstd(lam)

def ceil_clamp(d, r): return np.minimum(d, r*np.median(d))
def pedestal(d, r):
    q = 1.0/np.maximum(d, 1e-9); return 1.0/(q + np.median(q)/r)

def stats(dp, tf):
    K = dp.shape[0]; acc = []
    for k in range(K):
        d0 = dp[k].ravel(); v = d0 > 0
        dep = tf(1.0/d0[v])
        u = (1.0/dep); u = u/np.median(u)
        w = u*u
        acc.append([u.mean(), u.std(), (u < 0.5).mean(), u.min(),
                    w[u < 0.667].sum()/w.sum(),          # info share beyond 1.5x median depth
                    u[u < 1].std(),                      # contrast of the far half
                    np.percentile(u, 95)/np.percentile(u, 5),
                    (u**2).mean()/u.mean()**2])
    return np.mean(acc, 0)

NAMES = ['mean(u)', 'std(u)', 'P(u<.5)', 'min u', 'info>1.5x', 'std(u|u<1)', 'q95/q05', 'E[u2]/E[u]2']
rows = []
for pri, npz in (('omni', 'omni_fg2a05.npz'), ('base', 'vggt_fg2a05.npz')):
    dp = np.load(f'{CACHE}/{npz}')['dp']
    specs = [(pri, lambda d: d)]
    for r in (1.25, 1.35, 1.4, 1.42, 1.43, 1.45, 1.5, 2, 3):
        specs.append((f'{pri}_ceil{r:g}'.replace('.', 'p'), (lambda r: lambda d: ceil_clamp(d, r))(r)))
    for r in (0.6, 0.7, 0.8, 0.85, 0.9, 1, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.8, 2, 3):
        specs.append((f'{pri}_ped{r:g}'.replace('.', 'p'), (lambda r: lambda d: pedestal(d, r))(r)))
    for arm, tf in specs:
        if not os.path.exists(f'{SCENE}/{arm}/traj_full.txt'):
            continue
        ate, wob = wobble_ate(arm)
        rows.append((arm, ate, wob) + tuple(stats(dp, tf)))

print(f'  {"arm":<16}{"ATE":>7}{"wobble":>8}  ' + ''.join(f'{n:>12}' for n in NAMES))
for r in sorted(rows, key=lambda r: r[1]):
    print(f'  {r[0]:<16}{r[1]:7.2f}{r[2]:8.4f}  ' + ''.join(f'{v:12.4f}' for v in r[3:]))
A = np.array([r[1:] for r in rows], float)
print('\n  correlation with log(wobble), all arms / omni only / base only:')
lw = np.log(A[:, 1]); isb = np.array([r[0].startswith('base') for r in rows])
for i, n in enumerate(NAMES):
    c = lambda m: np.corrcoef(A[m, 2+i], lw[m])[0, 1]
    print(f'    {n:<14} {c(np.ones(len(A), bool)):+.3f}   {c(~isb):+.3f}   {c(isb):+.3f}')
