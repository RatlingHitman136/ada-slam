"""Global scale DRIFT against local scale JITTER: which one costs the ATE, and which one R fixes.

The local Sim(3) scale of an arm is measured per pose, split into a slow trend (drift) and the
residual around it (jitter), and then each component is REMOVED from the trajectory by rescaling
its incremental translations and re-integrating.  The ATE of each corrected path is the
attribution - no model, just the same trajectory with one error mode taken out.
"""
import os, sys, glob, numpy as np
sys.path.insert(0, sys.path[0])
from geo import umeyama_sim3, load_gt_tum

gp = load_gt_tum(os.environ.get('JDSA_GT', 'data/KITTI/00/traj_tum.txt'))[1]
W, MIN_WIN_M, TREND = 15, 3.0, 201        # local window, motion guard, trend smoothing (poses)

def local_scale(X, G):
    cum = np.cumsum(np.concatenate([[0], np.linalg.norm(np.diff(G, axis=0), axis=1)]))
    r = np.full(len(X), np.nan)
    for k in range(len(X)):
        i, j = max(0, k-W), min(len(X), k+W+1)
        if j-i >= 11 and cum[j-1] - cum[i] > MIN_WIN_M:
            r[k] = umeyama_sim3(X[i:j], G[i:j])[0]
    ok = np.isfinite(r)
    return np.interp(np.arange(len(X)), np.flatnonzero(ok), r[ok])

def smooth(v, n):
    ker = np.ones(n)/n
    pad = np.concatenate([np.full(n//2, v[0]), v, np.full(n//2, v[-1])])
    return np.convolve(pad, ker, mode='valid')[:len(v)]

def ate_of(X, G):
    s, R, t = umeyama_sim3(X, G)
    return float(np.sqrt((np.linalg.norm((s*(R@X.T).T+t)-G, axis=1)**2).mean()))

def reintegrate(X, c):
    d = np.diff(X, axis=0)*c[:-1, None]
    return np.concatenate([X[:1], X[:1] + np.cumsum(d, axis=0)])

def analyse(path):
    a = np.loadtxt(path); X, G = a[:, 1:4], gp[a[:, 0].astype(int)][:, :3]
    r = local_scale(X, G)
    lr = np.log(r); tr = smooth(lr, TREND)               # slow component
    jit = lr - tr                                        # fast component
    base = ate_of(X, G)
    no_drift = ate_of(reintegrate(X, np.exp(tr - tr.mean())), G)   # flatten the trend only
    no_jit = ate_of(reintegrate(X, np.exp(jit)), G)               # remove the jitter only
    both = ate_of(reintegrate(X, np.exp(lr - lr.mean())), G)
    tenth = len(X)//10
    drift_ratio = np.exp(np.mean(lr[-tenth:]) - np.mean(lr[:tenth]))
    return dict(ate=base, no_drift=no_drift, no_jit=no_jit, both=both,
                drift=drift_ratio, jitter=float(np.std(jit)), spread=float(np.std(lr)))

SETS = {'kitti_00_fg2a05_f0-1000': ['omni', 'omni_ceil1p45', 'omni_ceil1p5', 'omni_ceil2',
                                    'omni_soft1p45', 'omni_ped1p8', 'omni_ped1', 'base',
                                    'base_ped1', 'base_soft1', 'base_ceil1p45', 'base_ped1p5'],
        'kitti_00_fg2a05_f1000-2000': ['base', 'base_ped0p8', 'base_ped1', 'omni_ceil1p45']}
print(f'  {"arm":<28}{"ATE":>7}{"drift":>8}{"jitter":>8}   '
      f'{"ATE w/o drift":>14}{"w/o jitter":>12}{"w/o both":>10}')
rows = []
for scene, arms in SETS.items():
    print(f'  --- {scene}')
    for arm in arms:
        p = f'outputs/test/end2end/{scene}/{arm}/traj_full.txt'
        if not os.path.isfile(p):
            continue
        d = analyse(p); rows.append((scene, arm, d))
        print(f'  {arm:<28}{d["ate"]:7.2f}{d["drift"]:8.3f}{d["jitter"]:8.4f}   '
              f'{d["no_drift"]:13.2f}{d["no_jit"]:12.2f}{d["both"]:10.2f}')
A = np.array([[r[2]['ate'], abs(np.log(r[2]['drift'])), r[2]['jitter']] for r in rows])
print(f'\n  corr(ATE, |log drift|) {np.corrcoef(A[:,0], A[:,1])[0,1]:+.3f}   '
      f'corr(ATE, jitter) {np.corrcoef(A[:,0], A[:,2])[0,1]:+.3f}   n={len(A)}')
print(f'  mean ATE removed by flattening the drift: '
      f'{100*np.mean(1-np.array([r[2]["no_drift"]/r[2]["ate"] for r in rows])):.0f}%')
print(f'  mean ATE removed by removing the jitter : '
      f'{100*np.mean(1-np.array([r[2]["no_jit"]/r[2]["ate"] for r in rows])):.0f}%')
