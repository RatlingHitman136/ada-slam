"""Automatic R, decided: for every (prior, family) whose ATE optimum is known, invert EACH band
target and see which one lands on it.

One statistic grid per curve (25 tags), then linear interpolation - no bisection, no GT depth, no
extra SLAM run.  The question this answers is not "which statistic correlates" but "which one, set
to a fixed target, PICKS the tag someone else's sweep found".
"""
import os, sys, numpy as np
sys.path.insert(0, sys.path[0])
from geo import umeyama_sim3, load_gt_tum

CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
SCENE = 'outputs/test/end2end/kitti_00_fg2a05_f0-1000'
gp = load_gt_tum('data/KITTI/00/traj_tum.txt')[1]
TB = [(0, .5), (.5, 1), (1, 1.5), (1.5, 2.5), (2.5, 1e9)]
NAMES = ['near .0-.5', 'near .5-1', 'mid 1-1.5', 'far 1.5-2.5', 'far >2.5', 'push (>1.5)']
# targets: the median over the five bracketed optima (21_disagreement_profile.py)
TGT = [0.912, 0.971, 0.986, 1.241, 4.958, 1.888]

def ceil_clamp(d, r): return np.minimum(d, r*np.median(d))
def pedestal(d, r):
    q = 1.0/np.maximum(d, 1e-9); return 1.0/(q + np.median(q)/r)
def soft(d, r):
    q = 1.0/np.maximum(d, 1e-9); return 1.0/np.sqrt(q*q + (np.median(q)/r)**2)
FAM = {'ceil': ceil_clamp, 'ped': pedestal, 'soft': soft}

def basis(ht, wd):
    y = np.linspace(0, 1-1e-6, ht)[:, None]; x = np.linspace(0, 1-1e-6, wd)[None, :]
    return np.stack([(1-y)*(1-x)*np.ones((ht, wd)), (1-y)*x*np.ones((ht, wd)),
                     y*(1-x)*np.ones((ht, wd)), y*x*np.ones((ht, wd))], -1).reshape(-1, 4)

def bands(z, tf, B=None):
    dp, db = z['dp'], z['db']; K, ht, wd = dp.shape
    B = basis(ht, wd) if B is None else B
    acc = []
    for k in range(0, K, 2):                      # every other keyframe: 2x faster, same numbers
        p0 = dp[k].ravel(); t = db[k].ravel()
        v = (p0 > 0) & (t > 0)
        q = 1.0/np.maximum(tf(1.0/p0[v]), 1e-9)
        A = B[v]*q[:, None]
        s = np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@t[v])
        al = (B[v]@s)*q
        tz = (1.0/t[v]) / np.median(1.0/t[v])
        r = [np.median(al[(tz >= lo) & (tz < hi)]/t[v][(tz >= lo) & (tz < hi)])
             if ((tz >= lo) & (tz < hi)).sum() > 20 else np.nan for lo, hi in TB]
        r.append(np.median(al[tz >= 1.5]/t[v][tz >= 1.5]))
        acc.append(r)
    return np.nanmean(acc, 0)

CURVES = [('omni', 'ceil', 1.44, 'omni_fg2a05.npz'), ('omni', 'soft', 1.50, 'omni_fg2a05.npz'),
          ('omni', 'ped', 1.86, 'omni_fg2a05.npz'), ('base', 'soft', 0.98, 'vggt_fg2a05.npz'),
          ('base', 'ped', 0.99, 'vggt_fg2a05.npz')]
Z = {n: np.load(f'{CACHE}/{n}') for n in ('omni_fg2a05.npz', 'vggt_fg2a05.npz')}

print(f'  {"curve":<12}{"swept opt":>10}   ' + ''.join(f'{n:>13}' for n in NAMES))
print(f'  {"":<12}{"":>10}   ' + ''.join(f'{"tgt "+str(t):>13}' for t in TGT))
err = []
for pri, fam, opt, npz in CURVES:
    z = Z[npz]; B = basis(*z['dp'].shape[1:])
    grid = np.exp(np.linspace(np.log(0.6), np.log(3.2), 25))
    S = np.array([bands(z, lambda d, m=g, f=fam: FAM[f](d, m), B) for g in grid])
    pred = []
    for i in range(6):
        s = S[:, i]
        ok = np.isfinite(s)
        g, v = grid[ok], s[ok]
        o = np.argsort(v)
        p = np.interp(TGT[i], v[o], g[o]) if v.min() <= TGT[i] <= v.max() else np.nan
        # reject a target the statistic never crosses inside the family's usable range
        pred.append(p)
    err.append([abs(np.log(p/opt)) if np.isfinite(p) else np.nan for p in pred])
    print(f'  {pri+" @"+fam:<12}{opt:10.2f}   ' +
          ''.join((f'{p:13.2f}' if np.isfinite(p) else f'{"never":>13}') for p in pred))
E = np.array(err)
print(f'\n  {"|log(pred/opt)| mean":<24}' + ''.join(f'{np.nanmean(E[:,i]):13.3f}' for i in range(6)))
print(f'  {"worst curve":<24}' + ''.join(f'{np.nanmax(E[:,i]):13.3f}' for i in range(6)))
print(f'  {"curves it can invert":<24}' + ''.join(f'{np.isfinite(E[:,i]).sum():13d}' for i in range(6)))
