"""THE TEST: does the push follow a change of SCENE, or only of prior?

Frames 1000-2000 of kitti_00 move VGGT's pedestal optimum from 1.0 to ~0.8 with the prior and the
tracking config held fixed.  The rule was calibrated on f0-1000 and asked 1.02 there.  If it asks
~0.8 on this window's own dump, the statistic tracks the scene and an online controller is the
right implementation; if it asks ~1.0 again, it is bound to the prior and no zero-run scheme can
follow a window change.
"""
import os, sys, numpy as np
sys.path.insert(0, sys.path[0])
from geo import umeyama_sim3, load_gt_tum

CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
gp = load_gt_tum('data/KITTI/00/traj_tum.txt')[1]
TARGET, TH = 1.93, 1.5

def ate(scene, arm):
    a = np.loadtxt(f'outputs/test/end2end/{scene}/{arm}/traj_full.txt')
    X, G = a[:, 1:4], gp[a[:, 0].astype(int)][:, :3]
    s, R, t = umeyama_sim3(X, G)
    return float(np.sqrt((np.linalg.norm((s*(R@X.T).T+t)-G, axis=1)**2).mean()))

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

def stats(z, tf, B):
    """(push past TH x tracker median depth, near-band ratio below 0.5x) - per keyframe, averaged."""
    dp, db = z['dp'], z['db']; pu, nb = [], []
    for k in range(0, dp.shape[0], 2):
        p0 = dp[k].ravel(); t = db[k].ravel()
        v = (p0 > 0) & (t > 0)
        q = 1.0/np.maximum(tf(1.0/p0[v]), 1e-9)
        A = B[v]*q[:, None]
        s = np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@t[v])
        al = (B[v]@s)*q
        tz = (1.0/t[v]) / np.median(1.0/t[v])
        if (tz >= TH).sum() > 20:
            pu.append(np.median(al[tz >= TH]/t[v][tz >= TH]))
        if (tz < 0.5).sum() > 20:
            nb.append(np.median(al[tz < 0.5]/t[v][tz < 0.5]))
    return float(np.mean(pu)), float(np.mean(nb))

GRID = np.exp(np.linspace(np.log(0.45), np.log(3.5), 34))
SWEEPS = {
    ('f0-1000', 'ped'): ('kitti_00_fg2a05_f0-1000', 'vggt_fg2a05.npz',
                         [0.8, 0.9, 1, 1.1, 1.15, 1.25, 1.3, 1.5, 2]),
    ('f1000-2000', 'ped'): ('kitti_00_fg2a05_f1000-2000', 'vggt_f1k2k.npz',
                            [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.35]),
}
for (win, fam), (scene, npz, tags) in SWEEPS.items():
    z = np.load(f'{CACHE}/{npz}'); B = basis(*z['dp'].shape[1:])
    a = np.array([ate(scene, f'base_{fam}{t:g}'.replace('.', 'p')) for t in tags])
    tg = np.array(tags); i = int(np.argmin(a)); j = slice(max(0, i-1), i+2)
    c = np.polyfit(np.log(tg[j]), a[j], 2)
    opt = float(np.clip(np.exp(-c[1]/(2*c[0])), tg[j].min(), tg[j].max())) if c[0] > 0 else tg[i]
    S = np.array([stats(z, lambda d, m=g, f=fam: FAM[f](d, m), B)[0] for g in GRID])
    o = np.argsort(S)
    pred = np.interp(TARGET, S[o], GRID[o]) if S.min() <= TARGET <= S.max() else np.nan
    raw_p, raw_n = stats(z, lambda d: d, B)
    print(f'\n=== VGGT base @{fam}, {win}   ({z["dp"].shape[0]} keyframes)')
    print('  sweep  ' + '  '.join(f'{t:g}:{x:.2f}' for t, x in zip(tg, a)))
    print(f'  swept optimum          {opt:.2f}   (best measured {tg[i]:g} at {a[i]:.2f} m)')
    print(f'  untransformed push     {raw_p:.2f}   near band {raw_n:.3f}'
          f'   -> family: {"clamp is enough" if raw_n < 0.95 else "needs ped/soft"}')
    print(f'  RULE PREDICTS          {pred:.2f}   error {100*(pred/opt-1):+.0f}%')
    print(f'  push at the swept opt  {np.interp(opt, GRID, S):.3f}   (target {TARGET})')
