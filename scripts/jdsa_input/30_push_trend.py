"""If DRIFT is the ATE, can drift be seen without GT?

The push compares the served prior with the tracker.  If the prior is scale-consistent frame to
frame and the tracker's scale drifts, the push must trend over the run - which would make the
tracker's drift observable online, with no lidar and no GT trajectory.  Measured here against the
trajectory drift each dump's own run realised.
"""
import os, sys, numpy as np
sys.path.insert(0, sys.path[0])
from geo import umeyama_sim3, load_gt_tum

CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
gp = load_gt_tum('data/KITTI/00/traj_tum.txt')[1]
W, MIN_WIN_M = 15, 3.0

def basis(ht, wd):
    y = np.linspace(0, 1-1e-6, ht)[:, None]; x = np.linspace(0, 1-1e-6, wd)[None, :]
    return np.stack([(1-y)*(1-x)*np.ones((ht, wd)), (1-y)*x*np.ones((ht, wd)),
                     y*(1-x)*np.ones((ht, wd)), y*x*np.ones((ht, wd))], -1).reshape(-1, 4)

def push_series(z):
    """per keyframe: aligned served disparity / tracker disparity, on the tracker's far pixels."""
    dp, db = z['dp'], z['db']; K, ht, wd = dp.shape; B = basis(ht, wd)
    out = np.full(K, np.nan)
    for k in range(K):
        p = dp[k].ravel(); t = db[k].ravel()
        v = (p > 0) & (t > 0)
        q = p[v]
        A = B[v]*q[:, None]
        s = np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@t[v])
        al = (B[v]@s)*q
        tz = (1.0/t[v]) / np.median(1.0/t[v])
        if (tz >= 1.5).sum() > 20:
            out[k] = np.median(al[tz >= 1.5]/t[v][tz >= 1.5])
    return out

def traj_drift(path):
    a = np.loadtxt(path); X, G = a[:, 1:4], gp[a[:, 0].astype(int)][:, :3]
    cum = np.cumsum(np.concatenate([[0], np.linalg.norm(np.diff(G, axis=0), axis=1)]))
    r = np.full(len(X), np.nan)
    for k in range(len(X)):
        i, j = max(0, k-W), min(len(X), k+W+1)
        if j-i >= 11 and cum[j-1] - cum[i] > MIN_WIN_M:
            r[k] = umeyama_sim3(X[i:j], G[i:j])[0]
    r = r[np.isfinite(r)]; n = len(r)//10
    return float(np.exp(np.mean(np.log(r[-n:])) - np.mean(np.log(r[:n]))))

RUNS = [
    ('omni raw',        'omni_fg2a05',         'outputs/extract/kitti_00_fg2a05_f0-1000/normal'),
    ('omni@ceil1.5',    'omni_ceil15_fg2a05',  'outputs/extract/kitti_00_fg2a05_f0-1000/normal_ceil1p5'),
    ('vggt raw',        'vggt_fg2a05',         'outputs/extract/kitti_00_fg2a05_f0-1000/normal_vggt'),
    ('vggt-adapted',    'adapt_fg2a05',        'outputs/extract/kitti_00_fg2a05_f0-1000/normal_adapt'),
    ('vggt@ped1 f0-1k', 'insitu_ped1_f0k1k',   'outputs/extract/kitti_00_fg2a05_f0-1000/insitu_ped1'),
    ('vggt raw f1k-2k', 'vggt_f1k2k',          'outputs/extract/kitti_00_fg2a05_f1000-2000/basedump'),
    ('vggt@ped0.8 f1k-2k','insitu_ped0p8_f1k2k','outputs/extract/kitti_00_fg2a05_f1000-2000/insitu_ped0p8'),
]
print(f'  {"run":<22}{"push start":>12}{"push end":>10}{"push trend":>12}'
      f'{"traj drift":>12}{"agreement":>11}')
P, T = [], []
for lab, npz, run in RUNS:
    f = f'{run}/traj_full.txt'
    if not os.path.isfile(f) or not os.path.isfile(f'{CACHE}/{npz}.npz'):
        print(f'  {lab:<22} (missing)'); continue
    v = push_series(np.load(f'{CACHE}/{npz}.npz'))
    v = v[np.isfinite(v)]; n = max(len(v)//10, 5)
    a_, b_ = float(np.mean(v[:n])), float(np.mean(v[-n:]))
    pt, td = b_/a_, traj_drift(f)
    P.append(np.log(pt)); T.append(np.log(td))
    print(f'  {lab:<22}{a_:12.2f}{b_:10.2f}{pt:12.3f}{td:12.3f}{pt/td:11.2f}')
print(f'\n  corr(log push trend, log trajectory drift) = '
      f'{np.corrcoef(P, T)[0,1]:+.3f}   n={len(P)}')
print(f'  slope = {np.polyfit(T, P, 1)[0]:+.2f}  (1.0 would mean the push trend IS the drift)')
