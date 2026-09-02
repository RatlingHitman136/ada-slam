"""IN SITU: the push read on a run that was actually SERVED the pedestal its window's sweep chose.

Every push number before this one was counterfactual - a candidate transform scored against a
tracker field recorded under the RAW prior.  These two dumps remove that assumption, and the pair
decides between the two readings of the rule's 21% miss on f1000-2000:
    both near 1.93  -> the target holds and the offline INVERSION was the error (a controller fixes it)
    1.94 vs 1.87    -> the TARGET moves with the scene (no fixed-target rule can hit the optimum)
"""
import os, sys, numpy as np
sys.path.insert(0, sys.path[0])
CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
TH = 1.5

def basis(ht, wd):
    y = np.linspace(0, 1-1e-6, ht)[:, None]; x = np.linspace(0, 1-1e-6, wd)[None, :]
    return np.stack([(1-y)*(1-x)*np.ones((ht, wd)), (1-y)*x*np.ones((ht, wd)),
                     y*(1-x)*np.ones((ht, wd)), y*x*np.ones((ht, wd))], -1).reshape(-1, 4)

def measure(z):
    """push and near band, twice: with JDSA's OWN dumped grid, and with the refit the offline
    inversion uses - so the in-situ number is comparable to the counterfactual one."""
    dp, db, ds = z['dp'], z['db'], z['dscales']
    K, ht, wd = dp.shape; B = basis(ht, wd)
    out = []
    for k in range(0, K, 2):
        p = dp[k].ravel(); t = db[k].ravel()
        v = (p > 0) & (t > 0)
        q = p[v]
        dumped = (B[v] @ np.array([ds[k][0,0], ds[k][0,1], ds[k][1,0], ds[k][1,1]])) * q
        A = B[v]*q[:, None]
        s = np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@t[v])
        refit = (B[v]@s)*q
        tz = (1.0/t[v]) / np.median(1.0/t[v])
        far, near = tz >= TH, tz < 0.5
        if far.sum() > 20 and near.sum() > 20:
            out.append([np.median(dumped[far]/t[v][far]), np.median(refit[far]/t[v][far]),
                        np.median(refit[near]/t[v][near])])
    return np.mean(out, 0), len(out)

print(f'  {"dump":<34}{"served":>16}{"push (JDSA grid)":>19}{"push (refit)":>14}{"near band":>11}')
for nm, served, note in (
        ('vggt_fg2a05',        'raw',            'f0-1000, counterfactual source'),
        ('insitu_ped1_f0k1k',  '@ped1  (opt 0.99)', 'f0-1000, IN SITU'),
        ('vggt_f1k2k',         'raw',            'f1000-2000, counterfactual source'),
        ('insitu_ped0p8_f1k2k','@ped0.8 (opt 0.82)', 'f1000-2000, IN SITU')):
    m, n = measure(np.load(f'{CACHE}/{nm}.npz'))
    print(f'  {nm:<34}{served:>16}{m[0]:19.3f}{m[1]:14.3f}{m[2]:11.3f}   {note}')
print(f'\n  counterfactual estimates of the same two optima (25_window_test.py):'
      f'  f0-1000 1.942   f1000-2000 1.870')

# ---------------------------------------------------------------------------------------------
# The near band held tighter in situ than the push did (0.905 vs 0.884, against 1.926 vs 1.819).
# Would inverting IT have picked the tag on both windows?
def pedestal(d, r):
    q = 1.0/np.maximum(d, 1e-9); return 1.0/(q + np.median(q)/r)

def curve(z, stat, grid):
    dp, db = z['dp'], z['db']; K, ht, wd = dp.shape; B = basis(ht, wd)
    out = []
    for g in grid:
        acc = []
        for k in range(0, K, 2):
            p0 = dp[k].ravel(); t = db[k].ravel()
            v = (p0 > 0) & (t > 0)
            q = 1.0/np.maximum(pedestal(1.0/p0[v], g), 1e-9)
            A = B[v]*q[:, None]
            s = np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@t[v])
            al = (B[v]@s)*q
            tz = (1.0/t[v]) / np.median(1.0/t[v])
            m = (tz >= TH) if stat == 'push' else (tz < 0.5)
            if m.sum() > 20:
                acc.append(np.median(al[m]/t[v][m]))
        out.append(np.mean(acc))
    return np.array(out)

GRID = np.exp(np.linspace(np.log(0.45), np.log(3.0), 26))
print(f'\n  inverting each statistic on the RAW dump of each window (base@ped family)')
print(f'  {"window":<14}{"swept opt":>10}{"push->1.93":>12}{"near->0.912":>13}{"near->0.895":>13}')
for nm, lab, opt in (('vggt_fg2a05', 'f0-1000', 0.99), ('vggt_f1k2k', 'f1000-2000', 0.82)):
    z = np.load(f'{CACHE}/{nm}.npz')
    row = []
    for stat, tgt in (('push', 1.93), ('near', 0.912), ('near', 0.895)):
        v = curve(z, stat, GRID); o = np.argsort(v)
        row.append(np.interp(tgt, v[o], GRID[o]) if v.min() <= tgt <= v.max() else np.nan)
    print(f'  {lab:<14}{opt:10.2f}' + ''.join(f'{x:12.2f} ' for x in row))
