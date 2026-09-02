"""How sharply is the push target determined?  Sweep it and watch the tag error on all five curves.

The target is an EMPIRICAL constant, not a derived one, so its error bar is the thing to publish:
this prints the whole curve rather than the argmin.
"""
import os, sys, numpy as np
sys.path.insert(0, sys.path[0])
CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')

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

def push(z, tf, B, th=1.5):
    dp, db = z['dp'], z['db']; acc = []
    for k in range(0, dp.shape[0], 2):
        p0 = dp[k].ravel(); t = db[k].ravel()
        v = (p0 > 0) & (t > 0)
        q = 1.0/np.maximum(tf(1.0/p0[v]), 1e-9)          # served disparity
        A = B[v]*q[:, None]                              # JDSA's own 2x2 alignment, refitted
        s = np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@t[v])
        al = (B[v]@s)*q                                  # aligned served disparity
        tz = (1.0/t[v]) / np.median(1.0/t[v])            # tracker depth / tracker frame median
        far = tz >= th
        if far.sum() > 20:
            acc.append(np.median(al[far]/t[v][far]))     # median over the frame's far pixels
    return float(np.mean(acc))                           # mean over keyframes

CURVES = [('omni', 'ceil', 1.44, 'omni_fg2a05.npz'), ('omni', 'soft', 1.50, 'omni_fg2a05.npz'),
          ('omni', 'ped', 1.86, 'omni_fg2a05.npz'), ('base', 'soft', 0.98, 'vggt_fg2a05.npz'),
          ('base', 'ped', 0.99, 'vggt_fg2a05.npz')]
Z = {n: np.load(f'{CACHE}/{n}') for n in ('omni_fg2a05.npz', 'vggt_fg2a05.npz')}
GRID = np.exp(np.linspace(np.log(0.55), np.log(3.5), 30))
S = {}
for pri, fam, opt, npz in CURVES:
    B = basis(*Z[npz]['dp'].shape[1:])
    S[(pri, fam)] = np.array([push(Z[npz], lambda d, m=g, f=fam: FAM[f](d, m), B) for g in GRID])

def inv(pri, fam, target):
    v = S[(pri, fam)]; o = np.argsort(v)
    return np.interp(target, v[o], GRID[o]) if v.min() <= target <= v.max() else np.nan

print('  what the push reads at each curve\'s SWEPT optimum:')
for pri, fam, opt, npz in CURVES:
    o = np.argsort(S[(pri, fam)])
    print(f'    {pri} @{fam:<5} tag {opt:.2f} -> push {np.interp(opt, GRID, S[(pri,fam)]):.3f}')
print(f'\n  {"target":>8}' + ''.join(f'{p}@{f}'.rjust(11) for p, f, _, _ in CURVES) +
      f'{"mean err":>10}{"worst":>8}')
for t in np.arange(1.70, 2.16, 0.05):
    pr = [inv(p, f, t) for p, f, _, _ in CURVES]
    e = [abs(np.log(x/o)) for x, (_, _, o, _) in zip(pr, CURVES)]
    star = '  <-' if abs(t - 1.93) < 0.026 else ''
    print(f'  {t:8.2f}' + ''.join(f'{x:11.2f}' for x in pr) +
          f'{100*(np.exp(np.nanmean(e))-1):9.0f}%{100*(np.exp(np.nanmax(e))-1):7.0f}%{star}')
