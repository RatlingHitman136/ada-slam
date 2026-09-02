"""Placement table for the proposed arms: what each candidate transform would actually serve,
measured on the priors the dumps recorded, so the sweep points are chosen rather than guessed."""
import sys, os, numpy as np
# the per-keyframe caches 01_build_cache.py writes (86 MB, storage partition; $JDSA_CACHE overrides)
CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
BANDS = [(5, 10), (10, 20), (20, 40), (40, 80)]

def ceil_clamp(d, r): return np.minimum(d, r*np.median(d))
def pedestal(d, r):
    q = 1.0/np.maximum(d, 1e-9); return 1.0/(q + np.median(q)/r)
def soft(d, r):
    q = 1.0/np.maximum(d, 1e-9); return 1.0/np.sqrt(q*q + (np.median(q)/r)**2)
def mask(d, r):
    o = d.copy(); o[o > r*np.median(d)] = 0.0; return o

def row(dp, gt, tf):
    K = dp.shape[0]; out = []
    for k in range(K):
        d0 = dp[k].ravel(); g = gt[k].ravel()
        v = (d0 > 0) & (g > 0)
        dep0 = 1.0/d0[v]
        dep = tf(dep0)
        keep = dep > 0
        q = 1.0/dep[keep]; y = 1.0/g[v][keep]; r = g[v][keep]
        s = (q*y).sum()/(q*q).sum()
        al = 1.0/(s*q)
        near = dep0[keep] < np.median(dep0)
        out.append([np.median(al[(r >= lo) & (r < hi)]/r[(r >= lo) & (r < hi)])
                    if ((r >= lo) & (r < hi)).sum() > 20 else np.nan for lo, hi in BANDS]
                   + [np.percentile(al, 99), 1.0 - (dep[keep][near]/dep0[keep][near]).mean(),
                      1.0 - keep.mean()])
    return np.nanmedian(out, 0)

SPECS = [('raw', lambda d: d)]
SPECS += [(f'@ceil{r:g}', (lambda r: lambda d: ceil_clamp(d, r))(r)) for r in (1.45, 1.2)]
SPECS += [(f'@soft{r:g}', (lambda r: lambda d: soft(d, r))(r))
          for r in (2.0, 1.7, 1.45, 1.2, 1.0, 0.8)]
SPECS += [(f'@ped{r:g}', (lambda r: lambda d: pedestal(d, r))(r)) for r in (2.0, 1.8, 1.0)]
SPECS += [(f'@mask{r:g}', (lambda r: lambda d: mask(d, r))(r)) for r in (1.45, 1.2)]

for pri, npz in (('omni', 'omni_fg2a05.npz'), ('base', 'vggt_fg2a05.npz')):
    z = np.load(f'{CACHE}/{npz}'); dp, gt = z['dp'], z['gt']
    print(f'\n=== {pri}: what the transform would serve   (assert = aligned depth / true depth)')
    print(f'  {"spec":<10}' + ''.join(f'{lo}-{hi}m'.rjust(9) for lo, hi in BANDS) +
          f'{"p99 (m)":>9}{"near cost":>11}{"deleted":>9}')
    for nm, tf in SPECS:
        v = row(dp, gt, tf)
        print(f'  {nm:<10}' + ''.join(f'{x:9.3f}' for x in v[:4]) +
              f'{v[4]:9.1f}{100*v[5]:10.1f}%{100*v[6]:8.1f}%')
