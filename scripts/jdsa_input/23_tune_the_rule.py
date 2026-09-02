"""Squeeze the rule: which far-field threshold, and which target, picks the swept optimum on all
five curves - and does a target fitted on ONE prior transfer to the other?"""
import os, sys, numpy as np
sys.path.insert(0, sys.path[0])

CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
THR = [1.25, 1.5, 2.0, 3.0]          # tracker depth, in multiples of its own frame median

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

def push(z, tf, B):
    dp, db = z['dp'], z['db']; K = dp.shape[0]; acc = []
    for k in range(0, K, 2):
        p0 = dp[k].ravel(); t = db[k].ravel()
        v = (p0 > 0) & (t > 0)
        q = 1.0/np.maximum(tf(1.0/p0[v]), 1e-9)
        A = B[v]*q[:, None]
        s = np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@t[v])
        al = (B[v]@s)*q
        tz = (1.0/t[v]) / np.median(1.0/t[v])
        acc.append([np.median(al[tz >= th]/t[v][tz >= th]) if (tz >= th).sum() > 20 else np.nan
                    for th in THR])
    return np.nanmean(acc, 0)

CURVES = [('omni', 'ceil', 1.44, 'omni_fg2a05.npz'), ('omni', 'soft', 1.50, 'omni_fg2a05.npz'),
          ('omni', 'ped', 1.86, 'omni_fg2a05.npz'), ('base', 'soft', 0.98, 'vggt_fg2a05.npz'),
          ('base', 'ped', 0.99, 'vggt_fg2a05.npz')]
Z = {n: np.load(f'{CACHE}/{n}') for n in ('omni_fg2a05.npz', 'vggt_fg2a05.npz')}
GRID = np.exp(np.linspace(np.log(0.55), np.log(3.5), 30))

S, raw = {}, {}
for pri, fam, opt, npz in CURVES:
    z = Z[npz]; B = basis(*z['dp'].shape[1:])
    S[(pri, fam)] = np.array([push(z, lambda d, m=g, f=fam: FAM[f](d, m), B) for g in GRID])
    raw.setdefault(pri, push(z, lambda d: d, B))
print('  untransformed push, by far-field threshold:')
for p, v in raw.items():
    print(f'    {p:<5}' + ''.join(f'  >{th:g}x: {x:.2f}' for th, x in zip(THR, v)))

def invert(pri, fam, i, target):
    v = S[(pri, fam)][:, i]; ok = np.isfinite(v)
    g, v = GRID[ok], v[ok]; o = np.argsort(v)
    return np.interp(target, v[o], g[o]) if v.min() <= target <= v.max() else np.nan

print(f'\n  {"threshold":<11}{"target":>8}{"mean |log err|":>16}{"worst":>8}   per-curve prediction')
best = None
for i, th in enumerate(THR):
    lo = max(np.nanmin(S[k][:, i]) for k in S); hi = min(np.nanmax(S[k][:, i]) for k in S)
    if not np.isfinite(lo) or lo >= hi:
        print(f'  >{th:<10g}{"-":>8}{"no common range":>16}')
        continue
    cand = np.linspace(lo, hi, 200)
    def score(t):
        e = [abs(np.log(invert(p, f, i, t)/o)) for p, f, o, _ in CURVES]
        return np.nanmean(e), np.nanmax(e), e
    t_best = min(cand, key=lambda t: score(t)[0])
    m, w, e = score(t_best)
    print(f'  >{th:<10g}{t_best:8.2f}{m:16.3f}{w:8.3f}   ' +
          '  '.join(f'{p}@{f} {invert(p,f,i,t_best):.2f}/{o:.2f}' for (p, f, o, _) in CURVES))
    if best is None or m < best[0]:
        best = (m, th, t_best, i)

m, th, t, i = best
print(f'\n  BEST: threshold >{th:g}x tracker median depth, target {t:.2f}, mean tag error '
      f'{100*(np.exp(m)-1):.0f}%')
# transfer: fit the target on one prior, apply to the other
for fit_on in ('omni', 'base'):
    sub = [c for c in CURVES if c[0] == fit_on]
    lo = max(np.nanmin(S[(p, f)][:, i]) for p, f, _, _ in sub)
    hi = min(np.nanmax(S[(p, f)][:, i]) for p, f, _, _ in sub)
    t_sub = min(np.linspace(lo, hi, 200),
                key=lambda x: np.nanmean([abs(np.log(invert(p, f, i, x)/o)) for p, f, o, _ in sub]))
    oth = [c for c in CURVES if c[0] != fit_on]
    e_in = np.nanmean([abs(np.log(invert(p, f, i, t_sub)/o)) for p, f, o, _ in sub])
    e_out = np.nanmean([abs(np.log(invert(p, f, i, t_sub)/o)) for p, f, o, _ in oth])
    print(f'  target fitted on {fit_on:<5} = {t_sub:.2f}  ->  {100*(np.exp(e_in)-1):3.0f}% error on '
          f'{fit_on}, {100*(np.exp(e_out)-1):3.0f}% on the other prior')
