"""Can the tag be INFERRED instead of swept?

For every (prior, family) whose ATE-vs-tag curve is bracketed on disk, locate the optimum by a
parabola through its best three points, then evaluate a battery of statistics of the SERVED
distribution at that optimum.  A statistic that takes the same value at every curve's optimum is a
rule; one that does not, is not.  Everything in the battery is computable at run time - no GT
depth, no trajectory - because a rule that needs the answer is not a rule.
"""
import os, sys, numpy as np
sys.path.insert(0, sys.path[0])
from geo import umeyama_sim3, load_gt_tum

CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
SCENE = 'outputs/test/end2end/kitti_00_fg2a05_f0-1000'
gidx, gp = load_gt_tum('data/KITTI/00/traj_tum.txt')

def ate(arm):
    a = np.loadtxt(f'{SCENE}/{arm}/traj_full.txt')
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

NAMES = ['T95', 'T99', 'moved%', 'nearcost', 'floor', 'info>1.5x', 'push>1.5x', 'b/trackmed',
         'floor pct']

def stats(z, tf, B=None):
    dp, db = z['dp'], z['db']
    K, ht, wd = dp.shape
    B = basis(ht, wd) if B is None else B
    acc = []
    for k in range(K):
        p0 = dp[k].ravel(); t = db[k].ravel()
        v = (p0 > 0) & (t > 0)
        dep0 = 1.0/p0[v]
        dep = tf(dep0)
        q = 1.0/dep                                   # served disparity
        med_q = np.median(q)
        # the alignment JDSA would fit for THIS served field, against the tracker it recorded
        A = B[v]*q[:, None]
        s = np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@t[v])
        al = (B[v]@s)*q                               # aligned served disparity
        # FAR IS DEFINED BY THE TRACKER, not by the prior: the tracker's own map is what a rule
        # would have at run time, and a clamped prior has no tail of its own left to point at
        far = t[v] < (2.0/3.0)*np.median(t[v])
        near = dep0 < np.median(dep0)
        w = (q/med_q)**2
        acc.append([
            np.percentile(dep, 99.9)*0 + med_q/np.percentile(q, 5),       # T95 (depth p95/median)
            med_q/np.percentile(q, 1),                                    # T99
            (np.abs(dep-dep0) > 0.05*dep0).mean()*100,                    # moved%
            (1.0 - (dep[near]/dep0[near]).mean())*100,                    # near-half cost
            q.min()/med_q,                                                # served disparity floor
            w[q < 0.667*med_q].sum()/w.sum()*100,                         # info beyond 1.5x median
            np.median(al[far]/t[v][far]) if far.sum() > 20 else np.nan,   # push against tracker
            (np.median(q)/np.median(1.0/dep0) - 1) * np.median(al)/np.median(t[v]),  # b / tracker med
            (t[v] < np.median(al[far]) if far.sum() > 20 else t[v] < 0).mean()*100,  # floor pct
        ])
    return np.array(acc).mean(0)

CURVES = {
    ('omni', 'ceil'): ('omni_fg2a05.npz', [(1.25, 'omni_ceil1p25'), (1.35, 'omni_ceil1p35'),
        (1.4, 'omni_ceil1p4'), (1.45, 'omni_ceil1p45'), (1.5, 'omni_ceil1p5'),
        (2.0, 'omni_ceil2'), (3.0, 'omni_ceil3')]),
    ('omni', 'ped'): ('omni_fg2a05.npz', [(0.85, 'omni_ped0p85'), (1.0, 'omni_ped1'),
        (1.2, 'omni_ped1p2'), (1.4, 'omni_ped1p4'), (1.6, 'omni_ped1p6'), (1.8, 'omni_ped1p8'),
        (2.0, 'omni_ped2')]),
    ('base', 'ped'): ('vggt_fg2a05.npz', [(0.8, 'base_ped0p8'), (0.9, 'base_ped0p9'),
        (1.0, 'base_ped1'), (1.1, 'base_ped1p1'), (1.15, 'base_ped1p15'), (1.25, 'base_ped1p25'),
        (1.3, 'base_ped1p3'), (1.5, 'base_ped1p5'), (2.0, 'base_ped2')]),
}

rows = []
for (pri, fam), (npz, pts) in CURVES.items():
    z = np.load(f'{CACHE}/{npz}')
    tags = np.array([t for t, _ in pts]); ates = np.array([ate(a) for _, a in pts])
    i = int(np.argmin(ates))
    j = slice(max(0, i-1), min(len(tags), i+2))
    c = np.polyfit(np.log(tags[j]), ates[j], 2)
    t_opt = float(np.exp(-c[1]/(2*c[0]))) if c[0] > 0 else float(tags[i])
    t_opt = float(np.clip(t_opt, tags[j].min(), tags[j].max()))
    print(f'\n=== {pri} x @{fam}   measured curve')
    print('   ' + '  '.join(f'{t:g}:{a:.2f}' for t, a in zip(tags, ates)))
    print(f'   optimum tag {t_opt:.3f} (parabola through the best three; best measured '
          f'{tags[i]:g} at {ates[i]:.2f} m)')
    rows.append(((pri, fam, t_opt), stats(z, lambda d: FAM[fam](d, t_opt))))

print(f'\n{"statistic":<12}' + ''.join(f'{p} @{f} {t:.2f}'.rjust(20) for (p, f, t), _ in rows)
      + '   spread')
for i, n in enumerate(NAMES):
    v = np.array([r[1][i] for r in rows])
    sp = (v.max()-v.min())/np.abs(v.mean())
    print(f'{n:<12}' + ''.join(f'{x:20.3f}' for x in v) + f'   {100*sp:6.0f}%')

# ---------------------------------------------------------------------------------------------
# The candidate rule, applied where there is no sweep to read off: what does each prior ALREADY
# push, untransformed, and what tag would a push target of 1.9 ask for?
print('\n\nWHAT EACH PRIOR PUSHES UNTRANSFORMED, and the tag a push target of 1.9 would set')
print(f'  {"dump":<26}{"raw push":>10}{"@ceil tag":>11}{"@ped tag":>10}{"@soft tag":>11}')
for nm, npz in (('kitti omni', 'omni_fg2a05.npz'), ('kitti VGGT base', 'vggt_fg2a05.npz'),
                ('kitti VGGT adapted', 'adapt_fg2a05.npz'),
                ('kitti omni@ceil1.5 (run)', 'omni_ceil15_fg2a05.npz'),
                ('rellis omni', 'rellis_omni.npz')):
    z = np.load(f'{CACHE}/{npz}')
    raw = stats(z, lambda d: d)[6]
    out = []
    for fam in ('ceil', 'soft', 'ped'):
        lo, hi, hit = 1.02, 8.0, None
        if fam != 'ceil':
            lo = 0.2
        for _ in range(22):                      # bisection on a monotone statistic
            mid = np.sqrt(lo*hi)
            v = stats(z, lambda d, m=mid, f=fam: FAM[f](d, m))[6]
            if v > 1.9:
                lo = mid
            else:
                hi = mid
            hit = mid
        out.append(hit if abs(stats(z, lambda d, m=hit, f=fam: FAM[f](d, m))[6] - 1.9) < 0.25
                   else np.nan)
    print(f'  {nm:<26}{raw:10.2f}' + ''.join(f'{v:11.2f}' if np.isfinite(v) else f'{"none":>11}'
                                             for v in [out[0], out[2], out[1]]))

# ---------------------------------------------------------------------------------------------
# Does the push explain the WHOLE sweep, not just the three optima?  Every frozen arm on disk,
# each scored by the push its own transform would produce on its own prior's dump.
print('\n\nEVERY FROZEN ARM, ordered by measured ATE')
print(f'  {"arm":<18}{"push":>7}{"|log(push/1.9)|":>17}{"ATE":>8}')
ARMS = ([('omni', 'omni_fg2a05.npz', 'ceil', None)] +
        [(f'omni_ceil{t:g}'.replace('.', 'p'), 'omni_fg2a05.npz', 'ceil', t)
         for t in (1.25, 1.35, 1.4, 1.45, 1.5, 2, 3)] +
        [(f'omni_ped{t:g}'.replace('.', 'p'), 'omni_fg2a05.npz', 'ped', t)
         for t in (0.6, 0.7, 0.85, 1, 1.2, 1.4, 1.6, 1.8, 2)] +
        [('base', 'vggt_fg2a05.npz', 'ceil', None)] +
        [(f'base_ceil{t:g}'.replace('.', 'p'), 'vggt_fg2a05.npz', 'ceil', t) for t in (1.5, 2)] +
        [(f'base_ped{t:g}'.replace('.', 'p'), 'vggt_fg2a05.npz', 'ped', t)
         for t in (0.8, 0.9, 1, 1.1, 1.15, 1.25, 1.3, 1.5, 2, 3)])
Z = {n: np.load(f'{CACHE}/{n}') for n in ('omni_fg2a05.npz', 'vggt_fg2a05.npz')}
out = []
for arm, npz, fam, t in ARMS:
    if not os.path.isdir(f'{SCENE}/{arm}'):
        print(f'  {arm:<18} (no run on disk)'); continue
    tf = (lambda d: d) if t is None else (lambda d, m=t, f=fam: FAM[f](d, m))
    out.append((arm, stats(Z[npz], tf)[6], ate(arm)))
for arm, pu, a in sorted(out, key=lambda r: r[2]):
    print(f'  {arm:<18}{pu:7.2f}{abs(np.log(pu/1.9)):17.3f}{a:8.2f}')
A = np.array([[abs(np.log(p/1.9)), a] for _, p, a in out])
print(f'  corr(|log(push/1.9)|, ATE) = {np.corrcoef(A[:,0], A[:,1])[0,1]:+.3f}  over {len(A)} arms')
print(f'  corr(|log(push/1.9)|, log ATE) = '
      f'{np.corrcoef(A[:,0], np.log(A[:,1]))[0,1]:+.3f}')
import json
json.dump([[arm, round(float(p), 4), round(float(a), 3)] for arm, p, a in out],
          open('scripts/jdsa_input/report/push.json', 'w'))

# ---------------------------------------------------------------------------------------------
# Which statistic, at which target, explains the whole sweep?  For every statistic, fit
# log ATE ~ a + b |s - s*| with s* searched, and report R^2; then the best PAIR.
print('\n\nHOW WELL EACH STATISTIC EXPLAINS THE WHOLE SWEEP (30 arms)')
S = np.array([stats(Z[npz], (lambda d: d) if t is None else
                    (lambda d, m=t, f=fam: FAM[f](d, m)))
              for arm, npz, fam, t in ARMS if os.path.isdir(f'{SCENE}/{arm}')])
y = np.log(np.array([a for _, _, a in out]))
def r2(X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - ((y - X@b)**2).sum()/((y - y.mean())**2).sum()
best = {}
for i, n in enumerate(NAMES):
    s = S[:, i]
    if not np.isfinite(s).all():
        continue
    grid = np.linspace(s.min(), s.max(), 121)
    tgt, sc = max(((t, r2([np.abs(s-t)])) for t in grid), key=lambda z: z[1])
    best[n] = (tgt, sc, np.abs(s-tgt))
    print(f'  {n:<12} best target {tgt:8.3f}   R2 {sc:5.3f}')
print('\n  the best PAIRS:')
ks = list(best)
pairs = sorted(((r2([best[a][2], best[b][2]]), a, b) for i, a in enumerate(ks) for b in ks[i+1:]),
               reverse=True)[:4]
for sc, a, b in pairs:
    print(f'    {a} + {b:<12} R2 {sc:5.3f}   (alone {best[a][1]:.3f} / {best[b][1]:.3f})')

# ---------------------------------------------------------------------------------------------
# Could a controller hold the push at target ONLINE?  Only if one keyframe's push is a usable
# measurement rather than noise.
print('\n\nPER-KEYFRAME PUSH, for a controller that would have to read it live')
def per_kf_push(z, tf):
    dp, db = z['dp'], z['db']; K, ht, wd = dp.shape; B = basis(ht, wd); v_out = []
    for k in range(K):
        p0 = dp[k].ravel(); t = db[k].ravel(); v = (p0 > 0) & (t > 0)
        q = 1.0/tf(1.0/p0[v])
        A = B[v]*q[:, None]
        s = np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@t[v])
        al = (B[v]@s)*q
        far = t[v] < (2.0/3.0)*np.median(t[v])
        if far.sum() > 20:
            v_out.append(np.median(al[far]/t[v][far]))
    return np.array(v_out)
for nm, npz, fam, t in (('kitti omni raw', 'omni_fg2a05.npz', 'ceil', None),
                        ('kitti omni @ceil1.42', 'omni_fg2a05.npz', 'ceil', 1.42),
                        ('kitti omni@ceil1.5 (run)', 'omni_ceil15_fg2a05.npz', 'ceil', None),
                        ('kitti base raw', 'vggt_fg2a05.npz', 'ceil', None)):
    z = np.load(f'{CACHE}/{npz}')
    v = per_kf_push(z, (lambda d: d) if t is None else (lambda d, m=t, f=fam: FAM[f](d, m)))
    print(f'  {nm:<26} median {np.median(v):5.2f}  p10-p90 {np.percentile(v,10):5.2f}-'
          f'{np.percentile(v,90):5.2f}  frame-to-frame |diff| median '
          f'{np.median(np.abs(np.diff(v))):5.3f}  n={len(v)}')
