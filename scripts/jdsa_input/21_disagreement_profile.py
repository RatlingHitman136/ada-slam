"""Automatic R, take two: five curve optima instead of three, and the prior/tracker disagreement
resolved by RANGE instead of collapsed into one number.

Everything here is available during a run - the served prior, the tracker's own disparity, and the
scale grid JDSA fits between them - so any rule that falls out costs no extra SLAM run.
"""
import os, sys, numpy as np
sys.path.insert(0, sys.path[0])
from geo import umeyama_sim3, load_gt_tum

CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
SCENE = 'outputs/test/end2end/kitti_00_fg2a05_f0-1000'
gp = load_gt_tum('data/KITTI/00/traj_tum.txt')[1]
# tracker depth in multiples of the tracker's OWN frame median
TB = [(0, .5), (.5, 1), (1, 1.5), (1.5, 2.5), (2.5, 1e9)]

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
FAM = {'ceil': ceil_clamp, 'ped': pedestal, 'soft': soft, None: lambda d, r: d}

def basis(ht, wd):
    y = np.linspace(0, 1-1e-6, ht)[:, None]; x = np.linspace(0, 1-1e-6, wd)[None, :]
    return np.stack([(1-y)*(1-x)*np.ones((ht, wd)), (1-y)*x*np.ones((ht, wd)),
                     y*(1-x)*np.ones((ht, wd)), y*x*np.ones((ht, wd))], -1).reshape(-1, 4)

NAMES = ['near .0-.5', 'near .5-1', 'mid 1-1.5', 'far 1.5-2.5', 'far >2.5',
         'near cost', 'tail p99', 'std(u)']

def battery(z, tf):
    dp, db = z['dp'], z['db']; K, ht, wd = dp.shape; B = basis(ht, wd); acc = []
    for k in range(K):
        p0 = dp[k].ravel(); t = db[k].ravel()
        v = (p0 > 0) & (t > 0)
        dep0 = 1.0/p0[v]; dep = tf(dep0)
        q = 1.0/np.maximum(dep, 1e-9)
        A = B[v]*q[:, None]
        s = np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@t[v])
        al = (B[v]@s)*q                                   # aligned served disparity
        tz = (1.0/t[v]) / np.median(1.0/t[v])              # tracker depth, its own median units
        row = [np.median(al[(tz >= lo) & (tz < hi)]/t[v][(tz >= lo) & (tz < hi)])
               if ((tz >= lo) & (tz < hi)).sum() > 20 else np.nan for lo, hi in TB]
        near = dep0 < np.median(dep0)
        row += [100*(1.0 - (dep[near]/dep0[near]).mean()),
                np.median(q)/np.percentile(q, 1),
                (q/np.median(q)).std()]
        acc.append(row)
    return np.nanmean(acc, 0)

CURVES = {
    ('omni', 'ceil'): ('omni_fg2a05.npz', [1.25, 1.35, 1.4, 1.45, 1.5, 2, 3]),
    ('omni', 'soft'): ('omni_fg2a05.npz', [1.2, 1.45, 1.55, 1.7]),
    ('omni', 'ped'):  ('omni_fg2a05.npz', [0.85, 1, 1.2, 1.4, 1.6, 1.8, 2]),
    ('base', 'soft'): ('vggt_fg2a05.npz', [0.8, 0.9, 1, 1.2]),
    ('base', 'ped'):  ('vggt_fg2a05.npz', [0.8, 0.9, 1, 1.1, 1.15, 1.25, 1.3, 1.5, 2]),
    ('base', 'ceil'): ('vggt_fg2a05.npz', [1.45, 1.5, 2]),
}
def arm(pri, fam, t): return f'{pri}_{fam}{t:g}'.replace('.', 'p')

Z = {}
rows = []
print('CURVES, and where each one turns over')
for (pri, fam), (npz, tags) in CURVES.items():
    Z.setdefault(npz, np.load(f'{CACHE}/{npz}'))
    a = np.array([ate(arm(pri, fam, t)) for t in tags]); tg = np.array(tags)
    i = int(np.argmin(a))
    print(f'  {pri:<5} @{fam:<5} ' + '  '.join(f'{t:g}:{x:.2f}' for t, x in zip(tg, a)))
    if 0 < i < len(tg)-1:
        j = slice(i-1, i+2)
        c = np.polyfit(np.log(tg[j]), a[j], 2)
        opt = float(np.clip(np.exp(-c[1]/(2*c[0])), tg[j].min(), tg[j].max())) if c[0] > 0 else tg[i]
        rows.append((pri, fam, opt, a.min(), battery(Z[npz], lambda d, f=fam, m=opt: FAM[f](d, m))))
        print(f'        optimum {opt:.2f} at {a[i]:.2f} m')
    else:
        print(f'        optimum NOT bracketed (best is an endpoint, {tg[i]:g})')

print(f'\nTHE BATTERY AT EACH BRACKETED OPTIMUM (aligned prior / tracker disparity by tracker depth)')
print(f'  {"statistic":<13}' + ''.join(f'{p} @{f}'.rjust(13) for p, f, _, _, _ in rows) + '   spread')
for i, n in enumerate(NAMES):
    v = np.array([r[4][i] for r in rows])
    print(f'  {n:<13}' + ''.join(f'{x:13.3f}' for x in v) +
          f'   {100*(v.max()-v.min())/abs(v.mean()):6.0f}%')
print('  ' + '-'*13 + ''.join(f'{r[3]:13.2f}' for r in rows) + '   ATE at optimum')
print('  ' + f'{"tag":<13}' + ''.join(f'{r[2]:13.2f}' for r in rows))

# ---------------------------------------------------------------------------------------------
# If that profile is the target, it should (a) explain every arm, not just the optima, and (b) say
# in advance which FAMILY a prior needs - a clamp cannot move the near bands at all.
TGT = np.median(np.array([r[4][:4] for r in rows]), axis=0)
print(f'\nTARGET PROFILE (median over the five optima): ' +
      '  '.join(f'{n} {t:.3f}' for n, t in zip(NAMES[:4], TGT)))

ALL = ([('omni', None, None)] +
       [('omni', f, t) for f, ts in (('ceil', [1.25, 1.35, 1.4, 1.45, 1.5, 2, 3]),
                                     ('soft', [1.2, 1.45, 1.55, 1.7]),
                                     ('ped', [0.6, 0.7, 0.85, 1, 1.2, 1.4, 1.6, 1.8, 2]))
        for t in ts] +
       [('base', None, None)] +
       [('base', f, t) for f, ts in (('ceil', [1.45, 1.5, 2]), ('soft', [0.8, 0.9, 1, 1.2]),
                                     ('ped', [0.8, 0.9, 1, 1.1, 1.15, 1.25, 1.3, 1.5, 2, 3]))
        for t in ts])
NPZ = {'omni': 'omni_fg2a05.npz', 'base': 'vggt_fg2a05.npz'}
out = []
for pri, fam, t in ALL:
    a = pri if fam is None else arm(pri, fam, t)
    if not os.path.isfile(f'{SCENE}/{a}/traj_full.txt'):
        continue
    b = battery(Z[NPZ[pri]], lambda d, f=fam, m=t: FAM[f](d, m))
    out.append((a, pri, ate(a), b[:4], float(np.mean(np.abs(np.log(b[:4]/TGT))))))

print(f'\n  {"arm":<16}{"ATE":>7}{"distance":>10}   ' +
      ''.join(n.rjust(13) for n in NAMES[:4]))
for a, pri, t, b, d in sorted(out, key=lambda r: r[2]):
    print(f'  {a:<16}{t:7.2f}{d:10.4f}   ' + ''.join(f'{x:13.3f}' for x in b))
A = np.array([[np.log(r[2]), r[4]] for r in out]); ob = np.array([r[1] == 'omni' for r in out])
for lab, m in (('both', np.ones(len(A), bool)), ('omni', ob), ('base', ~ob)):
    c = np.corrcoef(A[m, 1], A[m, 0])[0, 1]
    print(f'  corr(profile distance, log ATE) {lab:<5} {c:+.3f}   R2 {c*c:5.3f}   n={m.sum()}')

# ---------------------------------------------------------------------------------------------
# The four-band distance is dominated by bands a CLAMP cannot move, which is why it cannot order
# the clamp family.  Redo it with all five bands, per-band, and test transfer: fit the weights on
# ONE prior and predict the other - the only honest test of a rule meant for an unseen prior.
print('\n\nPER-BAND, AND WHETHER IT TRANSFERS BETWEEN PRIORS')
B5, y, ob2 = [], [], []
for pri, fam, t in ALL:
    a = pri if fam is None else arm(pri, fam, t)
    if not os.path.isfile(f'{SCENE}/{a}/traj_full.txt'):
        continue
    B5.append(battery(Z[NPZ[pri]], lambda d, f=fam, m=t: FAM[f](d, m))[:5])
    y.append(np.log(ate(a))); ob2.append(pri == 'omni')
B5, y, ob2 = np.array(B5), np.array(y), np.array(ob2)

def fit(X, m):
    A = np.column_stack([np.ones(m.sum()), X[m]])
    b, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    return b
def r2(X, b, m):
    A = np.column_stack([np.ones(m.sum()), X[m]])
    return 1 - ((y[m]-A@b)**2).sum()/((y[m]-y[m].mean())**2).sum()

both = np.ones(len(y), bool)
print(f'  {"band":<14}{"best target":>12}{"R2 both":>9}{"R2 omni":>9}{"R2 base":>9}')
feats = []
for i, n in enumerate(NAMES[:5]):
    s = B5[:, i]
    grid = np.linspace(np.nanmin(s), np.nanmax(s), 121)
    tgt, sc = max(((g, r2(np.abs(np.log(s/g))[:, None], fit(np.abs(np.log(s/g))[:, None], both), both))
                   for g in grid if g > 0), key=lambda z: z[1])
    feats.append(np.abs(np.log(s/tgt)))
    print(f'  {n:<14}{tgt:12.3f}{sc:9.3f}'
          f'{r2(feats[-1][:, None], fit(feats[-1][:, None], ob2), ob2):9.3f}'
          f'{r2(feats[-1][:, None], fit(feats[-1][:, None], ~ob2), ~ob2):9.3f}')
X = np.column_stack(feats)
b_all = fit(X, both)
print(f'\n  all five bands together:      R2 {r2(X, b_all, both):.3f} (39 arms, 6 parameters)')
b_o, b_b = fit(X, ob2), fit(X, ~ob2)
print(f'  fitted on OMNI  -> omni {r2(X, b_o, ob2):.3f}   applied to base {r2(X, b_o, ~ob2):+.3f}')
print(f'  fitted on BASE  -> base {r2(X, b_b, ~ob2):.3f}   applied to omni {r2(X, b_b, ob2):+.3f}')
# the two-term rule the mechanism argues for: far push + how much the NEAR field was moved
X2 = X[:, [0, 4]]
print(f'\n  near-band + far-band only:    R2 {r2(X2, fit(X2, both), both):.3f}'
      f'   omni-fit -> base {r2(X2, fit(X2, ob2), ~ob2):+.3f}'
      f'   base-fit -> omni {r2(X2, fit(X2, ~ob2), ob2):+.3f}')
