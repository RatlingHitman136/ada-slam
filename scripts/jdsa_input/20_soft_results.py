"""The @soft / @mask batch, read against the rule that chose its tags.

Every arm in the scene, re-measured from its raw trajectory (ATE + wobble), joined to the
statistics of the field it was served: the push the tag realises, what the near field paid, and
where the served tail ended up.  The three columns are what the batch was run to separate.
"""
import os, sys, numpy as np
sys.path.insert(0, sys.path[0])
from geo import umeyama_sim3, load_gt_tum

CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
SCENE = 'outputs/test/end2end/kitti_00_fg2a05_f0-1000'
gidx, gp = load_gt_tum('data/KITTI/00/traj_tum.txt')
W, MIN_WIN_M = 15, 3.0

def ate_wobble(arm):
    a = np.loadtxt(f'{SCENE}/{arm}/traj_full.txt')
    X, G = a[:, 1:4], gp[a[:, 0].astype(int)][:, :3]
    s, R, t = umeyama_sim3(X, G)
    e = np.linalg.norm((s*(R@X.T).T+t)-G, axis=1)
    cum = np.cumsum(np.concatenate([[0], np.linalg.norm(np.diff(G, axis=0), axis=1)]))
    lam = [umeyama_sim3(X[max(0,k-W):k+W+1], G[max(0,k-W):k+W+1])[0]/s
           if cum[min(len(X), k+W+1)-1] - cum[max(0, k-W)] > MIN_WIN_M else np.nan
           for k in range(len(X))]
    return float(np.sqrt((e**2).mean())), float(np.nanstd(lam))

def ceil_clamp(d, r): return np.minimum(d, r*np.median(d))
def pedestal(d, r):
    q = 1.0/np.maximum(d, 1e-9); return 1.0/(q + np.median(q)/r)
def soft(d, r):
    q = 1.0/np.maximum(d, 1e-9); return 1.0/np.sqrt(q*q + (np.median(q)/r)**2)
def mask(d, r):
    o = d.copy(); o[o > r*np.median(d)] = 0.0; return o
FAM = {'ceil': ceil_clamp, 'ped': pedestal, 'soft': soft, 'mask': mask, None: lambda d, r: d}

def basis(ht, wd):
    y = np.linspace(0, 1-1e-6, ht)[:, None]; x = np.linspace(0, 1-1e-6, wd)[None, :]
    return np.stack([(1-y)*(1-x)*np.ones((ht, wd)), (1-y)*x*np.ones((ht, wd)),
                     y*(1-x)*np.ones((ht, wd)), y*x*np.ones((ht, wd))], -1).reshape(-1, 4)

def served(z, tf):
    """push, near-half cost %, served p99/median depth, % of the tracker's far pixels left mute."""
    dp, db = z['dp'], z['db']; K, ht, wd = dp.shape; B = basis(ht, wd); acc = []
    for k in range(K):
        p0 = dp[k].ravel(); t = db[k].ravel()
        v = (p0 > 0) & (t > 0)
        dep0 = 1.0/p0[v]; dep = tf(dep0)
        live = dep > 0
        q = np.where(live, 1.0/np.maximum(dep, 1e-9), 0.0)
        A = (B[v]*q[:, None])[live]
        s = np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@t[v][live])
        al = (B[v]@s)*q
        far = t[v] < (2.0/3.0)*np.median(t[v])
        near = dep0 < np.median(dep0)
        acc.append([np.median(al[far & live]/t[v][far & live]) if (far & live).sum() > 20 else np.nan,
                    100*(1.0 - (dep[near & live]/dep0[near & live]).mean()),
                    np.median(q[live])/np.percentile(q[live], 1),
                    100*(far & ~live).sum()/max(far.sum(), 1)])
    return np.nanmean(acc, 0)

ARMS = [('omni', None, None, 'omni_fg2a05.npz')]
ARMS += [(f'omni_{f}{t:g}'.replace('.', 'p'), f, t, 'omni_fg2a05.npz') for f, t in
         [('ceil', 1.25), ('ceil', 1.35), ('ceil', 1.4), ('ceil', 1.45), ('ceil', 1.5),
          ('ceil', 2), ('ceil', 3), ('soft', 1.2), ('soft', 1.45), ('soft', 1.7),
          ('ped', 0.6), ('ped', 0.7), ('ped', 0.85), ('ped', 1), ('ped', 1.2), ('ped', 1.4),
          ('ped', 1.6), ('ped', 1.8), ('ped', 2), ('mask', 1.45)]]
ARMS += [('base', None, None, 'vggt_fg2a05.npz')]
ARMS += [(f'base_{f}{t:g}'.replace('.', 'p'), f, t, 'vggt_fg2a05.npz') for f, t in
         [('ceil', 1.45), ('ceil', 1.5), ('ceil', 2), ('soft', 1), ('soft', 1.2),
          ('ped', 0.8), ('ped', 0.9), ('ped', 1), ('ped', 1.1), ('ped', 1.15), ('ped', 1.25),
          ('ped', 1.3), ('ped', 1.5), ('ped', 2), ('ped', 3), ('mask', 1.45)]]

Z = {n: np.load(f'{CACHE}/{n}') for n in ('omni_fg2a05.npz', 'vggt_fg2a05.npz')}
rows = []
for arm, fam, t, npz in ARMS:
    if not os.path.isfile(f'{SCENE}/{arm}/traj_full.txt'):
        continue
    a, w = ate_wobble(arm)
    pu, nc, t99, mute = served(Z[npz], lambda d, f=fam, m=t: FAM[f](d, m))
    rows.append((arm, fam or 'raw', a, w, pu, nc, t99, mute))

print(f'  {"arm":<16}{"family":>7}{"ATE":>7}{"wobble":>8}{"push":>7}{"near cost":>10}'
      f'{"tail p99":>9}{"mute far":>9}')
for r in sorted(rows, key=lambda r: r[2]):
    new = ' *' if ('soft' in r[0] or 'mask' in r[0] or r[0] == 'base_ceil1p45') else '  '
    print(f'{new}{r[0]:<16}{r[1]:>7}{r[2]:7.2f}{r[3]:8.4f}{r[4]:7.2f}{r[5]:9.1f}%'
          f'{r[6]:9.2f}{r[7]:8.1f}%')

A = np.array([[r[2], r[4], r[5]] for r in rows if 'mask' not in r[0]])
o = np.array(['omni' in r[0] and 'mask' not in r[0] for r in rows if 'mask' not in r[0]])
for lab, m in (('all', np.ones(len(A), bool)), ('omni', o), ('base', ~o)):
    print(f'\n  {lab:<5} n={m.sum():2d}  corr(log ATE, |log push/1.9|) '
          f'{np.corrcoef(np.log(A[m,0]), np.abs(np.log(A[m,1]/1.9)))[0,1]:+.3f}'
          f'   corr(log ATE, near cost) {np.corrcoef(np.log(A[m,0]), A[m,2])[0,1]:+.3f}')

# ---------------------------------------------------------------------------------------------
# With the new family in hand, does ANY statistic - even one that needs the lidar - order both
# priors at once?  Same protocol as before: log ATE ~ a + b|s - s*| with s* searched.
print('\n\nWHICH STATISTIC ORDERS BOTH PRIORS (34 transform arms; masks excluded)')
BANDS = [(5, 10), (10, 20), (20, 40)]
def full_stats(z, tf):
    dp, db, gt = z['dp'], z['db'], z['gt']; K, ht, wd = dp.shape; B = basis(ht, wd); acc = []
    for k in range(K):
        p0 = dp[k].ravel(); t = db[k].ravel(); g = gt[k].ravel()
        v = (p0 > 0) & (t > 0)
        dep0 = 1.0/p0[v]; dep = tf(dep0); live = dep > 0
        q = np.where(live, 1.0/np.maximum(dep, 1e-9), 0.0)
        A = (B[v]*q[:, None])[live]
        s = np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@t[v][live])
        al = (B[v]@s)*q
        far = t[v] < (2.0/3.0)*np.median(t[v]); near = dep0 < np.median(dep0)
        u = q[live]/np.median(q[live])
        # the lidar-based rows: what the served field ASSERTS, aligned to GT the way JDSA aligns
        gv = g[v] > 0
        qq, yy = q[live & gv], 1.0/g[v][live & gv]
        sc = (qq*yy).sum()/max((qq*qq).sum(), 1e-30)
        asrt = 1.0/(sc*qq); rr = g[v][live & gv]
        row = [np.median(al[far & live]/t[v][far & live]) if (far & live).sum() > 20 else np.nan,
               100*(1.0 - (dep[near & live]/dep0[near & live]).mean()),
               np.median(q[live])/np.percentile(q[live], 1),
               u.std(), q[live].min()/np.median(q[live])]
        for lo, hi in BANDS:
            m = (rr >= lo) & (rr < hi)
            row.append(np.median(asrt[m]/rr[m]) if m.sum() > 20 else np.nan)
        acc.append(row)
    return np.nanmean(acc, 0)

NM = ['push', 'near cost', 'tail p99', 'std(u)', 'floor',
      'assert 5-10m', 'assert 10-20m', 'assert 20-40m']
S, y, isb = [], [], []
for arm, fam, t, npz in ARMS:
    if not os.path.isfile(f'{SCENE}/{arm}/traj_full.txt') or fam == 'mask':
        continue
    S.append(full_stats(Z[npz], lambda d, f=fam, m=t: FAM[f](d, m)))
    y.append(ate_wobble(arm)[0]); isb.append(arm.startswith('base'))
S, y, isb = np.array(S), np.log(np.array(y)), np.array(isb)

def r2(cols, m=None):
    m = np.ones(len(y), bool) if m is None else m
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in cols])
    b, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    return 1 - ((y[m] - X@b)**2).sum()/((y[m] - y[m].mean())**2).sum()

best = {}
print(f'  {"statistic":<15}{"target":>9}{"R2 both":>9}{"R2 omni":>9}{"R2 base":>9}')
for i, n in enumerate(NM):
    s = S[:, i]
    if not np.isfinite(s).all():
        continue
    grid = np.linspace(s.min(), s.max(), 121)
    tgt, sc = max(((t, r2([np.abs(s-t)])) for t in grid), key=lambda z: z[1])
    best[n] = np.abs(s-tgt)
    print(f'  {n:<15}{tgt:9.3f}{sc:9.3f}{r2([np.abs(s-tgt)], ~isb):9.3f}'
          f'{r2([np.abs(s-tgt)], isb):9.3f}')
ks = list(best)
print('  best pairs over both priors: ' + ', '.join(
    f'{a}+{b} {sc:.2f}' for sc, a, b in sorted(
        ((r2([best[a], best[b]]), a, b) for i, a in enumerate(ks) for b in ks[i+1:]),
        reverse=True)[:3]))
