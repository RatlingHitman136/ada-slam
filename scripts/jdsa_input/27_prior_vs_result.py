"""How much of a prior's difference survives into the finished map?

Five extract runs, same scene, same window, same tracking config, priors that disagree wildly.
Each run's depth is put into metres by its OWN Sim(3) scale to the GT trajectory, so the arms are
comparable without borrowing anyone's gauge, and the comparison is made on the keyframes they
share.  The same statistic is applied twice: to what the priors SERVED (after JDSA's own alignment)
and to what the tracker FINISHED with.
"""
import os, sys, itertools, numpy as np
sys.path.insert(0, sys.path[0])
from geo import umeyama_sim3, c2w_from_w2c, load_gt_tum

CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
gp = load_gt_tum('data/KITTI/00/traj_tum.txt')[1]
ARMS = {'omni': 'omni_fg2a05', 'omni@ceil1.5': 'omni_ceil15_fg2a05', 'vggt': 'vggt_fg2a05',
        'vggt@ped1': 'insitu_ped1_f0k1k', 'vggt-adapted': 'adapt_fg2a05'}
BANDS = [(0, 10), (10, 20), (20, 40), (40, 80)]

def bilin(g, ht, wd):
    y = np.linspace(0, 1-1e-6, ht)[:, None]; x = np.linspace(0, 1-1e-6, wd)[None, :]
    return (1-y)*(1-x)*g[0,0] + (1-y)*x*g[0,1] + y*(1-x)*g[1,0] + y*x*g[1,1]

D = {}
for name, npz in ARMS.items():
    z = np.load(f'{CACHE}/{npz}.npz')
    ts, db, dp, ds = z['tstamp'], z['db'], z['dp'], z['dscales']
    K, ht, wd = db.shape
    X = np.array([c2w_from_w2c(p)[1] for p in z['poses']])
    s = umeyama_sim3(X, gp[ts][:, :3])[0]                    # SLAM units -> metres
    ate = np.sqrt((np.linalg.norm(umeyama_sim3(X, gp[ts][:, :3])[1] @ X.T * 0, axis=0)**2).mean())
    fused = s/np.maximum(db, 1e-9)                           # the finished map, metres
    served = np.stack([s/np.maximum(dp[k]*bilin(ds[k], ht, wd), 1e-9) for k in range(K)])
    D[name] = {'t': {int(t): i for i, t in enumerate(ts)}, 'fused': fused, 'served': served,
               'gt': z['gt'], 'valid': db > 0}
    print(f'  {name:<14} {K} keyframes, scale {s:.3f}, median served depth '
          f'{np.median(served[served > 0]):5.1f} m, median fused {np.median(fused):5.1f} m')

common = sorted(set.intersection(*[set(d['t']) for d in D.values()]))
print(f'\n  {len(common)} keyframes common to all five\n')

def compare(field):
    print(f'  === {field.upper()}: median |log ratio| between arms, as % (all valid pixels)')
    names = list(ARMS)
    print('  ' + ' '*15 + ''.join(n.rjust(15) for n in names))
    M = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j: continue
            acc = []
            for t in common:
                ia, ib = D[a]['t'][t], D[b]['t'][t]
                m = D[a]['valid'][ia] & D[b]['valid'][ib]
                if field == 'served':
                    m &= (D[a]['served'][ia] > 0) & (D[b]['served'][ib] > 0)
                if m.sum() < 100: continue
                acc.append(np.median(np.abs(np.log(D[a][field][ia][m]/D[b][field][ib][m]))))
            M[i, j] = M[j, i] = 100*(np.exp(np.mean(acc))-1)
        print(f'  {a:<15}' + ''.join((f'{M[i,j]:14.1f}%' if np.isfinite(M[i,j]) else f'{"-":>15}')
                                     for j in range(len(names))))
    return M

Mp = compare('served'); print(); Mf = compare('fused')
iu = np.triu_indices(len(ARMS), 1)
print(f'\n  served priors differ by {np.nanmean(Mp[iu]):.1f}% on average (worst pair '
      f'{np.nanmax(Mp[iu]):.1f}%)')
print(f'  finished maps differ by {np.nanmean(Mf[iu]):.1f}% on average (worst pair '
      f'{np.nanmax(Mf[iu]):.1f}%)')
print(f'  the tracker absorbs {100*(1-np.nanmean(Mf[iu])/np.nanmean(Mp[iu])):.0f}% of the '
      f'disagreement it was handed')

print('\n  === by GT range: how far apart the FINISHED maps are, and how far each is from lidar')
print(f'  {"band":<10}' + ''.join(f'{n:>15}' for n in ARMS) + f'{"spread of arms":>16}')
for lo, hi in BANDS:
    err, pair = [], []
    for name in ARMS:
        acc = []
        for t in common:
            i = D[name]['t'][t]
            g = D[name]['gt'][i]; f = D[name]['fused'][i]
            m = (g > 0) & (g >= lo) & (g < hi) & D[name]['valid'][i]
            if m.sum() > 20: acc.append(np.median(np.abs(np.log(f[m]/g[m]))))
        err.append(100*(np.exp(np.mean(acc))-1))
    for a, b in itertools.combinations(ARMS, 2):
        acc = []
        for t in common:
            ia, ib = D[a]['t'][t], D[b]['t'][t]
            g = D[a]['gt'][ia]
            m = (g > 0) & (g >= lo) & (g < hi) & D[a]['valid'][ia] & D[b]['valid'][ib]
            if m.sum() > 20:
                acc.append(np.median(np.abs(np.log(D[a]['fused'][ia][m]/D[b]['fused'][ib][m]))))
        pair.append(100*(np.exp(np.mean(acc))-1))
    print(f'  {lo}-{hi} m'.ljust(10) + ''.join(f'{e:14.1f}%' for e in err) +
          f'{np.mean(pair):15.1f}%')
print('    (first five columns: each arm against lidar; last: mean pairwise between arms)')

# ---------------------------------------------------------------------------------------------
# The numbers above are measured after ONE Sim(3) scale per run, so a run whose local gauge
# breathes inherits that as depth error everywhere. Re-measure with a scale fitted PER FRAME: what
# is left is the map's SHAPE, with the gauge taken out.
print('\n  === shape only: depth vs lidar after a PER-FRAME scale (median |log ratio|, %)')
print(f'  {"band":<10}' + ''.join(f'{n:>15}' for n in ARMS))
for lo, hi in BANDS:
    row = []
    for name in ARMS:
        acc = []
        for t in common:
            i = D[name]['t'][t]
            g = D[name]['gt'][i]; f = D[name]['fused'][i]
            v = (g > 0) & D[name]['valid'][i]
            m = v & (g >= lo) & (g < hi)
            if m.sum() > 20 and v.sum() > 100:
                s = np.exp(np.median(np.log(g[v]/f[v])))        # this frame's own scale
                acc.append(np.median(np.abs(np.log(s*f[m]/g[m]))))
        row.append(100*(np.exp(np.mean(acc))-1))
    print(f'  {lo}-{hi} m'.ljust(10) + ''.join(f'{x:14.1f}%' for x in row))

# ---------------------------------------------------------------------------------------------
# Two questions the numbers above raise. Is one scale per frame the RIGHT model - i.e. is the
# residual after it unbiased across range, or does a frame need a range-dependent correction?
# And how much do those per-frame scales differ from each other?
print('\n  === is the per-frame residual a BIAS or noise?  signed median log ratio, %')
print(f'  {"band":<10}' + ''.join(f'{n:>15}' for n in ARMS))
for lo, hi in BANDS:
    row = []
    for name in ARMS:
        acc = []
        for t in common:
            i = D[name]['t'][t]
            g = D[name]['gt'][i]; f = D[name]['fused'][i]
            v = (g > 0) & D[name]['valid'][i]
            m = v & (g >= lo) & (g < hi)
            if m.sum() > 20 and v.sum() > 100:
                s = np.exp(np.median(np.log(g[v]/f[v])))
                acc.append(np.median(np.log(s*f[m]/g[m])))
        row.append(100*(np.exp(np.mean(acc))-1))
    print(f'  {lo}-{hi} m'.ljust(10) + ''.join(f'{x:+14.1f}%' for x in row))

print('\n  === and how much do the PER-FRAME scales differ from each other?')
print(f'  {"arm":<16}{"scale std across frames":>26}{"p5-p95":>18}{"ATE":>8}')
import glob
for name, npz in ARMS.items():
    sc = []
    for t in common:
        i = D[name]['t'][t]
        g = D[name]['gt'][i]; f = D[name]['fused'][i]
        v = (g > 0) & D[name]['valid'][i]
        if v.sum() > 100:
            sc.append(np.median(g[v]/f[v]))
    sc = np.array(sc)
    print(f'  {name:<16}{100*sc.std()/sc.mean():25.1f}%'
          f'{np.percentile(sc,5)/sc.mean():9.2f}-{np.percentile(sc,95)/sc.mean():.2f}')

# ---------------------------------------------------------------------------------------------
# The shape comparison above lives on lidar-valid pixels - 56% of the frame, none of it sky. Where
# the arms' maps might actually differ is where BA is blind and the lidar cannot check. Compare the
# fused maps to each other by the TRACKER's own depth band, per frame scale removed, no GT needed.
print('\n  === where do the finished maps actually differ?  pairwise median |log ratio| of fused')
print('      depth, per-frame scale removed, binned by the tracker\'s own depth (x its frame median)')
TB = [(0, 1), (1, 1.5), (1.5, 2.5), (2.5, 5), (5, 1e9)]
names = list(ARMS)
print(f'  {"pair":<30}' + ''.join((f'{lo}-{hi}x' if hi < 1e8 else f'>{lo}x').rjust(11) for lo, hi in TB)
      + f'{"px share":>10}')
share = np.zeros(len(TB)); nsh = 0
for a, b in itertools.combinations(names, 2):
    acc = []
    for t in common:
        ia, ib = D[a]['t'][t], D[b]['t'][t]
        fa, fb = D[a]['fused'][ia], D[b]['fused'][ib]
        m0 = D[a]['valid'][ia] & D[b]['valid'][ib]
        if m0.sum() < 200: continue
        lr = np.log(fa[m0]/fb[m0]); lr = lr - np.median(lr)      # per-frame scale removed
        tz = fa[m0]/np.median(fa[m0])
        row = []
        for i, (lo, hi) in enumerate(TB):
            m = (tz >= lo) & (tz < hi)
            row.append(np.median(np.abs(lr[m])) if m.sum() > 20 else np.nan)
            if a == names[0] and b == names[1]:
                share[i] += m.mean()
        acc.append(row)
        if a == names[0] and b == names[1]: nsh += 1
    v = np.nanmean(acc, 0)
    print(f'  {a+" vs "+b:<30}' + ''.join(f'{100*(np.exp(x)-1):10.1f}%' for x in v))
print(f'  {"share of pixels":<30}' + ''.join(f'{100*x/nsh:10.1f}%' for x in share))
