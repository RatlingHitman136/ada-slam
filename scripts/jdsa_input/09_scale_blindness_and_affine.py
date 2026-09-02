"""Two things at once:
 (1) the SCALE-BLINDNESS check - dscales x the served median disparity should be one SLAM gauge,
     whatever units the prior arrived in;
 (2) the affine anatomy - per frame, fit GT disparity ~ a*d_prior + b under JDSA's own weighting,
     and report b in units of the frame's median served disparity.  b<0 means the served disparity
     carries a PEDESTAL that the data wants removed; b>0 means the data wants one added.
"""
import sys, numpy as np

def bilin2x2(g, ht, wd):
    y = np.linspace(0, 1-1e-6, ht)[:, None]; x = np.linspace(0, 1-1e-6, wd)[None, :]
    return (1-y)*(1-x)*g[0,0] + (1-y)*x*g[0,1] + y*(1-x)*g[1,0] + y*x*g[1,1]

print(f'{"arm":<22}{"med(dp)":>9}{"mean dscale":>12}{"product":>9}{"med(db)":>9}   '
      f'{"b/med(dp)":>10}{"impl.sat":>9}  {"resid a-only":>13}{"resid affine":>13}')
for path in sys.argv[1:]:
    z = np.load(path)
    dp, db, gt, ds = z['dp'], z['db'], z['gt'], z['dscales']
    K, ht, wd = dp.shape
    med_dp = np.array([np.median(dp[k][dp[k] > 0]) for k in range(K)])
    med_db = np.array([np.median(db[k][db[k] > 0]) for k in range(K)])
    al_med = np.array([np.median((dp[k]*bilin2x2(ds[k], ht, wd))[dp[k] > 0]) for k in range(K)])
    bs, r1, r2 = [], [], []
    for k in range(K):
        p = dp[k].ravel(); g = gt[k].ravel()
        v = (p > 0) & (g > 0)
        y = 1.0/g[v]; x = p[v]
        w = np.ones_like(x)                      # JDSA weights each pixel equally (far_gain=1)
        # scale-only
        a1 = (w*x*y).sum()/(w*x*x).sum()
        r1.append(np.sqrt((w*(y-a1*x)**2).mean())/np.median(y))
        # affine
        A = np.array([[ (w*x*x).sum(), (w*x).sum()], [(w*x).sum(), w.sum()]])
        rhs = np.array([(w*x*y).sum(), (w*y).sum()])
        a2, b2 = np.linalg.solve(A, rhs)
        r2.append(np.sqrt((w*(y-a2*x-b2)**2).mean())/np.median(y))
        bs.append(b2/(a2*med_dp[k]))             # b expressed in the prior's own median-disparity
    bs = np.array(bs)
    sat = np.where(bs < 0, np.nan, 1.0/np.maximum(bs, 1e-9))
    print(f'{path.split("/")[-1][:-4]:<22}{med_dp.mean():9.4f}{ds.mean():12.4f}'
          f'{al_med.mean():9.4f}{med_db.mean():9.4f}   {np.median(bs):+10.4f}'
          f'{np.nanmedian(sat) if np.isfinite(sat).any() else float("nan"):9.2f}'
          f'  {np.mean(r1):13.4f}{np.mean(r2):13.4f}')
