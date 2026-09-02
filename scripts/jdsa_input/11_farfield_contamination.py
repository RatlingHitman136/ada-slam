"""Does the far field contaminate the NEAR-field alignment?

JDSA models the prior/tracker ratio as ONE bilinear-in-image-position field s(x,y) (2x2 grid).
If the far field's ratio error is not bilinear, fitting it drags the near corners too.  Measured
here counterfactually: fit the same 4-parameter grid (a) to all valid pixels, (b) to near pixels
only, and report how much the near-field scale moves when the far field is let in - and, the part
that matters for a trajectory, how much that displacement VARIES from frame to frame.
"""
import sys, numpy as np

def basis(ht, wd):
    y = np.linspace(0, 1-1e-6, ht)[:, None]; x = np.linspace(0, 1-1e-6, wd)[None, :]
    return np.stack([((1-y)*(1-x))*np.ones((ht, wd)), (1-y)*x*np.ones((ht, wd)),
                     y*(1-x)*np.ones((ht, wd)), y*x*np.ones((ht, wd))], -1).reshape(-1, 4)

def fit(B, dp, tgt, m):
    A = B[m]*dp[m, None]
    return np.linalg.solve(A.T@A + 1e-12*np.eye(4), A.T@tgt[m])

def ceil_clamp(d, r): return np.minimum(d, r*np.median(d)) if r > 1 else d
def pedestal(d, r):
    q = 1.0/np.maximum(d, 1e-9); return 1.0/(q + np.median(q)/r)
def mask_far(d, r):
    o = d.copy(); o[o > r*np.median(d)] = 0.0; return o

TFS = [('raw', lambda d: d)]
TFS += [(f'ceil{r:g}', (lambda r: lambda d: ceil_clamp(d, r))(r)) for r in (3, 2, 1.5, 1.2)]
TFS += [(f'ped{r:g}',  (lambda r: lambda d: pedestal(d, r))(r)) for r in (3, 2, 1.35, 1)]
TFS += [(f'mask{r:g}', (lambda r: lambda d: mask_far(d, r))(r)) for r in (3, 2, 1.5)]

for path in sys.argv[1:]:
    z = np.load(path); dp, db = z['dp'], z['db']
    K, ht, wd = dp.shape
    B = basis(ht, wd)
    print(f'\n=== {path.split("/")[-1][:-4]}  (target: this run\'s own tracker disparity)')
    print(f'  {"transform":<10}{"near-scale shift":>18}{"frame-to-frame std":>20}'
          f'{"grid tilt t/b":>15}{"fit resid":>11}')
    for nm, tf in TFS:
        sh, tilt, res = [], [], []
        for k in range(K):
            d0 = 1.0/np.maximum(dp[k].ravel(), 1e-9); d0[dp[k].ravel() <= 0] = 0
            v0 = dp[k].ravel() > 0
            dep = tf(d0[v0])
            d = np.zeros(ht*wd); d[np.flatnonzero(v0)] = np.where(dep > 0, 1.0/np.maximum(dep, 1e-9), 0)
            t = db[k].ravel()
            m_all = (d > 0) & (t > 0)
            med = np.median(d[m_all])
            m_near = m_all & (d > med)            # nearer than the frame median
            if m_near.sum() < 200: continue
            s_all, s_near = fit(B, d, t, m_all), fit(B, d, t, m_near)
            fa, fn = B[m_near]@s_all, B[m_near]@s_near
            sh.append(np.median((fa-fn)/fn))
            tilt.append((s_all[0]+s_all[1])/(s_all[2]+s_all[3]))
            r = t[m_all] - (B[m_all]@s_all)*d[m_all]
            res.append(np.sqrt((r**2).mean())/np.median(t[m_all]))
        sh = np.array(sh)
        print(f'  {nm:<10}{np.median(sh):+18.4f}{sh.std():20.4f}{np.mean(tilt):15.3f}'
              f'{np.mean(res):11.4f}')
