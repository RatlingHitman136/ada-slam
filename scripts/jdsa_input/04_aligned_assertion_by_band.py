"""What the prior asserts, after JDSA's OWN alignment.

JDSA fits a multiplicative scale field s (2x2 bilinear) by weighted least squares in DISPARITY,
with per-pixel weight alpha*far and Jacobian d(residual)/ds ~ d_prior, so the scale parameter's
information is w * d_prior^2.  Here that same fit is done against GT disparity, and the residual
is reported as a function of GT range: bias_B (systematic) and wander_B (cross-frame std) - the
second being what a trajectory gauge can drift on.
"""
import sys, numpy as np

BANDS = [(2, 5), (5, 10), (10, 20), (20, 40), (40, 80)]

def far_weight(dpv, fg):
    if fg <= 1.0:
        return np.ones_like(dpv)
    return np.clip(np.median(dpv)/np.clip(dpv, 1e-9, None), 1.0, fg)

def fit_scale(dp_k, tgt_k, w):
    """min_s sum w (tgt - s*dp)^2  ->  s = sum w dp tgt / sum w dp^2."""
    return (w*dp_k*tgt_k).sum()/max((w*dp_k*dp_k).sum(), 1e-30)

def run(path, fg=1.0, label=None):
    z = np.load(path)
    dp, db, gt = z['dp'], z['db'], z['gt']
    K = dp.shape[0]
    eps = np.full((K, len(BANDS)), np.nan)
    sg = np.zeros(K); info = np.zeros((K, len(BANDS)))
    for k in range(K):
        p = dp[k].ravel(); g = gt[k].ravel()
        v = (p > 0) & (g > 0)
        gdisp = 1.0/g[v]
        wf = far_weight(dp[k].ravel()[v], fg)
        s = fit_scale(p[v], gdisp, wf)          # JDSA's alignment, but onto GT
        sg[k] = s
        al = s*p[v]                              # aligned prior disparity
        r = g[v]
        for i, (lo, hi) in enumerate(BANDS):
            m = (r >= lo) & (r < hi)
            if m.sum() > 20:
                # ratio of aligned-prior DEPTH to GT depth in this band
                eps[k, i] = np.median((1.0/al[m])/r[m])
            info[k, i] = (wf[m]*p[v][m]**2).sum()
    info = info/np.maximum(info.sum(1, keepdims=True), 1e-30)
    nm = label or path.split('/')[-1][:-4]
    print(f'\n--- {nm}  far_gain={fg}  K={K}')
    print('   band        aligned-prior depth / GT     info share of the scale fit')
    for i, (lo, hi) in enumerate(BANDS):
        e = eps[:, i]; e = e[np.isfinite(e)]
        if len(e) < 20:
            print(f'   {lo:>3}-{hi:<3} m   (too few frames)'); continue
        print(f'   {lo:>3}-{hi:<3} m   bias {np.median(e):6.3f}   wander(std) {e.std():.3f}   '
              f'p10 {np.percentile(e,10):5.3f} p90 {np.percentile(e,90):5.3f}   '
              f'{info[:,i].mean()*100:5.1f}%   n={len(e)}')
    return eps, sg

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--fg')]
    fgs = [float(a[4:]) for a in sys.argv[1:] if a.startswith('--fg')] or [1.0]
    for p in args:
        for fg in fgs:
            run(p, fg)
