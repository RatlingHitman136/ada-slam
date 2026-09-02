"""Descriptive: what disparity distribution does JDSA actually receive, and where does its
scale fit take its information from?  All per-keyframe, at the 1/8 grid JDSA runs on."""
import sys, numpy as np

def bilin2x2(g, ht, wd):
    """Reproduce get_prior_depth_aligned's bilinear lift of the 2x2 grid to (ht,wd)."""
    y = np.linspace(0, 1 - 1e-6, ht)[:, None]
    x = np.linspace(0, 1 - 1e-6, wd)[None, :]
    return ((1-y)*(1-x)*g[0,0] + (1-y)*x*g[0,1] + y*(1-x)*g[1,0] + y*x*g[1,1])

for path in sys.argv[1:]:
    z = np.load(path)
    dp, db, gt, ds = z['dp'], z['db'], z['gt'], z['dscales']
    K, ht, wd = dp.shape
    name = path.split('/')[-1].replace('.npz','')

    # ---- prior disparity shape, per frame, in units of that frame's own median ----
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    P = np.zeros((K, len(qs))); G = np.zeros((K, len(qs))); B = np.zeros((K, len(qs)))
    frac_info = np.zeros((K, 5)); cover = np.zeros(K)
    bins = [0, 10, 20, 40, 80, 1e9]
    for k in range(K):
        p = dp[k].ravel(); b = db[k].ravel(); g = gt[k].ravel()
        m = (p > 0) & (b > 0)
        P[k] = np.percentile(p[m]/np.median(p[m]), qs)
        B[k] = np.percentile(b[m]/np.median(b[m]), qs)
        v = m & (g > 0)
        gd = 1.0/g[v]
        G[k] = np.percentile(gd/np.median(gd), qs)
        cover[k] = v.mean()
        # JDSA's scale-fit information per pixel is (dJso/ds)^2 ~ dp^2 : where does it live?
        w = p[v]**2
        r = g[v]
        for i in range(5):
            frac_info[k, i] = w[(r >= bins[i]) & (r < bins[i+1])].sum()/w.sum()

    print(f'\n=== {name}   K={K}  GT coverage of the 1/8 grid {cover.mean()*100:.1f}%')
    print('  per-frame disparity quantiles / that frame\'s median   (mean over keyframes)')
    hdr = '   '.join(f'q{q:<3}' for q in qs)
    print(f'    {"":10}{hdr}')
    for lbl, A in (('prior', P), ('BA', B), ('GT(lidar)', G)):
        print(f'    {lbl:10}' + '  '.join(f'{v:6.3f}' for v in A.mean(0)))
    # implied hard depth ceiling: a prior whose min disparity is f x its median saturates there
    print(f'    prior min-disparity pedestal: {P[:,0].mean():.4f} x median disparity '
          f'-> depths saturate at {1/max(P[:,0].mean(),1e-9):.2f} x median depth '
          f'(per-frame range {1/P[:,0].max():.1f}..{1/max(P[:,0].min(),1e-9):.1f})')
    print(f'    GT   min-disparity floor    : {G[:,0].mean():.4f} x median  '
          f'-> {1/max(G[:,0].mean(),1e-9):.2f} x median depth (lidar-limited)')
    print('  where JDSA\'s scale fit takes its information (mean share of sum dp^2 by GT range)')
    print('    <10m {:.3f} | 10-20m {:.3f} | 20-40m {:.3f} | 40-80m {:.3f} | >80m {:.3f}'
          .format(*frac_info.mean(0)))
    # the fitted grid itself
    dsm = ds[:K]
    print(f'  fitted dscales: mean {dsm.mean():.4f}  top-row/bottom-row '
          f'{dsm[:,0].mean()/dsm[:,1].mean():.4f}  left/right {dsm[:,:,0].mean()/dsm[:,:,1].mean():.4f}'
          f'  spread(max-min)/mean {np.mean((dsm.max((1,2))-dsm.min((1,2)))/dsm.mean((1,2))):.4f}')
