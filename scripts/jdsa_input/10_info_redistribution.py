"""The whole frame, not just the lidar-covered part: how a served-depth transform redistributes
JDSA's alignment information.  The scale parameter's Fisher weight is w*d_prior^2, so a transform
that raises far disparities raises their LEVERAGE quadratically - which is a different mechanism
from making them more accurate."""
import sys, numpy as np

BINS = [0, .5, 1, 1.5, 2, 3, 5, 10, 1e9]      # prior depth in multiples of the frame median

def ceil_clamp(depth, r):
    return np.minimum(depth, r*np.median(depth)) if r and r > 1.0 else depth

def pedestal(depth, r):
    if r is None: return depth
    q = 1.0/np.maximum(depth, 1e-9)
    return 1.0/(q + np.median(q)/r)

def mask_far(depth, r):
    """the third option: say NOTHING beyond r x the frame median (JDSA's m goes to 0 there)."""
    d = depth.copy(); d[d > r*np.median(depth)] = 0.0
    return d

def profile(dp, tf, fg=1.0):
    K, ht, wd = dp.shape
    px = np.zeros(len(BINS)-1); inf = np.zeros(len(BINS)-1); touched = 0.0; floor = []
    for k in range(K):
        d0 = dp[k].ravel(); v = d0 > 0
        dep = 1.0/d0[v]
        dep2 = tf(dep)
        keep = dep2 > 0
        u = dep2[keep]/np.median(dep[keep] if keep.all() else dep)    # median of the PRE-transform
        q = 1.0/dep2[keep]
        med_q = np.median(q)
        w = np.ones_like(q)
        if fg > 1.0:
            w = np.clip(med_q/q, 1.0, fg)
        wi = w*(q/med_q)**2
        touched += (np.abs(dep2[keep] - dep[keep]) > 1e-6*dep[keep]).mean() + (~keep).mean()
        floor.append(q.min()/med_q)
        idx = np.digitize(u, BINS) - 1
        for i in range(len(BINS)-1):
            m = idx == i
            px[i] += m.mean(); inf[i] += wi[m].sum()/wi.sum()
    return px/K, inf/K, touched/K, float(np.mean(floor))

name = {'raw': lambda d: d}
for r in (3.0, 2.0, 1.5, 1.2):
    name[f'ceil{r:g}'] = (lambda r: (lambda d: ceil_clamp(d, r)))(r)
for r in (3.0, 2.0, 1.35, 1.0, 0.5):
    name[f'ped{r:g}'] = (lambda r: (lambda d: pedestal(d, r)))(r)
for r in (3.0, 2.0, 1.5):
    name[f'mask{r:g}'] = (lambda r: (lambda d: mask_far(d, r)))(r)

for path in sys.argv[1:]:
    z = np.load(path); dp = z['dp']
    print(f'\n=== {path.split("/")[-1][:-4]}   pixel share / INFO share of the JDSA scale fit, '
          f'by served depth in multiples of the frame median')
    hdr = ''.join(f'{"<"+str(BINS[i+1]) if i==0 else (str(BINS[i])+"-"+str(BINS[i+1]) if BINS[i+1]<1e8 else ">"+str(BINS[i])):>11}'
                  for i in range(len(BINS)-1))
    print(f'  {"transform":<12}{"floor":>7}{"touched":>8}  {hdr}')
    for nm, tf in name.items():
        px, inf, tch, fl = profile(dp, tf)
        cells = ''.join(f'{p*100:5.1f}/{i*100:5.1f}' for p, i in zip(px, inf))
        print(f'  {nm:<12}{fl:7.3f}{tch*100:7.1f}%  {cells}')
