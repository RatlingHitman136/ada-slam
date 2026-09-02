"""Re-measure every arm on disk from its RAW trajectory: Sim(3) ATE, the local-scale wobble
(sliding-window Sim3 scale), and the spread of the 100 m chord ratio.  Nothing is read from any
results.json - only traj_full.txt and the GT trajectory."""
import sys, os, glob, numpy as np
sys.path.insert(0, sys.path[0])
from geo import umeyama_sim3, load_gt_tum

# GT trajectory: KITTI by default, any scene through $JDSA_GT
gidx, gp = load_gt_tum(os.environ.get('JDSA_GT', 'data/KITTI/00/traj_tum.txt'))
W = 15
MIN_WIN_M = float(os.environ.get('JDSA_MIN_WIN_M', 3.0))   # GT path length a window must cover

def metrics(path):
    a = np.loadtxt(path)
    idx = a[:, 0].astype(int); X = a[:, 1:4]
    G = gp[idx][:, :3]
    n = len(idx)
    s, R, t = umeyama_sim3(X, G)
    err = np.linalg.norm((s*(R@X.T).T+t)-G, axis=1)
    # a window the platform barely moved through cannot identify a scale at all - on a scene
    # with stops (RELLIS) that is most of them, and their noise swamps the real signal
    step = np.concatenate([[0], np.linalg.norm(np.diff(G, axis=0), axis=1)])
    cum = np.cumsum(step)
    lam = np.full(n, np.nan)
    for k in range(n):
        i, j = max(0, k-W), min(n, k+W+1)
        if j-i >= 11 and cum[j-1] - cum[i] > MIN_WIN_M:
            lam[k] = umeyama_sim3(X[i:j], G[i:j])[0]/s
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1))])
    ch = []
    for i in range(n):
        j = np.searchsorted(dist, dist[i]+100.0)
        if j >= n: break
        ch.append(np.linalg.norm(X[j]-X[i])*s/max(np.linalg.norm(G[j]-G[i]), 1e-9))
    ch = np.array(ch) if ch else np.array([np.nan])
    return (np.sqrt((err**2).mean()), np.nanstd(lam), np.nanstd(ch), np.nanmean(ch),
            int(np.isfinite(lam).sum()))

for scene in sys.argv[1:]:
    rows = []
    for d in sorted(glob.glob(f'outputs/test/end2end/{scene}/*/')):
        f = os.path.join(d, 'traj_full.txt')
        if not os.path.exists(f):
            continue
        try:
            rows.append((os.path.basename(d.rstrip('/')),) + metrics(f))
        except Exception as e:
            print('skip', d, e)
    rows.sort(key=lambda r: r[1])
    print(f'\n=== {scene}   ({len(rows)} arms)   ATE / wobble = local-scale std / 100 m chord')
    print(f'  {"arm":<48}{"ATE":>7}{"wobble":>9}{"chord100":>18}{"n_win":>7}')
    for nm, ate, wob, cs, cm, n in rows:
        print(f'  {nm:<48}{ate:7.2f}{wob:9.4f}   {cm:6.3f} +- {cs:.3f}{n:8d}')
    A = np.array([[r[1], r[2]] for r in rows])
    if len(A) > 3:
        print(f'  corr(ATE, wobble) = {np.corrcoef(A[:,0], A[:,1])[0,1]:+.4f}   '
              f'ATE/wobble ratio: median {np.median(A[:,0]/A[:,1]):.1f} '
              f'p10 {np.percentile(A[:,0]/A[:,1],10):.1f} p90 {np.percentile(A[:,0]/A[:,1],90):.1f}')
