import sys, os, glob, json, numpy as np
sys.path.insert(0, sys.path[0])
from geo import umeyama_sim3, load_gt_tum
gidx, gp = load_gt_tum('data/KITTI/00/traj_tum.txt')
W = 15
out = []
for scene in ('kitti_00_fg2a05_f0-1000', 'kitti_00_f0-1000'):
    for d in sorted(glob.glob(f'outputs/test/end2end/{scene}/*/')):
        f = os.path.join(d, 'traj_full.txt')
        if not os.path.exists(f): continue
        a = np.loadtxt(f); idx = a[:,0].astype(int); X = a[:,1:4]; G = gp[idx][:,:3]
        s, R, t = umeyama_sim3(X, G)
        e = np.linalg.norm((s*(R@X.T).T+t)-G, axis=1)
        lam = [umeyama_sim3(X[max(0,k-W):k+W+1], G[max(0,k-W):k+W+1])[0]/s for k in range(len(X))]
        out.append({'scene': 'a05fg2' if 'fg2a05' in scene else 'a01',
                    'arm': os.path.basename(d.rstrip('/')),
                    'ate': round(float(np.sqrt((e**2).mean())), 3),
                    'wob': round(float(np.nanstd(lam)), 4)})
json.dump(out, open(sys.argv[1], 'w'))
print(len(out), 'arms')
