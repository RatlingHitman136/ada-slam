"""Build a small per-keyframe cache from a slam_depth.npz + KITTI GT depth, at the 1/8 JDSA grid."""
import sys, os, numpy as np, cv2

NPZ, OUT = sys.argv[1], sys.argv[2]
# the GT depth directory MUST match the dump's scene - the frame indices in `tstamp` index it, so
# the wrong directory silently produces a plausible-looking `gt` field for another sequence
GT = sys.argv[3] if len(sys.argv) > 3 else 'data/KITTI/00/depths'
PNG_SCALE = float(sys.argv[4]) if len(sys.argv) > 4 else 256.0

d = np.load(NPZ)
ts = d['tstamp'].astype(int)
dp = d['disps_prior'].astype(np.float64)   # prior disparity, prior units, 1/8 grid
db = d['disps'].astype(np.float64)         # BA disparity, SLAM units, 1/8 grid
ds = d['dscales'].astype(np.float64)       # 2x2 per-keyframe scale grid
poses = d['poses'].astype(np.float64)      # w2c, (tx,ty,tz,qx,qy,qz,qw)
intr = d['intrinsics'].astype(np.float64)  # full-stream res (848x256)
K, ht, wd = dp.shape
H, W = d['disps_up'].shape[1:]

gt = np.zeros((K, ht, wd), np.float64)
for i, t in enumerate(ts):
    g = cv2.imread(f'{GT}/{t:06d}.png', cv2.IMREAD_ANYDEPTH).astype(np.float64) / PNG_SCALE
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
    gt[i] = g[3::8, 3::8]

np.savez_compressed(OUT, tstamp=ts, dp=dp, db=db, dscales=ds, poses=poses,
                    intrinsics=intr, gt=gt, hw=np.array([H, W]))
print(OUT, K, ht, wd, 'gt valid frac', float((gt > 0).mean()))
