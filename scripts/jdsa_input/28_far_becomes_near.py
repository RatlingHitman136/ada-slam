"""How can the far field tie two keyframes together, when nothing weights far pixels?

It does not tie them as far pixels.  It ties them because they ARE the near pixels of later
keyframes: under forward motion a point crosses the whole range in a few keyframes, and the
photometric edge that joins those keyframes has to reconcile the free depth it had when it was far
with the parallax-pinned depth it has once it is near.  Measured with GT poses and GT depth only -
no SLAM geometry, no prior.
"""
import os, sys, numpy as np
sys.path.insert(0, sys.path[0])
from geo import quat2R, load_gt_tum

CACHE = os.environ.get('JDSA_CACHE', '/storage/user/treh/adaslam_analysis/jdsa_input')
gp = load_gt_tum('data/KITTI/00/traj_tum.txt')[1]
z = np.load(f'{CACHE}/omni_fg2a05.npz')
gt, ts = z['gt'], z['tstamp']
fx, fy, cx, cy = z['intrinsics']
K, ht, wd = gt.shape
jj, ii = np.meshgrid(np.arange(wd), np.arange(ht))
xn, yn = (8*jj+3-cx)/fx, (8*ii+3-cy)/fy
FAR, NEAR = 2.5, 1.5          # multiples of the frame's own median GT depth

print(f'  keyframes {K}, mean spacing {np.mean(np.diff(ts)):.1f} frames')
print(f'\n  a pixel that is beyond {FAR}x the median depth in keyframe k - where is it at k+n?')
print(f'  {"n":>3}{"still in view":>15}{"median band":>14}{"reached <"+str(NEAR)+"x":>16}'
      f'{"mean GT travel":>16}')
for n in (1, 2, 3, 5, 8, 12, 20):
    inview, band, crossed, travel = [], [], [], []
    for a in range(0, K-n, 3):
        b = a+n
        Ra, ta = quat2R(gp[ts[a]][3:]), gp[ts[a]][:3]
        Rb, tb = quat2R(gp[ts[b]][3:]), gp[ts[b]][:3]
        ga = gt[a]; med_a = np.median(ga[ga > 0])
        v = (ga > FAR*med_a)
        if v.sum() < 30: continue
        Xa = np.stack([xn*ga, yn*ga, ga], -1)[v]
        Xb = ((Xa @ Ra.T + ta) - tb) @ Rb
        Z = Xb[:, 2]
        u = (fx*Xb[:, 0]/np.maximum(Z, 1e-6) + cx - 3)/8
        w_ = (fy*Xb[:, 1]/np.maximum(Z, 1e-6) + cy - 3)/8
        ok = (Z > 1) & (u >= 0) & (u < wd) & (w_ >= 0) & (w_ < ht)
        gb = gt[b]; med_b = np.median(gb[gb > 0])
        inview.append(ok.mean())
        if ok.sum() > 10:
            band.append(np.median(Z[ok])/med_b)
            crossed.append((Z[ok] < NEAR*med_b).mean())
        travel.append(np.linalg.norm(gp[ts[b]][:3] - gp[ts[a]][:3]))
    print(f'  {n:3d}{100*np.mean(inview):14.0f}%{np.mean(band):14.2f}x'
          f'{100*np.mean(crossed):15.0f}%{np.mean(travel):14.1f} m')

print(f'\n  and the converse - of the pixels in the WELL-WEIGHTED band (0.5-{NEAR}x) at keyframe k,')
print(f'  what share were beyond {FAR}x in an earlier keyframe within the last n?')
print(f'  {"n":>3}{"share arriving from the far band":>36}')
for n in (3, 5, 8, 12, 20):
    share = []
    for b in range(n, K, 3):
        gb = gt[b]; med_b = np.median(gb[gb > 0])
        tgt = (gb > 0.5*med_b) & (gb < NEAR*med_b)
        if tgt.sum() < 50: continue
        Rb, tb = quat2R(gp[ts[b]][3:]), gp[ts[b]][:3]
        Xb = np.stack([xn*gb, yn*gb, gb], -1)[tgt]
        hit = np.zeros(len(Xb), bool)
        for a in range(max(0, b-n), b):
            Ra, ta = quat2R(gp[ts[a]][3:]), gp[ts[a]][:3]
            Xa = ((Xb @ Rb.T + tb) - ta) @ Ra
            Z = Xa[:, 2]
            u = (fx*Xa[:, 0]/np.maximum(Z, 1e-6) + cx - 3)/8
            w_ = (fy*Xa[:, 1]/np.maximum(Z, 1e-6) + cy - 3)/8
            ga = gt[a]; med_a = np.median(ga[ga > 0])
            hit |= (Z > FAR*med_a) & (u >= 0) & (u < wd) & (w_ >= 0) & (w_ < ht)
        share.append(hit.mean())
    print(f'  {n:3d}{100*np.mean(share):35.0f}%')
