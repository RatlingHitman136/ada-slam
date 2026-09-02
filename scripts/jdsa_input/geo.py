import numpy as np

def quat2R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def c2w_from_w2c(p):
    """p = (tx,ty,tz,qx,qy,qz,qw) world->cam  ->  (R_wc, t_wc) camera centre in world."""
    R = quat2R(p[3:7]); t = p[:3]
    Rw = R.T; tw = -Rw @ t
    return Rw, tw

def umeyama_sim3(X, Y):
    """scale s, R, t minimising |Y - (s R X + t)|^2 ; X,Y (N,3)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    S = Yc.T @ Xc / len(X)
    U, D, Vt = np.linalg.svd(S)
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2] = -1
    R = U @ np.diag(d) @ Vt
    varx = (Xc**2).sum() / len(X)
    s = (D * d).sum() / varx
    t = my - s * R @ mx
    return s, R, t

def load_gt_tum(path):
    a = np.loadtxt(path)
    return a[:, 0].astype(int), a[:, 1:]     # idx, (tx,ty,tz,qx,qy,qz,qw) cam->world
