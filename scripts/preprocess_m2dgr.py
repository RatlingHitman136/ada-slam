"""Preprocess an M2DGR sequence (ROS bag + RTK ground truth) into this repo's layout.

    python scripts/preprocess_m2dgr.py --inspect --bag data/M2DGR/_raw/gate_01.bag

    python scripts/preprocess_m2dgr.py \
        --bag data/M2DGR/_raw/gate_01.bag --gt data/M2DGR/_raw/gate_01_gt.txt \
        --dst data/M2DGR/gate_01 --selftest

    python scripts/preprocess_m2dgr.py --bag ... --gt ... --dst ... --jobs 16

Produces preprocess_rellis3d.py's shape:

    colors/%06d.png   undistorted RealSense d435i colour, 640x480 at --scale 1
    depths/%06d.png   same index and size, uint16 * --depth_png_scale, 0 = no lidar return
    traj_tum.txt      "<index> tx ty tz qx qy qz qw", camera-to-world, world = local ENU
    calib.txt         "fx fy cx cy"   (distortion is removed from the images, not carried)

CAMERA.  M2DGR carries seven 190-deg fisheye cameras and one pinhole: the RealSense d435i
colour stream on /camera/color/image_raw/compressed.  We use the pinhole, because everything
downstream (calib.txt, the tracker, the VGGT/Omnidata priors) assumes a pinhole model and
rectifying a 190-deg fisheye into one throws away most of the field of view.  M2DGR's own
my_params_camera.yaml declares this camera `model_type: PINHOLE` with k1/k2 only, and upstream
ships separate ORB-Pinhole (d435i) and ORB-Fisheye configs, so this is the supported path.

DEPTH comes from the Velodyne VLP-32C, projected into the camera and interpolated over a
Delaunay triangulation exactly as preprocess_rellis3d.py does it.  Two M2DGR-specific errors
are corrected first, both worth a few centimetres:

  * the sweep is captured over a 100 ms rotation while the platform moves, so points are
    motion-skewed within one message;
  * the camera runs at 15 Hz and the lidar at 10 Hz, so the nearest sweep is up to 50 ms away
    from the frame it is being pasted onto.

--deskew gt (the default) fixes both at once by carrying every point to the world frame through
the ground truth at its own capture time, then into the camera at the frame's exact timestamp.
This makes the depth depend on the pose ground truth, which is standard (KITTI's depth benchmark
accumulates scans the same way) but worth stating: --deskew none keeps the two independent and
--selftest prints what the choice is worth.

POSE ground truth is NOT written through unchanged; see gt_to_camera() for the steps and
--selftest for what each one is worth.

CALIBRATION CAVEAT.  M2DGR publishes the d435i-to-lidar rotation as exact 0/+-1 entries - the
nominal mounting, not a measured rotation (the file even carries a "%to be changed" note over
the first camera block).  Perturbing it on gate_01 and scoring a photometric warp prefers about
+1 deg of roll, consistently on two disjoint halves of the sequence; pitch is not pinned down
(one half wants +1 deg, the other runs to the edge of the search) and yaw is already optimal.  A
pose-free edge-alignment score over the same sequence is too noisy to confirm any of it.  One
degree of roll displaces a corner pixel by about 7 px, so this is worth knowing, but a single
night-time sequence is not grounds for overriding the dataset's own calibration: the published
extrinsic is used as-is and --extrinsic_rpy exists to experiment.  Re-check on a daylight
sequence before changing the default.
"""
import argparse
import os

# Before numpy is imported: one worker per frame is the parallelism we want, and a BLAS that
# spawns its own 64 threads in the parent only makes forking dangerous and each worker slower.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import multiprocessing as mp                                            # noqa: E402
from collections import deque                                          # noqa: E402

import cv2
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation, Slerp

IMAGE_TOPIC = '/camera/color/image_raw/compressed'
CLOUD_TOPIC = '/velodyne_points'
MIN_Z = 0.5                    # m in front of the camera; also culls the rear half of the sweep
WARP_MAX_Z = 40.0              # m; the warp check only scores points near enough to actually move

# ---------------------------------------------------------------------------- calibration
# Verbatim from M2DGR's calibration_results.txt (SJTU-ViSYS/M2DGR@main).  The intrinsic block
# for the d435i is stored column-major in that file; it is transposed here.  Every "Extrinsic
# [to LIDAR]" 4x4 is T_lidar_sensor: it takes points from the sensor frame into the lidar
# frame, and its translation is the sensor's origin expressed in the lidar frame.
K_COLOR = np.array([[617.971050917033, 0.0, 327.710279392468],
                    [0.0, 616.445131524790, 253.976983707814],
                    [0.0, 0.0, 1.0]])
D_COLOR = np.array([0.148000794688248, -0.217835187249065, 0.0, 0.0])   # k1 k2 p1 p2
WH_COLOR = (640, 480)

# d435i colour -> lidar.  The rotation sends camera +z (optical forward) to lidar +x, camera +x
# to lidar -y and camera +y to lidar -z, i.e. the lidar frame is x-forward, y-left, z-up.
T_LIDAR_CAM = np.array([[0.0, 0.0, 1.0, 0.30456],
                        [-1.0, 0.0, 0.0, 0.00065],
                        [0.0, -1.0, 0.0, 0.65376],
                        [0.0, 0.0, 0.0, 1.0]])

# Ublox/Xsens GNSS-INS -> lidar.  Rotation is identity, so the frame the RTK ground truth
# reports attitude in is already lidar-aligned; only a lever arm separates them.
T_LIDAR_GNSS = np.array([[1.0, 0.0, 0.0, -0.09825],
                         [0.0, 1.0, 0.0, 0.00582],
                         [0.0, 0.0, 1.0, 0.72673],
                         [0.0, 0.0, 0.0, 1.0]])

WGS84_A, WGS84_F = 6378137.0, 1.0 / 298.257223563


# ---------------------------------------------------------------------------- ground truth

def ecef_to_enu_frame(X0):
    """Rotation from ECEF to the local ENU frame at X0, and that point's geodetic coordinates.

    Rigid, so it changes no distance, angle or scale in the trajectory - it only moves the
    origin to the first fix and aligns z with local gravity, which keeps the numbers near 0
    instead of near 6.4e6 where float32 has ~0.5 m of resolution.
    """
    e2 = WGS84_F * (2 - WGS84_F)
    x, y, z = X0
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat, h = np.arctan2(z, p * (1 - e2)), 0.0
    for _ in range(8):
        N = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    R = np.array([[-so, co, 0.0],
                  [-sl * co, -sl * so, cl],
                  [cl * co, cl * so, sl]])
    return R, np.degrees(lat), np.degrees(lon), h


def load_gt(path):
    """M2DGR ground truth -> (t, positions, Rotation, meta), untouched except ECEF -> ENU.

    The file is TUM-shaped, "timestamp tx ty tz qx qy qz qw" at 100 Hz for the RTK/INS
    sequences.  Positions are ECEF; the quaternion is already the body-in-ENU attitude (the
    check is in --selftest: body +x sits on the ENU velocity, body +z on the vertical), so it
    is carried through with no frame change at all.

    Over a 600 m sequence the local ENU frame at the far end differs from the one at the origin
    by 600/R_earth = 9.4e-5 rad = 0.0054 deg, which is far below the INS attitude noise, so a
    single frame for the whole run is used rather than a per-sample one.
    """
    d = np.loadtxt(path)
    t, X, q = d[:, 0], d[:, 1:4], d[:, 4:8]
    if np.linalg.norm(q, axis=1).min() < 1e-6:
        raise SystemExit(f'{path} has zero quaternions - this is one of M2DGR\'s Leica '
                         f'sequences (hall/lift/door), which publish position only and cannot '
                         f'give a camera pose.  Use an RTK sequence (street/gate/circle/walk) '
                         f'or a mocap one (room/room_dark).')
    ecef = np.linalg.norm(X, axis=1).mean() > 1e6
    meta = dict(frame='ECEF->ENU' if ecef else 'local as published',
                hz=len(t) / (t[-1] - t[0]), n=len(t), span=t[-1] - t[0])
    if ecef:
        R, lat, lon, h = ecef_to_enu_frame(X[0])
        X = (R @ (X - X[0]).T).T
        meta.update(lat=lat, lon=lon, alt=h)
    else:
        X = X - X[0]
    return t, X, Rotation.from_quat(q), meta


def gt_to_camera(X_gt, R_gt):
    """Ground-truth body poses -> camera-to-world 4x4, one per ground-truth sample.

    T_enu_cam = T_enu_gnss . T_gnss_lidar . T_lidar_cam

    The ground truth is the pose of the GNSS/INS body, not of the camera.  The two are 0.409 m
    apart on the rig (0.403 m of it forward), and because that lever arm rotates with the
    platform it is a real, heading-dependent offset of up to 0.8 m peak-to-peak - not a constant
    that a Sim(3) alignment would absorb.  Both extrinsics are M2DGR's own published ones.
    """
    A = np.linalg.inv(T_LIDAR_GNSS) @ T_LIDAR_CAM          # body -> camera, constant
    T = np.tile(np.eye(4), (len(X_gt), 1, 1))
    T[:, :3, :3] = R_gt.as_matrix()
    T[:, :3, 3] = X_gt
    return T @ A, A


class PoseInterp:
    """Ground truth sampled at arbitrary times: linear on position, slerp on rotation.

    The camera shutter does not fire on ground-truth ticks, so every frame needs a pose between
    two samples.  At 100 Hz and ~0.8 m/s the bracketing samples are 10 ms and ~8 mm apart, and
    the trajectory is smooth at that scale, so the interpolation error is well under a
    centimetre; --selftest reports the measured residual of a leave-one-out check.
    """

    def __init__(self, t, X, R):
        self.t, self.X, self.R = t, X, R
        self.slerp = Slerp(t, R)

    def __call__(self, ts):
        ts = np.clip(np.atleast_1d(ts), self.t[0], self.t[-1])
        P = np.stack([np.interp(ts, self.t, self.X[:, i]) for i in range(3)], axis=1)
        T = np.tile(np.eye(4), (len(ts), 1, 1))
        T[:, :3, :3] = self.slerp(ts).as_matrix()
        T[:, :3, 3] = P
        return T

    def covers(self, ts):
        return (np.asarray(ts) >= self.t[0]) & (np.asarray(ts) <= self.t[-1])


# ---------------------------------------------------------------------------- bag reading

PC2_DTYPE = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def _stamp(msg):
    s = msg.header.stamp
    return s.sec + (getattr(s, 'nanosec', None) or getattr(s, 'nsec', 0)) * 1e-9


def pc2_fields(msg):
    names = [f.name for f in msg.fields]
    dt = np.dtype({'names': names,
                   'formats': [PC2_DTYPE[f.datatype] for f in msg.fields],
                   'offsets': [f.offset for f in msg.fields],
                   'itemsize': msg.point_step})
    return np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height), names


def parse_cloud(msg):
    """PointCloud2 -> (xyz float64 (N,3), per-point time offset in seconds or None, t_header)."""
    arr, names = pc2_fields(msg)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float64)
    finite = np.isfinite(xyz).all(1) & (np.abs(xyz).max(1) > 1e-6)
    tf = next((n for n in ('time', 'timestamp', 't') if n in names), None)
    t0 = _stamp(msg)
    dt = None
    if tf is not None:
        dt = arr[tf].astype(np.float64)[finite]
        if np.nanmax(np.abs(dt)) > 1e6:                    # absolute stamps, not offsets
            dt = dt - t0
    return xyz[finite], dt, t0


def open_bag(path):
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    return Reader(path), get_typestore(Stores.ROS1_NOETIC)


def inspect(path):
    """Topics, counts, rates and the point-cloud field layout - run this before anything else."""
    reader, ts = open_bag(path)
    with reader:
        dur = (reader.end_time - reader.start_time) / 1e9
        print(f'bag      : {path}  ({os.path.getsize(path) / 2**30:.1f} GiB)')
        print(f'duration : {dur:.1f} s, {reader.message_count} messages\n')
        print(f'{"topic":<46}{"type":<34}{"msgs":>8}{"Hz":>7}')
        for c in sorted(reader.connections, key=lambda c: -c.msgcount):
            print(f'{c.topic.strip():<46}{c.msgtype:<34}{c.msgcount:>8}{c.msgcount / dur:>7.1f}')
        for topic, want in ((IMAGE_TOPIC, 'image'), (CLOUD_TOPIC, 'cloud')):
            conns = [c for c in reader.connections if c.topic.strip() == topic]
            if not conns:
                print(f'\n!! {topic} is MISSING from this bag')
                continue
            _, _, raw = next(iter(reader.messages(connections=conns)))
            msg = ts.deserialize_ros1(raw, conns[0].msgtype)
            if want == 'image':
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                print(f'\n{topic}\n  format {msg.format}, decodes to {img.shape[1]}x{img.shape[0]}'
                      f', first stamp {_stamp(msg):.3f}')
            else:
                arr, names = pc2_fields(msg)
                print(f'\n{topic}\n  {msg.width * msg.height} points, fields {names}, '
                      f'point_step {msg.point_step}, first stamp {_stamp(msg):.3f}')
                tf = next((n for n in ('time', 'timestamp', 't') if n in names), None)
                if tf:
                    v = arr[tf].astype(np.float64)
                    print(f'  per-point "{tf}" spans {v.min():.4f} .. {v.max():.4f} '
                          f'({(v.max() - v.min()) * 1000:.1f} ms) -> deskew is possible')
                else:
                    print('  no per-point time field -> --deskew gt will use the header stamp '
                          'for the whole sweep')


def stream(path, stride, limit, max_dt, nsweeps=1, keep=24):
    """Yield (n, jpeg_bytes, t_cam, [(xyz, dt, t0), ...]) with the sweeps covering each frame.

    One sequential pass over the bag.  Only the nsweeps sweeps nearest the frame are handed on,
    for two reasons: each job crosses a process boundary and a whole VLP-32C sweep is ~1.4 MB, so
    shipping the history would cost more in pickling than the depth work itself; and a fixed
    count keeps the beam density the same on every frame.  Taking "everything within max_dt"
    instead makes it alternate between one and two sweeps as the 14.4 Hz camera beats against the
    10 Hz lidar, which would put a sawtooth into every depth-coverage statistic.

    A frame is released only once a sweep later than t_cam + max_dt has arrived, which is the
    point at which no sweep still to come can be nearer.  Releasing it as soon as ANY
    later-stamped sweep arrives is not enough: the bag delivers messages in log-time order while
    the pairing is done on header stamps, and the two are skewed by roughly half a second here,
    so clouds run ahead of the camera frames they belong to.  With a short history the matching
    sweep was already evicted by the time the frame was released - that silently left 77 of
    gate_01's 2490 frames with no depth at all, each of which did have a sweep 30 ms away.
    """
    reader, ts = open_bag(path)
    with reader:
        want = {IMAGE_TOPIC, CLOUD_TOPIC}
        conns = [c for c in reader.connections if c.topic.strip() in want]
        got = {c.topic.strip() for c in conns}
        for topic in want:
            if topic not in got:
                raise SystemExit(f'{path} has no {topic}; run --inspect to see what it does have')
        sweeps, pend, n, seen = deque(maxlen=keep), deque(), 0, 0

        def near(t_cam):
            ok = [s for s in sweeps if abs(s[2] - t_cam) <= max_dt]
            return sorted(ok, key=lambda s: abs(s[2] - t_cam))[:nsweeps]

        for conn, _, raw in reader.messages(connections=conns):
            msg = ts.deserialize_ros1(raw, conn.msgtype)
            if conn.topic.strip() == CLOUD_TOPIC:
                sweeps.append(parse_cloud(msg))
                while pend and sweeps[-1][2] > pend[0][0] + max_dt:
                    t_cam, data = pend.popleft()
                    yield n, data, t_cam, near(t_cam)
                    n += 1
                    if limit and n >= limit:
                        return
            else:
                if seen % stride == 0:
                    pend.append((_stamp(msg), bytes(msg.data)))
                seen += 1
        while pend:
            t_cam, data = pend.popleft()
            yield n, data, t_cam, near(t_cam)
            n += 1
            if limit and n >= limit:
                return


# ---------------------------------------------------------------------------- lidar -> depth

TBINS = 32


def sweep_to_camera(sweeps, t_cam, cfg, interp):
    """Lidar points of the sweep(s) around t_cam, expressed in the camera frame at t_cam.

    --deskew none  : nearest sweep, static extrinsic.  Ignores both the 100 ms rotation and the
                     up-to-50 ms camera/lidar offset.
    --deskew gt    : every point is carried to the world through the ground-truth pose at the
                     time it was actually measured, then back into the camera at t_cam.  Points
                     are bucketed into TBINS time bins (100 ms / 32 = 3 ms, under 3 mm of
                     motion) so this costs 32 pose evaluations, not one per point.
    """
    if not sweeps:
        return np.zeros((0, 3))
    if cfg['deskew'] == 'none' or interp is None:
        xyz, _, _ = min(sweeps, key=lambda s: abs(s[2] - t_cam))
        return xyz @ cfg['E'][:3, :3].T + cfg['E'][:3, 3]

    if not interp.covers(t_cam):
        return np.zeros((0, 3))
    T_wc = interp(np.array([t_cam]))[0] @ cfg['A']         # camera-to-world at the frame time
    T_cw = np.linalg.inv(T_wc)
    body_to_lidar = np.linalg.inv(T_LIDAR_GNSS)            # T_gnss_lidar
    out = []
    for xyz, dt, t0 in sweeps:
        if abs(t0 - t_cam) > cfg['max_dt'] or not len(xyz):
            continue
        tp = np.full(len(xyz), t0) if dt is None else t0 + dt
        if not interp.covers(t0):
            continue
        lo, hi = float(tp.min()), float(tp.max())
        edges = np.linspace(lo, hi + 1e-9, TBINS + 1)
        which = np.clip(np.searchsorted(edges, tp, 'right') - 1, 0, TBINS - 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        T_wl = interp(centres) @ body_to_lidar             # lidar-to-world at each bin centre
        M = T_cw @ T_wl                                    # lidar(t_bin) -> camera(t_cam)
        Ms = M[which]
        out.append(np.einsum('nij,nj->ni', Ms[:, :3, :3], xyz) + Ms[:, :3, 3])
    if not out:
        return np.zeros((0, 3))
    return np.concatenate(out)


def project(P, K, hw, max_range):
    """Camera-frame points -> (u, v, z) inside the frame, nearest kept per pixel.

    The z-buffer matters: the sweep is 360 deg and sees past occluders the camera cannot see
    around, so two points can land on one pixel with very different depths.
    """
    h, w = hw
    P = P[(P[:, 2] > MIN_Z) & (P[:, 2] < max_range)]
    if not len(P):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    uv = (K @ P.T).T
    u, v, z = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2], P[:, 2]
    m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[m], v[m], z[m]
    if not len(z):
        return u, v, z
    order = np.argsort(z)                                  # nearest first
    _, first = np.unique((v.astype(int) * w + u.astype(int))[order], return_index=True)
    keep = order[first]
    return u[keep], v[keep], z[keep]


def densify(u, v, z, hw, max_edge, max_ratio):
    """Linear interpolation of the projected beams over their Delaunay triangulation.

    Triangles that bridge a gap (longest edge > max_edge px) or straddle a depth discontinuity
    (max(z)/min(z) > max_ratio) are dropped, so sky and silhouettes stay 0 rather than being
    filled with fictitious surface.  Identical to preprocess_rellis3d.py's.
    """
    h, w = hw
    if len(z) < 4:
        return np.zeros(hw, np.float64)
    tri = Delaunay(np.c_[u, v])
    yy, xx = np.mgrid[0:h, 0:w]
    pix = np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]
    s = tri.find_simplex(pix)
    out = np.zeros(h * w)
    ok = s >= 0                                            # outside the hull stays invalid
    if not ok.any():
        return out.reshape(hw)
    T = tri.transform[s[ok]]
    b = np.einsum('ijk,ik->ij', T[:, :2], pix[ok] - T[:, 2])
    out[ok] = (np.c_[b, 1 - b.sum(1)] * z[tri.simplices[s[ok]]]).sum(1)
    corners = np.c_[u, v][tri.simplices]
    edge = np.max([np.linalg.norm(corners[:, i] - corners[:, j], axis=1)
                   for i, j in ((0, 1), (1, 2), (2, 0))], axis=0)
    zs = z[tri.simplices]
    bad = (edge > max_edge) | (zs.max(1) / np.maximum(zs.min(1), 1e-6) > max_ratio)
    out[ok] = np.where(bad[s[ok]], 0.0, out[ok])
    return out.reshape(hw)


def sparse_image(u, v, z, hw):
    """The raw projection, one pixel per beam return - what --fill none writes."""
    d = np.zeros(hw)
    if len(z):
        d[v.astype(int), u.astype(int)] = z
    return d


# ---------------------------------------------------------------------------- per-frame work

_INTERP = None
_CFG = None


def _init(gt, cfg):
    """Workers get the pose table and the config once, not once per frame.

    cfg carries the two 640x480 float32 undistort maps; passing it through partial() would
    re-pickle 2.4 MB for every frame in the sequence.
    """
    global _INTERP, _CFG
    _INTERP = None if gt is None else PoseInterp(*gt)
    _CFG = cfg


def process(job, cfg=None):
    """One frame: jpeg -> undistorted png, lidar -> depth png."""
    cv2.setNumThreads(1)
    cfg = cfg if cfg is not None else _CFG
    n, data, t_cam, sweeps = job
    h, w = cfg['hw']
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f'frame {n}: could not decode the compressed image')
    img = cv2.remap(img, cfg['map1'], cfg['map2'], cv2.INTER_LINEAR)
    if (w, h) != (WH_COLOR[0], WH_COLOR[1]):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{cfg["dst"]}/colors/{n:06d}.png', img)

    P = sweep_to_camera(sweeps, t_cam, cfg, _INTERP)
    u, v, z = project(P, cfg['K'], cfg['hw'], cfg['max_range'])
    d = (sparse_image(u, v, z, cfg['hw']) if cfg['fill'] == 'none'
         else densify(u, v, z, cfg['hw'], cfg['max_edge'], cfg['max_ratio']))
    raw = np.clip(d * cfg['png_scale'], 0, 65535).astype(np.uint16)
    cv2.imwrite(f'{cfg["dst"]}/depths/{n:06d}.png', raw)
    valid = d > 0
    return (len(z), float(valid.mean()), float(d[valid].min()) if valid.any() else 0.0,
            float(d[valid].max()) if valid.any() else 0.0, t_cam)


# ---------------------------------------------------------------------------- checks

def check_gt_frame(t, X, R, hz):
    """Is the published quaternion really the body-in-ENU attitude?  Measured, not assumed.

    If it is, the body +x axis lies along the direction of travel and the body +z axis is the
    local vertical.  Two things would otherwise make a correct attitude look wrong:

      * the velocity direction is meaningless while the platform is nearly stopped, and single
        RTK epochs are noisy, so the velocity is smoothed over a second and slow samples are
        dropped before the angle is taken;
      * several sequences reverse (upstream calls them "back and forth" / "loop back"), which
        flips the sign of the cosine while the attitude stays perfectly correct.  The mean is
        therefore reported alongside the mean of |cos| and the +1/-1 split: a correct attitude
        on a reversing platform is bimodal, a wrong one is spread out.
    """
    w = max(3, int(hz))
    k = np.ones(w) / w
    Xs = np.stack([np.convolve(X[:, i], k, mode='same') for i in range(3)], axis=1)
    V = np.gradient(Xs, axis=0)
    sp = np.linalg.norm(V, axis=1) * hz
    m = sp > 0.3
    m[:w] = m[-w:] = False
    if m.sum() < 32:
        print('\nground-truth frame check: the platform never moves fast enough to score')
        return
    idx = np.flatnonzero(m)
    M = R[idx].as_matrix()
    v = V[idx] / np.linalg.norm(V[idx], axis=1, keepdims=True)
    cx = (M[:, :, 0] * v).sum(1)
    print(f'\nground-truth frame check ({m.sum()} moving samples, velocity smoothed over '
          f'{w / hz:.1f} s):')
    for name, ax in (('body +x on the travel direction', 0), ('body +y on it', 1),
                     ('body +z on it', 2)):
        print(f'  {name:<34} mean cos {(M[:, :, ax] * v).sum(1).mean():+.3f}')
    print(f'  {"body +z on the ENU vertical":<34} mean |cos| {np.abs(M[:, 2, 2]).mean():.3f}')
    print(f'  {"|cos| of body +x on travel":<34} mean     {np.abs(cx).mean():.3f}   '
          f'forward {100 * (cx > 0.8).mean():.1f}%  reversing {100 * (cx < -0.8).mean():.1f}%'
          f'  neither {100 * (np.abs(cx) < 0.8).mean():.1f}%')
    print('  verdict: |cos| on travel and |cos| on the vertical must both be near 1.0.  A large '
          '"reversing" share is the platform driving backwards, not a bad attitude.')


def scan_gt_jumps(t, X, hz):
    """RTK discontinuities in the published track: (time, size, residual 0.5 s later).

    Not every M2DGR sequence has a continuous ground truth.  A step far larger than the local
    median is either a one-epoch spike, which is harmless, or a receiver re-initialisation,
    which permanently offsets the rest of the track and cannot be repaired by dropping samples -
    street_07 has thirteen of these, the largest 115 m.  The two are told apart by extrapolating
    the motion from before the jump and asking whether the track has come back half a second
    later.
    """
    d = np.linalg.norm(np.diff(X, axis=0), axis=1)
    med = float(np.median(d))
    bad = np.flatnonzero(d > max(20 * med, 0.05))
    groups, cur = [], []
    for b in bad:
        if cur and b - cur[-1] > hz // 4:
            groups.append(cur)
            cur = []
        cur.append(b)
    if cur:
        groups.append(cur)
    w, k, out = max(5, int(hz * 0.2)), max(10, int(hz * 0.5)), []
    for g in groups:
        i, j = g[0], g[-1] + 1
        if i - w < 0 or j + k >= len(X):
            continue
        v = (X[i] - X[i - w]) / w
        size = float(np.linalg.norm(X[j] - X[i] - v))
        after = float(np.linalg.norm(X[j + k] - (X[i] + v * (j + k - i))))
        out.append((float(t[i] - t[0]), size, after))
    return out, med


def report_gt_jumps(events, med):
    """Print the integrity scan; returns True if the track is continuous enough to use."""
    perm = [e for e in events if e[2] > 0.5 * e[1] and e[1] > 0.10]
    if not events:
        print(f'gt health : continuous, median step {med * 1000:.1f} mm, no anomalies')
        return True
    print(f'gt health : median step {med * 1000:.1f} mm, {len(events)} anomalies '
          f'({len(perm)} permanent)')
    for tt, size, after in events[:10]:
        kind = 'RE-INIT' if (after > 0.5 * size and size > 0.10) else 'spike'
        print(f'            t+{tt:8.2f}s  step {size:8.2f} m  still off {after:7.2f} m '
              f'0.5 s later  [{kind}]')
    if perm:
        print(f'!! this sequence re-initialises: {len(perm)} permanent offsets, largest '
              f'{max(e[1] for e in perm):.2f} m.  The track is piecewise-consistent only, so '
              f'ATE and scale drift measured across a break are meaningless.')
    return not perm


def check_interp(t, X, R):
    """Leave-one-out: drop every 2nd ground-truth sample, interpolate it back, score the error.

    This is the cost of PoseInterp at the actual ground-truth rate, and it upper-bounds the cost
    at the camera timestamps (which sit inside a half-as-wide bracket).
    """
    keep = np.arange(0, len(t), 2)
    held = np.arange(1, len(t) - 1, 2)
    pi = PoseInterp(t[keep], X[keep], R[keep])
    T = pi(t[held])
    ep = np.linalg.norm(T[:, :3, 3] - X[held], axis=1)
    er = (Rotation.from_matrix(T[:, :3, :3]) * R[held].inv()).magnitude()
    print('\ninterpolation self-check, holding out every 2nd ground-truth sample:')
    print(f'  position  median {1000 * np.median(ep):.2f} mm   p95 {1000 * np.percentile(ep, 95):.2f} mm'
          f'   max {1000 * ep.max():.2f} mm')
    print(f'  rotation  median {np.degrees(np.median(er)) * 1000:.2f} mdeg '
          f'  p95 {np.degrees(np.percentile(er, 95)) * 1000:.2f} mdeg'
          f'   max {np.degrees(er.max()) * 1000:.2f} mdeg')
    print('  (the real gap is half this wide, so these are upper bounds)')


def selftest_depth(probe, cfg, interp, rng):
    """Leave-one-out: interpolate from 80% of the beams, score at the held-out 20%.

    This measures what densify() invents between beams.  It does NOT measure the extrinsic,
    the time alignment or the lidar's own accuracy - warp_test covers those.
    """
    print('\ndepth self-test - interpolation accuracy against held-out beams:')
    print(f'  {"frame":>7}{"beams":>8}{"covered":>11}{"MAE m":>9}{"median m":>11}'
          f'{"AbsRel":>9}{"d<1.25":>9}')
    rows = []
    for n, data, t_cam, sweeps in probe:
        P = sweep_to_camera(sweeps, t_cam, cfg, interp)
        u, v, z = project(P, cfg['K'], cfg['hw'], cfg['max_range'])
        if len(z) < 64:
            continue
        hold = rng.random(len(z)) < 0.2
        if hold.sum() < 32:
            continue
        d = densify(u[~hold], v[~hold], z[~hold], cfg['hw'], cfg['max_edge'], cfg['max_ratio'])
        pred = d[v[hold].astype(int), u[hold].astype(int)]
        m = pred > 0
        if m.sum() < 32:
            continue
        gt = z[hold][m]
        err = np.abs(pred[m] - gt)
        d125 = np.mean(np.maximum(pred[m] / gt, gt / pred[m]) < 1.25)
        rows.append((err.mean(), np.mean(err / gt), d125))
        print(f'  {n:>7}{len(z):>8}{f"{m.sum()}/{hold.sum()}":>11}{err.mean():>9.3f}'
              f'{np.median(err):>11.3f}{np.mean(err / gt):>9.4f}{d125:>9.4f}')
    if rows:
        a = np.array(rows)
        print(f'  {"mean":>7}{"":>8}{"":>11}{a[:, 0].mean():>9.3f}{"":>11}'
              f'{a[:, 1].mean():>9.4f}{a[:, 2].mean():>9.4f}')


def warp_test(probe, cfg, interp, T_wc_at):
    """Warp a frame into a later one with the written calib, poses and lidar depth.

    Covers intrinsics, extrinsics, pose convention and depth scale at once: the ratio of
    unwarped to warped photometric error must exceed 1.0 or one of them is wrong.  Only frames
    that actually moved are informative - with the camera static the warp is a no-op that can
    only add resampling blur.

    Scored on points closer than WARP_MAX_Z rather than the whole --max_range: a point 100 m
    away barely moves over half a second, so including the far field drives the ratio towards
    1.0 whether the geometry is right or not.
    """
    h, w = cfg['hw']
    print('\nwarp check - unwarped/warped photometric MAE, must be > 1.0:')
    rows = []
    frames = {n: (data, t_cam, sweeps) for n, data, t_cam, sweeps in probe}
    for n in sorted(frames):
        later = [n + k for k in (5, 10, 20) if n + k in frames]
        if len(later) < 3:
            continue
        data, t_cam, sweeps = frames[n]
        moved = np.linalg.norm(T_wc_at(frames[later[-1]][1])[:3, 3] - T_wc_at(t_cam)[:3, 3])
        if moved < 0.2:
            continue
        P = sweep_to_camera(sweeps, t_cam, cfg, interp)
        u, v, z = project(P, cfg['K'], cfg['hw'], WARP_MAX_Z)
        if len(z) < 100:
            continue
        Q0 = np.c_[(u - cfg['K'][0, 2]) * z / cfg['K'][0, 0],
                   (v - cfg['K'][1, 2]) * z / cfg['K'][1, 1], z]

        def grey(d):
            img = cv2.remap(cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR),
                            cfg['map1'], cfg['map2'], cv2.INTER_LINEAR)
            if (w, h) != (WH_COLOR[0], WH_COLOR[1]):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        def smp(I, a, b):
            return cv2.remap(I, a.astype(np.float32).reshape(1, -1),
                             b.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
        I0 = grey(data)
        s0 = smp(I0, u, v)
        cells = {}
        for k in later:
            T = np.linalg.inv(T_wc_at(frames[k][1])) @ T_wc_at(t_cam)
            Q = Q0 @ T[:3, :3].T + T[:3, 3]
            f = Q[:, 2] > MIN_Z
            uv = (cfg['K'] @ Q[f].T).T
            un, vn = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
            g = (un > 1) & (un < w - 2) & (vn > 1) & (vn < h - 2)
            if g.sum() < 100:
                continue
            IN = grey(frames[k][0])
            warped = np.abs(s0[f][g] - smp(IN, un[g], vn[g])).mean()
            still = np.abs(s0[f][g] - smp(IN, u[f][g], v[f][g])).mean()
            cells[k - n] = still / warped
        if cells:
            rows.append((n, moved, cells))
    if not rows:
        print('  no frame moved far enough to score')
        return
    print(f'  {"gap":>6}{"frames":>9}{"median":>9}{"mean":>8}{"min":>8}{"max":>8}'
          f'{"below 1.0":>11}')
    for k in (5, 10, 20):
        v = np.array([r[2][k] for r in rows if k in r[2]])
        if not len(v):
            continue
        print(f'  {f"+{k}":>6}{len(v):>9}{np.median(v):>9.2f}{v.mean():>8.2f}'
              f'{v.min():>8.2f}{v.max():>8.2f}{f"{100 * (v < 1).mean():.0f}%":>11}')
    md = np.mean([r[1] for r in rows])
    print(f'  over {len(rows)} scored frames, {md:.2f} m of motion at the +20 gap')


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True, help='an M2DGR rosbag, e.g. .../street_02.bag')
    ap.add_argument('--gt', default=None, help='the matching ground-truth txt')
    ap.add_argument('--dst', default=None, help='output sequence directory')
    ap.add_argument('--inspect', action='store_true',
                    help='list the bag topics and the cloud field layout, then exit')
    ap.add_argument('--scale', type=float, default=1.0, help='image downscale factor')
    ap.add_argument('--deskew', default='gt', choices=['gt', 'none'],
                    help="'gt' carries each point through the ground truth at its own capture "
                         "time into the camera at the frame time; 'none' pastes the nearest "
                         "sweep with a static extrinsic")
    ap.add_argument('--max_dt', type=float, default=0.075,
                    help='s; drop a sweep whose header is further than this from the frame')
    ap.add_argument('--sweeps', type=int, default=1,
                    help='how many of the nearest sweeps to merge into one frame. 1 keeps the '
                         'beam density identical on every frame; 2 roughly doubles the valid '
                         'depth but ghosts anything that moves, and is only sound with '
                         '--deskew gt, which brings both sweeps to the frame time')
    ap.add_argument('--extrinsic_rpy', type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=('ROLL', 'PITCH', 'YAW'),
                    help='deg; pre-rotate the published lidar->camera extrinsic. M2DGR gives '
                         'the d435i rotation as exact 0/+-1 entries, i.e. the nominal mounting '
                         'rather than a measured one, and a photometric warp on gate_01 prefers '
                         'about +1 deg of roll (see the module docstring). The default changes '
                         'nothing')
    ap.add_argument('--fill', default='linear', choices=['linear', 'none'],
                    help="'linear' interpolates the beams; 'none' writes the raw projection")
    ap.add_argument('--max_edge', type=float, default=32.0,
                    help='px, at the OUTPUT resolution; longer triangle edges are dropped')
    ap.add_argument('--max_ratio', type=float, default=1.3,
                    help='drop a triangle whose max/min depth exceeds this')
    ap.add_argument('--max_range', type=float, default=100.0, help='m; cull farther returns')
    ap.add_argument('--depth_png_scale', type=float, default=256.0,
                    help='metres = px / this. 256 saturates at 256 m; the repo default 6553.5 '
                         'would clip at 10 m')
    ap.add_argument('--stride', type=int, default=1, help='keep every Nth camera frame')
    ap.add_argument('--limit', type=int, default=0, help='stop after N frames (0 = all)')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument('--selftest', action='store_true',
                    help='ground-truth frame check, interpolation residual, depth accuracy and '
                         'the warp check, then exit')
    ap.add_argument('--force', action='store_true', help='overwrite a non-empty --dst')
    args = ap.parse_args()

    if args.inspect:
        inspect(args.bag)
        return
    if not args.gt:
        raise SystemExit('--gt is required unless --inspect')

    t_gt, X_gt, R_gt, meta = load_gt(args.gt)
    T_lidar_cam = T_LIDAR_CAM.copy()
    if any(args.extrinsic_rpy):
        T_lidar_cam[:3, :3] = (Rotation.from_rotvec(np.radians(args.extrinsic_rpy)).as_matrix()
                               @ T_LIDAR_CAM[:3, :3])
    A = np.linalg.inv(T_LIDAR_GNSS) @ T_lidar_cam          # body -> camera
    interp_body = PoseInterp(t_gt, X_gt, R_gt)
    lever = np.linalg.norm(A[:3, 3])

    s = args.scale
    K = np.array([[K_COLOR[0, 0] * s, 0, K_COLOR[0, 2] * s],
                  [0, K_COLOR[1, 1] * s, K_COLOR[1, 2] * s], [0, 0, 1]])
    map1, map2 = cv2.initUndistortRectifyMap(K_COLOR, D_COLOR, None, K_COLOR,
                                             WH_COLOR, cv2.CV_32FC1)
    hw = (int(round(WH_COLOR[1] * s)), int(round(WH_COLOR[0] * s)))
    cfg = {'dst': args.dst, 'hw': hw, 'K': K, 'E': np.linalg.inv(T_lidar_cam), 'A': A,
           'fill': args.fill, 'max_edge': args.max_edge, 'max_ratio': args.max_ratio,
           'max_range': args.max_range, 'png_scale': args.depth_png_scale,
           'deskew': args.deskew, 'max_dt': args.max_dt, 'map1': map1, 'map2': map2}

    fov = (2 * np.degrees(np.arctan(WH_COLOR[0] / 2 / K_COLOR[0, 0])),
           2 * np.degrees(np.arctan(WH_COLOR[1] / 2 / K_COLOR[1, 1])))
    step = np.linalg.norm(np.diff(X_gt, axis=0), axis=1)
    print(f'bag      : {args.bag}')
    print(f'camera   : RealSense d435i colour (pinhole), {IMAGE_TOPIC}')
    print(f'calib    : fx={K[0, 0]:.3f} fy={K[1, 1]:.3f} cx={K[0, 2]:.3f} cy={K[1, 2]:.3f}  '
          f'FoV {fov[0]:.1f} x {fov[1]:.1f} deg')
    gx, gy = np.meshgrid(np.arange(WH_COLOR[0]), np.arange(WH_COLOR[1]))
    dmax = float(np.hypot(map1 - gx, map2 - gy).max())
    print(f'distort  : k1={D_COLOR[0]:+.4f} k2={D_COLOR[1]:+.4f}, removed by remap; max pixel '
          f'displacement {dmax:.1f} px')
    print(f'ground truth: {args.gt}')
    print(f'           {meta["n"]} samples, {meta["hz"]:.0f} Hz, {meta["span"]:.1f} s, '
          f'frame {meta["frame"]}')
    if 'lat' in meta:
        print(f'           ENU origin at lat {meta["lat"]:.6f} lon {meta["lon"]:.6f} '
              f'alt {meta["alt"]:.1f} m')
    print(f'           body->camera lever arm {lever:.3f} m '
          f'({A[0, 3]:+.3f}, {A[1, 3]:+.3f}, {A[2, 3]:+.3f}) applied')
    print(f'motion   : {step.sum():.0f} m path, '
          f'{np.linalg.norm(X_gt[-1] - X_gt[0]):.0f} m net, '
          f'{step.sum() / meta["span"]:.2f} m/s mean')
    jumps, med = scan_gt_jumps(t_gt, X_gt, int(round(meta['hz'])))
    report_gt_jumps(jumps, med)
    print(f'depth    : {args.fill}, deskew {args.deskew}, max_edge {args.max_edge} px, '
          f'max_ratio {args.max_ratio}, range < {args.max_range} m, '
          f'scale {args.depth_png_scale} (saturates at {65535 / args.depth_png_scale:.0f} m)')

    interp = interp_body if args.deskew == 'gt' else None

    def T_wc_at(t):
        return interp_body(np.array([t]))[0] @ A

    if args.selftest:
        check_gt_frame(t_gt, X_gt, R_gt, int(round(meta['hz'])))
        check_interp(t_gt, X_gt, R_gt)
        # three bursts of 41 consecutive frames: the depth check wants variety, and the warp
        # check needs neighbours 5, 10 and 20 frames ahead INSIDE the same burst - with 21-frame
        # bursts only the first frame of each had all three, so the ratio was an average of
        # three samples and far too noisy to read.
        probe, bursts = [], (200, 600, 1000)
        for job in stream(args.bag, max(args.stride, 1), 0, args.max_dt, args.sweeps):
            n = job[0]
            if any(b <= n <= b + 40 for b in bursts):
                probe.append(job)
            if n > bursts[-1] + 40:
                break
        if not probe:
            raise SystemExit('no frames could be paired with a lidar sweep')
        print(f'\nprobing {len(probe)} frames in bursts at {bursts}')
        for label, mode in (('deskew gt  (the default)', interp_body), ('deskew none', None)):
            print(f'\n================ {label} ================')
            cfg2 = dict(cfg, deskew='gt' if mode is not None else 'none')
            selftest_depth(probe[::5], cfg2, mode, np.random.default_rng(0))
            warp_test(probe, cfg2, mode, T_wc_at)
        return

    if not args.dst:
        raise SystemExit('--dst is required unless --inspect or --selftest')
    for sub in ('colors', 'depths'):
        d = f'{args.dst}/{sub}'
        if os.path.isdir(d) and os.listdir(d) and not args.force:
            raise SystemExit(f'{d} is not empty; pass --force to overwrite it')
        os.makedirs(d, exist_ok=True)

    print(f'\nwriting with {args.jobs} processes...')
    from tqdm import tqdm
    gt_arg = (t_gt, X_gt, R_gt) if args.deskew == 'gt' else None
    gen = stream(args.bag, max(args.stride, 1), args.limit, args.max_dt, args.sweeps)
    # spawn, not the Linux default fork: by this point the parent has already run numpy and
    # scipy over the ground truth, so its BLAS has a thread pool, and forking a process with
    # live threads can hand the child a mutex that is locked and will never be released.  The
    # workers then sit at 0% CPU for ever - which is exactly what fork did here.
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.jobs, initializer=_init, initargs=(gt_arg, cfg)) as pool:
        stats = list(tqdm(pool.imap(process, gen, chunksize=4), desc='frames'))
    stats = np.array(stats, dtype=np.float64)
    t_cam = stats[:, 4]

    T = interp_body(t_cam) @ A                             # camera-to-world at every frame
    q = Rotation.from_matrix(T[:, :3, :3]).as_quat()
    np.savetxt(f'{args.dst}/traj_tum.txt',
               np.column_stack([np.arange(len(T)), T[:, :3, 3], q]))
    with open(f'{args.dst}/calib.txt', 'w') as f:
        f.write(f'{K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}')
    outside = int((~interp_body.covers(t_cam)).sum())
    with open(f'{args.dst}/preprocess_info.txt', 'w') as f:
        f.write(f'bag {args.bag}\ngt {args.gt}\ncamera d435i colour (pinhole) {IMAGE_TOPIC}\n'
                f'cloud {CLOUD_TOPIC}\nscale {s}\nfill {args.fill}\ndeskew {args.deskew}\n'
                f'max_dt {args.max_dt}\nmax_edge {args.max_edge}\nmax_ratio {args.max_ratio}\n'
                f'max_range {args.max_range}\ndepth_png_scale {args.depth_png_scale}\n'
                f'stride {args.stride}\nframes {len(stats)}\n'
                f'resolution {hw[1]}x{hw[0]}\ngt_frame {meta["frame"]}\n'
                f'lever_arm_m {lever:.4f}\nframes_outside_gt {outside}\n'
                f'frames_without_lidar {int((stats[:, 0] == 0).sum())}\n'
                f'mean_depth_coverage {stats[:, 1].mean():.4f}\n'
                f'extrinsic_rpy_deg {" ".join(str(x) for x in args.extrinsic_rpy)}\n')

    cov, beams = stats[:, 1], stats[:, 0]
    blank = int((beams == 0).sum())
    print(f'\nframes   : {len(stats)}, {hw[1]}x{hw[0]}, '
          f'{t_cam[-1] - t_cam[0]:.1f} s at {len(stats) / (t_cam[-1] - t_cam[0]):.1f} Hz')
    print(f'beams    : {beams.mean():.0f} per frame in view (min {beams.min():.0f}, '
          f'max {beams.max():.0f})')
    print(f'depth    : {100 * cov.mean():.1f}% of pixels valid '
          f'(min {100 * cov.min():.1f}%, max {100 * cov.max():.1f}%), '
          f'range {stats[:, 2][stats[:, 2] > 0].min():.2f} .. {stats[:, 3].max():.1f} m')
    if blank:
        print(f'!! {blank} frames ({100 * blank / len(stats):.2f}%) got no lidar at all - their '
              f'depth png is entirely zero.  Raise --max_dt, or --sweeps if the lidar itself '
              f'dropped out.')
    else:
        print(f'           every frame paired with a sweep')
    if outside:
        print(f'!! {outside} frames fall outside the ground-truth interval; their poses are '
              f'clamped to the endpoints')
    print(f'wrote    : {args.dst}/{{colors,depths}}/, traj_tum.txt, calib.txt, '
          f'preprocess_info.txt')


if __name__ == '__main__':
    main()
