"""Helpers more than one stage needs - the neutral ground they import instead of each other."""
import os

import cv2
import numpy as np

DEPTH_DIR, MASK_DIR = 'depth_slam', 'mask_slam'   # extract writes them, adapt reads them

# ---------------------------------------------------------------- the outputs/ layout (7.1)
# outputs/<stage>/<scene>/<experiment>/, the experiment holding what the NEXT stage consumes and
# nothing else. Here rather than in a stage package because the drivers need the names too.
EXTRACT_RUN_SUBDIR = 'full'                        # the raw SLAM run; deletable afterwards
ADAPT_CKPT_SUBDIR = 'checkpoints'
ADAPTER_FILE = 'adapter.safetensors'               # what an adapt experiment hands to its readers
TEST_KINDS = ('end2end', 'prior')
HANDOFF_UP = ('traj_full.txt', 'intrinsics.npy')   # COPIED up from full/, so full/ stays complete

# An online run produces BOTH an adapter and the arm that trained it, and end2end/config.py:arm_name
# would name them the same directory - so the live arm carries this suffix. Without it, later
# testing the frozen final adapter as an ordinary END2END_PRIORS entry would overwrite the live
# run's trajectory with a different experiment's (13).
ONLINE_ARM_SUFFIX = '_live'


def camera_points(depth, K):
    """(H, W) depth -> (N, 3) points in THAT camera's own frame, valid pixels only.

    `K` is the 3x3 intrinsic matrix at the DEPTH MAP's resolution, which is what both callers
    already hold: online/target.py:_intrinsics rescales the tracker's to VGGT's input size, and
    adapt/data.py does the same to intrinsics.npy.

    Valid is `depth > 0` and nothing else - deliberately NOT the supervision mask. See gauge_scale.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = np.nonzero(depth > 0)
    Z = depth[v, u].astype(np.float64)
    return np.stack([(u - cx) * Z / fx, (v - cy) * Z / fy, Z], axis=1)


def transform_points(X, T):
    """(N, 3) points through a 4x4 rigid transform. One definition, two callers."""
    return X @ T[:3, :3].T + T[:3, 3]


def gauge_scale(points, clamp=2.0):
    """VGGT's target-normalization statistic, made robust for a SLAM point cloud.

    VGGT normalizes its training GT by the ARITHMETIC MEAN Euclidean distance of every valid point
    to the origin, with the origin at the first camera and all frames of the sample pooled
    (thirdparty/vggt/training/train_utils/normalization.py:96-107). Dividing our target by the same
    statistic puts it in the gauge the model was pretrained to emit, and - the property this whole
    change rests on - makes the target EXACTLY invariant to a global rescale of itself, because the
    statistic scales with the depth it is computed from. The tracker's map units inflate ~1.97x
    across KITTI 00 and that inflation cancels, with no anchor, window or history.

    WHY THE CLAMP, i.e. why not VGGT's statistic verbatim. Their training GT is dense AND correct;
    ours is 1/disps_up, whose far-field holes reach 1000, and an arithmetic mean of Euclidean
    distance is carried entirely by them. Measured against metric velodyne over 436 keyframes, the
    plain mean is a 32x NOISIER estimator of the map's local scale than a clamped one
    (sd of the step-to-step log ratio 0.695 vs 0.0247), spanning a factor of 58 rather than 2.2.

    Clamping at `clamp` x the frame's own MEDIAN distance - not at any absolute distance - is what
    keeps the rescale invariance: an absolute threshold would select a different physical set of
    pixels as the map inflated and the cancellation would break. It is the same relative-threshold
    discipline ceil_ratio/ped_ratio and relative_loss already follow. At clamp=2 the result is
    VGGT's own statistic divided by a near-constant 1.122 (+/-5.2% over the run), which the adapter
    absorbs as one global bias.

    Returns None when there is nothing to measure, so callers distinguish it from a small gauge.
    """
    if len(points) == 0:
        return None
    r = np.linalg.norm(points, axis=1)
    med = np.median(r)
    if not np.isfinite(med) or med <= 0:
        return None
    return float(np.minimum(r, clamp * med).mean())


def context_keyframes(t, n_ctx, stride=1):
    """The `n_ctx` keyframes before `t` at `stride` spacing, ASCENDING, clipped at 0.

    Here rather than in either stage because BOTH ends need it and must not disagree: the trainer
    builds a sample with it (online/target.py:sample) and the extractor builds the served sequence
    with it (end2end/prior.py:context_stack). Two copies would be two chances to drift, and the
    whole point of context_kf is that the two forwards see the same shape.

    WHY STRIDE EXISTS. Consecutive keyframes are ~1.65 m apart on KITTI 00, so a 3-keyframe sample
    spans ~3 m - and the local Sim(3) scale drifts only -0.136% per keyframe, i.e. -0.41% across
    that whole sample. depth_loss then re-fits scale INSIDE the sample and pose_loss normalises
    both translation sets by their own mean norms, so even that 0.41% is divided out. Meanwhile
    the ATE is ~100% cumulative scale drift (measured: factor 0.56 over the sequence, and
    corr(local-scale sd, ATE) = 0.997). Striding is the cheapest way to put some of that drift
    INSIDE the sample where the loss can see it: at stride k the span is k times wider, so
    stride 5 carries ~1.4% and stride 10 ~2.7%.

    The cost is visual overlap - VGGT's cross-frame attention and the pose target both need the
    frames to see the same scene. At ~1.65 m per keyframe, stride 5 is a ~16 m baseline and
    stride 10 ~33 m; well beyond that the sample stops being a sequence at all.

    stride=1 reproduces `range(max(0, t - n_ctx), t)` exactly, so every run recorded before this
    argument existed is unchanged.
    """
    ix = [t - stride * i for i in range(n_ctx, 0, -1) if t - stride * i >= 0]
    return ix


def extract_run_dir(exp_dir):
    """The untouched HI-SLAM2 run inside an extract experiment directory."""
    return f'{exp_dir}/{EXTRACT_RUN_SUBDIR}'


def experiment_dir(root, stage, scene, name):
    """`<root>/<stage>/<scene>/<name>`. Pure string work - the PARAMETERS block calls it."""
    return f'{root}/{stage}/{scene}/{name}'


def test_dir(root, kind, scene):
    """`<root>/test/<kind>/<scene>` - every test of `kind` on `scene`, one subdirectory each."""
    if kind not in TEST_KINDS:
        raise ValueError(f'unknown test kind {kind!r}; choose from {TEST_KINDS}')
    return f'{root}/test/{kind}/{scene}'


def require_name(knob, value):
    """An experiment name must be set and must be one path component."""
    if not value or not str(value).strip():
        raise SystemExit(f'{knob} must be set - it names this experiment inside its scene '
                         f'directory, and every experiment needs a name of its own')
    if '/' in str(value) or str(value) in ('.', '..'):
        raise SystemExit(f'{knob}={value!r} must be a single directory name, not a path')
    return value


def stream_resize(img, res):
    """The resize the tracker sees. ONE definition - every consumer must agree pixel for pixel.

    `res` is a pixel budget, not a shape (9.6): both dims scale by sqrt(res / h0*w0), floored to a
    multiple of 8. That floor is not aspect-preserving, but slam/stream.py rescales the intrinsics
    with the actual ratios, so it is image shear rather than a calibration error.
    """
    h0, w0 = img.shape[:2]
    h1 = int(h0 * np.sqrt(res / (h0 * w0)))
    w1 = int(w0 * np.sqrt(res / (h0 * w0)))
    return cv2.resize(img, (w1 - w1 % 8, h1 - h1 % 8))


def probe_stream_hw(image_dir, res):
    """The (H, W) the tracker will run at, measured on the first frame.

    Touches the filesystem, so callers resolve it once - after chdir, before any Process is spawned.
    """
    files = sorted(os.listdir(image_dir))
    if not files:
        raise SystemExit(f'{image_dir} is empty - cannot determine the tracking resolution')
    img = cv2.imread(os.path.join(image_dir, files[0]))
    if img is None:
        raise SystemExit(f'could not read {os.path.join(image_dir, files[0])}')
    return tuple(stream_resize(img, res).shape[:2])
