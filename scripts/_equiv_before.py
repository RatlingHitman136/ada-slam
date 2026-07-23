"""Self-contained extract -> adapt -> test pipeline for the VGGT depth-prior experiment.

    python scripts/run_pipeline.py          # from the repo root, with the adaslam venv active

Three stages, run in ONE process, each skipped if its output already exists:

  1 extract  HI-SLAM2 on the first FRACTION% of the sequence, dumping its own post-global-BA
             depth, then exported to per-keyframe depth/mask/image + the accuracy table
  2 adapt    LoRA-adapt VGGT on that depth, on a TRAIN subset of the keyframes, reporting
             depth L1 on a held-out VAL subset
  3 test     two (or three) full-sequence arms differing ONLY in the depth prior, then a
             side-by-side comparison split at the frame the adapter's training data ended

Every parameter is a CAPITAL constant in the block below - no command line, no environment.
Dataset preprocessing is deliberately NOT here; run scripts/preprocess_tum.py first.

This file imports nothing from demo.py or the other scripts/ drivers: the code it needs from
them is inlined (see ARCHITECTURE.md for the provenance). It does import the system under test
(hislam2/, thirdparty/vggt) and drives three standalone CLIs as subprocesses: evo_ape,
tsdf_integrate.py and scripts/eval_recon.py.
"""
import os    # nopep8
import sys   # nopep8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # nopep8
sys.path.insert(0, _ROOT)                                                 # nopep8
sys.path.insert(0, os.path.join(_ROOT, 'hislam2'))                        # nopep8
sys.path.insert(0, os.path.join(_ROOT, 'thirdparty/vggt'))                # nopep8
import contextlib
import gc
import json
import math
import re
import resource
import subprocess
import time
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process, Queue
from tqdm import tqdm

# ==============================================================================
#  PARAMETERS
# ==============================================================================

# ---------------------------------------------------------------- data (preprocessing is NOT here)
SCENE   = 'rgbd_dataset_freiburg1_room'
DATA    = f'data/TUM/{SCENE}'          # preprocess_tum.py's output layout
COLORS  = f'{DATA}/colors'
DEPTHS  = f'{DATA}/depths'             # None if the dataset has no GT depth
GT_TRAJ = f'{DATA}/traj_tum.txt'
CALIB   = f'{DATA}/calib.txt'
GT_MESH = None                         # None -> skip TSDF + eval_recon (TUM ships no GT mesh)
CONFIG  = 'config/tum_config.yaml'
DROID_WEIGHTS = 'pretrained_models/droid.pth'

# Undistortion/cropping normally happens offline in preprocess_tum.py. Doing it here instead
# would make split_render_metrics() compare undistorted renders against distorted GT, because
# it re-derives the GT frame with stream_resize() only (ARCHITECTURE.md §10.1).
UNDISTORT   = False
CROP_BORDER = 0

# ---------------------------------------------------------------- run control
STAGES           = ('adapt',)
SKIP_EXISTING    = False               # reuse a stage's output if it is already on disk
MIN_FREE_VRAM_MB = 10000               # shared GPU: re-checked before every GPU stage
FRACTION         = 40                  # % of the sequence the adapter trains on; also SPLIT_AT
START            = 0
OUT_EXTRACT      = f'outputs/tum/{SCENE}_p{FRACTION}'
OUT_TEST         = f'outputs/tum_ab_p{FRACTION}'
STREAM_RES       = 341 * 640           # tracking resolution budget
DEPTH_PNG_SCALE  = 6553.5              # 16-bit depth PNG scale used across the repo

# ---------------------------------------------------------------- extract: keyframe production
# EXTRACT-ONLY, and the EXTRACT_ prefix is the whole point: these four go into a generated
# extract_config.yaml that ONLY the extract run is given (stage_extract). Every A/B arm is handed
# the unmodified CONFIG above, so denser training data can never be mistaken for a tracking change
# in the comparison - and the arms stay comparable with runs made before these knobs existed.
# stage_test asserts this rather than trusting it.
#
# Two gates decide the count, and the SECOND one usually wins. MotionFilter proposes a keyframe
# once the mean flow since the last one exceeds EXTRACT_KF_MOTION_THRESH (motion_filter.py:112-113),
# then TrackFrontend deletes it again if it lands closer than EXTRACT_KF_REDUNDANT_THRESH to its
# neighbour (track_frontend.py:49-52, and :93-99 during init, where it prunes on that alone). So
# lowering EXTRACT_KF_MOTION_THRESH by itself just proposes keyframes that are immediately pruned.
# Measured over 204 TUM frames:
#     (motion, redundant) = (2.4, 4.0) -> 43 kf   (1.2, 4.0) -> 45 kf   (1.2, 1.5) -> 83 kf
# To densify, lower BOTH; EXTRACT_KF_REDUNDANT_THRESH is the one that moves the number.
EXTRACT_KF_MOTION_THRESH    = 1.2
EXTRACT_KF_INIT_THRESH      = 4.0   # the same gate before initialisation
EXTRACT_KF_REDUNDANT_THRESH = 2.0
EXTRACT_KF_COVIS_THRESH     = 0.1   # extra keyframes inserted in terminate(); LOWER -> more
EXTRACT_BUFFER              = 500   # hard cap; MUST exceed the count (no overflow guard exists)
                            # any of the four thresholds may be None = inherit CONFIG unchanged

# ---------------------------------------------------------------- extract: export
DEPTH_SOURCE        = 'slam'  # 'rendered' (Gaussian expected depth) | 'slam' (1/disps_up)
MASK_FILTER_THRESH  = 0.005       # depth_filter disparity agreement
MASK_MIN_COUNT      = 1           # min agreeing neighbours out of 6
MASK_MIN_DISP_RATIO = 0.5         # drop pixels below this fraction of the frame's mean disparity

# ---------------------------------------------------------------- adapt (LoRA on VGGT)
VGGT_WEIGHTS = 'pretrained_models/vggt'
VGGT_HW      = (378, 518)     # dims %14; MUST match the tracking stream's aspect (§9.3)
LORA_RANK, LORA_ALPHA = 8, 16
LORA_TARGETS     = ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')
LORA_PATCH_EMBED = False
EPOCHS = 1
BATCH_SIZE = 2
LR, WEIGHT_DECAY, GRAD_CLIP, LAMBDA_POSE = 1e-4, 0.0, 1.0, 1.0
DEPTH_SPACE, COUPLED_SCALE = 'disparity', True     # 'depth' | 'disparity'
P_SINGLE_VIEW, MAX_LEFT, MAX_RIGHT, RADIUS = 1, 4, 4, 8
MIN_MASK_PIXELS, LOG_EVERY = 16, 20
SEED = 0

# ---- adapter train / val split over the exported keyframes ----
LORA_TRAIN_FRAC   = 0.8       # 1.0 = train on every keyframe, no val set
LORA_SPLIT_MODE   = 'stride'  # 'stride' (every Nth held out) | 'contiguous' (tail) | 'random'
LORA_EVAL_ON_VAL  = True      # depth L1 on held-out keyframes, base vs adapted
LORA_EVAL_ON_TRAIN = True     # also on the train subset, so the train/val gap is visible
LORA_EVAL_EVERY_EPOCH = True  # False = only before training and after the last epoch
LORA_EVAL_MAX_KF  = 5       # evenly subsample each eval subset to at most this many; 0 = no cap
LORA_KEEP_BEST    = False     # False = save the last epoch (report-only, the default);
                              # True  = snapshot the LoRA state whenever val L1 improves and
                              #         save that instead

# ---------------------------------------------------------------- test (A/B arms)
ARMS        = ('omnidata', 'vggt_lora')   # 'vggt_base' = stock VGGT-1B, the §10.2 third arm
TEST_LENGTH = 100000                      # 100000 = whole sequence
TEST_BUFFER = 500
VOXEL_SIZE  = 0.01                        # pinned for ALL arms (§9.3)
VOXEL_FALLBACKS = (0.01, 0.02)            # marching cubes OOMs on a busy shared GPU
MESH_WEIGHT = 2.0
OMNI_NORMAL_CKPT = 'pretrained_models/omnidata_dpt_normal_v2.ckpt'
OMNI_NORMAL_HW   = (512, 512)

# ==============================================================================

ARM_DIRS = {'omnidata': 'omnidata', 'vggt_lora': 'vggt', 'vggt_base': 'vggt_base'}
# short names for the comparison table's column headers; the full label goes in ab_results.json
ARM_NAMES = {'omnidata': 'Omnidata', 'vggt_lora': 'VGGT+LoRA', 'vggt_base': 'VGGT-base'}

# DepthVideo share_memory_()s every buffer; the default fd limit is not enough (demo.py:12-14 did
# this as an import side effect, which no longer happens now that demo.py is not imported).
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))

# evo_ape / tsdf_integrate.py / eval_recon.py are invoked through the shell and need the venv's
# bin on PATH, which is empty when the venv was not activated but this file was run with its python
os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')


# ------------------------------------------------------------------ small helpers

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def free_vram(tag=''):
    """Drop everything the finished stage held. In-process stages otherwise accumulate.

    One cost cannot be reclaimed, and it is worth budgeting for. With `Tracking.pgba.active`
    (true for TUM, false for Replica), Hi2 spawns the PGBA process and hands it the DepthVideo
    buffers over CUDA IPC, then `terminate()`s it - abruptly, so the producer side never learns
    the blocks are free and they stay pinned in IPC limbo for the life of the process. Measured
    on TUM at buffer=500: 1.29 GiB retained after one SLAM run and +1.26 GiB per run after that,
    versus 0.04 GiB flat with pgba off. Only 0.03 GiB of it is reachable from Python, so no
    gc/empty_cache/ipc_collect call touches it.

    Consequence: extract + two arms strands ~3.8 GiB by the end. That fits alongside a VGGT arm's
    ~10 GiB peak on this 24 GB card, but if it does not on yours, run one STAGES entry per
    process - a fresh process starts from zero.
    """
    gc.collect()
    torch.cuda.ipc_collect()      # reclaims blocks whose consumer *did* exit cleanly
    torch.cuda.empty_cache()
    if tag:
        held = torch.cuda.memory_allocated() / 2**30
        note = '  <- pgba IPC limbo, not reclaimable in-process' if held > 1.0 else ''
        print(f'  [vram] after {tag}: {held:.2f} GiB allocated, '
              f'{torch.cuda.memory_reserved()/2**30:.2f} GiB reserved{note}')


def gpu_gate():
    """The GPU is shared and the stages are far apart in time - re-check before each one."""
    r = sh('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits')
    used, total = (int(x) for x in r.stdout.splitlines()[0].replace(',', '').split())
    if total - used < MIN_FREE_VRAM_MB:
        raise SystemExit(f'only {total - used} MiB VRAM free (need {MIN_FREE_VRAM_MB}); another '
                         f'job is running. Lower MIN_FREE_VRAM_MB to override.')
    print(f'GPU free  : {total - used} / {total} MiB')


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def tee(path):
    """Print to stdout and to a file at once - export.txt is read back by other tooling."""
    with open(path, 'w') as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            yield


def banner(title):
    print(f'\n{"=" * 78}\n=== {title}\n{"=" * 78}')


def stream_resize(img, res=STREAM_RES):
    """The resize the tracker sees. ONE definition, used by the reader, the LoRA data loader and
    the render metrics - they must agree or renders and GT stop lining up pixel for pixel."""
    h0, w0 = img.shape[:2]
    h1 = int(h0 * np.sqrt(res / (h0 * w0)))
    w1 = int(w0 * np.sqrt(res / (h0 * w0)))
    return cv2.resize(img, (w1 - w1 % 8, h1 - h1 % 8))


# ==============================================================================
#  STAGE 1a - SLAM
# ==============================================================================

def mono_stream(queue, imagedir, calib, start, length, undistort, cropborder, res):
    """Image reader process: decode, optionally undistort, resize, hand to the tracker.

    Runs in a spawned child, which re-imports this module and therefore sees the constants as
    written in the file - never a caller's override. Everything it needs is passed in.
    """
    calib = np.loadtxt(calib, delimiter=' ')
    K = np.array([[calib[0], 0, calib[2]], [0, calib[1], calib[3]], [0, 0, 1]])
    image_list = sorted(os.listdir(imagedir))[start:start + length]

    for t, imfile in enumerate(image_list):
        image = cv2.cvtColor(cv2.imread(os.path.join(imagedir, imfile)), cv2.COLOR_BGR2RGB)
        intrinsics = torch.tensor(calib[:4])
        if len(calib) > 4 and undistort:
            image = cv2.undistort(image, K, calib[4:])
        if cropborder > 0:
            image = image[cropborder:-cropborder, cropborder:-cropborder]
            intrinsics[2:] -= cropborder

        h0, w0 = image.shape[:2]
        image = stream_resize(image, res)
        h1, w1 = image.shape[:2]
        intrinsics[[0, 2]] *= (w1 / w0)
        intrinsics[[1, 3]] *= (h1 / h0)

        queue.put((t, torch.as_tensor(image).permute(2, 0, 1)[None], intrinsics[None],
                   t == len(image_list) - 1))

    time.sleep(10)      # keep the queue's feeder thread alive until the consumer has drained it


def save_trajectory(hi2, traj_full, imagedir, output, start=0):
    """traj_kf.txt / traj_full.txt in TUM format (camera-to-world) + intrinsics.npy."""
    t = hi2.video.counter.value
    tstamps = hi2.video.tstamp[:t]
    import lietorch
    poses_wc = lietorch.SE3(hi2.video.poses[:t]).inv().data
    np.save(f'{output}/intrinsics.npy', hi2.video.intrinsics[0].cpu().numpy() * 8)

    # the timestamp is the number in the filename, so %06d names make timestamps frame indices
    tstamps_full = np.array([float(re.findall(r'[+]?(?:\d*\.\d+|\d+)', x)[-1])
                             for x in sorted(os.listdir(imagedir))[start:]])[..., np.newaxis]
    tstamps_kf = tstamps_full[tstamps.cpu().numpy().astype(int)]
    np.savetxt(f'{output}/traj_kf.txt',
               np.concatenate([tstamps_kf, poses_wc.cpu().numpy()], axis=1))
    if traj_full is not None:
        np.savetxt(f'{output}/traj_full.txt',
                   np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1))


def run_slam(out, config, length, buffer, gtdepthdir=None, dump_slam_depth=False):
    """demo.py's main loop. Any prior monkey-patch must already be installed on MotionFilter."""
    from hi2 import Hi2
    os.makedirs(out, exist_ok=True)

    queue = Queue(maxsize=8)
    reader = Process(target=mono_stream, args=(queue, COLORS, CALIB, START, length,
                                               UNDISTORT, CROP_BORDER, STREAM_RES))
    reader.start()

    N = len(os.listdir(COLORS))
    args = types.SimpleNamespace(
        weights=DROID_WEIGHTS, config=config, output=out, gtdepthdir=gtdepthdir,
        buffer=min(1000, N // 10 + 150) if buffer is None else buffer,
        droidvis=False, gsvis=False, dump_slam_depth=dump_slam_depth)

    hi2 = None
    pbar = tqdm(range(min(N, length)), desc='Processing keyframes')
    while True:
        t, image, intrinsics, is_last = queue.get()
        pbar.update()
        if hi2 is None:
            args.image_size = [image.shape[2], image.shape[3]]
            hi2 = Hi2(args)
        hi2.track(t, image, intrinsics=intrinsics, is_last=is_last)
        pbar.set_description(f'keyframe {hi2.video.counter.value} '
                             f'gs {hi2.gs.gaussians._xyz.shape[0]}')
        if is_last:
            pbar.close()
            break
    reader.join()

    traj = hi2.terminate()
    save_trajectory(hi2, traj, COLORS, out, start=START)
    n_kf = hi2.video.counter.value
    del hi2, traj, queue, reader           # the next stage needs the VRAM back
    free_vram()
    return n_kf


def write_extract_config(out):
    """Derived config carrying the keyframe-production knobs. Used by the extract stage ONLY.

    load_config() resolves `inherit_from` recursively and merges, so only the overridden keys need
    to appear here. It doubles as a record of what the extract run was actually told to do. The
    A/B arms deliberately use the unmodified CONFIG - these knobs shape the training-data run only,
    and stage_test asserts that this file never reaches an arm.
    """
    import yaml
    tracking = {}
    for section, keys in (('motion_filter', (('thresh', EXTRACT_KF_MOTION_THRESH),
                                             ('init_thresh', EXTRACT_KF_INIT_THRESH))),
                          ('frontend', (('keyframe_thresh', EXTRACT_KF_REDUNDANT_THRESH),)),
                          ('backend', (('covis_thresh', EXTRACT_KF_COVIS_THRESH),))):
        vals = {k: v for k, v in keys if v is not None}
        if vals:
            tracking[section] = vals

    cfg = {'inherit_from': os.path.abspath(CONFIG)}     # absolute: load_config resolves against cwd
    if tracking:
        cfg['Tracking'] = tracking
    os.makedirs(out, exist_ok=True)
    path = f'{out}/extract_config.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f'extract config: {path}  ({tracking if tracking else "no overrides"})')
    return path


# ==============================================================================
#  STAGE 1b - export the SLAM depth as training data
# ==============================================================================

def confidence_mask(poses, disps, intrinsics_full):
    """Multi-view consistency mask, following util/droid_visualization.py:104-110.

    droid_backends.depth_filter counts, per pixel, how many of 6 temporal neighbours agree on the
    reprojected disparity. The kernel bounds-checks against disps.size(0), so the arrays must be
    sliced to the real keyframe count - otherwise trailing keyframes match unused buffer slots
    that still hold the initial 1.0.
    """
    import droid_backends
    K = disps.shape[0]
    ix = torch.arange(K, device='cuda', dtype=torch.long)
    thresh = MASK_FILTER_THRESH * torch.ones(K, device='cuda', dtype=torch.float)
    count = droid_backends.depth_filter(poses, disps, intrinsics_full / 8.0, ix, thresh)
    return (count >= MASK_MIN_COUNT) & \
           (disps > MASK_MIN_DISP_RATIO * disps.mean(dim=[1, 2], keepdim=True))


def align_scale(pred, gt):
    """Median-ratio scale on flat arrays - SLAM units are arbitrary, GT is metric."""
    return float(np.median(gt) / np.median(pred))


def l1_per_frame(pairs):
    """One scale fitted per keyframe."""
    return np.mean([np.abs(g - align_scale(p, g) * p).mean() for g, p in pairs])


def l1_global(pairs):
    """One scale for the whole sequence, then averaged the same way as l1_per_frame.

    Averaging per frame in both keeps the only difference the scale fit itself, so the gap between
    the two columns isolates cross-frame scale drift rather than frame weighting.
    """
    s = align_scale(np.concatenate([p for _, p in pairs]),
                    np.concatenate([g for g, _ in pairs]))
    return np.mean([np.abs(g - s * p).mean() for g, p in pairs])


def export_slam_depth(out):
    """slam_depth.npz -> depth_<src>/ mask_<src>/ image/ poses_slam.txt, + the accuracy table."""
    from lietorch import SE3
    from geom.ba import get_prior_depth_aligned

    d = np.load(f'{out}/slam_depth.npz')
    tstamp, intrinsics = d['tstamp'], d['intrinsics']
    K, H, W = d['disps_up'].shape
    print(f'{K} keyframes, {H}x{W}, intrinsics fx={intrinsics[0]:.2f} fy={intrinsics[1]:.2f} '
          f'cx={intrinsics[2]:.2f} cy={intrinsics[3]:.2f}')

    poses = torch.from_numpy(d['poses']).cuda().contiguous()
    disps = torch.from_numpy(d['disps']).cuda().contiguous()
    intr = torch.from_numpy(intrinsics).cuda().contiguous()

    # 1/8-res consistency mask, nearest-upsampled to full res for use with disps_up
    mask_low = confidence_mask(poses, disps, intr)
    mask = F.interpolate(mask_low[:, None].float(), size=(H, W),
                         mode='nearest')[:, 0].cpu().numpy() > 0.5
    print(f'confidence mask (thresh={MASK_FILTER_THRESH}, min_count={MASK_MIN_COUNT}): '
          f'{100.0 * mask.mean():.1f}% of pixels kept')

    depth = 1.0 / np.clip(d['disps_up'], 1e-6, None)
    depth[~np.isfinite(depth)] = 0.0

    # 'rendered' is the Gaussian map's expected depth after the colour refinement. It is the better
    # target on two counts: measurably closer to GT (0.0133 vs 0.0324 m on Replica room0), and
    # rendered from the SAME post-refinement trajectory that traj_full.txt holds, whereas
    # 1/disps_up is dumped before the refinement overwrites video.poses (hi2.py:155).
    ddir, mdir = f'depth_{DEPTH_SOURCE}', f'mask_{DEPTH_SOURCE}'
    for sub in (ddir, mdir, 'image'):
        os.makedirs(f'{out}/{sub}', exist_ok=True)

    kept, missing = [], []
    for i in range(K):
        idx = int(tstamp[i])
        if DEPTH_SOURCE == 'rendered':
            rf = f'{out}/renders/depth_after_opt/{idx:06d}.png'
            if not os.path.exists(rf):
                missing.append(idx)
                continue
            # dequantize once here so downstream keeps a single float32 .npy loader
            dep = cv2.imread(rf, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_PNG_SCALE
        else:
            dep = depth[i].astype(np.float32)

        np.save(f'{out}/{ddir}/{idx:06d}.npy', dep)
        cv2.imwrite(f'{out}/{mdir}/{idx:06d}.png', ((mask[i] & (dep > 0)) * 255).astype(np.uint8))
        rgb = d['images'][i].transpose(1, 2, 0)          # stored RGB (mono_stream converts)
        cv2.imwrite(f'{out}/image/{idx:06d}.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        kept.append(i)

    if not kept:
        raise SystemExit(f'no {DEPTH_SOURCE} depth found - {out}/renders/depth_after_opt/ is empty '
                         f'or absent, so the run probably died before eval_rendering')
    if missing:
        print(f'WARNING: {len(missing)} of {K} keyframes have no render and were skipped: '
              f'{missing[:8]}{" ..." if len(missing) > 8 else ""}')

    # only the exported keyframes: the adapt stage takes its keyframe list from this file and
    # would otherwise look for depth files that were never written.
    # same convention as save_trajectory: TUM, camera-to-world
    poses_wc = SE3(poses).inv().data.cpu().numpy()
    np.savetxt(f'{out}/poses_slam.txt',
               np.concatenate([tstamp[kept][:, None], poses_wc[kept]], axis=1))
    print(f'wrote {ddir}/ {mdir}/ image/ poses_slam.txt to {out} ({len(kept)} keyframes)')

    if DEPTHS is None:
        return len(kept)

    # ---- compare the three candidate supervision sources against GT ----
    gtfiles = sorted(os.listdir(DEPTHS))

    # JDSA-aligned Omnidata prior, reusing geom/ba.py's bilinear scale field.
    # Inherently 1/8-res in the pipeline; bilinearly upsampled here so all three are comparable.
    prior_al, _ = get_prior_depth_aligned(torch.from_numpy(d['disps_prior']).cuda(),
                                          torch.from_numpy(d['dscales']).cuda())
    prior_al = F.interpolate(prior_al[:, None], size=(H, W), mode='bilinear',
                             align_corners=False)[:, 0]
    prior_depth = (1.0 / prior_al.clamp(min=1e-6)).cpu().numpy()

    pairs = {k: [] for k in ('slam', 'rendered', 'prior')}
    for i in range(K):
        idx = int(tstamp[i])
        gt = cv2.imread(os.path.join(DEPTHS, gtfiles[idx]), cv2.IMREAD_ANYDEPTH) / DEPTH_PNG_SCALE
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        valid = (gt > 0) & mask[i]
        if valid.sum() == 0:
            continue

        srcs = [('slam', depth[i]), ('prior', prior_depth[i])]
        rf = f'{out}/renders/depth_after_opt/{idx:06d}.png'
        if os.path.exists(rf):
            srcs.append(('rendered', cv2.imread(rf, cv2.IMREAD_ANYDEPTH) / DEPTH_PNG_SCALE))

        for name, pred in srcs:
            v = valid & (pred > 0)
            if v.sum() > 0:
                pairs[name].append((gt[v], pred[v]))

    print(f'\nscale-aligned depth L1 (m) vs GT, masked, over {len(pairs["slam"])} keyframes')
    print(f'  {"source":<34} {"per-frame":>10} {"global":>10}')
    print(f'  {"-" * 56}')
    for name, label in (('slam', 'SLAM depth (1/disps_up)'),
                        ('rendered', f'Gaussian-rendered ({len(pairs["rendered"])} kf)'),
                        ('prior', 'JDSA-aligned Omnidata prior')):
        if pairs[name]:
            print(f'  {label:<34} {l1_per_frame(pairs[name]):>10.4f} '
                  f'{l1_global(pairs[name]):>10.4f}')
        else:
            print(f'  {label:<34} {"n/a":>10} {"n/a":>10}')
    return len(kept)


# ==============================================================================
#  STAGE 2 - LoRA adaptation of VGGT
# ==============================================================================

class LoRALinear(nn.Module):
    """y = W0 x + (B A) x * alpha/r, with B zero-initialised so the adapter starts as identity."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scaling


def inject_lora(model):
    """Wrap the targeted Linears inside the aggregator's frame/global blocks."""
    blocks = list(model.aggregator.frame_blocks) + list(model.aggregator.global_blocks)
    if LORA_PATCH_EMBED and hasattr(model.aggregator.patch_embed, 'blocks'):
        blocks += list(model.aggregator.patch_embed.blocks)

    n = 0
    for blk in blocks:
        for tgt in LORA_TARGETS:
            parent_path, _, leaf = tgt.rpartition('.')
            parent = blk.get_submodule(parent_path) if parent_path else blk
            child = getattr(parent, leaf, None)
            if isinstance(child, nn.Linear):
                setattr(parent, leaf, LoRALinear(child, LORA_RANK, LORA_ALPHA))
                n += 1
    return n


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith('.A') or k.endswith('.B')}


def load_vggt(adapter=None):
    """VGGT-1B with LoRA injected, optionally loading an adapter. Shared by stage 2 and stage 3."""
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained(VGGT_WEIGHTS)
    model.point_head, model.track_head = None, None       # not supervised
    for p in model.parameters():
        p.requires_grad_(False)
    n = inject_lora(model)
    if adapter is not None:
        missing = model.load_state_dict(load_file(adapter), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    return model.cuda(), n


def vggt_forward(model, images):
    """Aggregator once; depth head on frame 0 only; camera head on everything."""
    tok, ps_idx = model.aggregator(images[None])
    # this build caches only layers 4/11/17/23 and leaves the rest None to save memory
    # (aggregator.py:196) - the frame slice must preserve those Nones
    tok0 = [t[:, :1] if t is not None else None for t in tok]
    depth, _ = model.depth_head(tok0, images[None][:, :1], ps_idx)
    pose_enc = model.camera_head(tok)[-1]
    return depth[0, 0, :, :, 0], pose_enc[0]


def tum_to_c2w(row):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
    T[:3, 3] = row[1:4]
    return T


def split_keyframes(kf):
    """Train / val split over the exported keyframe list.

    'stride' holds out every Nth keyframe, so val covers the whole trained region and the val
    curve is not confounded by the scene changing. Its caveat is that a held-out keyframe's
    neighbours are visually near-identical to trained ones, so it measures memorisation more than
    generalisation; 'contiguous' is the opposite trade.
    """
    kf = list(kf)
    if LORA_TRAIN_FRAC >= 1.0 or len(kf) < 5:
        return kf, []
    if LORA_SPLIT_MODE == 'stride':
        val = kf[::max(2, int(round(1.0 / (1.0 - LORA_TRAIN_FRAC))))]
    elif LORA_SPLIT_MODE == 'contiguous':
        val = kf[int(round(len(kf) * LORA_TRAIN_FRAC)):]
    elif LORA_SPLIT_MODE == 'random':
        perm = np.random.default_rng(SEED).permutation(len(kf))
        val = sorted(kf[i] for i in perm[int(round(len(kf) * LORA_TRAIN_FRAC)):])
    else:
        raise SystemExit(f'LORA_SPLIT_MODE={LORA_SPLIT_MODE!r} is not stride/contiguous/random')
    vset = set(val)
    return [t for t in kf if t not in vset], val


class SceneData:
    """One keyframe = one sample, placed FIRST in the sequence so VGGT's predictions land in that
    keyframe's coordinate frame (verified: extrinsic[0] is identity to 5e-4). Around it we attach
    a random number of neighbouring non-keyframe frames, so the adapter works both monocular - the
    way MotionFilter.prior_extractor calls it - and with a few frames of context."""

    def __init__(self, scene_dir, image_dir):
        self.scene_dir, self.image_dir = scene_dir, image_dir
        self.files = sorted(os.listdir(image_dir))

        self.ddir, self.mdir = f'depth_{DEPTH_SOURCE}', f'mask_{DEPTH_SOURCE}'
        if not os.path.isdir(f'{scene_dir}/{self.ddir}'):
            raise SystemExit(f'{scene_dir}/{self.ddir} not found - re-run the extract stage with '
                             f'DEPTH_SOURCE = {DEPTH_SOURCE!r}')

        traj = np.loadtxt(f'{scene_dir}/traj_full.txt')
        self.c2w = {int(r[0]): tum_to_c2w(r) for r in traj}
        self.t_min, self.t_max = int(traj[0, 0]), int(traj[-1, 0])
        self.kf = [int(t) for t in np.loadtxt(f'{scene_dir}/poses_slam.txt')[:, 0]]
        self.train_kf, self.val_kf = split_keyframes(self.kf)

        # intrinsics: stored at the tracker's resolution, rescale to the VGGT input size
        fx, fy, cx, cy = np.load(f'{scene_dir}/intrinsics.npy')
        probe = stream_resize(cv2.imread(os.path.join(image_dir, self.files[0])))
        self.stream_hw = probe.shape[:2]
        sy, sx = VGGT_HW[0] / probe.shape[0], VGGT_HW[1] / probe.shape[1]
        self.K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], np.float64)

    def frame(self, t):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, self.files[t])),
                           cv2.COLOR_BGR2RGB)
        img = cv2.resize(stream_resize(img), (VGGT_HW[1], VGGT_HW[0]),
                         interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def kf_target(self, t):
        d = np.load(f'{self.scene_dir}/{self.ddir}/{t:06d}.npy')
        m = cv2.imread(f'{self.scene_dir}/{self.mdir}/{t:06d}.png', cv2.IMREAD_GRAYSCALE) > 127
        d = cv2.resize(d, (VGGT_HW[1], VGGT_HW[0]), interpolation=cv2.INTER_NEAREST)
        m = cv2.resize(m.astype(np.uint8), (VGGT_HW[1], VGGT_HW[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
        return torch.from_numpy(d).float(), torch.from_numpy(m & (d > 0))

    def neighbours(self, t, rng, n_left, n_right):
        """Random non-keyframe neighbours within RADIUS; edge keyframes take from the other side."""
        left = [x for x in range(max(t - RADIUS, self.t_min), t) if x in self.c2w]
        right = [x for x in range(t + 1, min(t + RADIUS, self.t_max) + 1) if x in self.c2w]
        want = n_left + n_right
        n_left, n_right = min(n_left, len(left)), min(n_right, len(right))
        # keyframes at the sequence ends have no frames on one side; make up the shortfall from
        # the other side so the requested context size is still met where the frames exist
        n_right = min(n_right + (want - n_left - n_right), len(right))
        n_left = min(n_left + (want - n_left - n_right), len(left))
        picks = list(rng.choice(left, n_left, replace=False)) + \
            list(rng.choice(right, n_right, replace=False))
        return sorted(int(x) for x in picks)

    def sample(self, rng, t=None, single=None):
        from vggt.utils.pose_enc import extri_intri_to_pose_encoding
        t = int(rng.choice(self.train_kf)) if t is None else t
        if single is None:
            single = rng.random() < P_SINGLE_VIEW
        nb = [] if single else self.neighbours(t, rng, rng.integers(1, MAX_LEFT + 1),
                                               rng.integers(1, MAX_RIGHT + 1))
        seq = [t] + nb

        images = torch.stack([self.frame(x) for x in seq])
        gt_depth, mask = self.kf_target(t)

        # rebase every pose so the keyframe is the world origin -> frame 0 is identity
        kf_c2w = self.c2w[t]
        extr = np.stack([(np.linalg.inv(self.c2w[x]) @ kf_c2w)[:3] for x in seq])
        K = np.broadcast_to(self.K, (len(seq), 3, 3))
        gt_enc = extri_intri_to_pose_encoding(
            torch.from_numpy(extr).float()[None], torch.from_numpy(K.copy()).float()[None],
            image_size_hw=VGGT_HW)[0]
        return images, gt_depth, mask, gt_enc, seq


def _median_scale(pred, gt, mask):
    """Median ratio. Deliberately NOT detached.

    Detaching makes the loss only *look* scale-invariant: the optimiser then sees a gradient that
    rewards shrinking the prediction, even though rescaling is a no-op once the scale is recomputed
    on the next forward pass. Letting the gradient flow through the ratio makes the loss genuinely
    invariant, so it can only push shape, never overall magnitude.
    """
    return gt[mask].median() / pred[mask].median().clamp(min=1e-6)


def depth_loss(pred_depth, gt_depth, mask, scale=None):
    """Returns (loss, depth-space scale). The returned scale is always expressed in DEPTH terms,
    even when the loss is computed in disparity, so callers can compare it against the pose scale."""
    if mask.sum() < MIN_MASK_PIXELS:
        return pred_depth.sum() * 0.0, None
    p, g = pred_depth.clamp(min=1e-3), gt_depth.clamp(min=1e-3)
    inv = DEPTH_SPACE == 'disparity'
    if inv:
        p, g = 1.0 / p, 1.0 / g
    # a scale supplied by the caller (COUPLED_SCALE) is a depth scale; in disparity space the
    # equivalent multiplier is its reciprocal
    s = _median_scale(p, g, mask) if scale is None else (1.0 / scale if inv else scale)
    loss = (g[mask] - s * p[mask]).abs().mean()
    return loss, (1.0 / s if inv else s)


def pose_loss(pred_enc, gt_enc):
    """Translation (independently norm'd) + quaternion, over the non-reference frames."""
    if pred_enc.shape[0] < 2:
        z = pred_enc.sum() * 0.0
        return z, z, None
    tp, tg = pred_enc[1:, :3], gt_enc[1:, :3]
    # same reasoning as _median_scale: normalising by a *detached* predicted norm lets the
    # translations collapse toward zero at no loss cost. Keep the gradient in the normaliser.
    np_, ng = tp.norm(dim=-1).mean().clamp(min=1e-6), tg.norm(dim=-1).mean().clamp(min=1e-6)
    l_t = F.huber_loss(tp / np_, tg / ng)

    qp = F.normalize(pred_enc[1:, 3:7], dim=-1)
    qg = F.normalize(gt_enc[1:, 3:7], dim=-1)
    l_r = (1.0 - (qp * qg).sum(-1).abs()).mean()      # abs handles quaternion sign ambiguity
    return l_t, l_r, (ng / np_).detach()


@torch.no_grad()
def eval_depth(model, data, kfs):
    """Scale-aligned masked depth L1 over a keyframe subset, same metric as the export table."""
    if not kfs:
        return None
    if LORA_EVAL_MAX_KF and len(kfs) > LORA_EVAL_MAX_KF:
        pick = np.linspace(0, len(kfs) - 1, LORA_EVAL_MAX_KF).round().astype(int)
        kfs = [kfs[i] for i in sorted(set(pick.tolist()))]
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(SEED)
    errs = []
    for t in kfs:
        images, gt, mask, _, _ = data.sample(rng, t=t, single=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred, _ = vggt_forward(model, images.cuda())
        l, _ = depth_loss(pred.float(), gt.cuda(), mask.cuda())
        errs.append(l.item())
    model.train(was_training)
    return float(np.mean(errs))


def train_lora(scene_dir):
    """LoRA-adapt VGGT on the exported depth + poses, reporting train/val depth L1."""
    from safetensors.torch import save_file

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    out = f'{scene_dir}/lora-vggt'
    os.makedirs(out, exist_ok=True)

    data = SceneData(scene_dir, COLORS)
    print(f'scene {scene_dir}: {len(data.kf)} keyframes, frames {data.t_min}..{data.t_max}, '
          f'supervised on {data.ddir}/')
    print(f'split {LORA_SPLIT_MODE} @ {LORA_TRAIN_FRAC}: {len(data.train_kf)} train / '
          f'{len(data.val_kf)} val keyframes')
    if not data.val_kf:
        print('  note: empty val set - val eval and LORA_KEEP_BEST are disabled')

    sh_, sw = data.stream_hw
    skew = (VGGT_HW[1] / VGGT_HW[0]) / (sw / sh_)
    print(f'stream {sw}x{sh_} (aspect {sw/sh_:.3f}) -> VGGT {VGGT_HW[1]}x{VGGT_HW[0]} '
          f'(aspect {VGGT_HW[1]/VGGT_HW[0]:.3f}), squash {skew:.3f}x')
    if not 0.95 < skew < 1.05:
        print(f'  WARNING: aspect ratios differ by {abs(1-skew)*100:.0f}%. SceneData.frame() '
              f'resizes without letterboxing, so VGGT sees a distorted image. Consider VGGT_HW = '
              f'({14*round(518*sh_/sw/14)}, 518)')

    model, n_wrapped = load_vggt()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'LoRA r={LORA_RANK} on {n_wrapped} Linears -> {n_train/1e6:.2f}M trainable '
          f'/ {n_total/1e9:.2f}B ({100*n_train/n_total:.2f}%)')

    def evaluate_subsets(tag):
        row = {'tag': tag}
        if LORA_EVAL_ON_TRAIN:
            row['train_l1'] = eval_depth(model, data, data.train_kf)
        if LORA_EVAL_ON_VAL:
            row['val_l1'] = eval_depth(model, data, data.val_kf)
        cells = '  '.join(f'{k.split("_")[0]} {v:.4f}' for k, v in row.items()
                          if k != 'tag' and v is not None)
        if cells:
            print(f'  depth L1 [{tag:>5}]  {cells}')
        return row

    print(f'evaluating base VGGT ({DEPTH_SPACE} space, masked, scale-aligned):')
    history = [evaluate_subsets('base')]

    model.train()                        # enables the aggregator's gradient checkpointing
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    log, t0 = [], time.time()
    best = {'val_l1': float('inf'), 'epoch': None, 'state': None}

    if not data.train_kf:
        raise SystemExit('no training keyframes - lower LORA_TRAIN_FRAC or check the export')
    steps_per_epoch = math.ceil(len(data.train_kf) / BATCH_SIZE)
    print(f'{len(data.train_kf)} train keyframes / batch {BATCH_SIZE} = {steps_per_epoch} '
          f'optimiser steps per epoch, {EPOCHS * steps_per_epoch} in total')

    for epoch in range(EPOCHS):
        # every training keyframe exactly once per epoch, in a fresh order each time
        order = [int(t) for t in rng.permutation(data.train_kf)]
        run = []
        for step in range(steps_per_epoch):
            batch = order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]   # the tail batch is shorter
            opt.zero_grad(set_to_none=True)
            acc = {'loss': [], 'l_depth': [], 'l_trans': [], 'l_rot': [], 'scale_ratio': [],
                   'S': []}

            for t in batch:
                images, gt, mask, gt_enc, seq = data.sample(rng, t=t)
                images, gt, mask, gt_enc = images.cuda(), gt.cuda(), mask.cuda(), gt_enc.cuda()

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred_depth, pred_enc = vggt_forward(model, images)
                pred_depth, pred_enc = pred_depth.float(), pred_enc.float()

                l_t, l_r, pose_scale = pose_loss(pred_enc, gt_enc)
                l_d, depth_scale = depth_loss(pred_depth, gt, mask,
                                              scale=pose_scale if COUPLED_SCALE else None)
                loss = l_d + LAMBDA_POSE * (l_t + l_r)

                # divide before backward: accumulating the MEAN over the batch is what makes this
                # equivalent to one wider batch, and keeps the gradient magnitude (and so GRAD_CLIP
                # and LR) independent of BATCH_SIZE
                (loss / len(batch)).backward()

                acc['loss'].append(loss.item())
                acc['l_depth'].append(l_d.item())
                acc['l_trans'].append(l_t.item())
                acc['l_rot'].append(l_r.item())
                acc['S'].append(len(seq))
                # depth and pose scale agreed to 1% on the pretrained model; if they diverge during
                # training the adapter is breaking depth/pose consistency
                if pose_scale is not None and depth_scale is not None:
                    acc['scale_ratio'].append((depth_scale / pose_scale).item())

            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            opt.step()

            rec = {'epoch': epoch, 'step': step, 'kfs': batch, 'S': acc['S'],
                   **{k: float(np.mean(v)) for k, v in acc.items()
                      if k not in ('S',) and v}}
            log.append(rec)
            run.append(rec['loss'])

            if step % LOG_EVERY == 0:
                print(f'  e{epoch} s{step:4d}/{steps_per_epoch}  '
                      f'loss {np.mean(run[-LOG_EVERY:]):.4f}  (d {rec["l_depth"]:.4f} '
                      f't {rec["l_trans"]:.4f} r {rec["l_rot"]:.4f})  B={len(batch)} '
                      f'S={acc["S"]}  {torch.cuda.max_memory_allocated()/2**30:.1f}GiB')
        print(f'epoch {epoch}: mean loss {np.mean(run):.4f} over {len(order)} keyframes  '
              f'({time.time()-t0:.0f}s elapsed)')

        if LORA_EVAL_EVERY_EPOCH or epoch == EPOCHS - 1:
            row = evaluate_subsets(f'e{epoch}')
            history.append(row)
            v = row.get('val_l1')
            if LORA_KEEP_BEST and v is not None and v < best['val_l1']:
                best = {'val_l1': v, 'epoch': epoch, 'state': lora_state_dict(model)}

    keep = LORA_KEEP_BEST and best['state'] is not None
    save_file(best['state'] if keep else lora_state_dict(model), f'{out}/adapter.safetensors')

    cfg = {'rank': LORA_RANK, 'alpha': LORA_ALPHA, 'targets': list(LORA_TARGETS),
           'lora_patch_embed': LORA_PATCH_EMBED, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
           'steps_per_epoch': steps_per_epoch, 'samples_per_epoch': len(data.train_kf),
           'lr': LR, 'weight_decay': WEIGHT_DECAY, 'grad_clip': GRAD_CLIP,
           'lambda_pose': LAMBDA_POSE, 'depth_space': DEPTH_SPACE, 'depth_source': DEPTH_SOURCE,
           'coupled_scale': COUPLED_SCALE, 'p_single_view': P_SINGLE_VIEW,
           'max_left': MAX_LEFT, 'max_right': MAX_RIGHT, 'radius': RADIUS,
           'vggt_hw': list(VGGT_HW), 'weights': VGGT_WEIGHTS, 'scene': scene_dir,
           'trainable_params': n_train, 'seed': SEED,
           'split_mode': LORA_SPLIT_MODE, 'train_frac': LORA_TRAIN_FRAC,
           'n_train_kf': len(data.train_kf), 'n_val_kf': len(data.val_kf),
           'val_kf': data.val_kf, 'keep_best': LORA_KEEP_BEST,
           'saved_epoch': best['epoch'] if keep else EPOCHS - 1,
           'eval_history': history}
    json.dump(cfg, open(f'{out}/config.json', 'w'), indent=2)
    json.dump(log, open(f'{out}/train_log.json', 'w'))

    # ---- summary: the val row is the one that means something ----
    print(f'\ndepth L1 (masked, scale-aligned, {DEPTH_SPACE} space):')
    print(f'  {"":<8}' + ''.join(f'{r["tag"]:>10}' for r in history))
    for key, name in (('train_l1', 'train'), ('val_l1', 'val')):
        if any(r.get(key) is not None for r in history):
            print(f'  {name:<8}' + ''.join(
                f'{r[key]:>10.4f}' if r.get(key) is not None else f'{"n/a":>10}' for r in history))
    if LORA_KEEP_BEST and best['epoch'] is not None:
        print(f'  saved epoch {best["epoch"]} (best val L1 {best["val_l1"]:.4f})')
    print(f'saved adapter ({n_train/1e6:.1f}M params) to {out}')

    del model, trainable, opt, data
    free_vram()


# ==============================================================================
#  STAGE 3 - the A/B arms
# ==============================================================================

_VGGT_MODEL = None      # held only for the duration of a VGGT arm
_VGGT_HW = VGGT_HW


@torch.no_grad()
def _vggt_depth(images):
    """Depth for a single frame. Skips camera_head, and runs the DPT head on frame 0 only."""
    tok, ps_idx = _VGGT_MODEL.aggregator(images[None])
    tok0 = [t[:, :1] if t is not None else None for t in tok]
    depth, _ = _VGGT_MODEL.depth_head(tok0, images[None][:, :1], ps_idx)
    return depth[0, 0, :, :, 0]


@torch.amp.autocast('cuda', enabled=True)   # matches upstream prior_extractor's decorator
@torch.no_grad()
def vggt_prior_extractor(self, im_tensor):
    """Drop-in for MotionFilter.prior_extractor: VGGT depth, Omnidata normals."""
    from midas.omnidata import OmnidataModel
    from torchvision import transforms
    input_size = im_tensor.shape[-2:]

    # --- normals: unchanged from upstream (motion_filter.py:70-72), minus the depth model ---
    if getattr(self, 'omni_normal', None) is None:
        self.omni_normal = OmnidataModel('normal', OMNI_NORMAL_CKPT, device='cuda:0')
    resized = transforms.Resize(OMNI_NORMAL_HW, antialias=True)(im_tensor).cuda()
    normal = self.omni_normal(resized) * 2.0 - 1.0
    normal = F.interpolate(normal, input_size, mode='bicubic').float().squeeze()

    # --- depth: VGGT ---
    # motion_filter.py:88-89 hands us an ImageNet-NORMALISED tensor, but VGGT expects [0,1] and
    # normalises internally (aggregator.py:205). Undo it, or VGGT sees doubly-normalised input.
    rgb = (im_tensor * self.STDV + self.MEAN).clamp(0, 1)
    rgb = F.interpolate(rgb, _VGGT_HW, mode='bilinear', align_corners=False)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        depth = _vggt_depth(rgb.cuda())
    # bilinear, not bicubic: bicubic can overshoot to negative depth at edges
    depth = F.interpolate(depth.float()[None, None], input_size, mode='bilinear',
                          align_corners=False).squeeze().clamp(min=1e-3)
    return depth, normal


def install_vggt_prior(adapter):
    """Patch MotionFilter so the depth prior comes from VGGT. Normals stay Omnidata, so depth is
    the only variable between the arms."""
    global _VGGT_MODEL, _VGGT_HW
    from motion_filter import MotionFilter

    _VGGT_HW = VGGT_HW
    # The adapter was trained at one input size and must be run at that size, so the value written
    # into its config.json wins over VGGT_HW. Only the un-adapted arm is free to take VGGT_HW.
    cfg = os.path.join(os.path.dirname(adapter or ''), 'config.json')
    if adapter and os.path.exists(cfg):
        _VGGT_HW = tuple(json.load(open(cfg))['vggt_hw'])
        if tuple(VGGT_HW) != _VGGT_HW:
            print(f'note: ignoring VGGT_HW, adapter was trained at {_VGGT_HW[0]},{_VGGT_HW[1]}')

    _VGGT_MODEL = load_vggt(adapter=adapter)[0].eval()
    MotionFilter.prior_extractor = vggt_prior_extractor
    which = f'LoRA-adapted VGGT ({adapter})' if adapter else 'base VGGT-1B (no adapter)'
    print(f'depth prior: {which} at {_VGGT_HW[1]}x{_VGGT_HW[0]}')
    print('normals    : Omnidata (unchanged, so depth is the only variable)')
    return f'{"VGGT+LoRA" if adapter else "base VGGT"} depth / Omnidata normals'


def run_ate(out):
    """evo with Sim3 alignment. Returns (overall rmse, per-frame errors, timestamps)."""
    sh(f'cd {out} && evo_ape tum {os.path.abspath(GT_TRAJ)} traj_full.txt -vas '
       f'--save_results evo.zip --no_warnings > ape.txt 2>&1')
    sh(f'rm -rf {out}/evo && unzip -q {out}/evo.zip -d {out}/evo')
    err = np.load(f'{out}/evo/error_array.npy')
    ts = np.load(f'{out}/evo/timestamps.npy')
    return float(np.sqrt((err ** 2).mean())), err, ts


def run_mesh(out):
    """TSDF fuse, Sim3-align (mandatory: SLAM scale is arbitrary), then score against the GT mesh."""
    import open3d as o3d
    w = MESH_WEIGHT

    # tsdf_integrate builds an Open3D VoxelBlockGrid on cuda:0; without releasing what the just
    # finished SLAM run is still holding, that subprocess OOMs and dies silently
    free_vram()

    # Open3D's marching-cubes allocates a large assistance structure on the GPU and fails at fine
    # voxel sizes when the shared card is busy. Retry coarser rather than losing the metric - but
    # record which size won, because the two arms MUST be compared at the same voxel size.
    raw = f'{out}/tsdf_mesh_w{w:.1f}.ply'
    voxel_used, r = None, None
    for vs in (VOXEL_SIZE, *VOXEL_FALLBACKS):
        if os.path.exists(raw):
            os.remove(raw)
        r = sh(f'cd {_ROOT} && python tsdf_integrate.py --result {out} '
               f'--voxel_size {vs} --weight {w}')
        if os.path.exists(raw):
            voxel_used = vs
            break
        print(f'  [mesh] tsdf_integrate failed at voxel_size={vs} (rc={r.returncode})')
    if voxel_used is None:
        print('   ', (r.stderr or r.stdout).strip().splitlines()[-2:])
        return None
    if voxel_used != VOXEL_SIZE:
        print(f'  [mesh] fell back to voxel_size={voxel_used}; the other arm must match')

    # without this the mesh sits in SLAM units and every number is ~50x off; ICP inside
    # eval_recon is rigid-only and cannot recover scale
    mesh = o3d.io.read_triangle_mesh(raw)
    mesh.transform(np.load(f'{out}/evo/alignment_transformation_sim3.npy'))
    aligned = f'{out}/tsdf_mesh_w{w:.1f}_aligned.ply'
    o3d.io.write_triangle_mesh(aligned, mesh)

    res = f'{out}/eval_recon.txt'
    r = sh(f'cd {_ROOT} && python scripts/eval_recon.py {aligned} {os.path.abspath(GT_MESH)} '
           f'--eval_3d --save {res}')
    if not os.path.exists(res):
        print(f'  [mesh] eval_recon failed (rc={r.returncode}):')
        print('   ', (r.stderr or r.stdout).strip().splitlines()[-3:])
        return None
    out_d = eval(open(res).read(), {'np': np, 'array': np.array})
    out_d['voxel_size'] = voxel_used
    return out_d


def split_render_metrics(out, split_at):
    """Recompute PSNR/SSIM and depth L1 per frame from the saved renders, then split.

    eval_rendering only reports sequence means, but it writes every render named by original frame
    index, so the seen/unseen breakdown can be recovered without re-rendering. One global depth
    scale is fitted across all frames (SLAM units are arbitrary) and the errors are then split -
    fitting per half would hide exactly the drift we are looking for.
    """
    from gaussian.utils.loss_utils import psnr, ssim
    files = sorted(os.listdir(COLORS))
    gtd = sorted(os.listdir(DEPTHS)) if DEPTHS else None
    rows = []

    for f in sorted(os.listdir(f'{out}/renders/image_after_opt')):
        idx = int(f[:-4])
        render = cv2.imread(f'{out}/renders/image_after_opt/{f}')
        gt = stream_resize(cv2.imread(os.path.join(COLORS, files[idx])))
        if render is None or gt is None or render.shape != gt.shape:
            continue
        r = torch.from_numpy(render[..., ::-1].copy()).permute(2, 0, 1).float().cuda() / 255.
        g = torch.from_numpy(gt[..., ::-1].copy()).permute(2, 0, 1).float().cuda() / 255.
        m = g > 0
        row = {'idx': idx,
               'psnr': psnr(r[m].unsqueeze(0), g[m].unsqueeze(0)).mean().item(),
               'ssim': ssim(r.unsqueeze(0), g.unsqueeze(0)).item()}

        dp = f'{out}/renders/depth_after_opt/{idx:06d}.png'
        if gtd and os.path.exists(dp):
            pred = cv2.imread(dp, cv2.IMREAD_ANYDEPTH) / DEPTH_PNG_SCALE
            gd = cv2.imread(os.path.join(DEPTHS, gtd[idx]),
                            cv2.IMREAD_ANYDEPTH) / DEPTH_PNG_SCALE
            gd = cv2.resize(gd, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            v = (gd > 0) & (pred > 0)
            if v.sum() > 0:
                row['_d'] = (gd[v], pred[v])
        rows.append(row)

    # one global median-ratio scale over every frame, then split the per-frame errors
    dv = [r['_d'] for r in rows if '_d' in r]
    if dv:
        s = np.median(np.concatenate([g for g, _ in dv])) / \
            np.median(np.concatenate([p for _, p in dv]))
        for r in rows:
            if '_d' in r:
                g, p = r.pop('_d')
                r['depth_l1'] = float(np.abs(g - s * p).mean())

    def agg(sel):
        sub = [r for r in rows if sel(r['idx'])]
        if not sub:
            return None
        o = {'n': len(sub), 'psnr': float(np.mean([r['psnr'] for r in sub])),
             'ssim': float(np.mean([r['ssim'] for r in sub]))}
        d = [r['depth_l1'] for r in sub if 'depth_l1' in r]
        if d:
            o['depth_l1'] = float(np.mean(d))
        return o

    return {'all': agg(lambda i: True), 'seen': agg(lambda i: i < split_at),
            'unseen': agg(lambda i: i >= split_at)}


def evaluate(out, label, split_at):
    res = {'label': label, 'output': out, 'split_at': split_at}

    ate, err, ts = run_ate(out)
    res['ate_all'] = ate
    for name, sel in (('seen', ts < split_at), ('unseen', ts >= split_at)):
        res[f'ate_{name}'] = float(np.sqrt((err[sel] ** 2).mean())) if sel.sum() else None

    pj = f'{out}/psnr/after_opt/final_result.json'
    if os.path.exists(pj):
        res['hislam2_eval'] = json.load(open(pj))

    res['render'] = split_render_metrics(out, split_at)
    res['mesh'] = run_mesh(out) if GT_MESH else None

    json.dump(res, open(f'{out}/ab_results.json', 'w'), indent=2, default=float)
    return res


def print_report(res):
    k = res['split_at']
    print(f"\n{'='*66}\n  {res['label']}  ->  {res['output']}\n{'='*66}")
    print(f"  {'metric':<22}{'all':>13}{f'seen <{k}':>13}{f'unseen >={k}':>15}")
    print(f"  {'-'*63}")
    print(f"  {'ATE RMSE (m)':<22}{res['ate_all']:>13.4f}"
          f"{res.get('ate_seen') or float('nan'):>13.4f}"
          f"{res.get('ate_unseen') or float('nan'):>15.4f}")
    r = res['render']
    for key, name in (('psnr', 'PSNR (dB)'), ('ssim', 'SSIM'), ('depth_l1', 'depth L1 (m)')):
        vals = [(r[s] or {}).get(key) for s in ('all', 'seen', 'unseen')]
        if any(v is not None for v in vals):
            cells = ''.join(f'{v:>13.4f}' if i < 2 else f'{v:>15.4f}'
                            if v is not None else f'{"n/a":>13}' for i, v in enumerate(vals))
            print(f'  {name:<22}{cells}')
    print(f"  {'frames evaluated':<22}{(r['all'] or {}).get('n', 0):>13}"
          f"{(r['seen'] or {}).get('n', 0):>13}{(r['unseen'] or {}).get('n', 0):>15}")
    if res.get('mesh'):
        m = res['mesh']
        print(f"\n  mesh (whole sequence, Sim3-aligned, voxel {m['voxel_size']}): "
              f"acc {m['mean precision']:.4f} m  comp {m['mean recall']:.4f} m  "
              f"comp-ratio {100*m['recall']:.1f}%  F {m['f-score']:.3f}")
    print()


def compare(labels, res):
    """Side-by-side table: baseline absolute, then absolute + delta for every other arm."""
    base = res[0]
    k = base['split_at']

    # an arm run at a different split or over a different frame count is not comparable, however
    # tempting the numbers look side by side
    for lbl, r in zip(labels[1:], res[1:]):
        if r['split_at'] != k:
            raise SystemExit(f"  !! {lbl} used split_at={r['split_at']}, baseline used {k} - "
                             'the arms are not comparable; delete its output and re-run')
        n0, n1 = (base['render']['all'] or {}).get('n'), (r['render']['all'] or {}).get('n')
        if n0 != n1:
            print(f'  !! {lbl} evaluated {n1} frames, baseline {n0} - arms are not comparable')

    print(f'  full-sequence comparison, split at frame {k}')
    print(f"  {'metric':<26}{labels[0]:>12}" +
          ''.join(f'{l:>12}{"delta":>11}' for l in labels[1:]))
    print('  ' + '-' * (26 + 12 + 23 * (len(labels) - 1)))

    def row(name, vals, better_low=True):
        if vals[0] is None:
            return
        line = f'  {name:<26}{vals[0]:>12.4f}'
        for v in vals[1:]:
            if v is None:
                line += f'{"n/a":>12}{"":>11}'
                continue
            d = v - vals[0]
            mark = ' ' if abs(d) < 1e-9 else ('+' if (d < 0) == better_low else '-')
            line += f'{v:>12.4f}{d:>+9.4f} {mark}'
        print(line)

    for s in ('all', 'seen', 'unseen'):
        row(f'ATE RMSE ({s})', [r.get(f'ate_{s}') for r in res])
    print()
    for s in ('all', 'seen', 'unseen'):
        for m, low in (('psnr', False), ('ssim', False), ('depth_l1', True)):
            row(f'{m} ({s})', [(r['render'].get(s) or {}).get(m) for r in res], low)
        print()

    meshes = [r.get('mesh') for r in res]
    if all(meshes):
        voxels = {m['voxel_size'] for m in meshes}
        if len(voxels) > 1:
            print(f'  !! voxel sizes differ ({sorted(voxels)}) - mesh numbers are NOT comparable; '
                  're-run with the same VOXEL_SIZE')
        else:
            for key, name, low in (('mean precision', 'mesh accuracy (m)', True),
                                   ('mean recall', 'mesh completion (m)', True),
                                   ('recall', 'mesh comp-ratio', False),
                                   ('f-score', 'mesh F-score', False)):
                row(name, [m[key] for m in meshes], low)
    elif any(meshes):
        print('  mesh metrics unavailable for at least one arm')

    print("\n  '+' better than baseline, '-' worse.")
    print("  'unseen' is the row that matters: it is the only evidence the adaptation")
    print('  generalises rather than having memorised the keyframes it trained on.')


# ==============================================================================
#  stages
# ==============================================================================

def stage_extract(extract_length):
    banner(f'1/3 extract  -> {OUT_EXTRACT}')
    # the ONLY run that gets the keyframe knobs; the arms in stage_test get CONFIG untouched
    cfg = write_extract_config(OUT_EXTRACT)

    if SKIP_EXISTING and os.path.exists(f'{OUT_EXTRACT}/slam_depth.npz'):
        print(f'{OUT_EXTRACT}/slam_depth.npz exists - skipping the SLAM run')
    else:
        gpu_gate()
        t0 = time.time()
        # GT depth goes to the EXPORT only, never to Hi2: eval_utils.py:50-52 zeroes the rendered
        # depth wherever GT is invalid, and on real sensors (TUM: 24% holes, on exactly the hard
        # surfaces) that would both shrink the training set and tie its mask to where the Kinect
        # happened to work. The export table masks on (gt > 0) & mask anyway.
        n_kf = run_slam(OUT_EXTRACT, cfg, extract_length, EXTRACT_BUFFER,
                        gtdepthdir=None, dump_slam_depth=True)
        print(f'=== SLAM done in {time.time()-t0:.0f}s: {n_kf} keyframes over {extract_length} '
              f'frames (1 per {extract_length/max(n_kf,1):.1f}). For more, lower '
              f'EXTRACT_KF_REDUNDANT_THRESH ({EXTRACT_KF_REDUNDANT_THRESH}) first, then '
              f'EXTRACT_KF_MOTION_THRESH ({EXTRACT_KF_MOTION_THRESH}) - the redundancy gate binds')
        if n_kf >= EXTRACT_BUFFER:
            print(f'WARNING: keyframe count hit EXTRACT_BUFFER ({EXTRACT_BUFFER})')

    with tee(f'{OUT_EXTRACT}/export.txt'):
        n_exported = export_slam_depth(OUT_EXTRACT)
    free_vram('extract')
    print(f'{n_exported} keyframes exported to {OUT_EXTRACT}/depth_{DEPTH_SOURCE}/')


def stage_adapt(adapter):
    banner(f'2/3 adapt  -> {adapter}')
    if SKIP_EXISTING and os.path.exists(adapter):
        print(f'{adapter} exists - skipping')
        return
    if not os.path.exists(f'{OUT_EXTRACT}/poses_slam.txt'):
        raise SystemExit(f'no {OUT_EXTRACT}/poses_slam.txt - run the extract stage first')
    gpu_gate()
    t0 = time.time()
    train_lora(OUT_EXTRACT)
    free_vram('adapt')
    print(f'=== adapt done in {time.time()-t0:.0f}s')


def stage_test(adapter, split_at):
    banner(f'3/3 test  -> {OUT_TEST}')
    from motion_filter import MotionFilter
    global _VGGT_MODEL

    # The arms must run stock tracking. The EXTRACT_KF_* knobs shape the training-data run only:
    # if the generated config leaked in here, a denser-keyframe extract would silently also mean
    # denser keyframes in the A/B, and neither arm would be comparable with any earlier run.
    # Cheap to assert, and the failure it prevents is invisible in the output.
    arm_config = CONFIG
    assert os.path.abspath(arm_config) != os.path.abspath(f'{OUT_EXTRACT}/extract_config.yaml'), \
        'the A/B arms must use the base CONFIG, not the extract run derived config'
    print(f'tracking config for every arm: {arm_config} (unmodified; the EXTRACT_KF_* knobs '
          f'apply to the extract run only)')

    # capture the stock prior ONCE: install_vggt_prior patches the class, and the patch would
    # otherwise leak into a later Omnidata arm and silently make it a second VGGT arm
    stock_prior = MotionFilter.prior_extractor

    labels, results = [], []
    for arm in ARMS:
        out = f'{OUT_TEST}/{SCENE}_{ARM_DIRS[arm]}'
        labels.append(ARM_NAMES[arm])
        if SKIP_EXISTING and os.path.exists(f'{out}/ab_results.json'):
            res = json.load(open(f'{out}/ab_results.json'))
            print(f'=== {arm}: ab_results.json exists, skipping ({res["label"]})')
            results.append(res)
            continue

        banner(f'arm {arm} -> {out}')
        MotionFilter.prior_extractor = stock_prior       # undo any previous arm's patch
        label = 'Omnidata depth (baseline)'
        if arm != 'omnidata':
            a = adapter if arm == 'vggt_lora' else None
            if a and not os.path.exists(a):
                raise SystemExit(f'no adapter at {a} - run the adapt stage first')
            label = install_vggt_prior(a)

        gpu_gate()
        t0 = time.time()
        n_kf = run_slam(out, arm_config, TEST_LENGTH, TEST_BUFFER, gtdepthdir=DEPTHS)
        print(f'{label}: SLAM done in {time.time()-t0:.0f}s, {n_kf} keyframes')

        _VGGT_MODEL = None                                # ~2.5 GB, not needed by the evaluation
        free_vram(f'arm {arm}')
        res = evaluate(out, label, split_at)
        print_report(res)
        results.append(res)
        free_vram(f'arm {arm} eval')

    MotionFilter.prior_extractor = stock_prior
    banner('comparison')
    compare(labels, results)


# ==============================================================================

def main():
    # must happen before any Process is started, and only once per process
    torch.multiprocessing.set_start_method('spawn', force=True)
    os.chdir(_ROOT)                       # every relative path above is repo-root relative

    needed = [COLORS, CALIB, CONFIG, DROID_WEIGHTS]
    if 'test' in STAGES:
        needed += [GT_TRAJ] + ([GT_MESH] if GT_MESH else [])   # run_ate / run_mesh need these
    for f in needed:
        if not os.path.exists(f):
            raise SystemExit(f'missing input: {f}')
    if UNDISTORT or CROP_BORDER:
        print('WARNING: undistorting at runtime - split_render_metrics re-derives the GT frame '
              'with a resize only, so renders and GT will not line up (ARCHITECTURE.md §10.1)')

    n_frames = len(os.listdir(COLORS))
    # every consumer indexes GT depth by RGB frame number, so these must be 1:1
    for name, path in (('depths', DEPTHS), ('traj', GT_TRAJ)):
        if path is None:
            continue
        n = len(os.listdir(path)) if os.path.isdir(path) else len(np.loadtxt(path))
        if n != n_frames:
            raise SystemExit(f'{path} has {n} entries but {COLORS} has {n_frames}; they must be '
                             f'1:1 by index. Re-run scripts/preprocess_tum.py.')

    extract_length = n_frames * FRACTION // 100
    if extract_length < 20:
        raise SystemExit(f'{n_frames} frames * {FRACTION}% = {extract_length}, too few to track')
    split_at = extract_length
    adapter = f'{OUT_EXTRACT}/lora-vggt/adapter.safetensors'

    print(f'sequence  : {SCENE}  ({n_frames} frames, {COLORS})')
    print(f'config    : {CONFIG}  calib {CALIB}')
    print(f'adapter   : trains on frames 0..{extract_length-1} ({FRACTION}%), '
          f'evaluated on 0..{n_frames-1}')
    print(f'target    : depth_{DEPTH_SOURCE}/   split at frame {split_at}')
    print(f'stages    : {" ".join(STAGES)}   arms: {" ".join(ARMS)}')

    t_all = time.time()
    if 'extract' in STAGES:
        stage_extract(extract_length)
    if 'adapt' in STAGES:
        stage_adapt(adapter)
    if 'test' in STAGES:
        stage_test(adapter, split_at)

    print(f'\nall stages done in {time.time()-t_all:.0f}s')
    print('\nread first:')
    print(f'  {OUT_EXTRACT}/export.txt   per-frame vs global depth L1 columns. The gap on the')
    print('                             Omnidata row is the cross-frame scale inconsistency this')
    print('                             track targets - if it is small, there was no headroom.')
    print("  the table above            'unseen' rows only; 'seen' is the adapter's training")


if __name__ == '__main__':
    main()
