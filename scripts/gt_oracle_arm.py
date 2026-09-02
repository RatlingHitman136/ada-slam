"""THE CEILING: one end2end arm whose depth prior is the dataset's ground truth.

    python scripts/gt_oracle_arm.py     # from the repo root, adaslam venv active

Every arm so far has asked "how good a depth prior can we LEARN". This one asks the prior question:
**how good would ATE be if the depth prior were perfect?** It replaces the prior generator with a
lookup into `DEPTHS/`, runs the ordinary full-sequence SLAM, and scores it with the same evo call
every other arm uses, into the same tree - so its number drops straight into the existing table
beside `omni` (31.279) and the adapted arms (22.7-28.3 on rellis_00000).

What that number means, and what it does not:

  * it is an UPPER BOUND on what any depth prior can buy, and therefore the yardstick that says
    whether the remaining 22.7 -> ? gap is worth chasing at all;
  * ITS OWN ATE IS NOT CIRCULAR. evo scores against traj_tum.txt, and `depths/` never enters that
    metric: the two are different products of the same rig, not the same quantity. GT depth is the
    64-beam Ouster sweep projected into the camera and Delaunay-densified; the GT trajectory is
    RELLIS's poses.txt composed with the camera-in-lidar extrinsic. Handing the tracker perfect
    STRUCTURE does not hand it its MOTION - it still has to solve for the trajectory, which is
    exactly the thing being measured. Sim(3) alignment (evo -vas) also fits scale, so the depth
    being metric buys nothing on the reported number; the whole advantage is shape and cross-frame
    consistency, which is what we want to bound.
  * IT IS STILL NOT A METHOD, for a narrower reason: its OUTPUTS are contaminated for downstream
    use. A LoRA trained on this run's `depth_slam/` would be learning LiDAR ground truth laundered
    through the tracker, and scoring that adapter on the same sequence's GT trajectory is leakage
    of a kind no deployable prior enjoys. **Never use this run as an adapt target or an extract -
    only as a number.** Hence its own arm directory and the `oracle` in ARM_NAME.
  * One honest second-order caveat: depth and trajectory share the rig and the extrinsic
    calibration A, so a systematic calibration error is common-mode here in a way it is not for a
    learned prior. That flatters the oracle slightly; it does not invalidate it as a ceiling.

THREE DESIGN DECISIONS, all of them load-bearing:

1. GT IS NOT DENSE. RELLIS depth covers only **52-74% of pixels** (LiDAR projection; sky, near-range
   and occlusion drop-outs are 0). A prior with holes is not "perfect", it is broken, so FILL
   decides what a quarter to a half of every prior contains. The default puts GT_MAX_DEPTH (250 m)
   there - "no data, treat as far" - because that is what JDSA handles best: it weights the 2x2
   scale grid by prior disparity SQUARED, so a 250 m hole carries 1/625 of a real pixel's weight
   and cannot capture the grid, while staying strictly positive keeps it clear of two unguarded
   divisions that a literal 0 would hit. The full argument is at the FILL knob; 'stock' (Omnidata,
   rescaled to GT) and 'nearest' remain available and answer slightly different questions.

2. NORMALS STAY OMNIDATA, as in every other arm (end2end/prior.py's rule), so depth remains the
   only variable and this number is comparable with the rest of the table. A fuller oracle would
   also derive normals from GT depth; that is a different experiment and it needs HI-SLAM2's normal
   convention pinned down first.

3. THE EXTRACTOR IS NOT TOLD WHICH FRAME IT IS LOOKING AT. `MotionFilter.prior_extractor(self,
   im_tensor)` receives pixels and nothing else, and `skip_blur` is ON for this scene
   (config/rellis_config.yaml inherits tum_config.yaml), so on a blur skip the tensor belongs to a
   CACHED EARLIER frame, not to the `tstamp` that triggered the call (motion_filter.py:115).
   Guessing the index would silently pair frame N's image with frame M's depth, which is the worst
   failure this script could have - it would look like a result. So we tap `MotionFilter.track`,
   fingerprint every image it is handed, and match the extractor's tensor back to its frame
   exactly; the round trip through ImageNet normalisation is lossless to ~1e-6, so a real match
   scores ~0 and anything above FP_TOL aborts the run rather than guessing.

Compare against: `outputs/test/end2end/rellis_00000/{omni,base,...}` - same tree, same split_at,
same evo invocation. scripts/export_end2end_results.py will pick it up with the others.
"""
import os    # nopep8
import sys   # nopep8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # nopep8
# repo root, so `adaslam` imports; its __init__ adds hislam2/ and thirdparty/vggt
sys.path.insert(0, _ROOT)                                                 # nopep8
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from adaslam.common import probe_stream_hw, require_name, test_dir
from adaslam.end2end import End2EndConfig
from adaslam.end2end.metrics import evaluate
from adaslam.adapt import LoRAConfig
from adaslam.pipeline import check_sequence, enter, scene_key, warn_runtime_undistort, window_frames
from adaslam.runtime import ensure_venv_on_path, gpu_gate, raise_fd_limit
from adaslam.slam import SlamConfig, SlamRunner, stock_prior_extractor

# ==============================================================================
#  PARAMETERS
# ==============================================================================

# ---------------------------------------------------------------- data
SCENE   = 'kitti_00_fg2a05'
DATA    = 'data/KITTI/00'
COLORS  = f'{DATA}/colors'
DEPTHS  = f'{DATA}/depths'             # THE PRIOR, here - not just the accuracy table
GT_TRAJ = f'{DATA}/traj_tum.txt'
CALIB   = f'{DATA}/calib.txt'
CONFIG  = 'config/kitti_fg2_a05_config.yaml'
DROID_WEIGHTS = 'pretrained_models/droid.pth'

DEPTH_PNG_SCALE = 256.0                # metres = png / this. MUST match the dataset (TUM 6553.5)
GT_MIN_DEPTH    = 0.5                  # below this a GT pixel is treated as a hole, not geometry
# The far end of everything this prior may emit: the ceiling of the valid GT range AND the value a
# hole is filled with (FILL='far'). One constant for both, so the invariant is simply "every pixel
# this arm returns lies in [GT_MIN_DEPTH, GT_MAX_DEPTH]" and 250 m reads as "no data, treat as far".
# Real RELLIS returns top out near 50 m, so raising the ceiling from 60 admits no new geometry - it
# only makes room for the fill. MUST stay <= 1000 m: the tracker floors its own disparity at 0.001
# (depth_video.py:270, ba.py:105/217), so a prior beyond that asks for something disps cannot reach.
GT_MAX_DEPTH    = 1000.0

UNDISTORT   = False                    # must match every other arm, or the pixels differ
CROP_BORDER = 0

# ---------------------------------------------------------------- run control
SKIP_EXISTING    = False                # reuse the arm if traj_full.txt is already there
MIN_FREE_VRAM_MB = 7000
START            = 0
STOP             = 1000                # exclusive; None = to the end
LENGTH           = 100000              # 100000 = the whole window
BUFFER           = 500                 # hard cap on keyframes; MUST exceed the count
STREAM_RES       = 341 * 640           # MUST equal the other arms' tracking budget
RENDER_EVAL      = False

# only so this arm lands in the same table as the init driver's: THAT driver splits at
# START + window*FRACTION//100 and results.json records it, and report.py refuses to put arms with
# different split_at side by side. There is no train/test split for an oracle - everything is
# "seen" in the sense that matters, which is that none of it is a held-out claim.
FRACTION = 7

SCENE_KEY = scene_key(SCENE, START, STOP)
OUT_ROOT  = 'outputs'
ARM_NAME  = 'gt_oracle'                # names outputs/test/end2end/<scene>/<ARM_NAME>

# ---------------------------------------------------------------- the oracle itself
# What goes where GT has no pixel. GT covers 52-74% of the frame on this dataset, so this is not a
# detail - it decides what a quarter to a half of every prior actually contains.
#
# WHY 'far' IS THE DEFAULT, from what JDSA does with a prior disparity (geom/ba.py:183-196):
#   * the scale-grid Jacobian is Jso = -m * disps_prior * Jbi, so the normal equations weight a
#     pixel by disps_prior SQUARED. A hole at GT_MAX_DEPTH is disparity 0.004 against a real pixel
#     at 10 m = 0.1, i.e. 625x less weight each; at ~35% holes the whole hole population is ~0.09%
#     of what the valid pixels contribute. It cannot capture the 2x2 grid, which is the failure
#     that would corrupt the whole keyframe's prior scale.
#   * the only rejection test is m = (disps_prior > 0), so a hole must be POSITIVE to stay out of
#     trouble elsewhere: track_frontend.py:42 initialises dscale as disps.median() /
#     disps_prior.median() over ALL pixels, which divides by zero once holes are the majority
#     (frame 2846 is 52.4% covered), and depth_video.py:71 writes 1.0/depth into disps_prior_up
#     with no guard at all. A small positive disparity keeps both finite; a 0 does not.
# The cost, stated plainly: a far-filled hole is silent in the SCALE grid but not in the disparity
# update - m=1 admits it with the same per-pixel weight alpha as a real measurement and pulls that
# pixel toward infinity. Correct for sky, wrong for the LiDAR's near-range and occlusion dropouts.
#
#   'far'     GT_MAX_DEPTH metres everywhere GT is 0. The default, for the reasons above.
#   'stock'   the upstream Omnidata depth, rescaled per frame by the median ratio on the pixels
#             where GT exists: the `omni` baseline's own answer in the holes. Plausible geometry
#             rather than "far", but it puts real disparity mass into the scale-grid solve.
#   'nearest' nearest valid GT pixel (distance transform). Self-consistent but invents the most:
#             it turns sky into whatever the horizon was, at horizon disparity.
FILL = 'far'

# A fingerprint distance above this means the extractor was handed an image `track` never saw, so
# the frame cannot be identified and the run is aborted. The normalise/un-normalise round trip is
# good to ~1e-6, so this is four orders of magnitude of headroom, not a tuned threshold.
FP_TOL = 1e-3

# ==============================================================================

# At module scope, not in main(): a spawned child re-executes this module and needs both.
raise_fd_limit()
ensure_venv_on_path()

# likewise at module scope: these bound what the prior may emit, and a violation would show up as a
# plausible ATE rather than as an error
if not 0 < GT_MIN_DEPTH < GT_MAX_DEPTH:
    raise SystemExit(f'gt_oracle: need 0 < GT_MIN_DEPTH ({GT_MIN_DEPTH}) < GT_MAX_DEPTH '
                     f'({GT_MAX_DEPTH})')
if GT_MAX_DEPTH > 1000.0:
    raise SystemExit(
        f'gt_oracle: GT_MAX_DEPTH={GT_MAX_DEPTH} exceeds 1000 m, the furthest the tracker can '
        f'represent - it floors its own disparity at 0.001 (depth_video.py:270, ba.py:105/217), so '
        f'a prior beyond that asks disps to converge somewhere it is clamped out of.')
if GT_MIN_DEPTH < 0.1:
    raise SystemExit(
        f'gt_oracle: GT_MIN_DEPTH={GT_MIN_DEPTH} is nearer than 0.1 m, where the tracker discards '
        f'its own disparity (> 10 -> 0, ba.py:216). JDSA weights the scale grid by disparity '
        f'SQUARED, so pixels this near dominate the solve for that keyframe.')


# ==============================================================================
#  Frame identification (design decision 3)
# ==============================================================================

def _fingerprint(rgb01):
    """A frame's identity: an 8x8 average-pooled thumbnail of the [0,1] RGB the tracker was given.

    Cheap, and exact enough to be an identity rather than a similarity - the query is a float
    round trip of a tensor already in the table, so the true match scores ~1e-6 and every other
    frame of a moving camera scores >1e-2.
    """
    x = rgb01.reshape(-1, *rgb01.shape[-3:]).float()
    return F.adaptive_avg_pool2d(x, 8).flatten().detach().cpu()


class FrameTap:
    """Every (tstamp, fingerprint) `MotionFilter.track` is handed, so the extractor can look up.

    The whole run, not a rolling window: hi2.terminate() calls prior_extractor again for the
    keyframes it inserts into low-covisibility gaps (hi2.py:143), and those images were streamed
    long before. ~192 floats per frame, so a 3k-frame sequence costs about 2 MB.
    """

    def __init__(self):
        self.tstamps = []
        self.fps = []
        self._stacked = None

    def record(self, tstamp, image_u8):
        self.tstamps.append(int(tstamp))
        self.fps.append(_fingerprint(image_u8.float() / 255.0))
        self._stacked = None

    def match(self, rgb01):
        """The tstamp whose image this is. Raises rather than guess - see the module docstring."""
        if not self.fps:
            raise SystemExit('gt_oracle: prior_extractor ran before MotionFilter.track - the tap '
                             'is installed too late to identify any frame')
        if self._stacked is None:
            self._stacked = torch.stack(self.fps)
        d = (self._stacked - _fingerprint(rgb01)[None]).abs().amax(dim=1)
        i = int(d.argmin())
        best = float(d[i])
        if best > FP_TOL:
            raise SystemExit(
                f'gt_oracle: could not identify the frame this prior call is for (closest of '
                f'{len(self.fps)} recorded frames is tstamp {self.tstamps[i]} at distance '
                f'{best:.2e} > FP_TOL={FP_TOL:g}). Pairing the wrong GT depth with an image would '
                f'produce a plausible-looking and completely meaningless ATE, so this aborts '
                f'instead.')
        return self.tstamps[i]

    def install(self):
        """Wrap MotionFilter.track. Returns the original so main() can restore it."""
        from motion_filter import MotionFilter
        tap, stock = self, MotionFilter.track

        def track(mf, tstamp, image, intrinsics=None, is_last=False):
            # BEFORE the call: track() may replace `tstamp`/`image` from its blur cache and then
            # invoke prior_extractor within the same call, so the pair must already be recorded
            tap.record(tstamp, image)
            return stock(mf, tstamp, image, intrinsics=intrinsics, is_last=is_last)

        MotionFilter.track = track
        return stock


# ==============================================================================
#  The prior
# ==============================================================================

def _resize_depth(depth, res):
    """stream_resize's geometry with NEAREST sampling - the tracker's pixels, no invented depth.

    Not stream_resize itself: that one interpolates, which would average across depth
    discontinuities and, worse, smear the 0 that marks an invalid pixel into its neighbours so
    holes stop being detectable.
    """
    h0, w0 = depth.shape[:2]
    h1 = int(h0 * np.sqrt(res / (h0 * w0)))
    w1 = int(w0 * np.sqrt(res / (h0 * w0)))
    return cv2.resize(depth, (w1 - w1 % 8, h1 - h1 % 8), interpolation=cv2.INTER_NEAREST)


class GtDepthPrior:
    """GT depth from disk + Omnidata normals. The same shape as end2end/prior.py:VggtPrior."""

    def __init__(self, cfg, files, stream_hw=None):
        self.cfg = cfg
        self.files = list(files)          # window_files order == the tstamps the tracker uses
        self.tap = FrameTap()
        self.label = f'GT depth ({FILL}-filled) / Omnidata normals'
        self._cache = {}
        self.n_calls = 0
        self.coverage = []

        # captured HERE, not inside the extractor: SlamRunner.run overwrites the class attribute
        # with ours, and a lazy fetch from inside our own extractor would fetch itself and recurse
        # forever (slam/stock_prior.py spells this out)
        self._stock = stock_prior_extractor()

        print(f'depth prior: GROUND TRUTH from {DEPTHS}/  (holes filled: {FILL})')
        print('normals    : Omnidata (unchanged, so depth is the only variable)')
        print('             THIS IS AN ORACLE. Its ATE is honest (evo scores against traj_tum.txt, '
              'not against depth),\n             but its OUTPUTS are not: never use this run as an '
              'adapt target or an extract.')
        if stream_hw is not None:
            print(f'             GT resized to {stream_hw[1]}x{stream_hw[0]} (NEAREST)')

    # ------------------------------------------------------------------ GT
    def gt_depth(self, tstamp):
        """(depth_m, valid) at tracking resolution for the frame the tracker calls `tstamp`.

        The file is found by NAME, not by START + tstamp: window_files() is what mono_stream
        streams and its order is the only definition of which frame a tstamp is.
        """
        if tstamp in self._cache:
            return self._cache[tstamp]
        if not 0 <= tstamp < len(self.files):
            raise SystemExit(f'gt_oracle: tstamp {tstamp} outside the {len(self.files)}-frame '
                             f'window - the tap and the file list disagree')
        stem = os.path.splitext(self.files[tstamp])[0]
        path = os.path.join(DEPTHS, f'{stem}.png')
        raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if raw is None:
            raise SystemExit(f'gt_oracle: {path} not found or unreadable (frame {stem}). Every '
                             f'streamed frame needs a GT depth; there is no fallback by design.')
        d = raw.astype(np.float32) / DEPTH_PNG_SCALE
        if UNDISTORT or CROP_BORDER:
            raise SystemExit('gt_oracle: UNDISTORT/CROP_BORDER are not applied to the GT depth, '
                             'so the prior would be misaligned with the image. Undistort offline '
                             'in the preprocess script instead (10.1).')
        d = _resize_depth(d, STREAM_RES)
        valid = (d >= GT_MIN_DEPTH) & (d <= GT_MAX_DEPTH) & np.isfinite(d)
        out = (torch.from_numpy(d), torch.from_numpy(valid))
        self._cache[tstamp] = out
        return out

    # ------------------------------------------------------------------ the hook
    def extractor(self):
        """A plain FUNCTION for MotionFilter.prior_extractor - never a bound method (9.3)."""
        prior = self

        @torch.amp.autocast('cuda', enabled=True)   # matches upstream prior_extractor's decorator
        @torch.no_grad()
        def prior_extractor(mf, im_tensor):
            # the stock arm's own output: normals verbatim, depth only as hole filler
            stock_depth, normal = prior._stock(mf, im_tensor)
            stock_depth = stock_depth.float()

            # which frame is this? see design decision 3
            rgb = (im_tensor * mf.STDV + mf.MEAN).clamp(0, 1)
            t = prior.tap.match(rgb)

            gt, valid = prior.gt_depth(t)
            gt = gt.to(stock_depth.device)
            valid = valid.to(stock_depth.device)
            if gt.shape != stock_depth.shape:
                raise SystemExit(
                    f'gt_oracle: GT depth is {tuple(gt.shape)} but the tracker runs at '
                    f'{tuple(stock_depth.shape)}. STREAM_RES here must equal the other arms\'.')

            n_valid = int(valid.sum())
            prior.coverage.append(n_valid / valid.numel())
            if n_valid < 64:
                raise SystemExit(f'gt_oracle: frame {t} has {n_valid} valid GT pixels - nothing to '
                                 f'anchor a fill scale to. Widen GT_MIN/MAX_DEPTH or drop the '
                                 f'frame from the window.')

            if FILL == 'far':
                filler = torch.full_like(stock_depth, float(GT_MAX_DEPTH))
            elif FILL == 'stock':
                # put the filler in GT's units before using it, or the two halves of the prior
                # disagree by a constant and JDSA sees a discontinuity at every hole edge
                s = gt[valid].median() / stock_depth[valid].median().clamp(min=1e-6)
                filler = s * stock_depth
            elif FILL == 'nearest':
                filler = prior._nearest_fill(gt, valid).to(stock_depth.device)
            else:
                raise SystemExit(f"gt_oracle: FILL={FILL!r} is not 'far' | 'stock' | 'nearest'")

            # BOUND THE FILLER, not the output. The obvious `.clamp(min=1e-3)` on the returned map
            # is a trap here: depth_video.py:73 inverts unconditionally, so 1e-3 m becomes DISPARITY
            # 1000, and JDSA weights the scale grid by disparity squared (ba.py:190-196) - one such
            # pixel carries 1e6 times a normal pixel's weight and drags the whole keyframe's 2x2
            # grid to fit it. A floor on the output would manufacture exactly that; clamping the
            # filler into the GT's own range prevents it. `gt` is already inside [GT_MIN_DEPTH,
            # GT_MAX_DEPTH] by the `valid` mask, so afterwards the whole map is, by construction -
            # no output clamp is needed and none is applied.
            # Inert for 'far' (the fill IS GT_MAX_DEPTH); it bites on 'stock', whose Omnidata
            # source can go near zero locally even after the median rescale.
            filler = filler.clamp(GT_MIN_DEPTH, GT_MAX_DEPTH)

            depth = torch.where(valid, gt, filler)
            prior.n_calls += 1
            return depth, normal

        return prior_extractor

    @staticmethod
    def _nearest_fill(gt, valid):
        """Nearest valid GT pixel, by distance transform on the hole mask.

        DIST_LABEL_PIXEL labels every pixel with the id of its closest ZERO pixel of the input, so
        the input is the hole mask INVERTED (valid = 0) and the label is the nearest valid pixel.
        The label ids are 1-based over those zeros, so the lookup table is built by reading the
        labels back at the valid pixels themselves.
        """
        v = valid.detach().cpu().numpy()
        g = gt.detach().cpu().numpy()
        _, idx = cv2.distanceTransformWithLabels(
            (~v).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(v)
        lut = np.zeros(int(idx.max()) + 1, np.float32)
        lut[idx[ys, xs]] = g[ys, xs]
        return torch.from_numpy(lut[idx])

    def release(self):
        self._cache.clear()


# ==============================================================================

def main():
    enter(_ROOT)                       # once per process, before any Process is started
    require_name('ARM_NAME', ARM_NAME)

    n_frames = check_sequence(COLORS, DEPTHS, GT_TRAJ,
                              required=[CALIB, CONFIG, DROID_WEIGHTS, GT_TRAJ])
    warn_runtime_undistort(UNDISTORT, CROP_BORDER)

    window = window_frames(n_frames, START, STOP)
    split_at = START + window * FRACTION // 100    # only to match the other arms' results.json

    slam = SlamConfig(weights=DROID_WEIGHTS, colors=COLORS, calib=CALIB,
                      start=START, stop=STOP, undistort=UNDISTORT, crop_border=CROP_BORDER,
                      stream_res=STREAM_RES, render_eval=RENDER_EVAL)
    # lora is unread by this arm - nothing here builds a VGGT - but End2EndConfig requires one and
    # evaluate() takes the whole config for gt_traj
    e2e = End2EndConfig(priors=(ARM_NAME,), length=LENGTH, buffer=BUFFER, gt_traj=GT_TRAJ,
                        lora=LoRAConfig(weights='pretrained_models/vggt', vggt_hw=None, rank=8,
                                        alpha=16, targets=('attn.qkv',), patch_embed=False),
                        omni_normal_ckpt='pretrained_models/omnidata_dpt_normal_v2.ckpt',
                        omni_normal_hw=(512, 512))

    out_root = test_dir(OUT_ROOT, 'end2end', SCENE_KEY)
    out = f'{out_root}/{ARM_NAME}'
    os.makedirs(out_root, exist_ok=True)

    stream_hw = probe_stream_hw(COLORS, STREAM_RES)
    print(f'sequence  : {SCENE}  ({n_frames} frames, {COLORS})')
    print(f'stream    : {stream_hw[1]}x{stream_hw[0]}   window {START}..'
          f'{START + window - 1}  split_at {split_at}')
    print(f'GT depth  : {DEPTHS}/  scale {DEPTH_PNG_SCALE}  kept {GT_MIN_DEPTH}-{GT_MAX_DEPTH} m')
    print(f'arm       : {out}')

    if SKIP_EXISTING and os.path.exists(f'{out}/traj_full.txt'):
        print(f'=== {ARM_NAME}: traj_full.txt exists, re-scoring only')
        res = evaluate(out, f'GT depth ({FILL}-filled) / Omnidata normals', split_at, e2e)
        print(f'\nATE (Sim3-aligned) = {res["ate_all"]:.3f} m   over {res["n_all"]} poses')
        return

    gpu_gate(MIN_FREE_VRAM_MB)

    from adaslam.slam.stream import window_files
    from motion_filter import MotionFilter

    prior = GtDepthPrior(e2e, window_files(slam)[:LENGTH], stream_hw)
    stock_track = prior.tap.install()      # must be live before the first track() call
    t0 = time.time()
    try:
        SlamRunner(slam).run(out, CONFIG, LENGTH, BUFFER, prior=prior)
    finally:
        MotionFilter.track = stock_track   # never leak the tap into another run in this process
        prior.release()

    cov = np.array(prior.coverage) if prior.coverage else np.zeros(1)
    print(f'\nprior served {prior.n_calls} keyframes in {time.time()-t0:.0f}s; '
          f'GT covered {cov.mean()*100:.1f}% of pixels on average '
          f'({cov.min()*100:.1f}-{cov.max()*100:.1f}%), the rest {FILL}-filled')

    res = evaluate(out, f'GT depth ({FILL}-filled) / Omnidata normals', split_at, e2e)
    print(f'\n=== CEILING: ATE (Sim3-aligned) = {res["ate_all"]:.3f} m over {res["n_all"]} poses')
    print('    compare with the arms in ' + out_root + ' at the same split_at:')
    print('      omni 31.279   base 33.224   best adapted ~22.7 (single draw, sd ~1.75)')
    print('    the gap between this row and the adapted ones is what a better prior could still buy')


if __name__ == '__main__':
    main()
