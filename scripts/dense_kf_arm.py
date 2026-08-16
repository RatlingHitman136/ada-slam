"""ONE end2end arm at a DIFFERENT KEYFRAME DENSITY - the control the comparison never had.

    python scripts/dense_kf_arm.py      # from the repo root, adaslam venv active

  -> outputs/test/end2end/<SCENE>/omni_dense/   traj_full.txt, results.json, evo/, ape.txt
                                                + dense_config.yaml, the config that produced it
  -> a two-row compare() table against the `omni` arm already on disk

Density only ever touched an adapter's TRAINING DATA: run_end2end_test hands every arm the
unmodified CONFIG on purpose (9.2.1), so the end2end stage cannot ask whether denser keyframing
improves the trajectory on its own. Hence a script of its own, changing ONE thing against `omni`.

No new logic - write_tracking_config, SlamRunner, make_prior and evaluate / print_report / compare
are the ordinary arm path, so the outputs read like any other arm's. The directory is INFERRED,
arm_name(DENSE_SPEC) (7.1), never typed.

To have it in a comparison table afterwards: add 'omnidata_dense' to any driver's END2END_PRIORS
with SKIP_EXISTING=True, and it is reused from disk at no GPU cost.
"""
import os    # nopep8
import sys   # nopep8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # nopep8
# repo root, so `adaslam` imports; its __init__ adds hislam2/ and thirdparty/vggt
sys.path.insert(0, _ROOT)                                                 # nopep8
import json
import time
from dataclasses import replace

from adaslam.adapt import LoRAConfig
from adaslam.common import test_dir
from adaslam.end2end import End2EndConfig
from adaslam.end2end.config import OMNIDATA, OMNIDATA_DENSE, arm_name
from adaslam.end2end.metrics import RESULTS, evaluate
from adaslam.end2end.report import compare, print_report
from adaslam.end2end.stage import make_prior
from adaslam.pipeline import (check_sequence, enter, resolve_lora, scene_key,
                              warn_runtime_undistort, window_frames)
from adaslam.print_utils import banner
from adaslam.runtime import ensure_venv_on_path, gpu_gate, raise_fd_limit
from adaslam.slam import SlamConfig, SlamRunner, write_tracking_config

# ==============================================================================
#  PARAMETERS
# ==============================================================================

# ---------------------------------------------------------------- data (preprocessing is NOT here)
SCENE   = 'rellis_00000'               # names the outputs/ tree
DATA    = 'data/RELLIS/00000'
COLORS  = f'{DATA}/colors'
GT_TRAJ = f'{DATA}/traj_tum.txt'
CALIB   = f'{DATA}/calib.txt'
CONFIG  = 'config/rellis_config.yaml'  # the base every arm shares - NOT edited, inherited from
DROID_WEIGHTS = 'pretrained_models/droid.pth'

# undistort offline in the preprocess script instead (10.1)
UNDISTORT   = False
CROP_BORDER = 0

# ---------------------------------------------------------------- what this arm is
# DENSE_SPEC names the arm (-> omni_dense), BASE_SPEC supplies the prior it shares with `omni`
DENSE_SPEC = OMNIDATA_DENSE
BASE_SPEC  = OMNIDATA
LABEL      = 'Omnidata depth, DENSE keyframing'

# the arm this is a control FOR: same prior, stock keyframing. A missing baseline is a hard stop
BASELINE_ARM = arm_name(BASE_SPEC)

# ---------------------------------------------------------------- keyframe density (the variable)
# identical to init_adapt_pipeline.py's EXTRACT; stock (from the base config) is 2.4/4.0/4.0/0.2
KF_MOTION_THRESH    = 1.2       # motion_filter.thresh - flow needed to propose a keyframe
KF_INIT_THRESH      = 4.0       # motion_filter.init_thresh - the same gate before initialisation
KF_REDUNDANT_THRESH = 2.0       # frontend.keyframe_thresh - the gate that moves the count (9.2.1)
KF_COVIS_THRESH     = 0.1       # backend.covis_thresh - extras inserted in terminate(); LOWER=more

# ---------------------------------------------------------------- run control
LENGTH           = 100000       # frames to run over; 100000 = the whole sequence
START            = 0
STOP             = None         # exclusive: the window is [START, STOP); None = to the end

# a windowed run keys its own outputs tree, or its omni/base would overwrite the full sequence's
SCENE_KEY = scene_key(SCENE, START, STOP)
STREAM_RES       = 341 * 640    # tracking resolution budget - must match every other arm
BUFFER           = 900          # hard cap, NO overflow guard here; the run prints its count
RENDER_EVAL      = False        # hi2.py's eval_rendering -> renders/ + psnr/ (11)
MIN_FREE_VRAM_MB = 7000
OUT_ROOT         = 'outputs'
SKIP_EXISTING    = True         # reuse a finished run; scoring re-runs when the split differs

# the seen/unseen boundary; MUST equal the baseline arm's recorded split_at or compare() refuses
SPLIT_AT = 200

# VGGT's input size - unread while BASE_SPEC is 'omnidata', but End2EndConfig requires a LoRAConfig
VGGT_HW = None

# ---------------------------------------------------------------- the SLAM run
SLAM = SlamConfig(
    weights=DROID_WEIGHTS, colors=COLORS, calib=CALIB, start=START, stop=STOP,
    undistort=UNDISTORT, crop_border=CROP_BORDER, stream_res=STREAM_RES,
    render_eval=RENDER_EVAL)

LORA = LoRAConfig(
    weights='pretrained_models/vggt',
    vggt_hw=VGGT_HW,
    rank=8, alpha=16,
    targets=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
    patch_embed=False)

# the real config, not a stand-in: it is what makes BASE_SPEC a knob instead of a fiction
END2END = End2EndConfig(
    priors=(BASE_SPEC,),
    length=LENGTH,
    buffer=BUFFER,
    gt_traj=GT_TRAJ,
    lora=LORA,
    omni_normal_ckpt='pretrained_models/omnidata_dpt_normal_v2.ckpt',
    omni_normal_hw=(512, 512))

# ==============================================================================


def baseline_results(out_root, split_at):
    """The stock-density arm's results.json, checked for comparability. Hard stop if absent."""
    path = f'{out_root}/{BASELINE_ARM}/{RESULTS}'
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found: this run is a control for the {BASELINE_ARM!r} arm '
                         f'and there is nothing to compare it against. Run any driver with '
                         f'{BASE_SPEC!r} in END2END_PRIORS first.')
    res = json.load(open(path))
    if res['split_at'] != split_at:
        raise SystemExit(f'{BASELINE_ARM} was scored at split_at={res["split_at"]} and this run '
                         f'would use {split_at} - compare() refuses that pair. Set SPLIT_AT = '
                         f'{res["split_at"]} (or delete {path} and let it be re-scored).')
    return res


def main():
    global LORA, END2END           # resolved below, exactly as the drivers do
    enter(_ROOT)
    ensure_venv_on_path()
    raise_fd_limit()

    banner(f'dense-keyframing control arm  ({DENSE_SPEC})')
    n_frames = check_sequence(COLORS, gt_traj=GT_TRAJ, required=(CONFIG, CALIB, DROID_WEIGHTS))
    warn_runtime_undistort(UNDISTORT, CROP_BORDER)
    window = window_frames(n_frames, START, STOP)
    length = min(LENGTH, window)
    # None = the end of the RUN, not of the sequence: a split the run never reached is no boundary
    split_at = (START + window) if SPLIT_AT is None else SPLIT_AT

    out_root = test_dir(OUT_ROOT, 'end2end', SCENE_KEY)
    out = f'{out_root}/{arm_name(DENSE_SPEC)}'
    base_res = baseline_results(out_root, split_at)

    LORA, stream_hw = resolve_lora(LORA, COLORS, STREAM_RES)   # after chdir, before any spawn
    END2END = replace(END2END, lora=LORA)   # unread at BASE_SPEC='omnidata'; correct if it changes

    print(f'sequence  : {SCENE}  ({n_frames} frames, {COLORS})')
    print(f'arm       : {arm_name(DENSE_SPEC)}  -> {out}')
    print(f'prior     : {BASE_SPEC}  (unchanged - keyframe density is the ONLY variable)')
    print(f'baseline  : {BASELINE_ARM}  ate_all {base_res["ate_all"]:.4f} over '
          f'{base_res.get("n_all")} poses, split_at {split_at}')
    print(f'density   : motion {KF_MOTION_THRESH} / init {KF_INIT_THRESH} / redundant '
          f'{KF_REDUNDANT_THRESH} / covis {KF_COVIS_THRESH}  (stock: 2.4 / 4.0 / 4.0 / 0.2)')
    print(f'run       : frames {START}..{START+length-1} of {n_frames}, buffer {BUFFER}'
          + (f'   WINDOW -> outputs tree {SCENE_KEY}' if SCENE_KEY != SCENE else ''))

    if SKIP_EXISTING and os.path.exists(f'{out}/traj_full.txt'):
        print(f'\n{out}/traj_full.txt exists - reusing the SLAM run, re-scoring at '
              f'split_at={split_at}')
    else:
        gpu_gate(MIN_FREE_VRAM_MB)
        # into the ARM's own directory, so the config that produced the trajectory sits beside it
        dense_config = write_tracking_config(
            out, CONFIG, motion_thresh=KF_MOTION_THRESH, init_thresh=KF_INIT_THRESH,
            keyframe_thresh=KF_REDUNDANT_THRESH, covis_thresh=KF_COVIS_THRESH,
            name='dense_config.yaml')

        # BASE_SPEC, never DENSE_SPEC: make_prior refuses the latter, and the density is the config
        prior = make_prior(BASE_SPEC, END2END, stream_hw)
        try:
            t0 = time.time()
            res = SlamRunner(SLAM).run(out, dense_config, length, BUFFER,
                                       gtdepthdir=None, prior=prior)
            per_kf = res.n_frames / max(res.n_kf, 1)
            print(f'\n{LABEL}: SLAM done in {time.time()-t0:.0f}s, {res.n_kf} keyframes over '
                  f'{res.n_frames} frames = {per_kf:.1f} frames/keyframe '
                  f'(the -24% extracts sit at ~5.0, stock at ~12.2)')
            if res.n_kf > 0.9 * BUFFER:
                print(f'  WARNING: {res.n_kf} keyframes against BUFFER={BUFFER} - too close. '
                      f'Raise BUFFER and re-run; there is no overflow guard.')
        finally:
            if prior is not None:
                prior.release()

    print_report(evaluate(out, LABEL, split_at, END2END))

    banner('comparison')
    print(f'  the ONLY difference between these two arms is keyframe density - same prior, same '
          f'stream,\n  same calibration, same tracking config otherwise\n')
    compare([BASELINE_ARM, arm_name(DENSE_SPEC)],
            [base_res, json.load(open(f'{out}/{RESULTS}'))])
    print(f'\n  to have this arm in another driver\'s table: add {DENSE_SPEC!r} to its '
          f'END2END_PRIORS\n  with SKIP_EXISTING=True - it is then reused from disk at no GPU cost')


if __name__ == '__main__':
    main()
