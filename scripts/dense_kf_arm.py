"""ONE end2end arm at a DIFFERENT KEYFRAME DENSITY - the control the comparison never had.

    python scripts/dense_kf_arm.py      # from the repo root, adaslam venv active

  -> outputs/test/end2end/<SCENE>/omni_dense/   traj_full.txt, results.json, evo/, ape.txt
                                                + dense_config.yaml, the config that produced it
  -> a two-row compare() table against the `omni` arm already on disk

WHY THIS IS NOT AN END2END_PRIORS ENTRY. Every adapter that reaches -24% on rellis_00000 trained on
a DENSIFIED extract - init_adapt_pipeline.py's EXTRACT halves the keyframe gates, giving ~5.0
frames per keyframe against stock's ~12.2 - while the best whole-sequence adapter reaches -20.45% at
a matched keyframe count and matched target shape quality. Density and span are perfectly confounded
in every extract on disk, and that 3.6-point difference is the largest unexplained gap left.

But density only ever touched an adapter's TRAINING DATA. Every arm ever scored ran at stock
keyframing: run_end2end_test hands every arm the unmodified CONFIG on purpose, and init's main()
asserts it, precisely so a denser training set cannot masquerade as a tracking change (9.2.1). So
"does denser keyframing improve the trajectory on its own?" has never been asked, and the end2end
stage cannot ask it - one arm_config serves every arm of a comparison.

Hence a script of its own. It changes ONE thing against `omni`, the four kf_* knobs, and it is the
only place that knows `omnidata_dense` means "BASE_SPEC's prior, tracked densely".

WHAT IT REUSES. Everything; there is no new logic here. write_tracking_config writes the generated
YAML, SlamRunner runs it (both already take the config and the output directory as arguments),
make_prior turns BASE_SPEC into a prior object, and evaluate / print_report / compare score it
exactly as an ordinary arm - same results.json shape, same evo layout, so ate_over_time.py and
export_end2end_results.py read it with no special case.

The arm's directory is INFERRED, `arm_name(DENSE_SPEC)` (7.1), never typed: end2end/config.py's
SENTINELS is what makes `omnidata_dense` listable in another driver's END2END_PRIORS, and a name
typed here would be a second naming rule to drift out of step with it.

AFTERWARDS, to have it in a comparison table: add 'omnidata_dense' to END2END_PRIORS in any driver,
with SKIP_EXISTING=True. It is then reused from disk at no GPU cost. Without the arm on disk (or
with SKIP_EXISTING off) end2end/stage.py:make_prior refuses rather than running a stock-density arm
into a directory named for a dense one.
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
from adaslam.pipeline import check_sequence, enter, resolve_lora, warn_runtime_undistort
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
# DENSE_SPEC names the arm (-> omni_dense), BASE_SPEC supplies the prior. They are separate because
# the whole point is that the two runs share a prior and differ only in keyframe density; pointing
# BASE_SPEC at 'vggt_base' or an adapter directory would need its own DENSE_SPEC sentinel first,
# since two specs inferring one directory is a hard error (End2EndConfig.__post_init__).
DENSE_SPEC = OMNIDATA_DENSE
BASE_SPEC  = OMNIDATA
LABEL      = 'Omnidata depth, DENSE keyframing'

# The arm this is a control FOR: same prior, stock keyframing. Read from disk for the table; the
# comparison is the entire deliverable, so a missing baseline is a hard stop rather than a warning.
BASELINE_ARM = arm_name(BASE_SPEC)

# ---------------------------------------------------------------- keyframe density (the variable)
# Identical to init_adapt_pipeline.py's EXTRACT, so this run's density is the density that trained
# the -24% adapters. Stock, inherited from tum_config.yaml, is 2.4 / 4.0 / 4.0 / 0.2.
KF_MOTION_THRESH    = 1.2       # motion_filter.thresh - flow needed to propose a keyframe
KF_INIT_THRESH      = 4.0       # motion_filter.init_thresh - the same gate before initialisation
KF_REDUNDANT_THRESH = 2.0       # frontend.keyframe_thresh - the gate that actually moves the
                                # count: track_frontend.py:49-52 prunes back whatever the motion
                                # filter proposes (9.2.1)
KF_COVIS_THRESH     = 0.1       # backend.covis_thresh - extras inserted in terminate(); LOWER=more

# ---------------------------------------------------------------- run control
LENGTH           = 100000       # frames to run over; 100000 = the whole sequence
START            = 0
STREAM_RES       = 341 * 640    # tracking resolution budget - must match every other arm
BUFFER           = 900          # NO overflow guard in SlamRunner.run (only run_extract warns).
                                # dense_kf_p40 gave 226 keyframes over 1138 frames, so the whole
                                # sequence should give ~565 - 2.4x stock's 233. Check the count
                                # this prints; if it approaches BUFFER, raise it and re-run.
RENDER_EVAL      = False        # hi2.py's eval_rendering -> renders/ + psnr/ (11)
MIN_FREE_VRAM_MB = 7000
OUT_ROOT         = 'outputs'
SKIP_EXISTING    = True         # reuse a finished run; scoring re-runs when the split differs

# The seen/unseen boundary. MUST equal the baseline arm's recorded split_at or compare() refuses
# the pair - which is the check working, not an obstacle. None = the whole sequence, which is what
# `omni` on rellis_00000 was scored at (2847).
SPLIT_AT = 200

# VGGT's input size. Unread while BASE_SPEC is 'omnidata' - no VGGT is built - but LoRAConfig is a
# required End2EndConfig field, so it is stated rather than faked. None = derive from the stream.
VGGT_HW = None

# ---------------------------------------------------------------- the SLAM run
SLAM = SlamConfig(
    weights=DROID_WEIGHTS, colors=COLORS, calib=CALIB, start=START,
    undistort=UNDISTORT, crop_border=CROP_BORDER, stream_res=STREAM_RES,
    render_eval=RENDER_EVAL)

LORA = LoRAConfig(
    weights='pretrained_models/vggt',
    vggt_hw=VGGT_HW,
    rank=8, alpha=16,
    targets=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
    patch_embed=False)

# evaluate() reads only gt_traj and make_prior reads nothing at all for 'omnidata', but the real
# config is built rather than a stand-in: it is what makes BASE_SPEC a knob instead of a fiction.
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
    """The stock-density arm's results.json, checked for comparability. Hard stop if absent.

    A dense arm on its own says nothing - the number only means something beside the arm it is a
    control for - so this is checked BEFORE the SLAM run rather than after it.
    """
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
    length = min(LENGTH, n_frames)
    split_at = n_frames if SPLIT_AT is None else SPLIT_AT

    out_root = test_dir(OUT_ROOT, 'end2end', SCENE)
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
    print(f'run       : frames 0..{length-1}, buffer {BUFFER}')

    if SKIP_EXISTING and os.path.exists(f'{out}/traj_full.txt'):
        print(f'\n{out}/traj_full.txt exists - reusing the SLAM run, re-scoring at '
              f'split_at={split_at}')
    else:
        gpu_gate(MIN_FREE_VRAM_MB)
        # into the ARM's own directory, so the config that produced the trajectory sits beside it -
        # the same provenance rule extract_config.yaml follows inside <exp>/full/ (7.1)
        dense_config = write_tracking_config(
            out, CONFIG, motion_thresh=KF_MOTION_THRESH, init_thresh=KF_INIT_THRESH,
            keyframe_thresh=KF_REDUNDANT_THRESH, covis_thresh=KF_COVIS_THRESH,
            name='dense_config.yaml')

        # BASE_SPEC, never DENSE_SPEC: make_prior refuses the latter by design, and rightly - it is
        # this script, not that function, that knows density comes from the config above
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
