"""INITIAL adaptation: extract a dense PREFIX of the sequence, adapt on it, test (9.1).

    python scripts/init_adapt_pipeline.py    # from the repo root, adaslam venv active

  1 extract  HI-SLAM2 over the first FRACTION%, keyframes DENSIFIED by the EXTRACT kf_* knobs
  2 adapt    LoRA-adapt VGGT on that depth, from stock VGGT-1B or another adapter
  3 end2end  one full-sequence arm per generator in END2END_PRIORS, then ATE side by side
  4 prior    the same generators vs GT depth directly, no SLAM run

Siblings: cont_adapt_pipeline.py (9.7) adapts on a thin sample of the whole sequence;
online_adapt_pipeline.py (13) adapts during the run itself. Run the preprocess script first.

The KNOB PANEL, not the implementation - every stage is a package under adaslam/. The config
literals below are rebuilt in every spawned reader child: primitives only, no disk, no computation.
"""
import os    # nopep8
import sys   # nopep8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # nopep8
# repo root, so `adaslam` imports; its __init__ adds hislam2/ and thirdparty/vggt
sys.path.insert(0, _ROOT)                                                 # nopep8
import time
from dataclasses import replace

from adaslam.adapt import AdaptConfig, LoRAConfig, run_adapt
from adaslam.common import (ADAPT_CKPT_SUBDIR, DEPTH_DIR, experiment_dir, extract_run_dir,
                            require_name, test_dir)
from adaslam.end2end import End2EndConfig, run_end2end_test
from adaslam.extract import ExtractConfig, run_extract
from adaslam.pipeline import (check_sequence, enter, print_arm_dirs, resolve_lora,
                              scene_key, warn_runtime_undistort, window_frames)
from adaslam.priortest import PriorTestConfig, run_prior_test
from adaslam.runtime import ensure_venv_on_path, gpu_gate, raise_fd_limit
from adaslam.slam import SlamConfig, SlamRunner

# ==============================================================================
#  PARAMETERS
# ==============================================================================

# ---------------------------------------------------------------- data (preprocessing is NOT here)
# TUM:    SCENE 'rgbd_dataset_freiburg1_room', DATA f'data/TUM/{SCENE}', config/tum_config.yaml,
#         DEPTH_PNG_SCALE 6553.5, PRIOR eval 0.1-10 m
SCENE   = 'rellis_00000'               # names the outputs/ tree
DATA    = 'data/RELLIS/00000'          # preprocess_rellis3d.py's output layout
COLORS  = f'{DATA}/colors'
DEPTHS  = f'{DATA}/depths'             # None if the dataset has no GT depth
GT_TRAJ = f'{DATA}/traj_tum.txt'
CALIB   = f'{DATA}/calib.txt'
CONFIG  = 'config/rellis_config.yaml'
DROID_WEIGHTS = 'pretrained_models/droid.pth'

# undistort offline in preprocess_tum.py instead - consumers re-derive a frame with stream_resize
# alone, so doing it here misaligns predictions and GT (10.1)
UNDISTORT   = False
CROP_BORDER = 0

# ---------------------------------------------------------------- run control
STAGES           = ('prior',)          # any subset; run in pipeline order
SKIP_EXISTING    = True                # reuse a stage's output if it is already on disk
MIN_FREE_VRAM_MB = 7000                # shared GPU: checked once at the start of main()
FRACTION         = 7                   # % of the window the adapter trains on; also SPLIT_AT
START            = 0
STOP             = None                # exclusive: the window is [START, STOP); None = to the end

# a windowed run keys its own outputs tree, or its omni/base would overwrite the full sequence's
SCENE_KEY = scene_key(SCENE, START, STOP)
STREAM_RES       = 341 * 640           # tracking resolution budget
DEPTH_PNG_SCALE  = 256.0               # metres = px / this; MUST match the dataset (TUM 6553.5)
RENDER_EVAL      = False               # hi2.py's eval_rendering -> renders/ + psnr/ (11)

# ---------------------------------------------------------------- experiment names
# both REQUIRED, unique within their scene only; lineage is recorded in the adapter's config.json
OUT_ROOT     = 'outputs'
EXTRACT_NAME = 'low_dense_kf'
ADAPT_NAME   = 'live_e3_w10_a16_w12_lag3_more_chkp_base'

# ---------------------------------------------------------------- stage I/O
# a stage RECEIVES these and reads no path global; pure string joins, no disk access in this block
OUT_EXTRACT  = experiment_dir(OUT_ROOT, 'extract', SCENE_KEY, EXTRACT_NAME)

ADAPT_IN     = OUT_EXTRACT                              # an extract export
ADAPT_IMAGES = COLORS                                   # keyframe RGB, indexed by frame number
ADAPT_OUT    = experiment_dir(OUT_ROOT, 'adapt', SCENE_KEY, ADAPT_NAME)
ADAPT_CKPT   = f'{ADAPT_OUT}/{ADAPT_CKPT_SUBDIR}'       # epoch_NNN/; cadence is checkpoint_every
ADAPT_INIT   = None                                     # adapter to CONTINUE from; None = stock

OUT_END2END  = test_dir(OUT_ROOT, 'end2end', SCENE_KEY)     # one subdirectory per prior generator
OUT_PRIOR    = test_dir(OUT_ROOT, 'prior', SCENE_KEY)       # same arm names, scored without SLAM

# VGGT's input size; an adapter's own recorded size wins over this. Resolved in main() (9.3)
VGGT_HW          = None                # None = derive | (378, 518) TUM | (294, 518) Replica

# ---------------------------------------------------------------- the SLAM runs
# what every invocation shares; what differs per run is an argument to SlamRunner.run() instead
SLAM = SlamConfig(
    weights=DROID_WEIGHTS, colors=COLORS, calib=CALIB, start=START, stop=STOP,
    undistort=UNDISTORT, crop_border=CROP_BORDER, stream_res=STREAM_RES,
    render_eval=RENDER_EVAL)

# ---------------------------------------------------------------- extract (stage 1)
# the kf_* knobs are EXTRACT-ONLY: a generated config only this run is given, asserted in main()
EXTRACT = ExtractConfig(
    kf_motion_thresh=1.2,           # motion_filter.thresh; any threshold may be None = inherit
    kf_init_thresh=4.0,             # the same gate before initialisation
    kf_redundant_thresh=2.0,        # the one that actually moves the keyframe count
    kf_covis_thresh=0.1,            # extras inserted in terminate(); LOWER -> more
    buffer=500,                     # hard cap; MUST exceed the count (no overflow guard)
    depth_png_scale=DEPTH_PNG_SCALE,
    mask_filter_thresh=0.005,       # depth_filter disparity agreement
    mask_min_count=2,               # min agreeing neighbours out of 6
    mask_min_disp_ratio=0.5,        # drop pixels below this fraction of the frame's mean disparity
    gt_depths=DEPTHS)               # the accuracy table ONLY - never reaches Hi2 (§9.3)

# ---------------------------------------------------------------- adapt (stage 2, LoRA on VGGT)
# LORA is the adapter STRUCTURE, recorded into its config.json; ADAPT is the training run
LORA = LoRAConfig(
    weights='pretrained_models/vggt',
    vggt_hw=VGGT_HW,           # None -> derived in main()
    rank=8, alpha=16,
    targets=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
    patch_embed=False)         # False = adapt only the alternating-attention stack

ADAPT = AdaptConfig(
    stream_res=STREAM_RES,
    p_single_view=1, max_left=4, max_right=4, radius=8,
    adapt_style='normal',      # 'normal' epochs | 'online' per arrival | 'wonline' sliding window
    epochs=10, batch_size=2,
    window_size=10,            # 'wonline' only
    lr=0.5e-4, weight_decay=0.0, grad_clip=1.0, lambda_pose=1.0,
    coupled_scale=True, min_mask_pixels=16, seed=0, log_every=20,
    # ---- which exported keyframes are trained on, and what the rest are for ----
    kf_fraction=1.0,           # 1.0 = every exported keyframe; < 1 = equidistant sample of them
    val_source='tail',         # 'tail' = the selection's tail | 'rest' = the keyframes it skipped
    train_frac=1.0,            # 'tail' ONLY; 1.0 = train on every keyframe, no val set
    eval_on_val=True,          # depth L1 on held-out keyframes, base vs adapted
    eval_on_train=True,        # also on the train subset, so the train/val gap is visible
    eval_every_epoch=False,     # False = only before training and after the last unit
    eval_max_kf=100,           # subsample each eval subset to at most this many; 0 = no cap
    keep_best=False,           # False = save the last epoch; True = snapshot on val improvement
    checkpoint_every=0)        # 0 = off; N = a loadable adapter dir in ADAPT_CKPT every N epochs

# ---------------------------------------------------------------- end2end test (stage 3)
# one entry per DEPTH-PRIOR GENERATOR; its arm directory is INFERRED from it, never typed (7.1)


# another adapt run's handoff directory; both prior lists below use it
def _a(name):
    return experiment_dir(OUT_ROOT, 'adapt', SCENE_KEY, name)


E8 = _a('live_e8_w10_a16_w12_lag3_base')

# comments are unit, last TRAINING frame and share of the sequence - read off train_log.json
END2END_PRIORS = (
    'omnidata', 'vggt_base',     # priors[0] is the baseline column
    f'{ADAPT_CKPT}/epoch_004',   # unit   4, frame  190,  6.7%
    f'{ADAPT_CKPT}/epoch_009',   # unit   9, frame  222,  7.8%
    f'{ADAPT_CKPT}/epoch_014',   # unit  14, frame  299, 10.5%
    f'{ADAPT_CKPT}/epoch_024',   # unit  24, frame  421, 14.8%
    f'{ADAPT_CKPT}/epoch_039',   # unit  39, frame  559, 19.7%
    f'{ADAPT_CKPT}/epoch_049',   # unit  49, frame  694, 24.4%
    f'{ADAPT_CKPT}/epoch_079',   # unit  79, frame 1033, 36.3%
    f'{ADAPT_CKPT}/epoch_124',   # unit 124, frame 1442, 50.7%
    f'{ADAPT_CKPT}/epoch_179',   # unit 179, frame 2044, 71.8%
    f'{ADAPT_CKPT}/epoch_229',   # unit 229, frame 2500, 87.8%
    f'{E8}/{ADAPT_CKPT_SUBDIR}/epoch_049',        # the other run at unit 49, frame 694, 24.4%
    ADAPT_OUT,                   # this run's final adapter, frozen
    E8,                          # the other run's final adapter, frozen
)

END2END = End2EndConfig(
    priors=END2END_PRIORS,
    length=100000,                    # 100000 = whole sequence
    buffer=500,
    gt_traj=GT_TRAJ,                  # evo_ape's reference; ATE is the whole table
    lora=LORA,
    omni_normal_ckpt='pretrained_models/omnidata_dpt_normal_v2.ckpt',
    omni_normal_hw=(512, 512))

# ---------------------------------------------------------------- prior test (stage 4)
# the same generators vs GT depth, no SLAM run - it attributes an end2end null (9.2.2)
PRIOR_PRIORS = (
    _a('live_e1_w10_a16_w12_lag3_base'),           # priors[0] is the baseline column
    _a('live_e5_w10_a16_w12_lag3_base'),
    _a('live_e10_w10_a16_w12_lag3_base'),
    _a('live_e12_w10_a16_w12_lag3_base'),
    _a('live_e1_w10_lr1.2_a16_w12_lag3_base'),
    _a('live_e3_w10_a16_w12_lag3_more_chkp_base'),
    _a('live_e8_w10_a16_w12_lag3_base'),
)

PRIOR = PriorTestConfig(
    priors=PRIOR_PRIORS,
    gt_depths=DEPTHS, depth_png_scale=DEPTH_PNG_SCALE,
    eval_min_depth=1.0,                # m; nothing is closer than a few m in these scenes
    eval_max_depth=50.0,               # m; past this the GT beam interpolation is extrapolation
    eval_samples_per_frame=20000,      # valid pixels kept per frame; 0 = all (needs the RAM)
    seed=0,
    lora=LORA,
    omni_normal_ckpt='pretrained_models/omnidata_dpt_normal_v2.ckpt',
    omni_normal_hw=(512, 512))

# ==============================================================================

# At module scope, not in main(): a spawned child re-executes this module and needs both.
raise_fd_limit()
ensure_venv_on_path()


# ==============================================================================

def main():
    # once per process, before any Process is started; every path above is repo-root relative
    enter(_ROOT)

    # these name directories every stage writes into, so a bad one must fail before any GPU work
    require_name('EXTRACT_NAME', EXTRACT_NAME)
    require_name('ADAPT_NAME', ADAPT_NAME)

    if 'prior' in STAGES and not DEPTHS:
        raise SystemExit('the prior stage scores against ground-truth depth; DEPTHS is None')
    n_frames = check_sequence(
        COLORS, DEPTHS, GT_TRAJ,
        required=[CALIB, CONFIG, DROID_WEIGHTS] + ([GT_TRAJ] if 'end2end' in STAGES else []))
    warn_runtime_undistort(UNDISTORT, CROP_BORDER)

    window = window_frames(n_frames, START, STOP)
    extract_length = window * FRACTION // 100     # a share of the WINDOW, not the sequence
    if extract_length < 20:
        raise SystemExit(f'{n_frames} frames * {FRACTION}% = {extract_length}, too few to track')
    split_at = START + extract_length     # ABSOLUTE: evo's timestamps are frame numbers

    # the arms must run stock tracking, never the extract run's denser generated config
    assert os.path.abspath(CONFIG) != \
        os.path.abspath(f'{extract_run_dir(OUT_EXTRACT)}/extract_config.yaml'), \
        'the arms must use the base CONFIG, not the extract run derived config'

    # both test trees; the arm directories inside them are created by the arms themselves
    for kind in ('end2end', 'prior'):
        os.makedirs(test_dir(OUT_ROOT, kind, SCENE_KEY), exist_ok=True)

    # here, not in PARAMETERS: deriving reads a frame, and this runs after chdir
    global LORA, END2END, PRIOR
    LORA, stream_hw = resolve_lora(LORA, COLORS, STREAM_RES)
    END2END = replace(END2END, lora=LORA)    # the vggt_base arm reads its size off this
    PRIOR = replace(PRIOR, lora=LORA)        # and so does the prior test's, for the same reason

    print(f'sequence  : {SCENE}  ({n_frames} frames, {COLORS})')
    print(f'config    : {CONFIG}  calib {CALIB}')
    print(f'stream    : {stream_hw[1]}x{stream_hw[0]} (aspect '
          f'{stream_hw[1]/stream_hw[0]:.3f})   VGGT input: {LORA.vggt_hw[1]}x{LORA.vggt_hw[0]}'
          f'{" (derived)" if VGGT_HW is None else " (pinned by VGGT_HW)"}')
    print(f'adapter   : trains on frames {START}..{split_at-1} ({FRACTION}% of the window), '
          f'evaluated on 0..{n_frames-1}')
    print(f'target    : {DEPTH_DIR}/   split at frame {split_at}')
    print(f'stages    : {" ".join(STAGES)}   render_eval {RENDER_EVAL}')
    print(f'outputs   : {OUT_EXTRACT}')
    print(f'            {ADAPT_OUT}')
    # arm directories are inferred, so print where each lands before a two-hour run
    print_arm_dirs(STAGES, (('end2end', END2END, OUT_END2END), ('prior', PRIOR, OUT_PRIOR)))

    # one VRAM check up front, before any GPU work or spawned Process
    gpu_gate(MIN_FREE_VRAM_MB)

    # ONE runner for every HI-SLAM2 invocation: the extract run and every arm
    runner = SlamRunner(SLAM)

    # every stage takes its paths as arguments, so where each writes is visible right here
    t_all = time.time()
    if 'extract' in STAGES:
        run_extract(runner, EXTRACT, OUT_EXTRACT, extract_length, CONFIG,
                    skip_existing=SKIP_EXISTING)
    if 'adapt' in STAGES:
        run_adapt(LORA, ADAPT, ADAPT_IN, ADAPT_IMAGES, ADAPT_OUT, ADAPT_CKPT,
                  init_adapter=ADAPT_INIT, skip_existing=SKIP_EXISTING)
    if 'end2end' in STAGES:
        run_end2end_test(runner, SLAM, END2END, OUT_END2END, CONFIG, split_at,
                         skip_existing=SKIP_EXISTING)
    if 'prior' in STAGES:
        run_prior_test(SLAM, PRIOR, OUT_PRIOR, skip_existing=SKIP_EXISTING)

    print(f'\nall stages done in {time.time()-t_all:.0f}s')
    print('\nread first:')
    print(f'  {OUT_EXTRACT}/export.txt   per-frame vs global depth L1 columns. The gap on the')
    print('                             Omnidata row is the cross-frame scale inconsistency this')
    print('                             track targets - if it is small, there was no headroom.')
    print("  the table above            'unseen' rows only; 'seen' is the adapter's training")


if __name__ == '__main__':
    main()
