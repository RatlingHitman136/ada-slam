"""INITIAL adaptation: extract a dense PREFIX of the sequence, adapt on it, test (9.1).

    python scripts/init_adapt_pipeline.py    # from the repo root, adaslam venv active

  1 extract  HI-SLAM2 over the first FRACTION%, keyframes DENSIFIED by the EXTRACT kf_* knobs
             -> per-keyframe depth/mask/image + accuracy table
  2 adapt    LoRA-adapt VGGT on that depth, from stock VGGT-1B or another adapter; depth L1 on a
             held-out val subset
  3 end2end  one full-sequence arm per generator in END2END_PRIORS, then ATE side by side
  4 prior    the same generators vs GT depth directly, no SLAM run

The sibling driver is scripts/cont_adapt_pipeline.py: the WHOLE sequence at stock keyframe
density, adapting on a thin equidistant sample of it (9.7). Both run the same stages.

This file is the KNOB PANEL, not the implementation: every stage is a package under adaslam/. No
CLI, no environment. Run preprocess_tum.py first.

The config literals below are rebuilt in every spawned reader child - primitives only, no file
access and no computation.
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
STAGES           = ('prior',)          # any subset; run in pipeline order.
                                       # E2 batch: prior stage only. The end2end block below is
                                       # left intact as E1's config - flip this back to
                                       # ('end2end',) to resume that ladder. The two stages write
                                       # to different trees (test/prior vs test/end2end), so an
                                       # E1 run on another host and this one cannot collide.
SKIP_EXISTING    = True                # reuse a stage's output if it is already on disk.
                                       # TRUE for this E0b batch. It was False for E0 batch 1, to
                                       # force the five arms scored 2026-08-02..04 to be recomputed
                                       # through the eval path as edited on 08-14; that gate
                                       # PASSED bit-exact (omni, base and normal_r8_e10 reproduce
                                       # to 0.000000 on every metric), so the old rows are valid
                                       # and nothing needs recomputing again. True also means a
                                       # re-run of this file is a no-op rather than a clobber -
                                       # which matters here, because outputs/ is shared storage.
MIN_FREE_VRAM_MB = 7000                # shared GPU: checked once at the start of main()
FRACTION         = 7                   # % of the sequence the adapter trains on; also SPLIT_AT.
                                       # 7 -> split_at = 2847*7//100 = 199, the p7 extract's
                                       # boundary (frames 0-198). That is the span the best small-
                                       # prefix offline arm (wonline_r8_e10_w10_p7, ATE 23.864)
                                       # trained on AND the span the earliest live checkpoint
                                       # scored below reached, so ate_unseen means the same thing
                                       # for both. Nothing here trains - only ate_all is compared.
START            = 0
STOP             = None                # exclusive: the window is [START, STOP), so (200, 2647) is
                                       # frames 200..2646 and (0, None) is the whole sequence.
                                       # A WINDOWED run gets its own outputs/ tree - see SCENE_KEY.

# WHERE A WINDOWED RUN'S OUTPUTS GO. end2end/config.py:arm_name maps 'omnidata' to `omni` whatever
# the window, so a windowed baseline would overwrite the full-sequence one and leave nothing to
# compare either against. The window therefore keys the SCENE directory instead: the full sequence
# keeps SCENE, anything else becomes SCENE_f<START>-<STOP> with its own omni/base, which
# SKIP_EXISTING fills on first use. Pure string work, so it belongs in this block (9.5 rule 3).
SCENE_KEY = scene_key(SCENE, START, STOP)
STREAM_RES       = 341 * 640           # tracking resolution budget
DEPTH_PNG_SCALE  = 256.0               # metres = px / this. MUST match the dataset: 6553.5
                                       # (TUM/Replica) saturates at 10 m, 256 at 256 m
RENDER_EVAL      = False               # hi2.py's eval_rendering -> renders/ + psnr/; nothing here
                                       # reads them and ATE is unaffected either way (11)

# ---------------------------------------------------------------- experiment names
# Both REQUIRED, and unique within their scene only. Lineage is DATA, not naming: an adapter's
# config.json holds the extract it trained on. FRACTION is not in the name - put it there yourself.
OUT_ROOT     = 'outputs'
EXTRACT_NAME = 'low_dense_kf'
# E1 scores a LIVE run's checkpoint ladder, so ADAPT_NAME names that run: it is what makes
# ADAPT_CKPT point at the 46 checkpoints below. No adapt stage runs here.
ADAPT_NAME   = 'live_e3_w10_a16_w12_lag3_more_chkp_base'

# ---------------------------------------------------------------- stage I/O
# A stage RECEIVES its paths and reads no path global, so it can be pointed at another run's
# results. Pure string joins - this block must not touch the disk.
OUT_EXTRACT  = experiment_dir(OUT_ROOT, 'extract', SCENE_KEY, EXTRACT_NAME)

ADAPT_IN     = OUT_EXTRACT                              # an extract export
ADAPT_IMAGES = COLORS                                   # keyframe RGB, indexed by frame number
ADAPT_OUT    = experiment_dir(OUT_ROOT, 'adapt', SCENE_KEY, ADAPT_NAME)
ADAPT_CKPT   = f'{ADAPT_OUT}/{ADAPT_CKPT_SUBDIR}'       # epoch_NNN/; cadence is checkpoint_every
ADAPT_INIT   = None                                     # the adapter to CONTINUE from; None =
                                                        # stock VGGT-1B. Same vocabulary as an
                                                        # END2END_PRIORS entry, e.g.
                                                        # experiment_dir(OUT_ROOT, 'adapt',
                                                        #                SCENE_KEY, 'x')

OUT_END2END  = test_dir(OUT_ROOT, 'end2end', SCENE_KEY)     # one subdirectory per prior generator
OUT_PRIOR    = test_dir(OUT_ROOT, 'prior', SCENE_KEY)       # same arm names, scored without SLAM

# VGGT's input size; an adapter's own recorded size wins over this. Derived in main(), not here -
# this block must not touch the disk (9.3).
VGGT_HW          = None                # None = derive | (378, 518) TUM | (294, 518) Replica

# ---------------------------------------------------------------- the SLAM runs
# What every invocation shares; what differs per run is an argument to SlamRunner.run() instead.
SLAM = SlamConfig(
    weights=DROID_WEIGHTS, colors=COLORS, calib=CALIB, start=START, stop=STOP,
    undistort=UNDISTORT, crop_border=CROP_BORDER, stream_res=STREAM_RES,
    render_eval=RENDER_EVAL)

# ---------------------------------------------------------------- extract (stage 1)
# The kf_* knobs are EXTRACT-ONLY: they go into a generated config only this run is given, so
# denser training data cannot look like a tracking change. main() asserts it.
EXTRACT = ExtractConfig(
    kf_motion_thresh=1.2,
    kf_init_thresh=4.0,             # the same gate before initialisation
    kf_redundant_thresh=2.0,        # the one that actually moves the keyframe count
    kf_covis_thresh=0.1,            # extras inserted in terminate(); LOWER -> more
    buffer=500,                     # hard cap; MUST exceed the count (no overflow guard)
                                    # any threshold may be None = inherit CONFIG
    depth_png_scale=DEPTH_PNG_SCALE,
    mask_filter_thresh=0.005,       # depth_filter disparity agreement
    mask_min_count=2,               # min agreeing neighbours out of 6
    mask_min_disp_ratio=0.5,        # drop pixels below this fraction of the frame's mean disparity
    gt_depths=DEPTHS)               # the accuracy table ONLY - never reaches Hi2 (§9.3)

# ---------------------------------------------------------------- adapt (stage 2, LoRA on VGGT)
# LORA is the adapter STRUCTURE, recorded into its config.json, so an arm always runs the model it
# was trained in. ADAPT is the training run, which no adapter re-reads.
LORA = LoRAConfig(
    weights='pretrained_models/vggt',
    vggt_hw=VGGT_HW,           # None -> derived in main()
    rank=8, alpha=16,
    targets=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
    patch_embed=False)         # False = adapt only the alternating-attention stack

ADAPT = AdaptConfig(
    stream_res=STREAM_RES,
    p_single_view=1, max_left=4, max_right=4, radius=8,
    adapt_style='normal',      # 'normal' = shuffled epochs over the train set; 'online' =
                               # keyframes in ARRIVAL order, `epochs` consecutive steps on each
                               # before the next arrives (batch_size unused); 'wonline' = a
                               # SLIDING WINDOW of the arrival + the `window_size`-1 keyframes
                               # before it, `epochs` shuffled batched passes over it, then slide
                               # by one (no partial warm-up window). Under the latter two the
                               # eval/checkpoint cadences count keyframes / windows, not epochs -
                               # set eval_every_epoch False
    epochs=10, batch_size=2,
    window_size=10,            # 'wonline' only
    lr=0.5e-4, weight_decay=0.0, grad_clip=1.0, lambda_pose=1.0,
    coupled_scale=True, min_mask_pixels=16, seed=0, log_every=20,
    # ---- which exported keyframes are trained on, and what the rest are for ----
    kf_fraction=1.0,           # 1.0 = train on EVERY exported keyframe. This pipeline densifies
                               # the keyframes on purpose (EXTRACT's kf_* knobs), so thinning them
                               # again here would undo that - cont_adapt_pipeline.py is the
                               # driver that samples instead
    val_source='tail',         # the contiguous last (1 - train_frac) of the selection, so val
                               # measures generalising FORWARD. 'rest' needs kf_fraction < 1.
    train_frac=1.0,            # 'tail' ONLY: val = the contiguous last (1 - train_frac) of the
                               # selection; 1.0 = train on every keyframe, no val set
    eval_on_val=True,          # depth L1 on held-out keyframes, base vs adapted
    eval_on_train=True,        # also on the train subset, so the train/val gap is visible
    eval_every_epoch=False,     # False = only before training and after the last epoch
    eval_max_kf=100,           # subsample each eval subset to at most this many; 0 = no cap
    keep_best=False,           # False = save the last epoch; True = snapshot on val improvement
    checkpoint_every=0)        # 0 = off; N = a loadable adapter dir in ADAPT_CKPT every N epochs

# ---------------------------------------------------------------- end2end test (stage 3)
# One entry per DEPTH-PRIOR GENERATOR, scored into a directory INFERRED from it, never typed:
#   'omnidata' -> .../omni | 'vggt_base' -> .../base | ADAPT_OUT -> .../<ADAPT_NAME> |
#   f'{ADAPT_CKPT}/epoch_005' -> .../<ADAPT_NAME>_chkp_005
# That is what makes an arm reusable. priors[0] is the baseline column.


# `_a` names another run's adapt handoff directory. Defined HERE rather than further down because
# both END2END_PRIORS and PRIOR_PRIORS use it and this one comes first.
def _a(name):
    return experiment_dir(OUT_ROOT, 'adapt', SCENE_KEY, name)


# The best live run so far: wonline, steps_per_kf 8, w10, a16, lr 1.2e-4, lag 3 -> live ATE 24.473.
E8 = _a('live_e8_w10_a16_w12_lag3_base')

# E1: WHERE IN A LIVE RUN DOES THE ADAPTER BECOME GOOD?
# `live_e3_w10_a16_w12_lag3_more_chkp_base` was run with checkpoint_every_kf=5 and left 46
# checkpoints spanning units 4..229 = 6.7%..87.8% of the sequence. Scoring them FROZEN turns the
# run into a curve of adapter quality against how much of the sequence it had seen, which is the
# one thing the live ATE cannot show. The comparison target is the offline arm trained on the SAME
# span: p7 23.864 / p10 23.498 / p20 23.769 / p40 23.750 (omni baseline 31.279).
#
# Why it matters: the live pipeline reaches 24.473 only at steps_per_kf=8, which costs 118 min of
# training INSIDE the scored run (training blocks tracking, online/prior.py:102-104) for ~7x more
# forward passes than the offline p7 arm needs. If this ladder is flat after unit ~20, most of that
# is avoidable; if it climbs to the end, the cost is structural.
#
# 10 of the 46, dense where the question lives (5 below 20% coverage), geometric after. Comments
# are unit, last TRAINING frame, and share of the sequence - read off train_log.json, not names.
END2END_PRIORS = (
    # priors[0] is the baseline column. Both already exist, so SKIP_EXISTING makes them cache
    # hits; they only re-score at the new split_at, which takes seconds and leaves ate_all alone.
    'omnidata', 'vggt_base',
    f'{ADAPT_CKPT}/epoch_004',   # unit   4, frame  190,  6.7%  <- vs offline p7  23.864
    f'{ADAPT_CKPT}/epoch_009',   # unit   9, frame  222,  7.8%
    f'{ADAPT_CKPT}/epoch_014',   # unit  14, frame  299, 10.5%  <- vs offline p10 23.498
    f'{ADAPT_CKPT}/epoch_024',   # unit  24, frame  421, 14.8%
    f'{ADAPT_CKPT}/epoch_039',   # unit  39, frame  559, 19.7%  <- near-replicate of the existing
                                 #            live_e3_w10_a16_w12_lag3_base_chkp_039 (26.793),
                                 #            which differs only in lr - a free reproducibility read
    f'{ADAPT_CKPT}/epoch_049',   # unit  49, frame  694, 24.4%  <- pairs with the e8 arm below
    f'{ADAPT_CKPT}/epoch_079',   # unit  79, frame 1033, 36.3%  <- vs offline p40 23.750
    f'{ADAPT_CKPT}/epoch_124',   # unit 124, frame 1442, 50.7%
    f'{ADAPT_CKPT}/epoch_179',   # unit 179, frame 2044, 71.8%
    f'{ADAPT_CKPT}/epoch_229',   # unit 229, frame 2500, 87.8%
    # The e8 run at the SAME 24.4% coverage (its epoch_049 also ends at frame 694). Same style,
    # window, alpha, lag and lr; ONLY steps_per_kf differs, 8 vs 3. The one controlled test of
    # whether repetition pays off EARLY or only accrues by the end of the run - which is what
    # decides whether the 118 min can be truncated.
    f'{E8}/{ADAPT_CKPT_SUBDIR}/epoch_049',        # unit 49, frame 694, 24.4%
    # Endpoints, one run each.
    ADAPT_OUT,                   # more_chkp FINAL, frozen. Its LIVE ATE is 25.913; frozen tells
                                 # us whether that came from the adapter or from the prior changing
                                 # during the run.
    E8,                          # the 24.473 arm, frozen. No frozen arm exists for it yet.
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
# The same generators vs GT depth, no SLAM run - minutes an arm. It attributes an end2end null:
# "no better prior" vs "HI-SLAM2 cannot feel the difference" (9.2.2).
# E0: does `scale_cv` (cross-frame scale consistency) predict ATE across EVERY training regime,
# or only across the offline arms it was first measured on? The existing table is 15 arms, all
# offline, all scored 2026-08-02..04 - it has never been shown a live or cont adapter. This set
# spans every style (normal / online / wonline), both drivers (init / cont / live), both inits
# (stock / warm-start) and the full ATE range 23.5..38.3, so the fit has real leverage.
#
# `_a` is defined up in the end2end block, which comes first; each entry is an adapt handoff
# directory, and its arm directory is INFERRED from the basename (end2end/config.py:arm_name).
#
# ---- ALREADY SCORED (E0 batch 1 + E0b batch 2, 27 arms, DONE). Do not re-add: outputs/ is
#      shared storage, so two hosts scoring the same arm would write one directory twice.
#      batch 1: 'omnidata', 'vggt_base', normal_r8_e10, wonline_r8_e5, online_r8_e3,
#        wonline_r8_e3_w10_p10, normal_r8_e20_p10, online_r8_e15_p10, wonline_r8_e20_w10_p6,
#        normal_r8_e5_p5, cont_normal_e10_pre50_b_base, cont_normal_e5_kf100_b_base,
#        cont_online_e10_kf10_b_wonline_r8_e20_w10_p6, live_e3_w10_a16_w12_lag3_base,
#        live_e3_a16_w13_lag3_base, live_e3_a16_w12_lag4_normal_r8_e20_p10,
#        live_e3_ctx2_a16_w12_lag3_base
#      batch 2 (p5..p9 operating point): wonline_r8_e10_w10_p7, wonline_e5_w10_p7_lr2,
#        normal_r8_e30_p7, online_e30_p7, wonline_r8_e10_w10_p6, normal_e20_p6_lr10,
#        online_e40_p6_lr5, wonline_e10_w10_p9_lr2, normal_r8_e40_p9, wonline_e80_w5_p5
#
# ---- E2 BATCH (this list): CV_depth across the live REPETITION sweep.
# E2 established that raising steps_per_kf moves live ATE 25.951 -> 24.473 (e1 -> e8). Offline,
# the same lever moved CV_depth 0.1355 -> 0.0174 at p6 (wonline e10 -> e20), and CV_depth is the
# one metric that discriminates WITHIN a matched family (+0.90/+0.95 at p6/p7). So: does the live
# sweep show the same mechanism, or does it reach 24.473 some other way?
#
# One family, so the comparison is matched: wonline, window 10, alpha 16, warmup/handover 12,
# lag 3, stock init, no ctx. ONLY steps_per_kf and lr differ. That also lets lr be partly
# deconfounded - §1.9b flagged it as confounded, because e8 is lr1.2e-4 while e10/e12 are lr1e-4 -
# since e1 and e3 each appear at BOTH learning rates below.
#
# live_e3_w10_a16_w12_lag3_base (e3, lr1e-4, CV_depth 0.1173) is already scored and deliberately
# omitted; it is the anchor this batch is read against.
PRIOR_PRIORS = (
    # -- lr 1.0e-4, rising repetition. priors[0] is the baseline column, so deltas read as
    #    "vs the least-repetition arm", which is the axis under test.
    _a('live_e1_w10_a16_w12_lag3_base'),          # e1   kf_vis  2330   25 min   live ATE 25.865
    _a('live_e5_w10_a16_w12_lag3_base'),          # e5   kf_vis 11550   77 min   live ATE 25.592
    _a('live_e10_w10_a16_w12_lag3_base'),         # e10  kf_vis 23000  142 min   live ATE 25.367
    _a('live_e12_w10_a16_w12_lag3_base'),         # e12  kf_vis 27960  173 min   live ATE 25.022
    # -- lr 1.2e-4. e1 and e3 pair with the lr1e-4 arms above/anchor at the SAME repetition,
    #    which is the only way to separate lr from steps_per_kf in the existing runs.
    _a('live_e1_w10_lr1.2_a16_w12_lag3_base'),    # e1   kf_vis  2330   23 min   live ATE 25.951
    _a('live_e3_w10_a16_w12_lag3_more_chkp_base'),# e3   kf_vis  6990   60 min   live ATE 25.913
    _a('live_e8_w10_a16_w12_lag3_base'),          # e8   kf_vis 18640  118 min   live ATE 24.473
                                                  #      THE ONE THAT MATTERS: did CV_depth fall
                                                  #      toward the 0.0174 the offline p6 winner
                                                  #      reaches, or is 24.473 reached some other way?
)

PRIOR = PriorTestConfig(
    priors=PRIOR_PRIORS,
    gt_depths=DEPTHS, depth_png_scale=DEPTH_PNG_SCALE,
    eval_min_depth=1.0,                # m; nothing is closer than a few m in these scenes
    eval_max_depth=50.0,               # m; past this the beams are far enough apart that the
                                       # interpolation between them is extrapolation
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
    # before any Process is started, and only once per process; every relative path above is
    # repo-root relative
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

    # the arms must run stock tracking, or a denser-keyframe extract would silently mean denser
    # keyframes in the comparison too. Asserted here, the one place both paths are in scope.
    assert os.path.abspath(CONFIG) != \
        os.path.abspath(f'{extract_run_dir(OUT_EXTRACT)}/extract_config.yaml'), \
        'the arms must use the base CONFIG, not the extract run derived config'

    # both test trees; the arm directories inside them are created by the arms themselves
    for kind in ('end2end', 'prior'):
        os.makedirs(test_dir(OUT_ROOT, kind, SCENE_KEY), exist_ok=True)

    # here, not in PARAMETERS: deriving reads a frame, which that block must not do. After chdir,
    # so relative COLORS resolves the same however invoked.
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
