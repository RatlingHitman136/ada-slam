"""CONTINUAL adaptation: extract the WHOLE sequence, adapt on a thin slice of it, test (9.7).

    python scripts/cont_adapt_pipeline.py    # from the repo root, adaslam venv active

  1 extract  HI-SLAM2 over the whole sequence at STOCK keyframe density - no kf_* overrides at
             all -> per-keyframe depth/mask/image + accuracy table
  2 adapt    LoRA-adapt VGGT on KF_FRACTION of those keyframes, taken EQUIDISTANT over the
             keyframe list, CONTINUING from ADAPT_INIT_NAME's adapter or from stock VGGT-1B.
             Validation is every keyframe the selection skipped.
  3 end2end  one full-sequence arm per generator in END2END_PRIORS, then ATE side by side
  4 prior    the same generators vs GT depth directly, no SLAM run

The sibling driver is scripts/init_adapt_pipeline.py: a DENSIFIED PREFIX of the sequence, adapted
on in full. Both run the same stages, which are the packages under adaslam/.

Two things follow from adapting on an equidistant sample rather than a prefix, and both are why
this is a separate driver rather than a mode of that one:

  * val is the COMPLEMENT of the training set, not its tail - "the keyframes it never trained on",
    interleaved through the whole sequence. That is the number that says whether a thin sample
    generalises to the frames between its samples.
  * there is no seen/unseen frontier. The training keyframes span the whole sequence, so no frame
    index separates trained from untrained and SPLIT_AT defaults to the whole thing: ate_all is
    the row the comparison reduces to, which 12.2 already argues is the only ATE comparable
    across arms anyway.

This file is the KNOB PANEL, not the implementation. No CLI, no environment. Run the dataset's
preprocess script first.

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
                              warn_runtime_undistort)
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

# undistort offline in the preprocess script instead - consumers re-derive a frame with
# stream_resize alone, so doing it here misaligns predictions and GT (10.1)
UNDISTORT   = False
CROP_BORDER = 0

# ---------------------------------------------------------------- run control
STAGES           = ('extract', 'adapt', 'end2end')   # any subset; run in pipeline order
SKIP_EXISTING    = True                # reuse a stage's output if it is already on disk
MIN_FREE_VRAM_MB = 7000                # shared GPU: checked once at the start of main()
EXTRACT_LENGTH   = 100000              # frames to extract over; 100000 = the WHOLE sequence,
                                       # which is the point of this driver
START            = 0
STREAM_RES       = 341 * 640           # tracking resolution budget
DEPTH_PNG_SCALE  = 256.0               # metres = px / this. MUST match the dataset: 6553.5
                                       # (TUM/Replica) saturates at 10 m, 256 at 256 m
RENDER_EVAL      = False               # hi2.py's eval_rendering -> renders/ + psnr/; nothing here
                                       # reads them and ATE is unaffected either way (11)

# ---------------------------------------------------------------- experiment names
# Both REQUIRED, and unique within their scene only. Lineage is DATA, not naming: an adapter's
# config.json holds the extract it trained on AND the adapter it continued from.
OUT_ROOT     = 'outputs'
EXTRACT_NAME = 'low_dense_kf'          # the whole sequence at STOCK keyframe density - one
                                       # expensive run, reused by every adapt run below it
ADAPT_NAME   = 'cont_normal_e5_kf100_b_base'

# ---------------------------------------------------------------- stage I/O
# A stage RECEIVES its paths and reads no path global, so it can be pointed at another run's
# results. Pure string joins - this block must not touch the disk.
OUT_EXTRACT  = experiment_dir(OUT_ROOT, 'extract', SCENE, EXTRACT_NAME)

ADAPT_IN     = OUT_EXTRACT                              # an extract export
ADAPT_IMAGES = COLORS                                   # keyframe RGB, indexed by frame number
ADAPT_OUT    = experiment_dir(OUT_ROOT, 'adapt', SCENE, ADAPT_NAME)
ADAPT_CKPT   = f'{ADAPT_OUT}/{ADAPT_CKPT_SUBDIR}'       # epoch_NNN/; cadence is checkpoint_every

# WHICH ADAPTER THIS RUN STARTS FROM. None = stock VGGT-1B ("base"); otherwise the name of an
# adapt experiment in this SCENE, i.e. the same vocabulary an END2END_PRIORS entry uses, so
# continuing from a run and testing it are spelled alike. A checkpoint works too:
#   f'{experiment_dir(OUT_ROOT, "adapt", SCENE, "x")}/{ADAPT_CKPT_SUBDIR}/epoch_005'
ADAPT_INIT_NAME = None
ADAPT_INIT = None if ADAPT_INIT_NAME is None else \
    experiment_dir(OUT_ROOT, 'adapt', SCENE, ADAPT_INIT_NAME)

OUT_END2END  = test_dir(OUT_ROOT, 'end2end', SCENE)     # one subdirectory per prior generator
OUT_PRIOR    = test_dir(OUT_ROOT, 'prior', SCENE)       # same arm names, scored without SLAM

# VGGT's input size; an adapter's own recorded size wins over this. Derived in main(), not here -
# this block must not touch the disk (9.3).
VGGT_HW          = None                # None = derive | (378, 518) TUM | (294, 518) Replica

# ---------------------------------------------------------------- the SLAM runs
# What every invocation shares; what differs per run is an argument to SlamRunner.run() instead.
SLAM = SlamConfig(
    weights=DROID_WEIGHTS, colors=COLORS, calib=CALIB, start=START,
    undistort=UNDISTORT, crop_border=CROP_BORDER, stream_res=STREAM_RES,
    render_eval=RENDER_EVAL)

# ---------------------------------------------------------------- extract (stage 1)
# ALL FOUR kf_* ARE None ON PURPOSE: None = inherit CONFIG, so this run keyframes exactly as
# stock HI-SLAM2 does and the generated YAML carries nothing but `inherit_from`. That is what
# "low density" means here - the init pipeline lowers these to densify, this one does not touch
# them, and the thinning happens later at ADAPT.kf_fraction instead.
EXTRACT = ExtractConfig(
    kf_motion_thresh=None,          # motion_filter.thresh
    kf_init_thresh=None,            # the same gate before initialisation
    kf_redundant_thresh=None,       # frontend.keyframe_thresh - the gate that binds
    kf_covis_thresh=None,           # extras inserted in terminate(); LOWER -> more
    buffer=900,                     # hard cap; MUST exceed the WHOLE sequence's keyframe count
                                    # (no overflow guard) - this run is not a prefix
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
    rank=8, alpha=8,
    targets=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
    patch_embed=False)         # False = adapt only the alternating-attention stack

KF_FRACTION = 1.0             # THE knob of this pipeline: train on 10% of the extract's
                               # keyframes, equidistant over the keyframe list

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
    epochs=5, batch_size=2,
    window_size=10,            # 'wonline' only
    lr=1e-4, weight_decay=0.0, grad_clip=1.0, lambda_pose=1.0,
    coupled_scale=True, min_mask_pixels=16, seed=0, log_every=20,
    # ---- which exported keyframes are trained on, and what the rest are for ----
    kf_fraction=KF_FRACTION,   # a THIN equidistant sample of the whole sequence
    val_source='tail',         # val = every keyframe the selection skipped, interleaved through
                               # the whole sequence. This is the row to read: it says whether a
                               # sparse sample generalises to the frames between its samples.
    train_frac=1.0,            # unread under val_source='rest' (that mode's val is the
                               # complement, not a tail)
    eval_on_val=True,          # depth L1 on the never-trained keyframes, base vs adapted
    eval_on_train=True,        # also on the train subset, so the train/val gap is visible
    eval_every_epoch=False,    # False = only before training and after the last epoch
    eval_max_kf=100,           # subsample each eval subset to at most this many; 0 = no cap.
                               # val is ~90% of the export here, so this cap is what keeps the
                               # eval cheap - it subsamples evenly, so it still spans the sequence
    keep_best=False,           # False = save the last epoch; True = snapshot on val improvement
    checkpoint_every=0)        # 0 = off; N = a loadable adapter dir in ADAPT_CKPT every N epochs

# ---------------------------------------------------------------- end2end test (stage 3)
# One entry per DEPTH-PRIOR GENERATOR, scored into a directory INFERRED from it, never typed:
#   'omnidata' -> .../omni | 'vggt_base' -> .../base | ADAPT_OUT -> .../<ADAPT_NAME> |
#   f'{ADAPT_CKPT}/epoch_005' -> .../<ADAPT_NAME>_chkp_005
# That is what makes an arm reusable. priors[0] is the baseline column.
END2END_PRIORS = ('omnidata', 'vggt_base', ADAPT_OUT, experiment_dir(OUT_ROOT, 'adapt', SCENE, 'online_r8_e15_p10'))

# The seen/unseen boundary. None = the whole sequence, i.e. everything counts as "seen" and the
# [unseen] table is empty - the honest default here, because an equidistant selection puts
# training keyframes across the WHOLE sequence and no frame index separates trained from
# untrained. ate_all is the row that means anything (12.2). Set a frame index to override.
SPLIT_AT = None

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
PRIOR_PRIORS = END2END_PRIORS          # same arms, so the two tests' directories line up

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

    extract_length = min(EXTRACT_LENGTH, n_frames)
    split_at = n_frames if SPLIT_AT is None else SPLIT_AT

    # the arms must run stock tracking, and so must the extract here - but they still read
    # different files, and the arms must never be handed the generated one
    assert os.path.abspath(CONFIG) != \
        os.path.abspath(f'{extract_run_dir(OUT_EXTRACT)}/extract_config.yaml'), \
        'the arms must use the base CONFIG, not the extract run derived config'

    # both test trees; the arm directories inside them are created by the arms themselves
    for kind in ('end2end', 'prior'):
        os.makedirs(test_dir(OUT_ROOT, kind, SCENE), exist_ok=True)

    # here, not in PARAMETERS: deriving reads a frame, which that block must not do. After chdir,
    # so relative COLORS resolves the same however invoked.
    global LORA, END2END, PRIOR
    LORA, stream_hw = resolve_lora(LORA, COLORS, STREAM_RES)
    END2END = replace(END2END, lora=LORA)    # the vggt_base arm reads its size off this
    PRIOR = replace(PRIOR, lora=LORA)        # and so does the prior test's, for the same reason

    print(f'sequence  : {SCENE}  ({n_frames} frames, {COLORS})')
    print(f'config    : {CONFIG}  calib {CALIB}  (STOCK keyframing - no kf_* overrides)')
    print(f'stream    : {stream_hw[1]}x{stream_hw[0]} (aspect '
          f'{stream_hw[1]/stream_hw[0]:.3f})   VGGT input: {LORA.vggt_hw[1]}x{LORA.vggt_hw[0]}'
          f'{" (derived)" if VGGT_HW is None else " (pinned by VGGT_HW)"}')
    print(f'extract   : frames 0..{extract_length-1} of {n_frames}')
    print(f'adapter   : trains on {KF_FRACTION:.0%} of those keyframes, equidistant; val is the '
          f'rest')
    print(f'            starts from '
          f'{ADAPT_INIT if ADAPT_INIT else "stock VGGT-1B (no adapter)"}')
    print(f'target    : {DEPTH_DIR}/   split at frame {split_at}'
          f'{" (= whole sequence, so [unseen] is empty)" if SPLIT_AT is None else ""}')
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
    print("  the adapt log's val row    depth L1 on the keyframes the sample SKIPPED - the")
    print('                             evidence a thin sample generalises between its samples')
    print("  the table above            the [all] block; [unseen] is empty unless SPLIT_AT is set")


if __name__ == '__main__':
    main()
