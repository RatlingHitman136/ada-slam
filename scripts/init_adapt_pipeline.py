"""INITIAL adaptation: extract a dense PREFIX of the sequence, adapt on it, test (9.1).

    python scripts/init_adapt_pipeline.py                 # the PARAMETERS block below
    python scripts/init_adapt_pipeline.py -c init_default # every parameter from run_configs/

  1 extract  HI-SLAM2 over the first FRACTION%, keyframes DENSIFIED by the EXTRACT kf_* knobs
  2 adapt    LoRA-adapt VGGT on that depth, from stock VGGT-1B or another adapter
  3 end2end  one full-sequence arm per generator in END2END_PRIORS, then ATE side by side
  4 prior    the same generators vs GT depth directly, no SLAM run

Siblings: cont_adapt_pipeline.py (9.7) adapts on a thin sample of the whole sequence;
online_adapt_pipeline.py (13) adapts during the run itself. Run the preprocess script first.

The KNOB PANEL, not the implementation - every stage is a package under adaslam/. The config
literals below are rebuilt in every spawned reader child: primitives only, no disk, no computation.

Every parameter below is also a key of run_configs/init_*.yaml: with -c that file states ALL of
them and the literals here are not read, which is what a queued job needs (13.7). All or nothing -
a key this driver never asks for and a parameter the file omits are both hard errors, reported
together before any GPU work. Without -c nothing changes and these literals are still the
documentation. run_configs/init_default.yaml is them, verbatim - copy it, do not edit it.
"""
import os    # nopep8
import sys   # nopep8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # nopep8
# repo root, so `adaslam` imports; its __init__ adds hislam2/ and thirdparty/vggt
sys.path.insert(0, _ROOT)                                                 # nopep8
import shutil
import time
from dataclasses import replace

from adaslam.runconfig import run_config
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

# -c init_NAME replaces every literal below with run_configs/init_NAME.yaml's value; without it
# P(...) returns the literal unchanged. The name is checked, so a queue cannot hand this driver a
# live_* or cont_* config meant for another one.
P = run_config(_ROOT, prefix='init_', doc=__doc__)

# ---------------------------------------------------------------- data (preprocessing is NOT here)
# TUM:    SCENE 'rgbd_dataset_freiburg1_room', tum_config.yaml, scale 6553.5, PRIOR eval 0.1-10 m
# RELLIS: SCENE 'rellis_00000', DATA 'data/RELLIS/00000', rellis_config.yaml, DEPTHS = <DATA>/depths
SCENE   = P('SCENE', 'kitti_00')       # names the outputs/ tree
DATA    = P('DATA', 'data/KITTI/00')   # preprocess_kitti.py's output layout
# spelled out rather than derived from DATA, so a run config can move any one of them alone
COLORS  = P('COLORS', f'{DATA}/colors')   # a SYMLINK to the mirror's image_2 - nothing was copied
DEPTHS  = P('DEPTHS', f'{DATA}/depths')   # velodyne GT, from preprocess_kitti.py --with-depth
                                       # (SemanticKitti's sweep projected into cam2 - the odometry
                                       # benchmark itself ships none). Reaches the extract accuracy
                                       # table and the prior stage ONLY, never Hi2 (9.3).
                                       # null here disables both and is still a valid setting
GT_TRAJ = P('GT_TRAJ', f'{DATA}/traj_tum.txt')
CALIB   = P('CALIB', f'{DATA}/calib.txt')
CONFIG  = P('CONFIG', 'config/kitti_config.yaml')
DROID_WEIGHTS = P('DROID_WEIGHTS', 'pretrained_models/droid.pth')

# undistort offline in the preprocess script instead, or predictions and GT misalign (10.1)
UNDISTORT   = P('UNDISTORT', False)
CROP_BORDER = P('CROP_BORDER', 0)

# ---------------------------------------------------------------- run control
STAGES           = P('STAGES', ('extract',))   # any subset; run in pipeline order. 'prior' scores
                                       # the generators against GT depth with NO SLAM run - about a
                                       # minute an arm against forty for an end2end one, so it is
                                       # the cheap way to ask whether an adapter's DEPTH improved
                                       # ('extract', 'adapt', 'end2end') is the full pipeline
SKIP_EXISTING    = P('SKIP_EXISTING', True)   # reuse a stage's output if already on disk
MIN_FREE_VRAM_MB = P('MIN_FREE_VRAM_MB', 10000)   # shared GPU: checked once at main()'s start.
                                       # High: the arms hold END2END.buffer=1500 (~3.7 GiB) on top
                                       # of VGGT-1B and its optimiser
FRACTION         = P('FRACTION', 100)  # % of the WINDOW the adapter trains on - NOT of the
                                       # sequence: extract_length = window * FRACTION // 100, so
                                       # at STOP=2000 this is 120 frames (it was 272 when the
                                       # window was the whole 4541). Changing STOP silently
                                       # changes the training set size; check both together.
                                       # Also SPLIT_AT. INT only: a float makes extract_length a
                                       # float and stream.py:window_files slices a list with it
START            = P('START', 0)
STOP             = P('STOP', 1000)     # exclusive: the window is [START, STOP); None = to
                                       # the end. 2000 of KITTI 00's 4541 frames is enough
                                       # for the experiment and halves every arm's runtime.
                                       # It also re-keys the whole outputs tree through
                                       # scene_key -> kitti_00_f0-2000, so these arms cannot
                                       # be confused with full-sequence ones (their evo
                                       # Sim(3) is fitted over the window alone)

# a windowed run keys its own outputs tree, or its omni/base would overwrite the full sequence's
SCENE_KEY = scene_key(SCENE, START, STOP)
STREAM_RES       = P('STREAM_RES', 341 * 640)   # tracking resolution budget
DEPTH_PNG_SCALE  = P('DEPTH_PNG_SCALE', 256.0)  # metres = px / this; MUST match the dataset
                                       # (TUM 6553.5). 256 is the KITTI convention and what
                                       # preprocess_kitti.py writes; it saturates at 256 m, well
                                       # past the 80 m the velodyne returns
RENDER_EVAL      = P('RENDER_EVAL', False)      # hi2.py's eval_rendering -> renders/ + psnr/ (11)

# ---------------------------------------------------------------- experiment names
# both REQUIRED, unique within their scene only; lineage is recorded in the adapter's config.json
OUT_ROOT     = P('OUT_ROOT', 'outputs')
EXTRACT_NAME = P('EXTRACT_NAME', 'normal_vggt')   # 'normal' is the OMNIDATA extract of the same
                                       # window; this is its VGGT twin - same tree, same config,
                                       # same kf_* knobs, so EXTRACT_PRIOR is the only difference
# The depth prior the EXTRACT run tracks under. null = Omnidata, upstream's and what every extract
# before this knob used. Any END2END_PRIORS spec works ('vggt_base', an adapt dir, '...@ceil2').
# It changes `disps_prior` in the dump, which is what makes a prior's own scale-grid behaviour
# readable offline (scripts/tmp_dscales_probe.py, scripts/viz_depth.py --source prior). NOTE the
# export's training depth is the TRACKER's depth either way, so an extract under a different prior
# is a different training set - do not mix two priors' exports into one adapt lineage unsaid.
EXTRACT_PRIOR = P('EXTRACT_PRIOR', 'vggt_base')
# RELLIS's wonline_r8_e3_w10_p10 with p10 -> p6 and a16 spelled out: alpha is the ONE field
# that differs from that arm, so the name must not claim a parity this run does not have
ADAPT_NAME   = P('ADAPT_NAME', 'wonline_a16_e12_w10_p10')

# ---------------------------------------------------------------- stage I/O
# a stage RECEIVES these and reads no path global; pure string joins, no disk access in this block
OUT_EXTRACT  = experiment_dir(OUT_ROOT, 'extract', SCENE_KEY, EXTRACT_NAME)

ADAPT_IN     = OUT_EXTRACT                              # an extract export
ADAPT_IMAGES = COLORS                                   # keyframe RGB, indexed by frame number
ADAPT_OUT    = experiment_dir(OUT_ROOT, 'adapt', SCENE_KEY, ADAPT_NAME)
ADAPT_CKPT   = f'{ADAPT_OUT}/{ADAPT_CKPT_SUBDIR}'       # epoch_NNN/; cadence is checkpoint_every
# an adapt experiment IN THIS SCENE to continue from; null = stock VGGT-1B. A name, not a path,
# so the file states the same thing the live driver's ADAPT_INIT_NAME does; the path is derived
ADAPT_INIT_NAME = P('ADAPT_INIT_NAME', None)
ADAPT_INIT   = experiment_dir(OUT_ROOT, 'adapt', SCENE_KEY, ADAPT_INIT_NAME) \
    if ADAPT_INIT_NAME else None

OUT_END2END  = test_dir(OUT_ROOT, 'end2end', SCENE_KEY)     # one subdirectory per prior generator
OUT_PRIOR    = test_dir(OUT_ROOT, 'prior', SCENE_KEY)       # same arm names, scored without SLAM

# VGGT's input size; an adapter's own recorded size wins over this. Resolved in main() (9.3)
VGGT_HW          = P('VGGT_HW', None)  # null = derive | [378, 518] TUM | [294, 518] Replica

# ---------------------------------------------------------------- the SLAM runs
# what every invocation shares; what differs per run is an argument to SlamRunner.run() instead.
# NO `slam:` section in a run config - every field of it is already a top-level key above, and one
# knob must have exactly one spelling in the file
SLAM = SlamConfig(
    weights=DROID_WEIGHTS, colors=COLORS, calib=CALIB, start=START, stop=STOP,
    undistort=UNDISTORT, crop_border=CROP_BORDER, stream_res=STREAM_RES,
    render_eval=RENDER_EVAL)

# ---------------------------------------------------------------- extract (stage 1)
# the kf_* knobs are EXTRACT-ONLY: a generated config only this run is given, asserted in main()
EXTRACT = P.over('extract', ExtractConfig(
    kf_motion_thresh=2.4,           # motion_filter.thresh; any threshold may be null = inherit
    kf_init_thresh=4.0,             # the same gate before initialisation
    kf_redundant_thresh=4.0,        # the one that actually moves the keyframe count
    kf_covis_thresh=0.2,            # extras inserted in terminate(); LOWER -> more
    buffer=1500,                     # hard cap; MUST exceed the count (no overflow guard)
    depth_png_scale=DEPTH_PNG_SCALE,
    mask_filter_thresh=0.005,       # depth_filter disparity agreement
    mask_min_count=2,               # min agreeing neighbours out of 6
    mask_min_disp_ratio=0.5,        # drop pixels below this fraction of the frame's mean disparity
    gt_depths=DEPTHS),              # the accuracy table ONLY - never reaches Hi2 (§9.3)
    fixed=('depth_png_scale', 'gt_depths'))   # DEPTH_PNG_SCALE / DEPTHS feed them

# ---------------------------------------------------------------- adapt (stage 2, LoRA on VGGT)
# LORA is the adapter STRUCTURE, recorded into its config.json; ADAPT is the training run
LORA = P.over('lora', LoRAConfig(
    weights='pretrained_models/vggt',
    vggt_hw=VGGT_HW,           # None -> derived in main()
    rank=8, alpha=16,
    targets=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
    patch_embed=False),        # False = adapt only the alternating-attention stack
    fixed=('vggt_hw',))        # the VGGT_HW key above feeds it

ADAPT = P.over('adapt', AdaptConfig(
    stream_res=STREAM_RES,
    p_single_view=1, max_left=4, max_right=4, radius=8,
    adapt_style='wonline',     # 'normal' epochs | 'online' per arrival | 'wonline' sliding window
    epochs=12, batch_size=2,
    window_size=10,            # 'wonline' only
    lr=1.0e-4, weight_decay=0.0, grad_clip=1.0, lambda_pose=1.0,
    coupled_scale=True, min_mask_pixels=16, seed=0, log_every=20,
    # ---- which exported keyframes are trained on, and what the rest are for ----
    kf_fraction=1.0,           # 1.0 = every exported keyframe; < 1 = equidistant sample of them
    val_source='tail',         # 'tail' = the selection's tail | 'rest' = the keyframes it skipped
    train_frac=0.9,            # 'tail' ONLY; the last 10% of the selection is the val set
    eval_on_val=True,          # depth L1 on held-out keyframes, base vs adapted
    eval_on_train=True,        # also on the train subset, so the train/val gap is visible
    eval_every_epoch=False,    # False = only before training and after the last unit
    eval_max_kf=100,           # subsample each eval subset to at most this many; 0 = no cap
    keep_best=False,           # False = save the last epoch; True = snapshot on val improvement
    checkpoint_every=0),       # 0 = off; N = a loadable adapter dir in ADAPT_CKPT every N epochs
    fixed=('stream_res',))     # the STREAM_RES key above feeds it, and SLAM must agree

# ---------------------------------------------------------------- end2end test (stage 3)


# another adapt run's handoff directory; both prior lists below use it
def _a(name):
    return experiment_dir(OUT_ROOT, 'adapt', SCENE_KEY, name)

# one entry per generator, arm directory INFERRED (7.1). Any entry may carry an '@ceil<tag>'
# modifier (14) - a clamped arm names its own directory, so it cannot overwrite its parent
END2END_PRIORS = P('END2END_PRIORS', (
    'omnidata', 'vggt_base',     # priors[0] is the baseline column
    ADAPT_OUT,                   # this run's final adapter, frozen
))

END2END = P.over('end2end', End2EndConfig(
    priors=END2END_PRIORS,
    length=100000,                    # 100000 = whole sequence
    buffer=1500,                      # the ARMS run all 4541 frames, not the 272-frame extract.
                                      # A hard cap with no overflow guard; ~3.7 GiB GPU +
                                      # 7.0 GiB CPU at 848x256
    gt_traj=GT_TRAJ,                  # evo_ape's reference; ATE is the whole table
    lora=LORA,
    omni_normal_ckpt='pretrained_models/omnidata_dpt_normal_v2.ckpt',
    omni_normal_hw=(512, 512)),
    fixed=('priors', 'gt_traj', 'lora'))   # END2END_PRIORS / GT_TRAJ / the lora section feed them

# ---------------------------------------------------------------- prior test (stage 4)
# The same generators vs GT depth, no SLAM run - it attributes an end2end null (9.2.2). Its own
# list, NOT END2END_PRIORS: scoring an arm here costs a minute rather than forty, so this stage can
# afford every adapter on the scene while an end2end table cannot.
#
# ORDER MATTERS TWICE. priors[0] is the baseline column, hence 'omnidata' first. And the table's
# seen/unseen split comes from the FIRST entry that is an adapter (priortest/config.py's
# resolve_split) - every live arm below records split_at=1000, so the split is the whole window,
# [unseen] is empty
# and [all] is the row to read (9.7). The offline wonline adapter trained to frame 100 and is put
# LAST for that reason: it still gets scored, and the report stars it as scored at a split that is
# not its own.
PRIOR_PRIORS = P('PRIOR_PRIORS', (
    'omnidata', 'vggt_base',          # the two controls: upstream's prior, and VGGT before adapting
    # every live (online-adapted) arm on this window, alphabetical so the table reads stably.
    # live_e40_a16_w12_lag7_low045_raw_r_fix has NO adapter.safetensors yet - adding it makes
    # End2EndConfig.check_priors_exist abort the whole run before any arm is scored
    _a('live_e10_w10_a16_w12_lag3_r_fix_base'),
    _a('live_e10_w10_a16_w12_lag5_low045_raw_base'),
    _a('live_e15_w10_a16_w12_lag3_r_fix_base'),
    _a('live_e15_w10_a16_w12_lag5_low045_raw_base'),
    _a('live_e3_w10_a16_w12_lag3_low045_raw_base'),
    _a('live_e40_a16_w12_lag3_low045_raw_base'),
    _a('live_e40_a16_w12_lag5_low045_raw_r_fix_base'),
    _a('live_e40_a16_w12_lag7_low045_raw_base'),
    ADAPT_OUT,                        # the offline wonline adapter, for contrast; split_at=100
))

PRIOR = P.over('prior', PriorTestConfig(
    priors=PRIOR_PRIORS,
    gt_depths=DEPTHS, depth_png_scale=DEPTH_PNG_SCALE,
    eval_min_depth=1.0,                # m; nothing is closer than a few m in these scenes
    eval_max_depth=50.0,               # m; past this the GT beam interpolation is extrapolation.
                                       # KITTI's velodyne does return to ~80 m - re-run with
                                       # 20/80 into a separate OUT_PRIOR to score the far field,
                                       # which is where this scene's pose accuracy actually lives.
                                       # These six fields are the frames.csv cache key
                                       # (PriorTestConfig.eval_spec): change one and every arm
                                       # already scored re-runs
    eval_samples_per_frame=20000,      # valid pixels kept per frame; 0 = all (needs the RAM)
    seed=0,
    lora=LORA,
    omni_normal_ckpt='pretrained_models/omnidata_dpt_normal_v2.ckpt',
    omni_normal_hw=(512, 512)),
    fixed=('priors', 'gt_depths', 'depth_png_scale', 'lora'))   # fed by the keys above

P.done()   # with -c: the file states every parameter above, and nothing else

# ==============================================================================

# At module scope, not in main(): a spawned child re-executes this module and needs both.
raise_fd_limit()
ensure_venv_on_path()


# ==============================================================================

def save_run_config(out):
    """Copy the -c file beside the run it produced, so a result records how it was made.

    Nothing to do without -c: there the driver's own literals are the record, and they are in git.
    """
    if P.path and os.path.isdir(out):
        shutil.copy(P.path, f'{out}/run_config.yaml')


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
        # built here and released here: the stage receives it and does not own it. None keeps the
        # stock Omnidata path untouched, so an EXTRACT_PRIOR of None is bit-identical to before
        from adaslam.end2end.stage import make_prior
        extract_prior = make_prior(EXTRACT_PRIOR, END2END, stream_hw) if EXTRACT_PRIOR else None
        try:
            run_extract(runner, EXTRACT, OUT_EXTRACT, extract_length, CONFIG,
                        skip_existing=SKIP_EXISTING, prior=extract_prior)
        finally:
            if extract_prior is not None and hasattr(extract_prior, 'release'):
                extract_prior.release()
        save_run_config(OUT_EXTRACT)
    if 'adapt' in STAGES:
        run_adapt(LORA, ADAPT, ADAPT_IN, ADAPT_IMAGES, ADAPT_OUT, ADAPT_CKPT,
                  init_adapter=ADAPT_INIT, skip_existing=SKIP_EXISTING)
        save_run_config(ADAPT_OUT)
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
