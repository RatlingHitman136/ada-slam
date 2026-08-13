"""CONTINUOUS adaptation: ONE SLAM run in which the depth prior adapts as it tracks (13).

    python scripts/online_adapt_pipeline.py    # from the repo root, adaslam venv active

  1 online   HI-SLAM2 over the whole sequence with a VGGT prior that ADAPTS ITSELF. The first
             WARMUP_KF keyframes are served by the fallback prior; from then on every arriving
             keyframe first triggers a burst of LoRA steps on the keyframes local BA has already
             settled, then is served by the weights that produced.
             -> an adapter in outputs/adapt/<SCENE>/<ONLINE_NAME>/
             -> an arm    in outputs/test/end2end/<SCENE>/<ONLINE_NAME>_live/
  2 end2end  the REFERENCE arms (omni, base, any frozen adapter) so the online arm has something
             to be compared against. Reused from disk when they have been run before, which they
             usually have - a scene's baselines are run once.

The two sibling drivers are the OFFLINE track and are unchanged: init_adapt_pipeline.py adapts on
a densified prefix, cont_adapt_pipeline.py on a thin equidistant sample of the whole sequence. Both
train on a finished export and then run a SECOND SLAM pass with the adapter frozen. This one has no
extract stage and no frozen pass at all - the run that produces the supervision is the run being
evaluated.

Three things follow from that, and they are why this is a driver of its own:

  * the supervision is LOCAL-BA depth. The offline export dumps after GLOBAL BA (hi2.py:155), the
    only instant where disps / disps_up / poses are mutually consistent. Live targets are noisier
    by construction.
  * the target is produced by BA that was itself pulled toward the prior being adapted, via JDSA.
    mono_depth_alpha is small (0.001-0.01) so the target is mostly photometric, but this is a
    self-training loop the offline pipeline does not have. Watch train_log.json's loss collapsing
    while ATE worsens.
  * there is no seen/unseen frontier for the ADAPTER - it learned across the whole sequence - so
    SPLIT_AT defaults to the whole thing and ate_all is the row the comparison reduces to (12.2).
    The one meaningful boundary, the handover frame, is recorded as `warmup_end_frame` in the
    adapter's config.json; 12.3 notes any split re-scores for free from evo/error_array.npy.

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

from adaslam.adapt import LoRAConfig
from adaslam.common import (ADAPT_CKPT_SUBDIR, ONLINE_ARM_SUFFIX, experiment_dir, require_name,
                            test_dir)
from adaslam.end2end import End2EndConfig, run_end2end_test
from adaslam.end2end.config import arm_name
from adaslam.online import OnlineConfig, run_online_adapt
from adaslam.pipeline import (check_sequence, enter, print_arm_dirs, resolve_lora,
                              warn_runtime_undistort)
from adaslam.print_utils import banner
from adaslam.end2end.report import compare
from adaslam.runtime import ensure_venv_on_path, gpu_gate, raise_fd_limit
from adaslam.slam import SlamConfig, SlamRunner

# ==============================================================================
#  PARAMETERS
# ==============================================================================

# ---------------------------------------------------------------- data (preprocessing is NOT here)
# TUM:    SCENE 'rgbd_dataset_freiburg1_room', DATA f'data/TUM/{SCENE}', config/tum_config.yaml,
#         DEPTH_PNG_SCALE 6553.5
SCENE   = 'rellis_00000'               # names the outputs/ tree
DATA    = 'data/RELLIS/00000'          # preprocess_rellis3d.py's output layout
COLORS  = f'{DATA}/colors'
DEPTHS  = f'{DATA}/depths'             # None if the dataset has no GT depth; unread by this driver
GT_TRAJ = f'{DATA}/traj_tum.txt'
CALIB   = f'{DATA}/calib.txt'
CONFIG  = 'config/rellis_config.yaml'
DROID_WEIGHTS = 'pretrained_models/droid.pth'

# undistort offline in the preprocess script instead (10.1)
UNDISTORT   = False
CROP_BORDER = 0

# ---------------------------------------------------------------- run control
STAGES           = ('online', 'end2end')   # any subset; run in pipeline order
SKIP_EXISTING    = True                # reuse a stage's output if it is already on disk
MIN_FREE_VRAM_MB = 12000               # HIGHER than the offline drivers' 7000: this run holds
                                       # VGGT-1B, its optimiser state AND the whole SLAM system at
                                       # once. Measure on a short LENGTH before trusting it.
LENGTH           = 100000              # frames to run over; 100000 = the whole sequence
START            = 0
STREAM_RES       = 341 * 640           # tracking resolution budget
BUFFER           = 900                 # keyframe buffer; must clear the sequence's keyframe count
RENDER_EVAL      = False               # hi2.py's eval_rendering -> renders/ + psnr/ (11)

# ---------------------------------------------------------------- experiment names
# REQUIRED, and unique within its scene only. Lineage is DATA, not naming: the adapter's
# config.json records the run that supervised it and the adapter it continued from.
OUT_ROOT    = 'outputs'
ONLINE_NAME = 'live_e3_w10_b5_a16_w12_lag3_base'

# ---------------------------------------------------------------- stage I/O
# A stage RECEIVES its paths and reads no path global. Pure string joins - no disk access here.
ADAPT_OUT  = experiment_dir(OUT_ROOT, 'adapt', SCENE, ONLINE_NAME)
ADAPT_CKPT = f'{ADAPT_OUT}/{ADAPT_CKPT_SUBDIR}'        # epoch_NNN/; cadence is checkpoint_every_kf

OUT_END2END = test_dir(OUT_ROOT, 'end2end', SCENE)     # one subdirectory per arm
# The live arm's own directory. The suffix is load-bearing: without it, later testing this run's
# FROZEN final adapter as an ordinary END2END_PRIORS entry would infer the same directory and
# overwrite this trajectory (common.py:ONLINE_ARM_SUFFIX).
ONLINE_ARM_NAME = f'{ONLINE_NAME}{ONLINE_ARM_SUFFIX}'
ONLINE_ARM      = f'{OUT_END2END}/{ONLINE_ARM_NAME}'

# WHICH ADAPTER THIS RUN STARTS FROM. None = stock VGGT-1B; otherwise the name of an adapt
# experiment in this SCENE - the same vocabulary an END2END_PRIORS entry uses. A checkpoint works:
#   f'{experiment_dir(OUT_ROOT, "adapt", SCENE, "x")}/{ADAPT_CKPT_SUBDIR}/epoch_005'
# With WARMUP_PRIOR = 'self' this is also what serves the warm-up keyframes.
ADAPT_INIT_NAME = None
ADAPT_INIT = None if ADAPT_INIT_NAME is None else \
    experiment_dir(OUT_ROOT, 'adapt', SCENE, ADAPT_INIT_NAME)

# VGGT's input size; an adapter's own recorded size wins over this. Derived in main(), not here -
# this block must not touch the disk (9.3).
VGGT_HW = None                         # None = derive | (378, 518) TUM | (294, 518) Replica

# ---------------------------------------------------------------- the SLAM runs
# What every invocation shares - the online run and every reference arm - so they cannot disagree
# about the stream, the calibration or the resolution.
SLAM = SlamConfig(
    weights=DROID_WEIGHTS, colors=COLORS, calib=CALIB, start=START,
    undistort=UNDISTORT, crop_border=CROP_BORDER, stream_res=STREAM_RES,
    render_eval=RENDER_EVAL)

# ---------------------------------------------------------------- the adapter structure
# Recorded into the adapter's config.json, so an arm always runs the model it was trained in.
LORA = LoRAConfig(
    weights='pretrained_models/vggt',
    vggt_hw=VGGT_HW,           # None -> derived in main()
    rank=8, alpha=16,
    targets=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
    patch_embed=False)         # False = adapt only the alternating-attention stack

# ---------------------------------------------------------------- online adaptation (stage 1)
ONLINE = OnlineConfig(
    # ---- warm-up: TWO gates - when the adapter starts LEARNING, and when it starts SERVING ----
    warmup_kf=12,              # keyframes before the first optimiser step, which lands at
                               # warmup_kf + 1. 12 is TrackFrontend.warmup, so learning begins
                               # right after initialisation, on keyframes local BA has settled.
    handover_kf=12,            # keyframes served by the FALLBACK; VGGT serves from here on. Must
                               # be >= warmup_kf; equal is the old single-gate behaviour. Between
                               # the two the adapter trains while Omnidata still drives, which is
                               # FREE - the steps run either way. Measured on rellis_00000: at 40
                               # units the adapter is already -14.3% vs omni as a frozen prior
                               # (chkp_039), stock VGGT is +6.2%, so the crossover is somewhere
                               # below keyframe 53 and 30 is a first bracket on it.
    warmup_prior='omnidata',   # 'omnidata' = upstream's prior, a genuinely different model
                               # 'self'     = the same VGGT this run adapts (with ADAPT_INIT set,
                               #              a pretrained adapter). Makes the split INERT: both
                               #              branches are then the same, adapting, model.

    # ---- schedule: the vocabulary of adapt/trainer.py:schedule, live ----
    adapt_style='wonline',      # 'online'  = the arriving keyframe alone, steps_per_kf steps
                               # 'wonline' = a SLIDING WINDOW of the arrival + the window_size-1
                               #             keyframes before it, steps_per_kf shuffled batched
                               #             passes, so each is revisited window_size times
    steps_per_kf=3,            # optimiser steps ('online') / shuffled passes ('wonline') per
                               # arriving keyframe. This is the runtime knob: it multiplies the
                               # forward+backward cost of the whole run.
    window_size=10,             # 'wonline' ONLY
    batch_size=5,              # 'wonline' ONLY - a keyframe arrives alone in 'online'
    lag=3,                     # keyframes back from the end the target is taken. 2 is
                               # track_frontend.py:65's own line: __update reports changes up to
                               # t1-2, so that is the newest index the repo already treats as
                               # settled enough to hand downstream.

    # ---- sample construction ----
    context_kf=0,              # previous KEYFRAMES appended after the target. 0 = monocular and
                               # depth-only, which is how prior_extractor itself calls the model.
                               # >0 also supervises poses from video.poses, at >0 x the VRAM.
    stream_res=STREAM_RES,     # must equal SLAM.stream_res

    # ---- optimisation ----
    lr=1e-4, weight_decay=0.0, grad_clip=1.0, lambda_pose=1.0,
    coupled_scale=True, min_mask_pixels=16, seed=0,
    log_every=2,               # every step: there are only steps_per_kf of them per keyframe

    # ---- supervision mask (the same knobs ExtractConfig uses for depth_slam/) ----
    mask_filter_thresh=0.005,  # depth_filter disparity agreement
    mask_min_count=2,          # min agreeing neighbours out of 6
    mask_min_disp_ratio=0.5,   # drop pixels below this fraction of the frame's mean disparity

    # ---- output ----
    checkpoint_every_kf=50)     # 0 = off; N = a loadable adapter dir every N adapted keyframes,
                               # each testable as an arm named <ONLINE_NAME>_chkp_NNN

# ---------------------------------------------------------------- reference arms (stage 2)
# The online arm needs something to be compared against. These are ordinary frozen arms and are
# reused from disk, so a scene pays for them once. [0] is the baseline column.
END2END_PRIORS = ('omnidata', 'vggt_base')

# The seen/unseen boundary. None = the whole sequence, i.e. everything counts as "seen" and the
# [unseen] table is empty - the honest default here, exactly as in cont_adapt_pipeline.py: the
# adapter learned across the WHOLE sequence, so no frame index separates trained from untrained.
SPLIT_AT = 200

END2END = End2EndConfig(
    priors=END2END_PRIORS,
    length=LENGTH,
    buffer=BUFFER,
    gt_traj=GT_TRAJ,                  # evo_ape's reference; ATE is the whole table
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

    require_name('ONLINE_NAME', ONLINE_NAME)

    n_frames = check_sequence(
        COLORS, DEPTHS, GT_TRAJ,
        required=[CALIB, CONFIG, DROID_WEIGHTS, GT_TRAJ])
    warn_runtime_undistort(UNDISTORT, CROP_BORDER)

    length = min(LENGTH, n_frames)
    split_at = n_frames if SPLIT_AT is None else SPLIT_AT

    os.makedirs(OUT_END2END, exist_ok=True)

    # here, not in PARAMETERS: deriving reads a frame, which that block must not do. After chdir,
    # so relative COLORS resolves the same however invoked.
    global LORA, END2END
    LORA, stream_hw = resolve_lora(LORA, COLORS, STREAM_RES)
    END2END = replace(END2END, lora=LORA)   # the vggt_base arm and the online prior both read it

    print(f'sequence  : {SCENE}  ({n_frames} frames, {COLORS})')
    print(f'config    : {CONFIG}  calib {CALIB}  (stock keyframing - no kf_* overrides anywhere)')
    print(f'stream    : {stream_hw[1]}x{stream_hw[0]} (aspect '
          f'{stream_hw[1]/stream_hw[0]:.3f})   VGGT input: {LORA.vggt_hw[1]}x{LORA.vggt_hw[0]}'
          f'{" (derived)" if VGGT_HW is None else " (pinned by VGGT_HW)"}')
    print(f'run       : frames 0..{length-1} of {n_frames}')
    print(f'warm-up   : {ONLINE.handover_kf} keyframes on {ONLINE.warmup_prior}; adaptation starts '
          f'at keyframe {ONLINE.warmup_kf + 1}'
          + (f' - SPLIT, {ONLINE.handover_kf - ONLINE.warmup_kf} keyframes of training before '
             f'handover' if ONLINE.handover_kf > ONLINE.warmup_kf else ''))
    print(f'schedule  : {ONLINE.adapt_style}, {ONLINE.steps_per_kf} steps per arriving keyframe'
          + (f', window {ONLINE.window_size} batch {ONLINE.batch_size}'
             if ONLINE.adapt_style == 'wonline' else ''))
    print(f'            starts from '
          f'{ADAPT_INIT if ADAPT_INIT else "stock VGGT-1B (no adapter)"}')
    print(f'split     : frame {split_at}'
          f'{" (= whole sequence, so [unseen] is empty)" if SPLIT_AT is None else ""}')
    print(f'stages    : {" ".join(STAGES)}   render_eval {RENDER_EVAL}')
    print(f'outputs   : {ADAPT_OUT}')
    print(f'            {ONLINE_ARM:<58} <- the live arm')
    print_arm_dirs(STAGES, (('end2end', END2END, OUT_END2END),))

    # one VRAM check up front, before any GPU work or spawned Process
    gpu_gate(MIN_FREE_VRAM_MB)

    # ONE runner for every HI-SLAM2 invocation: the online run and every reference arm
    runner = SlamRunner(SLAM)

    t_all = time.time()
    online_res = None
    if 'online' in STAGES:
        online_res = run_online_adapt(
            runner, ONLINE, END2END, ADAPT_OUT, ADAPT_CKPT, ONLINE_ARM, CONFIG,
            length, BUFFER, split_at, stream_hw=stream_hw, init_adapter=ADAPT_INIT,
            skip_existing=SKIP_EXISTING)

    if 'end2end' in STAGES:
        ref = run_end2end_test(runner, SLAM, END2END, OUT_END2END, CONFIG, split_at,
                               skip_existing=SKIP_EXISTING)
        labels = [arm_name(s) for s in END2END_PRIORS]
        if online_res is not None:
            banner('comparison including the online arm')
            compare([*labels, ONLINE_ARM_NAME], [*ref, online_res])

    print(f'\nall stages done in {time.time()-t_all:.0f}s')
    print('\nread first:')
    print('  the table above           ate_all. [unseen] is empty unless SPLIT_AT is set, because')
    print('                            the adapter learned across the whole sequence (12.2)')
    print(f'  {ADAPT_OUT}/train_log.json')
    print('                            the live loss trend. Collapsing toward zero while ATE gets')
    print('                            WORSE is the self-training failure mode to watch for')
    print(f'  scripts/ate_over_time.py -s {SCENE} {" ".join(arm_name(s) for s in END2END_PRIORS)} '
          f'{ONLINE_ARM_NAME} --bins 12')
    print('                            the shape that would show the adaptation doing something')
    print('                            is the online arm tracking the baselines through the')
    print('                            warm-up bins and separating after the handover frame')


if __name__ == '__main__':
    main()
