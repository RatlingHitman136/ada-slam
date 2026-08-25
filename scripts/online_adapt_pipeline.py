"""CONTINUOUS adaptation: ONE SLAM run in which the depth prior adapts as it tracks (13).

    python scripts/online_adapt_pipeline.py                  # the PARAMETERS block below
    python scripts/online_adapt_pipeline.py -c live_default  # every parameter from a run config

  1 online   HI-SLAM2 over the whole sequence with a VGGT prior that ADAPTS ITSELF: each arriving
             keyframe triggers a burst of LoRA steps on the keyframes local BA has settled, then
             is served by the weights that produced
             -> an adapter in outputs/adapt/<SCENE>/<ONLINE_NAME>/
             -> an arm    in outputs/test/end2end/<SCENE>/<ONLINE_NAME>_live/
  2 end2end  the REFERENCE arms (omni, base, any frozen adapter), reused from disk when a scene
             has already paid for them

Siblings: init_adapt_pipeline.py (9.1) and cont_adapt_pipeline.py (9.7) are the OFFLINE track -
they train on a finished export, then run a SECOND pass with the adapter frozen. Here the run that
produces the supervision is the run being evaluated. Run the preprocess script first.

The KNOB PANEL, not the implementation - the stage is adaslam/online/. The config literals below
are rebuilt in every spawned reader child: primitives only, no disk, no computation.

Every one of them is also a key of run_configs/live_*.yaml: with -c that file states ALL of them
and the literals stand aside, without it nothing changes at all (adaslam/runconfig.py, 13.7).
"""
import os    # nopep8
import sys   # nopep8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # nopep8
# repo root, so `adaslam` imports; its __init__ adds hislam2/ and thirdparty/vggt
sys.path.insert(0, _ROOT)                                                 # nopep8
import shutil
import time
from dataclasses import replace

from adaslam.adapt import LoRAConfig
from adaslam.common import (ADAPT_CKPT_SUBDIR, ONLINE_ARM_SUFFIX, experiment_dir, require_name,
                            test_dir)
from adaslam.end2end import End2EndConfig, run_end2end_test
from adaslam.end2end.config import arm_name
from adaslam.online import OnlineConfig, run_online_adapt
from adaslam.pipeline import (check_sequence, enter, print_arm_dirs, resolve_lora,
                              scene_key, warn_runtime_undistort, window_frames)
from adaslam.print_utils import banner
from adaslam.end2end.report import compare
from adaslam.runconfig import run_config
from adaslam.runtime import ensure_venv_on_path, gpu_gate, raise_fd_limit
from adaslam.slam import SlamConfig, SlamRunner

# ==============================================================================
#  PARAMETERS
# ==============================================================================
# Every literal below is also a KEY of a run config: `-c live_NAME` reads
# run_configs/live_NAME.yaml and takes ALL of them from there instead, which is what a queued job
# needs (adaslam/runconfig.py, 13.7). Without -c these literals stand, unchanged. At module scope
# because a spawned child re-executes this file and must rebuild the same values.
P = run_config(_ROOT, prefix='live_', doc=__doc__)

# ---------------------------------------------------------------- data (preprocessing is NOT here)
# TUM:    SCENE 'rgbd_dataset_freiburg1_room', config/tum_config.yaml, DEPTH_PNG_SCALE 6553.5
# RELLIS: SCENE 'rellis_00000', DATA 'data/RELLIS/00000', config/rellis_config.yaml, 2847 frames
SCENE   = P('SCENE', 'kitti_00')        # names the outputs/ tree
DATA    = P('DATA', 'data/KITTI/00')    # preprocess_kitti.py's output layout
COLORS  = P('COLORS', f'{DATA}/colors')  # a SYMLINK to the mirror's image_2 - nothing was copied
DEPTHS  = P('DEPTHS', None)             # KITTI ships no GT depth; unread by this driver
GT_TRAJ = P('GT_TRAJ', f'{DATA}/traj_tum.txt')
CALIB   = P('CALIB', f'{DATA}/calib.txt')
CONFIG  = P('CONFIG', 'config/kitti_config.yaml')
DROID_WEIGHTS = P('DROID_WEIGHTS', 'pretrained_models/droid.pth')

# undistort offline in the preprocess script instead (10.1)
UNDISTORT   = P('UNDISTORT', False)
CROP_BORDER = P('CROP_BORDER', 0)

# ---------------------------------------------------------------- run control
STAGES           = P('STAGES', ('online', 'end2end'))
SKIP_EXISTING    = P('SKIP_EXISTING', True)
MIN_FREE_VRAM_MB = P('MIN_FREE_VRAM_MB', 15000)
LENGTH           = P('LENGTH', 100000)

START            = P('START', 0)
STOP             = P('STOP', 1000)

# a windowed run keys its own outputs tree, or its omni/base would overwrite the full sequence's
SCENE_KEY = scene_key(SCENE, START, STOP)   # DERIVED, like every path below - not a config key
STREAM_RES       = P('STREAM_RES', 341 * 640)     # tracking resolution budget
BUFFER           = P('BUFFER', 1500)   # keyframe buffer; must clear the WINDOW's keyframe
                                       # count - a HARD CAP with no overflow guard. At 2.5 MiB
                                       # GPU + 4.8 MiB CPU per slot this is 6.1 + 11.6 GiB, and
                                       # over a 2000-frame window it cannot overflow at all.
                                       # Deliberately generous (RELLIS: 255 kf / 2847 frames);
                                       # trim once an extract reports the real keyframe rate
RENDER_EVAL      = P('RENDER_EVAL', False)   # hi2.py's eval_rendering -> renders/ + psnr/ (11)

# ---------------------------------------------------------------- experiment names
# REQUIRED, unique within its scene only; lineage is recorded in the adapter's config.json
OUT_ROOT    = P('OUT_ROOT', 'outputs')
# the SAME name as the RELLIS arm this is matched to, in a scene directory of its own - so the
# two are directly comparable and neither can overwrite the other
ONLINE_NAME = P('ONLINE_NAME', 'live_e3_w10_a16_w12_lag5_lr12_monoa10_base')

# ---------------------------------------------------------------- stage I/O
# a stage RECEIVES these and reads no path global; pure string joins, no disk access in this block
ADAPT_OUT  = experiment_dir(OUT_ROOT, 'adapt', SCENE_KEY, ONLINE_NAME)
ADAPT_CKPT = f'{ADAPT_OUT}/{ADAPT_CKPT_SUBDIR}'        # epoch_NNN/; cadence is checkpoint_every_kf

OUT_END2END = test_dir(OUT_ROOT, 'end2end', SCENE_KEY)     # one subdirectory per arm
# the suffix keeps a later FROZEN test of this adapter from overwriting the live trajectory (13.4)
ONLINE_ARM_NAME = f'{ONLINE_NAME}{ONLINE_ARM_SUFFIX}'
ONLINE_ARM      = f'{OUT_END2END}/{ONLINE_ARM_NAME}'

# the adapter this run CONTINUES from: an adapt experiment in this scene, or None for stock VGGT-1B
ADAPT_INIT_NAME = P('ADAPT_INIT_NAME', None)
ADAPT_INIT = None if ADAPT_INIT_NAME is None else \
    experiment_dir(OUT_ROOT, 'adapt', SCENE_KEY, ADAPT_INIT_NAME)

# VGGT's input size; an adapter's own recorded size wins over this. Resolved in main() (9.3)
VGGT_HW = P('VGGT_HW', None)           # None = derive | (378, 518) TUM | (294, 518) Replica

# ---------------------------------------------------------------- the SLAM runs
# what the online run and every reference arm share, so they cannot disagree about the stream.
# No 'slam' section in a run config: every field of it is a top-level key above
SLAM = SlamConfig(
    weights=DROID_WEIGHTS, colors=COLORS, calib=CALIB, start=START, stop=STOP,
    undistort=UNDISTORT, crop_border=CROP_BORDER, stream_res=STREAM_RES,
    render_eval=RENDER_EVAL)

# ---------------------------------------------------------------- the adapter structure
# recorded into the adapter's config.json, so an arm always runs the model it was trained in
LORA = P.over('lora', LoRAConfig(
    weights='pretrained_models/vggt',
    vggt_hw=VGGT_HW,           # None -> derived in main()
    rank=8, alpha=16,
    targets=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
    patch_embed=False),        # False = adapt only the alternating-attention stack
    fixed=('vggt_hw',))        # the VGGT_HW key above feeds it; refused inside the section

# ---------------------------------------------------------------- online adaptation (stage 1)
ONLINE = P.over('online', OnlineConfig(
    # ---- warm-up: TWO gates - when the adapter starts LEARNING, and when it starts SERVING ----
    warmup_kf=12,              # keyframes before the first optimiser step, which lands at +1
    handover_kf=12,            # keyframes served by the FALLBACK prior; must be >= warmup_kf
    warmup_prior='omnidata',   # 'omnidata' = upstream's prior | 'self' = the adapting VGGT itself

    # ---- schedule: the vocabulary of adapt/trainer.py:schedule, live ----
    adapt_style='wonline',      # 'online' = the arrival alone | 'wonline' = a sliding window
    steps_per_kf=3,            # steps ('online') / shuffled passes ('wonline') per arrival
    window_size=10,             # 'wonline' ONLY
    batch_size=2,              # 'wonline' ONLY - a keyframe arrives alone in 'online'
    lag=5,                     # keyframes back from the end the target is taken

    # ---- sample construction ----
    context_kf=0,              # previous KEYFRAMES appended after the target; 0 = monocular
    stream_res=STREAM_RES,     # must equal SLAM.stream_res

    # ---- optimisation ----
    lr=1.2e-4,
    weight_decay=0.0, grad_clip=1.0, lambda_pose=1.0,
    coupled_scale=True, min_mask_pixels=16, seed=0,
    log_every=5,               # every step: there are only steps_per_kf of them per keyframe

    # ---- supervision mask (the same knobs ExtractConfig uses for depth_slam/) ----
    mask_filter_thresh=0.005,  # depth_filter disparity agreement
    mask_min_count=2,          # min agreeing neighbours out of 6
    mask_min_disp_ratio=0.5,   # drop pixels below this fraction of the frame's mean disparity

    # ---- the loss gate: skip an arrival whose newest keyframe is not worth training on ----
    # both quantities are always measured into gate_log.json; this only picks the deciding one
    gate_metric='raw',         # 'rel' = relative loss, flat across a run | 'raw' = drifts with it
    gate_lo=0.0,             # 0 = off; skip BELOW this - the target already fits
    gate_hi=0.0,               # 0 = off; skip ABOVE this - the target is broken, not hard
                               # NOTE both bounds were calibrated on RELLIS, and 'raw' carries the
                               # tracker's own depth unit - which KITTI does not share. Read
                               # gate_log.json after the first run and re-derive them; the band can
                               # be re-chosen from one run's log without re-running.

    # ---- output ----
    checkpoint_every_kf=0),  # 0 = off; N = a loadable adapter dir every N adapted keyframes
    fixed=('stream_res',))   # the STREAM_RES key above feeds it, and SLAM must agree

# ---------------------------------------------------------------- reference arms (stage 2)
# ordinary frozen arms, reused from disk so a scene pays for them once; [0] is the baseline column
END2END_PRIORS = P('END2END_PRIORS', ('omnidata', 'vggt_base'))

# the seen/unseen boundary; None = the whole run, so [unseen] is empty and ate_all is the row (12.2).
# The RELLIS arm used 200, which was about its handover frame; KITTI's is not known until the run
# reports warmup_end_frame, and re-scoring at another split is one evo_ape over a finished
# trajectory (SKIP_EXISTING reuses the run and only re-evaluates).
SPLIT_AT = P('SPLIT_AT', None)

END2END = P.over('end2end', End2EndConfig(
    priors=END2END_PRIORS,
    length=LENGTH,
    buffer=BUFFER,
    gt_traj=GT_TRAJ,                  # evo_ape's reference; ATE is the whole table
    lora=LORA,
    omni_normal_ckpt='pretrained_models/omnidata_dpt_normal_v2.ckpt',
    omni_normal_hw=(512, 512)),
    fixed=('priors', 'length', 'buffer', 'gt_traj', 'lora'))   # all fed by the keys above

P.done()   # with -c: the file states every parameter above, and nothing else

# ==============================================================================

# At module scope, not in main(): a spawned child re-executes this module and needs both.
raise_fd_limit()
ensure_venv_on_path()


# ==============================================================================

def main():
    # once per process, before any Process is started; every path above is repo-root relative
    enter(_ROOT)

    require_name('ONLINE_NAME', ONLINE_NAME)

    n_frames = check_sequence(
        COLORS, DEPTHS, GT_TRAJ,
        required=[CALIB, CONFIG, DROID_WEIGHTS, GT_TRAJ])
    warn_runtime_undistort(UNDISTORT, CROP_BORDER)

    window = window_frames(n_frames, START, STOP)   # validated against the sequence
    length = min(LENGTH, window)
    # None = the end of the RUN, not of the sequence: a split the run never reached is no boundary
    split_at = (START + window) if SPLIT_AT is None else SPLIT_AT

    os.makedirs(OUT_END2END, exist_ok=True)
    if P.path:              # the knobs that produced this run, recorded beside its adapter
        os.makedirs(ADAPT_OUT, exist_ok=True)
        shutil.copy(P.path, f'{ADAPT_OUT}/run_config.yaml')

    # here, not in PARAMETERS: deriving reads a frame, and this runs after chdir
    global LORA, END2END
    LORA, stream_hw = resolve_lora(LORA, COLORS, STREAM_RES)
    END2END = replace(END2END, lora=LORA)   # the vggt_base arm and the online prior both read it

    print(f'knobs     : {P.path or "the PARAMETERS block in this file (no -c)"}')
    print(f'sequence  : {SCENE}  ({n_frames} frames, {COLORS})')
    print(f'config    : {CONFIG}  calib {CALIB}  (stock keyframing - no kf_* overrides anywhere)')
    print(f'stream    : {stream_hw[1]}x{stream_hw[0]} (aspect '
          f'{stream_hw[1]/stream_hw[0]:.3f})   VGGT input: {LORA.vggt_hw[1]}x{LORA.vggt_hw[0]}'
          f'{" (derived)" if VGGT_HW is None else " (pinned by VGGT_HW)"}')
    print(f'run       : frames {START}..{START+length-1} of {n_frames}'
          + (f'   WINDOW -> outputs tree {SCENE_KEY}' if SCENE_KEY != SCENE else ''))
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
          f'{" (= the whole run, so [unseen] is empty)" if SPLIT_AT is None else ""}')
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
