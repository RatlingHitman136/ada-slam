"""CONTINUOUS adaptation: ONE SLAM run in which the depth prior adapts as it tracks (13).

    python scripts/online_adapt_pipeline.py    # from the repo root, adaslam venv active

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
                              scene_key, warn_runtime_undistort, window_frames)
from adaslam.print_utils import banner
from adaslam.end2end.report import compare
from adaslam.runtime import ensure_venv_on_path, gpu_gate, raise_fd_limit
from adaslam.slam import SlamConfig, SlamRunner

# ==============================================================================
#  PARAMETERS
# ==============================================================================

# ---------------------------------------------------------------- data (preprocessing is NOT here)
# TUM: SCENE 'rgbd_dataset_freiburg1_room', config/tum_config.yaml, DEPTH_PNG_SCALE 6553.5
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
MIN_FREE_VRAM_MB = 12000               # high: VGGT-1B, its optimiser and SLAM are all resident
LENGTH           = 100000              # frames to run over; 100000 = the whole sequence
START            = 0
STOP             = None                # exclusive: the window is [START, STOP); None = to the end

# a windowed run keys its own outputs tree, or its omni/base would overwrite the full sequence's
SCENE_KEY = scene_key(SCENE, START, STOP)
STREAM_RES       = 341 * 640           # tracking resolution budget
BUFFER           = 900                 # keyframe buffer; must clear the sequence's keyframe count
RENDER_EVAL      = False               # hi2.py's eval_rendering -> renders/ + psnr/ (11)

# ---------------------------------------------------------------- experiment names
# REQUIRED, unique within its scene only; lineage is recorded in the adapter's config.json
OUT_ROOT    = 'outputs'
# the E3 coupling term live, against live_e8_w10_a16_w12_lag3_hi2_low035_raw_base (ATE 24.722,
# 2690 s): IDENTICAL in every field, coupling_lambda 0 -> 1. That arm is the best raw-gated live
# run on this scene, and the pair is the only comparison this run supports.
ONLINE_NAME = 'live_e1_w10_a16_w12_lag3_jdsa_l2_lr3__lo10_base'

# ---------------------------------------------------------------- stage I/O
# a stage RECEIVES these and reads no path global; pure string joins, no disk access in this block
ADAPT_OUT  = experiment_dir(OUT_ROOT, 'adapt', SCENE_KEY, ONLINE_NAME)
ADAPT_CKPT = f'{ADAPT_OUT}/{ADAPT_CKPT_SUBDIR}'        # epoch_NNN/; cadence is checkpoint_every_kf

OUT_END2END = test_dir(OUT_ROOT, 'end2end', SCENE_KEY)     # one subdirectory per arm
# the suffix keeps a later FROZEN test of this adapter from overwriting the live trajectory (13.4)
ONLINE_ARM_NAME = f'{ONLINE_NAME}{ONLINE_ARM_SUFFIX}'
ONLINE_ARM      = f'{OUT_END2END}/{ONLINE_ARM_NAME}'

# the adapter this run CONTINUES from: an adapt experiment in this scene, or None for stock VGGT-1B
ADAPT_INIT_NAME = None
ADAPT_INIT = None if ADAPT_INIT_NAME is None else \
    experiment_dir(OUT_ROOT, 'adapt', SCENE_KEY, ADAPT_INIT_NAME)

# VGGT's input size; an adapter's own recorded size wins over this. Resolved in main() (9.3)
VGGT_HW = None                         # None = derive | (378, 518) TUM | (294, 518) Replica

# ---------------------------------------------------------------- the SLAM runs
# what the online run and every reference arm share, so they cannot disagree about the stream
SLAM = SlamConfig(
    weights=DROID_WEIGHTS, colors=COLORS, calib=CALIB, start=START, stop=STOP,
    undistort=UNDISTORT, crop_border=CROP_BORDER, stream_res=STREAM_RES,
    render_eval=RENDER_EVAL)

# ---------------------------------------------------------------- the adapter structure
# recorded into the adapter's config.json, so an arm always runs the model it was trained in
LORA = LoRAConfig(
    weights='pretrained_models/vggt',
    vggt_hw=VGGT_HW,           # None -> derived in main()
    rank=8, alpha=16,
    targets=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
    patch_embed=False)         # False = adapt only the alternating-attention stack

# ---------------------------------------------------------------- online adaptation (stage 1)
ONLINE = OnlineConfig(
    # ---- warm-up: TWO gates - when the adapter starts LEARNING, and when it starts SERVING ----
    warmup_kf=12,              # keyframes before the first optimiser step, which lands at +1
    handover_kf=12,            # keyframes served by the FALLBACK prior; must be >= warmup_kf
    warmup_prior='omnidata',   # 'omnidata' = upstream's prior | 'self' = the adapting VGGT itself

    # ---- schedule: the vocabulary of adapt/trainer.py:schedule, live ----
    adapt_style='wonline',      # 'online' = the arrival alone | 'wonline' = a sliding window
    steps_per_kf=1,             # steps ('online') / shuffled passes ('wonline') per arrival
    window_size=10,             # 'wonline' ONLY
    batch_size=2,              # 'wonline' ONLY - a keyframe arrives alone in 'online'
    lag=3,                     # keyframes back from the end the target is taken

    # ---- sample construction ----
    context_kf=0,              # previous KEYFRAMES appended after the target; 0 = monocular
    stream_res=STREAM_RES,     # must equal SLAM.stream_res

    # ---- optimisation ----
    lr=1.0e-4,
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
    gate_lo=0.001,             # 0 = off; skip BELOW this - the target already fits
    gate_hi=0,               # 0 = off; skip ABOVE this - the target is broken, not hard

    # ---- WHICH OBJECTIVE (the knob) ----
    # 'normal'  masked L1 in DEPTH after a per-sample median scale. Every run before this knob
    #           existed, bit for bit.
    # 'coupled' 'normal' + E3's lambda*b^2 slope penalty. MEASURED AND NULL: three seeds put
    #           lambda=0 at 24.96 +/- 1.75 against lambda=1's 23.78 (t=0.83), and the placebo that
    #           reassigns the coefficients at random scored the same as lambda=0. Needs
    #           coupling_lambda > 0.
    # 'jdsa'    the residual the SOLVER cannot absorb. depth_loss aligns with one median scalar in
    #           depth; JDSA aligns with a 4-DOF bilinear field in DISPARITY, refit every BA
    #           iteration (geom/ba.py:161-196). This fits that same family and penalises what
    #           survives it, so the objective stops paying for what the solver discards.
    depth_loss='jdsa',
    jdsa_norm='l2',            # 'jdsa' ONLY: L2 weights far pixels and outliers hard, and the
                               # targets are masked tracker depth. L2 only once L1 works.
    jdsa_ridge=1e-6,           # 'jdsa' ONLY: ridge on the 4x4 normal equations, RELATIVE to their
                               # mean diagonal. Guards a mask confined to one image region, where
                               # the four corners are not determined.
    jdsa_lattice='full',       # 'jdsa' ONLY: 'full' = fit at vggt_hw (more points, better
                               # conditioned) | 'ba' = the [3::8,3::8] 1/64 subsample BA reads

    # ---- E3: penalise the depth->scale coupling (the objective change) ----
    # depth_loss aligns scale per sample, so L(c*p) = L(p) exactly and the per-frame output scale
    # has ZERO gradient. Its depth-coupled part - scale varying with how far the scene is, i.e.
    # range compression - is what correlates with ATE offline. This adds lambda * b^2 where b is
    # the slope of log(s_i) on log(median depth), fitted over the arrival's window.
    #
    # 0.0 IS THE OLD LOSS, bit for bit: the statistics pass is skipped entirely.
    #
    # WHAT THE OFFLINE RESULT SAYS, so this run is read honestly. On p10 the term is a NULL: over
    # three seeds lambda=0 spans 23.680-26.958 (sd 1.75) and lambda=1 sits at 23.782, i.e.
    # +1.18 +/- 1.43 m, t=0.83. A placebo that keeps the coefficients but reassigns them to random
    # keyframes scores 25.006, statistically identical to lambda=0. Offline, nothing here works.
    # This run exists because the LIVE setting is genuinely different - the window is short and its
    # targets are still moving under local BA - not because the offline evidence is encouraging.
    # Expect no effect; a single live pair cannot establish one either way.
    #
    # batch_size(2) < window_size(10) is DELIBERATE: it matches the reference arm, and changing two
    # knobs at once would make the comparison meaningless. The config will print a note about the
    # uncancelled per-step fragment (~5x here) - that is the honest description of this setting,
    # not a misconfiguration to fix. Offline, batch_size = window_size was not better.
    # Requires adapt_style='wonline' and window_size >= 3.
    coupling_lambda=0,
    coupling_axis='target',    # 'target' = median TARGET depth. Prefer it: the network does not
                               # control the axis. 'pred' lets it satisfy the penalty by collapsing
                               # every predicted median to one value = range collapse.
    coupling_min_var=1e-4,     # skip the term when the window has no depth spread (sum x~^2 below
                               # this), where b would be an ill-conditioned division
    coupling_shuffle=False,    # PLACEBO CONTROL - reassigns the coefficients to keyframes at random,
                               # keeping their magnitude and zero sum but destroying the link to
                               # depth. Only ever True for a control arm; needs coupling_lambda > 0.

    # ---- output ----
    checkpoint_every_kf=0)    # 0 = off; N = a loadable adapter dir every N adapted keyframes

# ---------------------------------------------------------------- reference arms (stage 2)
# ordinary frozen arms, reused from disk so a scene pays for them once; [0] is the baseline column
END2END_PRIORS = ('omnidata', 'vggt_base')

# the seen/unseen boundary; None = the whole run, so [unseen] is empty and ate_all is the row (12.2)
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

    # here, not in PARAMETERS: deriving reads a frame, and this runs after chdir
    global LORA, END2END
    LORA, stream_hw = resolve_lora(LORA, COLORS, STREAM_RES)
    END2END = replace(END2END, lora=LORA)   # the vggt_base arm and the online prior both read it

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
