"""N1, the PLACEBO experiment: is E3's ATE gain the slope constraint, or just the perturbation?

    python scripts/n1_placebo_pipeline.py    # from the repo root, adaslam venv active

A copy of init_adapt_pipeline.py that differs ONLY in carrying the N1 arm matrix below. It is a
separate file rather than an edit of that one because the config literals here are re-executed in
every spawned reader child, so editing the driver of a RUNNING arm would reach into that run.

WHY THIS EXPERIMENT EXISTS. coupling_lambda in [0.3, 3] is worth 3-4 m of ATE, but two results say
the stated mechanism is not what pays for it:

  * batch_size = window_size applies the derived gradient EXACTLY, and is worse than the fragmented
    batches that apply it wrong (23.7-28.3 vs 22.7-24.0);
  * CV_depth, the quantity the term minimises, correlates +0.02 with ATE over 17 p10 arms, and the
    best-consistency adapter ever trained is the worst arm.

Both follow if the gain comes from the SIZE of the per-sample pull rather than from which keyframe
it lands on. coupling_shuffle reassigns the fitted coefficients to keyframes at random - same
magnitudes, same zero sum, same lambda, no connection to depth - so a shuffled arm and its
unshuffled twin differ in exactly the thing the term claims to be doing.

RESULT SO FAR (both placebo arms done): 25.035 and 24.978, against lambda=0's 26.958 and lambda=1's
22.772/24.791. The placebo leaves b at +0.87/+0.81 - the UNADAPTED BASE MODEL's value, above
lambda=0's +0.622 - so it does not reduce the slope at all, and gains 1.95 m anyway. Whatever the
term is buying, it is not the slope constraint.

It is not the perturbation's size either: across every arm the pull spans 39x inside a 2.2 m ATE
band, and the two placebo seeds differ 5x in magnitude and 0.06 m in ATE. What lambda=0 uniquely
lacks is ANY gradient on the per-frame scale channel - its d/d(log s_i) is identically zero.

Caveat on the placebo as a control: mean|l_coup| came out 1.37 and 7.45 against the unshuffled arms'
0.19 and 0.26, so it is NOT magnitude-matched. The real term self-limits (coef scales with b, and
the term reduces b); the placebo never reduces b, so the brake is gone and the per-frame scales
random-walk. Read its 1.2 m shortfall against lambda=1 with that in mind.

Set ARM below, run, repeat. RefeWrence arms at this configuration (alpha=16, e3/w10/bs2, lr1e-4):
lambda=0 seed 0 -> 26.958;  lambda=1 seed 0 -> 22.772 and 24.791 (a replicate pair).

  1 adapt    LoRA-adapt VGGT on the extract's depth, from stock VGGT-1B
  2 end2end  one full-sequence arm per generator in END2END_PRIORS, then ATE side by side
  3 prior    the same generators vs GT depth directly, no SLAM run

STAGES omits 'extract' on purpose: dense_kf_p10 is shared with init_adapt_pipeline.py and is
already on disk, so every arm here reuses it and only stages 1-3 cost anything.

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
# TUM: SCENE 'rgbd_dataset_freiburg1_room', tum_config.yaml, scale 6553.5, PRIOR eval 0.1-10 m
SCENE   = 'rellis_00000'               # names the outputs/ tree
DATA    = 'data/RELLIS/00000'          # preprocess_rellis3d.py's output layout
COLORS  = f'{DATA}/colors'
DEPTHS  = f'{DATA}/depths'             # None if the dataset has no GT depth
GT_TRAJ = f'{DATA}/traj_tum.txt'
CALIB   = f'{DATA}/calib.txt'
CONFIG  = 'config/rellis_config.yaml'
DROID_WEIGHTS = 'pretrained_models/droid.pth'

# undistort offline in the preprocess script instead, or predictions and GT misalign (10.1)
UNDISTORT   = False
CROP_BORDER = 0

# ---------------------------------------------------------------- run control
STAGES           = ('adapt', 'end2end', 'prior')          # any subset; run in pipeline order
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
EXTRACT_NAME = 'dense_kf_p10'

# ---------------------------------------------------------------- THE N1 ARM MATRIX
# One line to change per run. Everything else below is identical across the five arms, so the only
# thing separating a placebo from its control is what this table says.
#
# S0 was run from init_adapt_pipeline.py (same settings, same name) - it is listed here so the
# matrix is complete and so a re-run lands in the same directory rather than a second one.
#
#   arm  lambda  shuffle  seed   status                     what it contributes
#   S0     1      True      0    DONE, ATE 25.035           placebo draw 1
#   S1     1      True      1    DONE, ATE 24.978           placebo draw 2
#   S2     1      True      2    optional                   placebo draw 3
#   C1     0      -         1    >>> RUN THIS <<<           lambda=0 draw 2
#   C2     0      -         2    >>> THEN THIS <<<          lambda=0 draw 3
#
# C1/C2 ARE NOW THE BLOCKING ONES, not the placebos. lambda=0 has exactly ONE draw on disk (seed 0,
# ATE 26.958), and every headline number is measured against it: the placebo's +1.95 m, lambda=1's
# +3.2 m, "the term helps" at all. The placebo pair just showed that run-to-run spread is not a
# single number - 0.057 m between two DIFFERENT-seed placebo draws, against 2.019 m between two
# SAME-seed unshuffled ones - so a lone baseline draw cannot be assumed representative. If lambda=0
# lands near 25 at seeds 1-2, most of the experiment evaporates and no further arm is worth running.
#
# `seed` reaches BOTH the LoRA init (torch.manual_seed, adapt/model.py:34) and the data order
# (adapt/trainer.py:127), so these are genuinely independent draws rather than reruns.
ARM = 'C2'

_ARMS = {                          # arm: (adapt name, coupling_lambda, coupling_shuffle, seed)
    'S0': ('wonline_a16_e3_w10_l1_shuf_s0_p10', 1.0, True,  0),
    'S1': ('wonline_a16_e3_w10_l1_shuf_s1_p10', 1.0, True,  1),
    'S2': ('wonline_a16_e3_w10_l1_shuf_s2_p10', 1.0, True,  2),
    'C1': ('wonline_a16_e3_w10_l0_s1_p10',      0.0, False, 1),
    'C2': ('wonline_a16_e3_w10_l0_s2_p10',      0.0, False, 2),
}
if ARM not in _ARMS:
    raise SystemExit(f'ARM={ARM!r} is not one of {sorted(_ARMS)}')
ADAPT_NAME, ARM_LAMBDA, ARM_SHUFFLE, ARM_SEED = _ARMS[ARM]

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
    adapt_style='wonline',     # 'normal' epochs | 'online' per arrival | 'wonline' sliding window
    epochs=3, batch_size=2,
    window_size=10,            # 'wonline' only
    lr=1.0e-4, weight_decay=0.0, grad_clip=1.0, lambda_pose=1.0,
    coupled_scale=True, min_mask_pixels=16, seed=ARM_SEED, log_every=20,
    # ---- which exported keyframes are trained on, and what the rest are for ----
    kf_fraction=1.0,           # 1.0 = every exported keyframe; < 1 = equidistant sample of them
    val_source='tail',         # 'tail' = the selection's tail | 'rest' = the keyframes it skipped
    train_frac=1.0,            # 'tail' ONLY; 1.0 = train on every keyframe, no val set
    eval_on_val=True,          # depth L1 on held-out keyframes, base vs adapted
    eval_on_train=True,        # also on the train subset, so the train/val gap is visible
    eval_every_epoch=False,    # False = only before training and after the last unit
    eval_max_kf=100,           # subsample each eval subset to at most this many; 0 = no cap
    keep_best=False,           # False = save the last epoch; True = snapshot on val improvement
    checkpoint_every=0,        # 0 = off; N = a loadable adapter dir in ADAPT_CKPT every N epochs
    # ---- E3: penalise the depth->scale coupling (the objective change) ----
    # depth_loss aligns scale per sample, so L(c*p) = L(p) exactly and the per-frame output scale
    # has ZERO gradient. This adds lambda * b^2, where b is the slope of log(s_i) on log(median
    # depth) fitted over the window, to put a gradient on its depth-coupled part.
    #
    # 0.0 IS THE OLD LOSS, bit for bit: the statistics pass is skipped entirely.
    # Requires adapt_style='wonline' (the slope needs a window; batch_size 2 cannot carry a fit).
    #
    # MEASURED, and it does NOT mean what it was built to mean. lambda in [0.3, 3] is worth 3-4 m
    # of ATE, but: batch_size = window_size, which applies the derived gradient exactly, is WORSE
    # than the fragmented batches that apply it wrong (23.7-28.3 vs 22.7-24.0); and CV_depth, the
    # quantity the term minimises, correlates +0.02 with ATE over 17 p10 arms. The gain looks like
    # the SIZE of the per-sample pull, not its direction - which is what coupling_shuffle tests.
    coupling_lambda=ARM_LAMBDA,
    coupling_axis='target',    # 'target' = median TARGET depth. Prefer it: the network does not
                               # control the axis. 'pred' lets it satisfy the penalty by
                               # collapsing every predicted median to one value = range collapse.
    coupling_min_var=1e-4,     # skip the term when the window has no depth spread (sum x~^2 below
                               # this), where b would be an ill-conditioned division
    # ---- the placebo control (N1) ----
    # True reassigns the fitted coefficients to the window's keyframes at RANDOM: same magnitudes,
    # same zero sum, same lambda, no connection to depth. An arm with this on is a CONTROL, and its
    # comparison is the unshuffled arm at the same lambda and seed, not the lambda=0 baseline.
    #   placebo ~= lambda=1 (23.8)  -> the depth pairing is incidental; the gain is the perturbation
    #   placebo ~= lambda=0 (27.0)  -> the slope constraint really is what pays
    # Run at alpha=16, where the lambda effect (3.2 m) clears the ~2 m run-to-run floor; at alpha=8
    # it is only ~1.1 m and a placebo could not be read against it.
    coupling_shuffle=ARM_SHUFFLE)

# ---------------------------------------------------------------- end2end test (stage 3)


# another adapt run's handoff directory; both prior lists below use it
def _a(name):
    return experiment_dir(OUT_ROOT, 'adapt', SCENE_KEY, name)

# one entry per generator, arm directory INFERRED (7.1); comments are unit, frame, share of sequence
END2END_PRIORS = (
    'omnidata', 'vggt_base',     # priors[0] is the baseline column
    ADAPT_OUT,                   # this run's final adapter, frozen
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
PRIOR_PRIORS = END2END_PRIORS

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
