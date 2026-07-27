"""The extract -> adapt -> test pipeline for the VGGT depth-prior experiment.

    python scripts/run_pipeline.py          # from the repo root, with the adaslam venv active

Four stages, run in ONE process, each skipped if its output already exists:

  1 extract  HI-SLAM2 on the first FRACTION% of the sequence, dumping its own post-global-BA
             depth, then exported to per-keyframe depth/mask/image + the accuracy table
  2 adapt    LoRA-adapt VGGT on that depth, on a TRAIN subset of the keyframes, reporting
             depth L1 on a held-out VAL subset
  3 end2end  one full-sequence arm per depth-prior generator in END2END_PRIORS - omnidata, stock
             VGGT, and any number of adapters or their checkpoints - then a side-by-side
             comparison split at the frame the adapter's training data ended
  4 prior    the same generators scored against GT depth DIRECTLY, no SLAM run: minutes an arm
             rather than forty, and it is what says whether an end2end null means "no better
             prior" or "HI-SLAM2 cannot feel the difference" (ARCHITECTURE.md 9.2.2)

Each writes into its own tree under outputs/, keyed by stage then scene then experiment, because
the fan-out is real - a scene has several extracts, each has several adapts, each has several
tests (ARCHITECTURE.md 7.1):

  outputs/extract/<scene>/<EXTRACT_NAME>/   the handoff to adapt + full/ (the whole SLAM run)
  outputs/adapt/<scene>/<ADAPT_NAME>/       the handoff to the tests + checkpoints/
  outputs/test/end2end/<scene>/<arm>/       one directory per prior generator, name INFERRED
  outputs/test/prior/<scene>/<arm>/         the same generators, scored without a SLAM run

The two stage names are yours and are required; the arm directories are inferred from the adapter
each uses, which is what makes an arm reusable - this scene's omnidata baseline is run once and
every later comparison finds it instead of repeating it.

This file is the KNOB PANEL, not the implementation. Every stage lives in a package under
adaslam/ - slam/ (the single interface to Hi2), extract/, adapt/, end2end/ - and nothing in
any of them carries a hyperparameter default of its own. That is the point: a value is written here,
once, and travels into exactly the configs that need it. There is no command line and no
environment. Dataset preprocessing is deliberately NOT here; run scripts/preprocess_tum.py first.

The five config literals below are rebuilt in every spawned child (torch.multiprocessing's 'spawn'
re-executes this module for the image reader), so none of them may do file access or computation -
keep them primitives.
"""
import os    # nopep8
import sys   # nopep8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # nopep8
# the repo root, so `adaslam` is importable; adaslam/__init__.py adds hislam2/ and thirdparty/vggt
sys.path.insert(0, _ROOT)                                                 # nopep8
import time
from dataclasses import replace

import numpy as np
import torch

from adaslam.adapt import AdaptConfig, LoRAConfig, LoRAVGGT
from adaslam.common import (ADAPT_CKPT_SUBDIR, experiment_dir, extract_run_dir, probe_stream_hw,
                            require_name, test_dir)
from adaslam.end2end import End2EndConfig, run_end2end_test
from adaslam.extract import ExtractConfig, run_extract
from adaslam.priortest import PriorTestConfig, run_prior_test
from adaslam.print_utils import banner
from adaslam.runtime import ensure_venv_on_path, free_vram, gpu_gate, raise_fd_limit
from adaslam.slam import SlamConfig, SlamRunner

# ==============================================================================
#  PARAMETERS
# ==============================================================================

# ---------------------------------------------------------------- data (preprocessing is NOT here)
SCENE   = 'rgbd_dataset_freiburg1_room'
DATA    = f'data/TUM/{SCENE}'          # preprocess_tum.py's output layout
COLORS  = f'{DATA}/colors'
DEPTHS  = f'{DATA}/depths'             # None if the dataset has no GT depth
GT_TRAJ = f'{DATA}/traj_tum.txt'
CALIB   = f'{DATA}/calib.txt'
GT_MESH = None                         # None -> skip TSDF + eval_recon (TUM ships no GT mesh)
CONFIG  = 'config/tum_config.yaml'
DROID_WEIGHTS = 'pretrained_models/droid.pth'

# Undistortion/cropping normally happens offline in preprocess_tum.py. Doing it here instead
# would make split_render_metrics() compare undistorted renders against distorted GT, because
# it re-derives the GT frame with stream_resize() only (ARCHITECTURE.md §10.1).
UNDISTORT   = False
CROP_BORDER = 0

# ---------------------------------------------------------------- run control
STAGES           = ('extract', 'adapt', 'prior', 'end2end')   # any subset; run in pipeline order
SKIP_EXISTING    = True                # reuse a stage's output if it is already on disk
MIN_FREE_VRAM_MB = 10000               # shared GPU: checked once at the start of main()
FRACTION         = 40                  # % of the sequence the adapter trains on; also SPLIT_AT
START            = 0
STREAM_RES       = 341 * 640           # tracking resolution budget
DEPTH_PNG_SCALE  = 6553.5              # 16-bit depth PNG scale used across the repo
DEPTH_SOURCE     = 'slam'  # which exported target SUPERVISES the adaptation:
                           # 'rendered' (Gaussian expected depth) | 'slam' (1/disps_up).
                           # Both are always exported - see EXTRACT.depth_sources - so changing
                           # this costs an adapt run, not another SLAM run.

# ---------------------------------------------------------------- experiment names
# outputs/ is one directory per STAGE, then one per SCENE, then one per EXPERIMENT, because the
# fan-out is real: a scene has several extracts, each has several adapts, each has several tests.
#
#   outputs/extract/<SCENE>/<EXTRACT_NAME>/
#   outputs/adapt/<SCENE>/<ADAPT_NAME>/
#   outputs/test/end2end/<SCENE>/<arm>/     <arm> is INFERRED - see END2END_PRIORS below
#
# Both names are REQUIRED (main() checks). The scene is a directory of its own, so a name only has
# to be unique within its scene: it need not carry the scene, nor chain the run before it. Lineage
# is recorded as DATA instead - an adapter's config.json holds the extract directory it trained on
# (adapt/trainer.py:113). FRACTION is not in the name either; put it there yourself if you vary it.
OUT_ROOT     = 'outputs'
EXTRACT_NAME = 'dense_kf_p40'
ADAPT_NAME   = 'r8_e10_depth'

# ---------------------------------------------------------------- stage I/O
# A stage RECEIVES its paths and reads no path global. That is what lets one be pointed at another
# run's results - adapt on one extract's export, write the adapter somewhere else - without moving
# the others. These are pure string joins: this block is re-executed in every spawned reader child
# and must not touch the filesystem.
OUT_EXTRACT  = experiment_dir(OUT_ROOT, 'extract', SCENE, EXTRACT_NAME)

ADAPT_IN     = OUT_EXTRACT                              # extract export to train on: depth_<src>/
                                                        # mask_<src>/ poses_slam.txt traj_full.txt
                                                        # intrinsics.npy
ADAPT_IMAGES = COLORS                                   # keyframe RGB, indexed by frame number
ADAPT_OUT    = experiment_dir(OUT_ROOT, 'adapt', SCENE, ADAPT_NAME)
                                                        # adapter.safetensors config.json train_log
ADAPT_CKPT   = f'{ADAPT_OUT}/{ADAPT_CKPT_SUBDIR}'       # epoch_NNN/; ADAPT.checkpoint_every says
                                                        # how often, and 0 there turns them off
                                                        # (a None here with a cadence set is an
                                                        # error, not a way to disable them)

OUT_END2END  = test_dir(OUT_ROOT, 'end2end', SCENE)     # one subdirectory per prior generator
OUT_PRIOR    = test_dir(OUT_ROOT, 'prior', SCENE)       # same arm names, scored without SLAM

# VGGT's input size. None = derive it in main() from the tracking stream's aspect ratio, which is
# what you want: nothing letterboxes anywhere, so a value that does not match the stream squashes
# the image off VGGT's training distribution, and the correct value is a pure function of the
# stream (adapt/config.py:vggt_hw_for). State a value instead to pin it - an adapter's own
# recorded size always wins over both (§9.3). It is NOT derived here: this block is re-executed in
# every spawned reader child and must not touch the filesystem.
VGGT_HW          = None                # or e.g. (378, 518) for TUM, (294, 518) for Replica

# ---------------------------------------------------------------- the SLAM runs
# What is identical across all three invocations - the extract run and both arms. Everything that
# differs between them (tracking YAML, output dir, length, buffer, gtdepthdir, dump_slam_depth,
# depth prior) is an argument to SlamRunner.run(), so the differences stay visible at the call
# site instead of hiding in an object. One runner is built in main() and handed to both stages.
SLAM = SlamConfig(
    weights=DROID_WEIGHTS, colors=COLORS, calib=CALIB, start=START,
    undistort=UNDISTORT, crop_border=CROP_BORDER, stream_res=STREAM_RES)

# ---------------------------------------------------------------- extract (stage 1)
# The four kf_* knobs are EXTRACT-ONLY, and that is the whole point: they go into a generated
# extract_config.yaml that ONLY the extract run is given. Every A/B arm is handed the unmodified
# CONFIG above, so denser training data can never be mistaken for a tracking change in the
# comparison - and the arms stay comparable with runs made before these knobs existed. main()
# asserts this rather than trusting it. See ExtractConfig's docstring for which gate binds.
EXTRACT = ExtractConfig(
    kf_motion_thresh=1.2,
    kf_init_thresh=4.0,             # the same gate before initialisation
    kf_redundant_thresh=2.0,        # the one that actually moves the keyframe count
    kf_covis_thresh=0.1,            # extra keyframes inserted in terminate(); LOWER -> more
    buffer=500,                     # hard cap; MUST exceed the count (no overflow guard exists)
                                    # any of the four thresholds may be None = inherit CONFIG
    # every source is exported, because all of them are handoff artifacts to adapt: choosing one
    # here would mean another 40-minute SLAM run the day you want the other. ADAPT.depth_source
    # picks which of them supervises.
    depth_sources=('slam', 'rendered'),
    depth_png_scale=DEPTH_PNG_SCALE,
    mask_filter_thresh=0.005,       # depth_filter disparity agreement
    mask_min_count=2,               # min agreeing neighbours out of 6
    mask_min_disp_ratio=0.5,        # drop pixels below this fraction of the frame's mean disparity
    gt_depths=DEPTHS)               # the accuracy table ONLY - never reaches Hi2 (§9.3)

# ---------------------------------------------------------------- adapt (stage 2, LoRA on VGGT)
# LORA is the model/adapter STRUCTURE - it is what gets recorded into the adapter's config.json
# and read back by LoRAVGGT.from_adapter, so an arm always runs the model its adapter was trained
# in even if these values move afterwards. ADAPT is the training run, which no adapter re-reads.
LORA = LoRAConfig(
    weights='pretrained_models/vggt',
    vggt_hw=VGGT_HW,           # None -> derived in main(); see the constant above
    rank=8, alpha=8,
    targets=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'),
    patch_embed=False)         # False = adapt only the alternating-attention stack

ADAPT = AdaptConfig(
    depth_source=DEPTH_SOURCE, stream_res=STREAM_RES,
    p_single_view=1, max_left=4, max_right=4, radius=8,
    epochs=10, batch_size=2,
    lr=1e-4, weight_decay=0.0, grad_clip=1.0, lambda_pose=1.0,
    depth_space='depth',   # 'depth' | 'disparity'
    coupled_scale=True, min_mask_pixels=16, seed=0, log_every=20,
    # ---- train / val split over the exported keyframes ----
    train_frac=0.8,            # 1.0 = train on every keyframe, no val set
    split_mode='stride',       # 'stride' (every Nth held out) | 'contiguous' (tail) | 'random'
    eval_on_val=True,          # depth L1 on held-out keyframes, base vs adapted
    eval_on_train=True,        # also on the train subset, so the train/val gap is visible
    eval_every_epoch=True,     # False = only before training and after the last epoch
    eval_max_kf=100,           # evenly subsample each eval subset to at most this many; 0 = no cap
    keep_best=False,           # False = save the last epoch (report-only, the default);
                               # True  = snapshot whenever val L1 improves and save that instead
    checkpoint_every=5)        # 0 = off; N = a full loadable adapter dir in ADAPT_CKPT/epoch_NNN
                               # every N epochs, on top of the one final save

# ---------------------------------------------------------------- end2end test (stage 3)
# One entry per DEPTH-PRIOR GENERATOR to compare. Each is either a sentinel or the handoff
# directory of an adapt run (or one of its checkpoints), and each scores into a directory whose
# name is INFERRED from it, never typed:
#
#   'omnidata'                            -> outputs/test/end2end/<SCENE>/omni
#   'vggt_base'                           -> .../base                (stock VGGT-1B, §10.2's arm)
#   ADAPT_OUT                             -> .../<ADAPT_NAME>
#   f'{ADAPT_CKPT}/epoch_005'             -> .../<ADAPT_NAME>_chkp_005
#
# That is what makes an arm reusable: this scene's omnidata baseline is run once and every later
# comparison finds it. priors[0] is the baseline column of the comparison table. Nothing stops you
# listing an adapter from another scene - only the inferred names have to stay distinct, which
# End2EndConfig checks.
#
# `lora=LORA` is not decoration: the 'vggt_base' arm has no adapter to read a structure back from,
# so it silently takes these values - above all vggt_hw, which must match the adapted arms' or they
# differ in input resolution as well as in adaptation (§9.3).
END2END_PRIORS = ('omnidata', 'vggt_base', ADAPT_OUT)

END2END = End2EndConfig(
    priors=END2END_PRIORS,
    length=100000,                    # 100000 = whole sequence
    buffer=500,
    gt_traj=GT_TRAJ, gt_mesh=GT_MESH, gt_depths=DEPTHS, depth_png_scale=DEPTH_PNG_SCALE,
    voxel_size=0.01,                  # pinned for ALL arms (§9.3)
    voxel_fallbacks=(0.01, 0.02),     # marching cubes OOMs on a busy shared GPU
    mesh_weight=2.0,
    lora=LORA,
    omni_normal_ckpt='pretrained_models/omnidata_dpt_normal_v2.ckpt',
    omni_normal_hw=(512, 512))

# ---------------------------------------------------------------- prior test (stage 4)
# The same generators, scored against GT depth directly, with NO SLAM run - minutes per arm rather
# than forty. It exists to attribute an end2end null (§9.4): "swapping the prior changed nothing"
# is either "the new prior is no better" or "HI-SLAM2 is insensitive to the way it is better", and
# only a measurement of the priors themselves tells those apart.
#
# The seen/unseen boundary comes from the FIRST adapter in the list and is applied to every arm,
# sentinels included - omni's and base's unseen rows are the control for "is the back of the
# sequence simply harder?", without which an adapter's unseen number cannot be read.
PRIOR_PRIORS = END2END_PRIORS          # same arms, so the two tests' directories line up

PRIOR = PriorTestConfig(
    priors=PRIOR_PRIORS,
    gt_depths=DEPTHS, depth_png_scale=DEPTH_PNG_SCALE,
    eval_min_depth=0.1,                # metres; below this TUM's Kinect reports noise, not geometry
    eval_max_depth=10.0,               # metres; the NYU/KITTI convention, and past the room anyway
    eval_samples_per_frame=20000,      # valid pixels kept per frame; 0 = all (needs the RAM)
    seed=0,
    lora=LORA,
    omni_normal_ckpt='pretrained_models/omnidata_dpt_normal_v2.ckpt',
    omni_normal_hw=(512, 512))

# ==============================================================================

# Both at module scope, not in main(): a spawned child re-executes this module and needs the same
# fd limit, and the subprocess CLIs need the venv's bin on PATH wherever they are reached from.
raise_fd_limit()
ensure_venv_on_path()


# ==============================================================================
#  stages
# ==============================================================================

def stage_extract(runner, out_dir, extract_length):
    """SLAM over the first `extract_length` frames, exported into the `out_dir` experiment.

    The run itself lands in out_dir/full and only the handoff artifacts reach the top level.
    """
    banner(f'extract  -> {out_dir}')
    run_extract(runner, EXTRACT, out_dir, extract_length, CONFIG, skip_existing=SKIP_EXISTING)


def stage_adapt(in_dir, image_dir, out_dir, ckpt_dir):
    """LoRA-adapt VGGT on `in_dir`'s export of `image_dir`, writing the adapter into `out_dir`.

    Takes its four paths as arguments and reads no path global, so it can be pointed at any
    earlier extract run. LORA / ADAPT / SKIP_EXISTING still arrive as globals: they are configs,
    not paths. Returns the adapter path.
    """
    adapter = f'{out_dir}/adapter.safetensors'
    banner(f'adapt  {in_dir} -> {adapter}')
    if SKIP_EXISTING and os.path.exists(adapter):
        print(f'{adapter} exists - skipping')
        return adapter

    # Checked here rather than left to SceneData, because in_dir and image_dir are now free to be
    # any pair and a wrong one otherwise dies deep inside the first sample.
    for f in (f'{in_dir}/poses_slam.txt', f'{in_dir}/traj_full.txt', f'{in_dir}/intrinsics.npy',
              f'{in_dir}/depth_{ADAPT.depth_source}', image_dir):
        if not os.path.exists(f):
            raise SystemExit(f'adapt input missing: {f}   (in_dir={in_dir} must be an extract '
                             f'export made with depth_source={ADAPT.depth_source!r})')
    # SceneData.frame() indexes sorted(os.listdir(image_dir)) by frame number, so a mismatched
    # pair would be an IndexError - or worse, silently the wrong image
    last_kf = int(np.loadtxt(f'{in_dir}/poses_slam.txt')[:, 0].max())
    n_img = len(os.listdir(image_dir))
    if last_kf >= n_img:
        raise SystemExit(f'{in_dir} has keyframe {last_kf} but {image_dir} holds {n_img} frames; '
                         f'the export and the images must be the same sequence.')

    t0 = time.time()
    # seed=ADAPT.seed must be given to the CONSTRUCTOR: the adapter's A matrices are initialised
    # when LoRA is injected, so seeding any later does not reproduce a run
    lora = LoRAVGGT(LORA, seed=ADAPT.seed)
    summary = lora.train(in_dir, image_dir, out_dir, ADAPT, ckpt_dir=ckpt_dir)
    # the one save, here rather than inside the loop - and before release(), which invalidates it
    print(f'saved adapter to {lora.save(out_dir, state=summary["state"], extra=summary["run"])}')
    lora.release()
    free_vram('adapt')
    print(f'=== adapt done in {time.time()-t0:.0f}s')
    return adapter


def stage_end2end(runner, out_root, arm_config, split_at):
    """One arm per entry in END2END.priors into `out_root`/<inferred name>, then the comparison.

    No `adapter` argument: each prior carries its own, which is what lets one comparison hold
    several adapters and their checkpoints. `arm_config` is the tracking YAML every arm gets;
    main() checks it is the base CONFIG and not the extract run's generated one, where both paths
    are visible side by side.
    """
    banner(f'end2end  -> {out_root}')
    print(f'tracking config for every arm: {arm_config} (unmodified; the EXTRACT kf_* knobs '
          f'apply to the extract run only)')
    run_end2end_test(runner, SLAM, END2END, out_root, arm_config, split_at,
                     skip_existing=SKIP_EXISTING)


def stage_prior(out_root):
    """Score every entry in PRIOR.priors against GT depth into `out_root`/<inferred name>.

    No runner and no split_at argument: there is no SLAM run, and the seen/unseen boundary is
    resolved from the priors themselves (priortest/config.py:resolve_split).
    """
    banner(f'prior test  -> {out_root}')
    run_prior_test(SLAM, PRIOR, out_root, skip_existing=SKIP_EXISTING)


# ==============================================================================

def main():
    # must happen before any Process is started, and only once per process
    torch.multiprocessing.set_start_method('spawn', force=True)
    os.chdir(_ROOT)                       # every relative path above is repo-root relative

    # names the directories every stage writes into, so a bad one must fail before any GPU work
    require_name('EXTRACT_NAME', EXTRACT_NAME)
    require_name('ADAPT_NAME', ADAPT_NAME)

    needed = [COLORS, CALIB, CONFIG, DROID_WEIGHTS]
    if 'end2end' in STAGES:
        needed += [GT_TRAJ] + ([GT_MESH] if GT_MESH else [])   # run_ate / run_mesh need these
    if 'prior' in STAGES and not DEPTHS:
        raise SystemExit('the prior stage scores against ground-truth depth; DEPTHS is None')
    for f in needed:
        if not os.path.exists(f):
            raise SystemExit(f'missing input: {f}')
    if UNDISTORT or CROP_BORDER:
        print('WARNING: undistorting at runtime - split_render_metrics re-derives the GT frame '
              'with a resize only, so renders and GT will not line up (ARCHITECTURE.md §10.1)')

    n_frames = len(os.listdir(COLORS))
    # every consumer indexes GT depth by RGB frame number, so these must be 1:1
    for name, path in (('depths', DEPTHS), ('traj', GT_TRAJ)):
        if path is None:
            continue
        n = len(os.listdir(path)) if os.path.isdir(path) else len(np.loadtxt(path))
        if n != n_frames:
            raise SystemExit(f'{path} has {n} entries but {COLORS} has {n_frames}; they must be '
                             f'1:1 by index. Re-run scripts/preprocess_tum.py.')

    extract_length = n_frames * FRACTION // 100
    if extract_length < 20:
        raise SystemExit(f'{n_frames} frames * {FRACTION}% = {extract_length}, too few to track')
    split_at = extract_length

    # The arms must run stock tracking. The EXTRACT kf_* knobs shape the training-data run only:
    # if the generated config leaked in here, a denser-keyframe extract would silently also mean
    # denser keyframes in the comparison, and no arm would be comparable with any earlier run.
    # Asserted here, the one place both paths are in scope.
    assert os.path.abspath(CONFIG) != \
        os.path.abspath(f'{extract_run_dir(OUT_EXTRACT)}/extract_config.yaml'), \
        'the arms must use the base CONFIG, not the extract run derived config'

    # both test trees; the arm directories inside them are created by the arms themselves
    for kind in ('end2end', 'prior'):
        os.makedirs(test_dir(OUT_ROOT, kind, SCENE), exist_ok=True)

    # Resolve vggt_hw HERE, not in the PARAMETERS block: deriving reads a frame, and that block is
    # re-executed by every spawned reader child (which never touches LORA - it only needs SLAM).
    # After chdir, so the relative COLORS resolves however the script was invoked.
    global LORA, END2END, PRIOR
    stream_hw = probe_stream_hw(COLORS, STREAM_RES)
    LORA = LORA.resolved(stream_hw)
    END2END = replace(END2END, lora=LORA)    # the vggt_base arm reads its size off this
    PRIOR = replace(PRIOR, lora=LORA)        # and so does the prior test's, for the same reason

    print(f'sequence  : {SCENE}  ({n_frames} frames, {COLORS})')
    print(f'config    : {CONFIG}  calib {CALIB}')
    print(f'stream    : {stream_hw[1]}x{stream_hw[0]} (aspect '
          f'{stream_hw[1]/stream_hw[0]:.3f})   VGGT input: {LORA.vggt_hw[1]}x{LORA.vggt_hw[0]}'
          f'{" (derived)" if VGGT_HW is None else " (pinned by VGGT_HW)"}')
    print(f'adapter   : trains on frames 0..{extract_length-1} ({FRACTION}%), '
          f'evaluated on 0..{n_frames-1}')
    print(f'target    : depth_{DEPTH_SOURCE}/   split at frame {split_at}')
    print(f'stages    : {" ".join(STAGES)}')
    print(f'outputs   : {OUT_EXTRACT}')
    print(f'            {ADAPT_OUT}')
    # the arm directories are inferred, so print them: this is the only place to see, before a
    # two-hour run, which prior generator lands where and that none of them collided
    for kind, cfg_, root in (('end2end', END2END, OUT_END2END), ('prior', PRIOR, OUT_PRIOR)):
        if kind in STAGES:
            for spec, d in cfg_.arm_dirs(root).items():
                print(f'            {d:<58} <- {spec}')

    # one VRAM check up front, before any GPU work or spawned Process; the stages no longer re-gate
    gpu_gate(MIN_FREE_VRAM_MB)

    # ONE runner for every HI-SLAM2 invocation in the pipeline: the extract run and both arms
    runner = SlamRunner(SLAM)

    t_all = time.time()
    if 'extract' in STAGES:
        stage_extract(runner, OUT_EXTRACT, extract_length)
    if 'adapt' in STAGES:
        stage_adapt(ADAPT_IN, ADAPT_IMAGES, ADAPT_OUT, ADAPT_CKPT)
    if 'end2end' in STAGES:
        stage_end2end(runner, OUT_END2END, CONFIG, split_at)
    if 'prior' in STAGES:
        stage_prior(OUT_PRIOR)

    print(f'\nall stages done in {time.time()-t_all:.0f}s')
    print('\nread first:')
    print(f'  {OUT_EXTRACT}/export.txt   per-frame vs global depth L1 columns. The gap on the')
    print('                             Omnidata row is the cross-frame scale inconsistency this')
    print('                             track targets - if it is small, there was no headroom.')
    print("  the table above            'unseen' rows only; 'seen' is the adapter's training")


if __name__ == '__main__':
    main()
