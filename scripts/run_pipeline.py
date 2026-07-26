"""The extract -> adapt -> test pipeline for the VGGT depth-prior experiment.

    python scripts/run_pipeline.py          # from the repo root, with the adaslam venv active

Three stages, run in ONE process, each skipped if its output already exists:

  1 extract  HI-SLAM2 on the first FRACTION% of the sequence, dumping its own post-global-BA
             depth, then exported to per-keyframe depth/mask/image + the accuracy table
  2 adapt    LoRA-adapt VGGT on that depth, on a TRAIN subset of the keyframes, reporting
             depth L1 on a held-out VAL subset
  3 test     two (or three) full-sequence arms differing ONLY in the depth prior, then a
             side-by-side comparison split at the frame the adapter's training data ended

This file is the KNOB PANEL, not the implementation. Every stage lives in a package under
adaslam/ - slam/ (the single interface to Hi2), extract/, adapt/, abtest/ - and nothing in any of
them carries a hyperparameter default of its own. That is the point: a value is written down here,
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

from adaslam.abtest import TestConfig, run_ab_test
from adaslam.adapt import AdaptConfig, LoRAConfig, LoRAVGGT
from adaslam.common import probe_stream_hw
from adaslam.extract import ExtractConfig, run_extract
from adaslam.runtime import banner, ensure_venv_on_path, free_vram, gpu_gate, raise_fd_limit
from adaslam.slam import SlamConfig, SlamRunner

# ==============================================================================
#  PARAMETERS
# ==============================================================================

# ---------------------------------------------------------------- data (preprocessing is NOT here)
SCENE   = 'rgbd_dataset_freiburg1_desk2'
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
STAGES           = ('extract', 'adapt', 'test')
SKIP_EXISTING    = True                # reuse a stage's output if it is already on disk
MIN_FREE_VRAM_MB = 10000               # shared GPU: checked once at the start of main()
FRACTION         = 100                 # % of the sequence the adapter trains on; also SPLIT_AT
START            = 0
OUT_EXTRACT      = f'outputs/tum/{SCENE}_p{FRACTION}'
OUT_TEST         = f'outputs/tum_ab_p{FRACTION}'
STREAM_RES       = 341 * 640           # tracking resolution budget
DEPTH_PNG_SCALE  = 6553.5              # 16-bit depth PNG scale used across the repo
DEPTH_SOURCE     = 'slam'  # 'rendered' (Gaussian expected depth) | 'slam' (1/disps_up)

# ---------------------------------------------------------------- stage I/O
# A stage RECEIVES its paths and reads no path global. That is what lets one be pointed at another
# run's results - adapt on one extract's export, write the adapter somewhere else - without moving
# OUT_EXTRACT, which the other stages also key off. The defaults below reproduce the layout §7
# documents, so an unedited run is unchanged. Only `adapt` is wired this way so far; extract and
# test still derive their paths from OUT_EXTRACT / OUT_TEST.
ADAPT_IN     = OUT_EXTRACT                              # extract export to train on: depth_<src>/
                                                        # mask_<src>/ poses_slam.txt traj_full.txt
                                                        # intrinsics.npy
ADAPT_IMAGES = COLORS                                   # keyframe RGB, indexed by frame number
ADAPT_OUT    = f'{OUT_EXTRACT}/lora-vggt'               # adapter.safetensors config.json train_log
ADAPT_CKPT   = f'{OUT_EXTRACT}/lora-vggt-checkpoints'   # epoch_NNN/; ADAPT.checkpoint_every says
                                                        # how often, and 0 there turns them off
                                                        # (a None here with a cadence set is an
                                                        # error, not a way to disable them)

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
    kf_motion_thresh=1.5,
    kf_init_thresh=4.0,             # the same gate before initialisation
    kf_redundant_thresh=3.0,        # the one that actually moves the keyframe count
    kf_covis_thresh=0.05,            # extra keyframes inserted in terminate(); LOWER -> more
    buffer=500,                     # hard cap; MUST exceed the count (no overflow guard exists)
                                    # any of the four thresholds may be None = inherit CONFIG
    depth_source=DEPTH_SOURCE, depth_png_scale=DEPTH_PNG_SCALE,
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

# ---------------------------------------------------------------- test (stage 3, the A/B arms)
# `lora=LORA` is not decoration: the 'vggt_base' arm has no adapter to read a structure back from,
# so it silently takes these values - above all vggt_hw, which must match the adapted arm's or the
# two VGGT arms differ in input resolution as well as in adaptation (§9.3).
TEST = TestConfig(
    arms=('omnidata', 'vggt_lora'),   # 'vggt_base' = stock VGGT-1B, the §10.2 third arm
    out_root=OUT_TEST, scene=SCENE,
    length=100000,                    # 100000 = whole sequence
    buffer=500,
    gt_traj=GT_TRAJ, gt_mesh=GT_MESH, gt_depths=DEPTHS, depth_png_scale=DEPTH_PNG_SCALE,
    voxel_size=0.01,                  # pinned for ALL arms (§9.3)
    voxel_fallbacks=(0.01, 0.02),     # marching cubes OOMs on a busy shared GPU
    mesh_weight=2.0,
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

def stage_extract(runner, extract_length):
    banner(f'1/3 extract  -> {OUT_EXTRACT}')
    run_extract(runner, EXTRACT, OUT_EXTRACT, extract_length, CONFIG,
                skip_existing=SKIP_EXISTING)


def stage_adapt(in_dir, image_dir, out_dir, ckpt_dir):
    """LoRA-adapt VGGT on `in_dir`'s export of `image_dir`, writing the adapter into `out_dir`.

    Takes its four paths as arguments and reads no path global, so it can be pointed at any
    earlier extract run. LORA / ADAPT / SKIP_EXISTING still arrive as globals: they are configs,
    not paths. Returns the adapter path.
    """
    adapter = f'{out_dir}/adapter.safetensors'
    banner(f'2/3 adapt  {in_dir} -> {adapter}')
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


def stage_test(runner, adapter, split_at):
    banner(f'3/3 test  -> {OUT_TEST}')

    # The arms must run stock tracking. The EXTRACT kf_* knobs shape the training-data run only:
    # if the generated config leaked in here, a denser-keyframe extract would silently also mean
    # denser keyframes in the A/B, and neither arm would be comparable with any earlier run.
    # Asserted here rather than inside abtest/, because this is where both paths are visible.
    arm_config = CONFIG
    assert os.path.abspath(arm_config) != os.path.abspath(f'{OUT_EXTRACT}/extract_config.yaml'), \
        'the A/B arms must use the base CONFIG, not the extract run derived config'
    print(f'tracking config for every arm: {arm_config} (unmodified; the EXTRACT kf_* knobs '
          f'apply to the extract run only)')

    run_ab_test(runner, SLAM, TEST, arm_config, adapter, split_at, skip_existing=SKIP_EXISTING)


# ==============================================================================

def main():
    # must happen before any Process is started, and only once per process
    torch.multiprocessing.set_start_method('spawn', force=True)
    os.chdir(_ROOT)                       # every relative path above is repo-root relative

    needed = [COLORS, CALIB, CONFIG, DROID_WEIGHTS]
    if 'test' in STAGES:
        needed += [GT_TRAJ] + ([GT_MESH] if GT_MESH else [])   # run_ate / run_mesh need these
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
    adapter = f'{ADAPT_OUT}/adapter.safetensors'      # stage_test's input, wherever adapt put it

    # Resolve vggt_hw HERE, not in the PARAMETERS block: deriving reads a frame, and that block is
    # re-executed by every spawned reader child (which never touches LORA - it only needs SLAM).
    # After chdir, so the relative COLORS resolves however the script was invoked.
    global LORA, TEST
    stream_hw = probe_stream_hw(COLORS, STREAM_RES)
    LORA = LORA.resolved(stream_hw)
    TEST = replace(TEST, lora=LORA)          # the vggt_base arm reads its size off this

    print(f'sequence  : {SCENE}  ({n_frames} frames, {COLORS})')
    print(f'config    : {CONFIG}  calib {CALIB}')
    print(f'stream    : {stream_hw[1]}x{stream_hw[0]} (aspect '
          f'{stream_hw[1]/stream_hw[0]:.3f})   VGGT input: {LORA.vggt_hw[1]}x{LORA.vggt_hw[0]}'
          f'{" (derived)" if VGGT_HW is None else " (pinned by VGGT_HW)"}')
    print(f'adapter   : trains on frames 0..{extract_length-1} ({FRACTION}%), '
          f'evaluated on 0..{n_frames-1}')
    print(f'target    : depth_{DEPTH_SOURCE}/   split at frame {split_at}')
    print(f'stages    : {" ".join(STAGES)}   arms: {" ".join(TEST.arms)}')

    # one VRAM check up front, before any GPU work or spawned Process; the stages no longer re-gate
    gpu_gate(MIN_FREE_VRAM_MB)

    # ONE runner for every HI-SLAM2 invocation in the pipeline: the extract run and both arms
    runner = SlamRunner(SLAM)

    t_all = time.time()
    if 'extract' in STAGES:
        stage_extract(runner, extract_length)
    if 'adapt' in STAGES:
        adapter = stage_adapt(ADAPT_IN, ADAPT_IMAGES, ADAPT_OUT, ADAPT_CKPT)
    if 'test' in STAGES:
        stage_test(runner, adapter, split_at)

    print(f'\nall stages done in {time.time()-t_all:.0f}s')
    print('\nread first:')
    print(f'  {OUT_EXTRACT}/export.txt   per-frame vs global depth L1 columns. The gap on the')
    print('                             Omnidata row is the cross-frame scale inconsistency this')
    print('                             track targets - if it is small, there was no headroom.')
    print("  the table above            'unseen' rows only; 'seen' is the adapter's training")


if __name__ == '__main__':
    main()
