"""Helpers shared by more than one pipeline stage.

Deliberately outside any of them: extract, adapt and end2end all need these, and this is the
neutral ground they can import from without depending on each other.
"""
import os

import cv2
import numpy as np

# Which of the extract stage's export targets supervises the adaptation. Lives here, not in a
# stage: extract writes depth_<src>/ and adapt reads it, so one tuple has to bound both.
DEPTH_SOURCES = ('slam', 'rendered')

# ---------------------------------------------------------------- the outputs/ layout
# outputs/ is one directory per STAGE, then one per SCENE, then one per EXPERIMENT, because the
# fan-out is real: one scene -> several extracts -> several adapts each -> several tests each. An
# experiment directory holds what the NEXT stage consumes and nothing else; the raw HI-SLAM2 run
# goes in a subdirectory, so it can be deleted to reclaim space without breaking the stage after it.
#
#   outputs/extract/<scene>/<exp>/ {depth_<src>/ mask_<src>/ image/ poses_slam.txt traj_full.txt
#                                   intrinsics.npy export.txt} + full/<the whole SLAM run>
#   outputs/adapt/<scene>/<exp>/   {adapter.safetensors config.json train_log.json} + checkpoints/
#   outputs/test/end2end/<scene>/<arm>/  one depth-prior generator's run; <arm> is INFERRED from
#                                        the adapter it uses (end2end/config.py:arm_name)
#   outputs/test/prior/<scene>/<arm>/    the same generators scored against GT with no SLAM run
#
# The scene level is what lets an experiment name be short: it only has to be unique within its
# scene, so nothing has to chain the scene into the name. Lineage is recorded as DATA instead - an
# adapter's config.json carries the extract directory it trained on (adapt/trainer.py:113).
#
# These names are here rather than in a stage package because run_pipeline.py needs them too - it
# asserts the arms are not handed the extract run's generated config, which lives inside full/.
EXTRACT_RUN_SUBDIR = 'full'
ADAPT_CKPT_SUBDIR = 'checkpoints'
TEST_KINDS = ('end2end', 'prior')
# Written by the SLAM run into full/, needed by adapt at the experiment's top level. COPIED up
# rather than moved: full/ has to stay a complete run for the split to mean anything.
HANDOFF_UP = ('traj_full.txt', 'intrinsics.npy')


def extract_run_dir(exp_dir):
    """The untouched HI-SLAM2 run inside an extract experiment directory."""
    return f'{exp_dir}/{EXTRACT_RUN_SUBDIR}'


def experiment_dir(root, stage, scene, name):
    """`<root>/<stage>/<scene>/<name>` - one experiment of `stage` on `scene`.

    Pure string work, so run_pipeline.py's PARAMETERS block may call it: that block is re-executed
    in every spawned reader child and must not touch the filesystem. Validate `name` with
    require_name() in main(), where raising is useful.
    """
    return f'{root}/{stage}/{scene}/{name}'


def test_dir(root, kind, scene):
    """`<root>/test/<kind>/<scene>` - every test of `kind` on `scene`, one subdirectory each."""
    if kind not in TEST_KINDS:
        raise ValueError(f'unknown test kind {kind!r}; choose from {TEST_KINDS}')
    return f'{root}/test/{kind}/{scene}'


def require_name(knob, value):
    """An experiment name must be set and must be one path component.

    Names are mandatory now that the scene is a directory of its own: an empty name would put the
    experiment's files directly in the scene directory, where the next run would overwrite them.
    """
    if not value or not str(value).strip():
        raise SystemExit(f'{knob} must be set - it names this experiment inside its scene '
                         f'directory, and every experiment needs a name of its own')
    if '/' in str(value) or str(value) in ('.', '..'):
        raise SystemExit(f'{knob}={value!r} must be a single directory name, not a path')
    return value


def stream_resize(img, res):
    """The resize the tracker sees. ONE definition, used by the reader, the LoRA data loader and
    the render metrics - they must agree or renders and GT stop lining up pixel for pixel.

    `res` is the resolution budget (PIXELS, not a shape): 341*640 is the scalar 218240, and both
    dims are scaled by sqrt(res / h0*w0) then floored to a multiple of 8. The floor is not
    aspect-preserving - Replica's 1200x680 drifts from aspect 1.765 to 1.791 - but slam/stream.py
    rescales the intrinsics with the ACTUAL ratios, so that is image shear, not a calibration
    error.
    """
    h0, w0 = img.shape[:2]
    h1 = int(h0 * np.sqrt(res / (h0 * w0)))
    w1 = int(w0 * np.sqrt(res / (h0 * w0)))
    return cv2.resize(img, (w1 - w1 % 8, h1 - h1 % 8))


def probe_stream_hw(image_dir, res):
    """The (H, W) the tracker will actually run at, measured by resizing the first frame.

    Cheap (one imread) but NOT free, and it touches the filesystem - so callers resolve it once,
    after chdir and before any Process is spawned, never in a module-level config literal.
    """
    files = sorted(os.listdir(image_dir))
    if not files:
        raise SystemExit(f'{image_dir} is empty - cannot determine the tracking resolution')
    img = cv2.imread(os.path.join(image_dir, files[0]))
    if img is None:
        raise SystemExit(f'could not read {os.path.join(image_dir, files[0])}')
    return tuple(stream_resize(img, res).shape[:2])
