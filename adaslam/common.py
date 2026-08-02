"""Helpers more than one stage needs - the neutral ground they import instead of each other."""
import os

import cv2
import numpy as np

DEPTH_DIR, MASK_DIR = 'depth_slam', 'mask_slam'   # extract writes them, adapt reads them

# ---------------------------------------------------------------- the outputs/ layout (7.1)
# outputs/<stage>/<scene>/<experiment>/, the experiment holding what the NEXT stage consumes and
# nothing else. Here rather than in a stage package because run_pipeline.py needs the names too.
EXTRACT_RUN_SUBDIR = 'full'                        # the raw SLAM run; deletable afterwards
ADAPT_CKPT_SUBDIR = 'checkpoints'
TEST_KINDS = ('end2end', 'prior')
HANDOFF_UP = ('traj_full.txt', 'intrinsics.npy')   # COPIED up from full/, so full/ stays complete


def extract_run_dir(exp_dir):
    """The untouched HI-SLAM2 run inside an extract experiment directory."""
    return f'{exp_dir}/{EXTRACT_RUN_SUBDIR}'


def experiment_dir(root, stage, scene, name):
    """`<root>/<stage>/<scene>/<name>`. Pure string work - the PARAMETERS block calls it."""
    return f'{root}/{stage}/{scene}/{name}'


def test_dir(root, kind, scene):
    """`<root>/test/<kind>/<scene>` - every test of `kind` on `scene`, one subdirectory each."""
    if kind not in TEST_KINDS:
        raise ValueError(f'unknown test kind {kind!r}; choose from {TEST_KINDS}')
    return f'{root}/test/{kind}/{scene}'


def require_name(knob, value):
    """An experiment name must be set and must be one path component."""
    if not value or not str(value).strip():
        raise SystemExit(f'{knob} must be set - it names this experiment inside its scene '
                         f'directory, and every experiment needs a name of its own')
    if '/' in str(value) or str(value) in ('.', '..'):
        raise SystemExit(f'{knob}={value!r} must be a single directory name, not a path')
    return value


def stream_resize(img, res):
    """The resize the tracker sees. ONE definition - every consumer must agree pixel for pixel.

    `res` is a pixel budget, not a shape (9.6): both dims scale by sqrt(res / h0*w0), floored to a
    multiple of 8. That floor is not aspect-preserving, but slam/stream.py rescales the intrinsics
    with the actual ratios, so it is image shear rather than a calibration error.
    """
    h0, w0 = img.shape[:2]
    h1 = int(h0 * np.sqrt(res / (h0 * w0)))
    w1 = int(w0 * np.sqrt(res / (h0 * w0)))
    return cv2.resize(img, (w1 - w1 % 8, h1 - h1 % 8))


def probe_stream_hw(image_dir, res):
    """The (H, W) the tracker will run at, measured on the first frame.

    Touches the filesystem, so callers resolve it once - after chdir, before any Process is spawned.
    """
    files = sorted(os.listdir(image_dir))
    if not files:
        raise SystemExit(f'{image_dir} is empty - cannot determine the tracking resolution')
    img = cv2.imread(os.path.join(image_dir, files[0]))
    if img is None:
        raise SystemExit(f'could not read {os.path.join(image_dir, files[0])}')
    return tuple(stream_resize(img, res).shape[:2])
