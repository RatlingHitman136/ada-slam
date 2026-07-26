"""Helpers shared by more than one pipeline stage.

Deliberately outside any of them: extract, adapt and abtest all need these, and this is the
neutral ground they can import from without depending on each other.
"""
import os

import cv2
import numpy as np

# Which of the extract stage's export targets supervises the adaptation. Lives here, not in a
# stage: extract writes depth_<src>/ and adapt reads it, so one tuple has to bound both.
DEPTH_SOURCES = ('slam', 'rendered')


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
