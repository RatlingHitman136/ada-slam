"""Helpers shared by more than one pipeline stage.

Deliberately outside adapt/: as the extract and test stages follow the adapt stage out of
scripts/run_pipeline.py, this is the neutral ground they can all import from without depending
on each other.
"""
import cv2
import numpy as np


def stream_resize(img, res):
    """The resize the tracker sees. ONE definition, used by the reader, the LoRA data loader and
    the render metrics - they must agree or renders and GT stop lining up pixel for pixel.

    `res` is the resolution budget (pixels); both output dims are made divisible by 8.
    """
    h0, w0 = img.shape[:2]
    h1 = int(h0 * np.sqrt(res / (h0 * w0)))
    w1 = int(w0 * np.sqrt(res / (h0 * w0)))
    return cv2.resize(img, (w1 - w1 % 8, h1 - h1 % 8))
