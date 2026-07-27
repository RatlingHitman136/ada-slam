"""The image reader: decode, optionally undistort, resize, hand to the tracker.

mono_stream runs in a spawned child, so everything it needs arrives through its arguments. It is
handed the SlamConfig itself rather than seven loose scalars: frozen dataclasses of primitives
pickle by value, so the child runs the exact object the parent built - not, as before, whatever
the constants happened to say when the child re-executed the driver module.
"""
import os
import time

import cv2
import numpy as np
import torch

from ..common import stream_resize


def load_calib(cfg):
    """(calib row, K) from cfg.calib - loaded once, then passed to load_frame per frame."""
    calib = np.loadtxt(cfg.calib, delimiter=' ')
    K = np.array([[calib[0], 0, calib[2]], [0, calib[1], calib[3]], [0, 0, 1]])
    return calib, K


def load_frame(cfg, path, calib, K):
    """One colour file -> (RGB image at tracking resolution, intrinsics for it).

    ONE definition of what the tracker is shown, because more than the tracker looks at it: the
    prior probe (prior_probe.py) scores depth priors on exactly these pixels, and a probe that
    undistorted differently, or resized differently, would be measuring a different image than the
    one the SLAM run sees.
    """
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    intrinsics = torch.tensor(calib[:4])
    if len(calib) > 4 and cfg.undistort:
        image = cv2.undistort(image, K, calib[4:])
    if cfg.crop_border > 0:
        image = image[cfg.crop_border:-cfg.crop_border, cfg.crop_border:-cfg.crop_border]
        intrinsics[2:] -= cfg.crop_border

    h0, w0 = image.shape[:2]
    image = stream_resize(image, cfg.stream_res)
    h1, w1 = image.shape[:2]
    intrinsics[[0, 2]] *= (w1 / w0)
    intrinsics[[1, 3]] *= (h1 / h0)
    return image, intrinsics


def mono_stream(queue, cfg, length):
    """Push (t, image, intrinsics, is_last) for `length` frames from cfg.start onwards.

    `t` is the index within this run, not the frame number, and it is what Hi2 stores as the
    keyframe timestamp - save_trajectory maps it back through the filename list.
    """
    calib, K = load_calib(cfg)
    image_list = sorted(os.listdir(cfg.colors))[cfg.start:cfg.start + length]

    for t, imfile in enumerate(image_list):
        image, intrinsics = load_frame(cfg, os.path.join(cfg.colors, imfile), calib, K)
        queue.put((t, torch.as_tensor(image).permute(2, 0, 1)[None], intrinsics[None],
                   t == len(image_list) - 1))

    time.sleep(10)      # keep the queue's feeder thread alive until the consumer has drained it
