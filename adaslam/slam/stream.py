"""The image reader: decode, optionally undistort, resize, hand to the tracker.

mono_stream runs in a spawned child, so it is handed the SlamConfig itself - a frozen dataclass of
primitives pickles by value, so the child runs the exact object the parent built.
"""
import os

import cv2
import numpy as np
import torch

from ..common import stream_resize

# How long the reader waits for the consumer to say it has every frame. Not a drain estimate - the
# consumer sets the Event the moment it does - only a ceiling for the case where it crashed first.
DRAIN_TIMEOUT = 600


def load_calib(cfg):
    """(calib row, K) from cfg.calib - loaded once, then passed to load_frame per frame."""
    calib = np.loadtxt(cfg.calib, delimiter=' ')
    K = np.array([[calib[0], 0, calib[2]], [0, calib[1], calib[3]], [0, 0, 1]])
    return calib, K


def load_frame(cfg, path, calib, K):
    """One colour file -> (RGB image at tracking resolution, intrinsics for it).

    ONE definition of what the tracker is shown: PriorProbe scores priors on exactly these pixels.
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


def window_files(cfg):
    """The colour filenames inside cfg's [start, stop) window, sorted.

    ONE definition, because three callers need it and a disagreement would be invisible:
    mono_stream streams this list, save_trajectory turns it into timestamps, and SlamRunner.run
    sizes the progress bar by it. Note what it is NOT - the whole directory:
    len(os.listdir(colors)) over-reports the moment start or stop is set.
    """
    return sorted(os.listdir(cfg.colors))[cfg.start:cfg.stop]


def mono_stream(queue, cfg, length, drained):
    """Push (t, image, intrinsics, is_last) for the first `length` frames of cfg's window.

    `t` is the index within this run, not the frame number, and it is what Hi2 stores as the
    keyframe timestamp - save_trajectory maps it back through the filename list.

    Two slices, not one: [start:stop] is the experiment's WINDOW and `length` caps how much of it
    this particular call consumes (the extract stage runs a prefix, the arms run all of it). Doing
    it as [start : start + length] instead would make `length` silently override `stop`.

    `drained` is the consumer's "I have every frame" Event, and this process MUST NOT return before
    it is set - see below.
    """
    calib, K = load_calib(cfg)
    image_list = window_files(cfg)[:length]

    for t, imfile in enumerate(image_list):
        image, intrinsics = load_frame(cfg, os.path.join(cfg.colors, imfile), calib, K)
        queue.put((t, torch.as_tensor(image).permute(2, 0, 1)[None], intrinsics[None],
                   t == len(image_list) - 1))

    # THIS PROCESS MUST OUTLIVE THE LAST get(). Every queued tensor is rebuilt in the consumer
    # from a file descriptor fetched over THIS process's resource_sharer socket, so whatever is
    # still in flight when it exits dies with it - surfacing there, confusingly, as
    # "FileNotFoundError: [Errno 2]" inside torch's rebuild_storage_fd rather than as anything
    # about the reader.
    #
    # It used to be a 10 s sleep, and that is a guess at how long the consumer needs. Queue's
    # maxsize means the last put returns with up to that many frames still unread, and ONE arriving
    # keyframe holds the consumer for a full adaptation burst - 49 s at steps_per_kf=10 (13),
    # measured 1.0 s/step. So the tail routinely outlasts any fixed sleep, and does so at the very
    # end of a multi-hour run. Wait to be told instead.
    #
    # The timeout only bounds the wait if the consumer died before setting it; runner.py also marks
    # this process daemonic, so a crashed parent kills it rather than blocking on it at exit.
    drained.wait(timeout=DRAIN_TIMEOUT)
