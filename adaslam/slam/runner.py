"""SlamRunner - the one way HI-SLAM2 is invoked.

The invariant, checkable in one line (9.3) - every hit must be under adaslam/slam/:

    grep -rn 'from hi2 import\\|from motion_filter import' adaslam/

hi2, motion_filter and lietorch are imported inside the functions that use them: spawn re-imports
this module in the reader child.
"""
import os
import re
import types
from dataclasses import dataclass

import numpy as np
from torch.multiprocessing import Process, Queue
from tqdm import tqdm

from ..runtime import free_vram

from .stream import mono_stream, window_files

# Every attribute Hi2 reads off its args namespace. Asserted, not trusted: a future Hi2 that reads
# one more fails here instead of silently seeing whatever the SimpleNamespace carried.
HI2_ARGS = ('weights', 'config', 'output', 'gtdepthdir', 'buffer', 'droidvis', 'gsvis',
            'dump_slam_depth', 'render_eval', 'image_size')


@dataclass
class SlamResult:
    """What a finished run leaves behind. The trajectories are already on disk."""
    out: str          # the output directory
    n_kf: int         # keyframes the tracker kept
    n_frames: int     # frames actually streamed


def save_trajectory(hi2, traj_full, cfg, out):
    """traj_kf.txt / traj_full.txt in TUM format (camera-to-world) + intrinsics.npy."""
    import lietorch
    t = hi2.video.counter.value
    tstamps = hi2.video.tstamp[:t]
    poses_wc = lietorch.SE3(hi2.video.poses[:t]).inv().data
    np.save(f'{out}/intrinsics.npy', hi2.video.intrinsics[0].cpu().numpy() * 8)

    # the timestamp is the number in the filename, so %06d names make timestamps frame indices -
    # ABSOLUTE ones, whatever the window, which is what makes a windowed run's trajectory line up
    # with GT, with evo's timestamps.npy and with adapt/data.py's pose dict
    tstamps_full = np.array([float(re.findall(r'[+]?(?:\d*\.\d+|\d+)', x)[-1])
                             for x in window_files(cfg)])[..., np.newaxis]
    tstamps_kf = tstamps_full[tstamps.cpu().numpy().astype(int)]
    np.savetxt(f'{out}/traj_kf.txt',
               np.concatenate([tstamps_kf, poses_wc.cpu().numpy()], axis=1))
    if traj_full is not None:
        np.savetxt(f'{out}/traj_full.txt',
                   np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1))


class SlamRunner:
    """Runs HI-SLAM2 over cfg's sequence, once per call to run().

    One instance per experiment, shared by every stage, so the arms cannot disagree about the
    stream, the calibration or the resolution.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, out, config, length, buffer, *,
            gtdepthdir=None, dump_slam_depth=False, prior=None):
        """One full SLAM run: stream -> track -> terminate -> write trajectories.

        `config` is the tracking YAML, `prior` an optional object exposing `.extractor()`.
        `gtdepthdir` is stated per call, never inherited: it corrupts a run whose renders become
        training data (9.3).
        """
        from hi2 import Hi2
        from motion_filter import MotionFilter
        cfg = self.cfg
        os.makedirs(out, exist_ok=True)

        # Snapshot, install, restore in the finally - so no arm inherits another's prior, even if
        # this one raises. The try opens on the very next line for exactly that reason.
        stock_prior = MotionFilter.prior_extractor
        if prior is not None:
            MotionFilter.prior_extractor = prior.extractor()

        try:
            queue = reader = None
            queue = Queue(maxsize=8)
            reader = Process(target=mono_stream, args=(queue, cfg, length))
            reader.start()

            # the WINDOW, not the directory: len(os.listdir(colors)) over-reports the moment start
            # or stop is set, and this number sizes the progress bar and SlamResult.n_frames
            n_frames = min(len(window_files(cfg)), length)
            args = types.SimpleNamespace(
                weights=cfg.weights, config=config, output=out, gtdepthdir=gtdepthdir,
                buffer=min(1000, n_frames // 10 + 150) if buffer is None else buffer,
                droidvis=False, gsvis=False, dump_slam_depth=dump_slam_depth,
                render_eval=cfg.render_eval)

            hi2 = None
            pbar = tqdm(range(n_frames), desc='Processing keyframes')
            while True:
                t, image, intrinsics, is_last = queue.get()
                pbar.update()
                if hi2 is None:
                    args.image_size = [image.shape[2], image.shape[3]]
                    assert tuple(sorted(vars(args))) == tuple(sorted(HI2_ARGS)), vars(args)
                    hi2 = Hi2(args)
                hi2.track(t, image, intrinsics=intrinsics, is_last=is_last)
                pbar.set_description(f'keyframe {hi2.video.counter.value} '
                                     f'gs {hi2.gs.gaussians._xyz.shape[0]}')
                if is_last:
                    pbar.close()
                    break
            reader.join()

            traj = hi2.terminate()
            save_trajectory(hi2, traj, cfg, out)
            n_kf = hi2.video.counter.value
            del hi2, traj                      # the next stage needs the VRAM back
        finally:
            # after terminate(), never before: hi2.py:143 runs the extractor once more for the
            # keyframes terminate() inserts into low-covisibility gaps
            MotionFilter.prior_extractor = stock_prior

        del queue, reader
        free_vram()
        return SlamResult(out=out, n_kf=n_kf, n_frames=n_frames)
