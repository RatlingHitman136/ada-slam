"""SlamRunner - the one way HI-SLAM2 is invoked.

This PACKAGE is the only place in adaslam/ that imports `hi2` or `motion_filter`. That is an
invariant worth grepping for - it is what "every invocation goes through one interface" means in
practice - and it is checkable in one line: every hit must be under adaslam/slam/.

    grep -rn 'from hi2 import\\|from motion_filter import' adaslam/

Two files match, and only two. This one imports both, to run SLAM. prior_probe.py imports
MotionFilter for a different reason: the stock depth prior IS one of its methods, and the prior
test has to call that exact function rather than a re-implementation of it. Anything outside
adaslam/slam/ matching that grep has broken the invariant.

Three call sites reach it - the extract run and one per end2end arm - and they differ only in their
arguments. The depth prior is one of those arguments rather than something a caller patches on
before calling, which fixes a real hazard: hi2.py:143 calls prior_extractor again inside
terminate(), for the covis-inserted keyframes, so the patch has to survive that call and be undone
afterwards. Doing it in a finally here makes both true by construction; the previous arrangement
(patch before the call, restore at the top of the next loop iteration) happened to be correct only
because nothing ran in between.

Module-level imports stay light on purpose. `spawn` re-imports this module in the reader child
(the target function lives in the sibling stream.py), so hi2, motion_filter and lietorch - the
last of which loads a CUDA extension - are all imported inside the functions that use them.
"""
import os
import re
import types
from dataclasses import dataclass

import numpy as np
from torch.multiprocessing import Process, Queue
from tqdm import tqdm

from ..runtime import free_vram

from .stream import mono_stream

# Every attribute Hi2 reads off its args namespace (hi2.py: 23, 24, 29, 41, 47, 157, 184 and the
# getattrs at 155 and 182). Asserted rather than trusted, so a future Hi2 that starts reading
# args.start fails here instead of silently seeing whatever a SimpleNamespace happened to carry.
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

    # the timestamp is the number in the filename, so %06d names make timestamps frame indices
    tstamps_full = np.array([float(re.findall(r'[+]?(?:\d*\.\d+|\d+)', x)[-1])
                             for x in sorted(os.listdir(cfg.colors))[cfg.start:]])[..., np.newaxis]
    tstamps_kf = tstamps_full[tstamps.cpu().numpy().astype(int)]
    np.savetxt(f'{out}/traj_kf.txt',
               np.concatenate([tstamps_kf, poses_wc.cpu().numpy()], axis=1))
    if traj_full is not None:
        np.savetxt(f'{out}/traj_full.txt',
                   np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1))


class SlamRunner:
    """Runs HI-SLAM2 over cfg's sequence, once per call to run().

    One instance is built per experiment and shared by every stage, which is what keeps the A/B
    arms comparable: they cannot disagree about the stream, the calibration or the resolution,
    because there is only one description of those and it is not a per-call argument.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, out, config, length, buffer, *,
            gtdepthdir=None, dump_slam_depth=False, prior=None):
        """One full SLAM run: stream -> track -> terminate -> write trajectories.

        `config` is the tracking YAML, `prior` an optional depth-prior object exposing
        `.extractor()` (see end2end/prior.py). `gtdepthdir` is stated per call and never
        inherited: passing it on a run whose renders become training data corrupts them (9.3).
        It only bites while cfg.render_eval is True - Hi2 reads it nowhere else - so with the
        toggle off every caller may pass None, and does.
        """
        from hi2 import Hi2
        from motion_filter import MotionFilter
        cfg = self.cfg
        os.makedirs(out, exist_ok=True)

        # Snapshot before installing, restore in the finally below. Both directions matter: an arm
        # must not inherit the previous arm's prior, and terminate() still calls the extractor.
        # The try opens on the very next line for a reason - anything that raises between the
        # install and the finally would leave the patch on the class for the next arm to inherit.
        stock_prior = MotionFilter.prior_extractor
        if prior is not None:
            MotionFilter.prior_extractor = prior.extractor()

        try:
            queue = reader = None
            queue = Queue(maxsize=8)
            reader = Process(target=mono_stream, args=(queue, cfg, length))
            reader.start()

            n_frames = len(os.listdir(cfg.colors))
            args = types.SimpleNamespace(
                weights=cfg.weights, config=config, output=out, gtdepthdir=gtdepthdir,
                buffer=min(1000, n_frames // 10 + 150) if buffer is None else buffer,
                droidvis=False, gsvis=False, dump_slam_depth=dump_slam_depth,
                render_eval=cfg.render_eval)

            hi2 = None
            pbar = tqdm(range(min(n_frames, length)), desc='Processing keyframes')
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
        return SlamResult(out=out, n_kf=n_kf, n_frames=min(n_frames, length))
