"""Live supervision out of the shared DepthVideo - the online counterpart of adapt/data.py.

There is no export directory and no traj_full.txt here: the depth, the mask, the images and the
poses are read straight off the tracker's own state, which the patched prior_extractor reaches as
`mf.video` (13). What comes out is EXACTLY the 5-tuple SceneData.sample returns, so
adapt/losses.py is reused unchanged.

Two things about that state are load-bearing:

  * indices are NEVER cached across calls. track_frontend.py:52 removes keyframe t1-2 and
    decrements counter, shifting everything after it, so every window is derived from
    counter.value read fresh inside the call that uses it.
  * poses/disps must be SLICED to counter.value before depth_filter sees them, or trailing
    keyframes agree with unused buffer slots still holding disps = 1.0 (extract/export.py:33-35).
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..extract.export import confidence_mask


def settled(video, lag):
    """The newest keyframe safe to train on, or None if the map is not that long yet.

    `lag` keyframes back from the end. The arriving keyframe has not been through BA at all when
    its prior is extracted, and the one before it is still inside the local window - lag=2 is
    track_frontend.py:65's own line, the last index __update reports as changed.
    """
    hi = video.counter.value - 1 - lag
    return hi if hi >= 0 else None


def unit_keyframes(video, cfg):
    """The keyframes one arrival trains on: [hi] in 'online', the sliding window in 'wonline'.

    The window is the arrival plus the window_size-1 keyframes before it, clipped at the start of
    the sequence - so early arrivals train on a short window rather than being skipped.
    """
    hi = settled(video, cfg.lag)
    if hi is None:
        return []
    if cfg.adapt_style == 'wonline':
        return list(range(max(0, hi - cfg.window_size + 1), hi + 1))
    return [hi]


def context_keyframes(t, n_ctx):
    """The `n_ctx` keyframes before `t`, ascending. Keyframes, not the frame neighbours
    adapt/data.py uses: non-keyframe images live on Hi2, which the extractor cannot reach."""
    return list(range(max(0, t - n_ctx), t))


class LiveSampler:
    """Builds training samples from a DepthVideo. Holds only sizes - the state is the video's."""

    def __init__(self, cfg, vggt_hw):
        self.cfg = cfg
        self.hw = tuple(vggt_hw)
        self._K = None
        self.stream_hw = None

    def _intrinsics(self, video):
        """The tracker's intrinsics at VGGT's input size. Cached: they never change mid-run.

        video.intrinsics is stored divided by 8 (motion_filter.py:83), so x8 recovers the
        full-resolution ones save_trajectory writes to intrinsics.npy - which is what
        adapt/data.py:99-103 rescales, by the same two ratios.
        """
        if self._K is None:
            self.stream_hw = (video.ht, video.wd)
            fx, fy, cx, cy = (video.intrinsics[0] * 8).detach().cpu().numpy().astype(np.float64)
            sy = self.hw[0] / self.stream_hw[0]
            sx = self.hw[1] / self.stream_hw[1]
            self._K = np.array([[fx * sx, 0, cx * sx],
                                [0, fy * sy, cy * sy],
                                [0, 0, 1]], np.float64)
        return self._K

    def frame(self, video, i):
        """One keyframe's RGB at VGGT's input size, pixel for pixel as SceneData.frame.

        video.images[i] is already stream_resize'd (the reader wrote it) and already RGB
        (mono_stream converts - extract/export.py:83), so ONLY the second resize is left, and it
        must be the same INTER_AREA adapt/data.py:116-117 uses.
        """
        img = video.images[i].numpy().transpose(1, 2, 0)          # (3,H,W) uint8 -> HWC
        img = cv2.resize(np.ascontiguousarray(img), (self.hw[1], self.hw[0]),
                         interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def kf_target(self, video, t):
        """(depth, mask) at VGGT's input size - 1/disps_up under the multi-view consistency mask.

        The same quantity extract/export.py writes to depth_slam/ + mask_slam/, taken after LOCAL
        BA rather than after global BA. That is the price of a single stage and is worth
        remembering when the numbers are read.
        """
        n = video.counter.value
        d = 1.0 / np.clip(video.disps_up[t].numpy(), 1e-6, None)
        d[~np.isfinite(d)] = 0.0
        d = cv2.resize(d.astype(np.float32), (self.hw[1], self.hw[0]),
                       interpolation=cv2.INTER_NEAREST)
        # the training side of the far-field ceiling (14.6): clamp the target at the same ratio
        # the serving clamp uses, over its VALID pixels only - the median must not read the
        # zero-filled holes, and min() cannot lift a zero, so the mask below is unaffected
        if self.cfg.ceil_target:
            valid = d > 0
            if valid.any():
                np.minimum(d, self.cfg.ceil_ratio * np.median(d[valid]), out=d)

        low = confidence_mask(video.poses[:n], video.disps[:n], video.intrinsics[0] * 8,
                              self.cfg, ix=[t])
        m = F.interpolate(low[:, None].float(), size=self.hw, mode='nearest')[0, 0]
        m = m.cpu().numpy() > 0.5
        return torch.from_numpy(d), torch.from_numpy(m & (d > 0))

    def pose_encoding(self, video, seq):
        """VGGT's pose encoding for `seq`, rebased so seq[0] is the world origin.

        Exactly adapt/data.py:156-162, with the poses read off video.poses (world->cam) instead of
        traj_full.txt. Those are LIVE SLAM poses, still being refined - the offline loader's come
        from the post-refinement trajectory.
        """
        from lietorch import SE3
        from vggt.utils.pose_enc import extri_intri_to_pose_encoding

        idx = torch.as_tensor(seq, device=video.poses.device, dtype=torch.long)
        c2w = SE3(video.poses[idx]).inv().matrix().detach().cpu().numpy().astype(np.float64)
        extr = np.stack([(np.linalg.inv(c2w[j]) @ c2w[0])[:3] for j in range(len(seq))])
        K = np.broadcast_to(self._intrinsics(video), (len(seq), 3, 3))
        return extri_intri_to_pose_encoding(
            torch.from_numpy(extr).float()[None], torch.from_numpy(K.copy()).float()[None],
            image_size_hw=self.hw)[0]

    def sample(self, video, t):
        """(images, gt_depth, mask, gt_enc, seq) for keyframe `t`, target placed FIRST.

        First because VGGT predicts in frame 0's coordinate frame - the same invariant
        adapt/data.py:75-78 documents and verifies.
        """
        self._intrinsics(video)                      # caches stream_hw on the first call
        seq = [int(t)] + context_keyframes(int(t), self.cfg.context_kf)
        images = torch.stack([self.frame(video, i) for i in seq])
        gt_depth, mask = self.kf_target(video, int(t))
        # pose_loss returns zeros below 2 frames; skip the encoding entirely at context_kf=0
        gt_enc = self.pose_encoding(video, seq) if len(seq) > 1 else torch.zeros(1, 9)
        return images, gt_depth, mask, gt_enc, seq
