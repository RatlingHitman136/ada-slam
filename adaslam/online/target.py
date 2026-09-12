"""Live supervision out of the shared DepthVideo - the online counterpart of adapt/data.py.

There is no export directory and no traj_full.txt here: the depth, the mask, the images and the
poses are read straight off the tracker's own state, which the patched prior_extractor reaches as
`mf.video` (13). What comes out is EXACTLY the 6-tuple SceneData.sample returns, so
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

from ..common import camera_points, context_keyframes, gauge_scale, transform_points
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

    def _kf_depth(self, video, i):
        """One keyframe's DENSE depth at VGGT's input size - 1/disps_up, no ceiling, no mask.

        Shared by kf_target, which then clamps and masks it, and by _gauge, which must see NEITHER:
        the gauge's footprint is every valid pixel by design (see _gauge).
        """
        d = 1.0 / np.clip(video.disps_up[i].numpy(), 1e-6, None)
        d[~np.isfinite(d)] = 0.0
        # cv2 resizes one plane at a time; N is 1-3 here so a loop costs nothing
        return cv2.resize(d.astype(np.float32), (self.hw[1], self.hw[0]),
                          interpolation=cv2.INTER_NEAREST)

    def _gauge(self, video, seq):
        """The sample's normalization scale (common.py:gauge_scale), or None to skip the sample.

        Points from EVERY frame of `seq`, expressed in the TARGET camera's frame (seq[0]) and
        pooled - VGGT's own convention, which normalizes over all S frames of the sample with the
        origin at the first camera. Measured at context_kf 2, pooling moves the gauge 11.0% on
        average (sd 6.3%) against a target-only statistic, so it is not an optional refinement.

        THE FOOTPRINT IS EVERY VALID-DEPTH PIXEL, NOT THE SUPERVISION MASK, and that is the whole
        design. The two answer different questions: the mask says which pixels are trustworthy
        enough to contribute a GRADIENT, while the gauge is one scalar that above all must be
        STABLE. mask_slam covers ~5% of pixels with 60% coverage variance keyframe to keyframe -
        variance driven by parallax and texture, nothing to do with scene scale - so a gauge built
        on it would move whenever the mask moved, injecting exactly the keyframe-to-keyframe noise
        this term exists to remove. Every valid-depth pixel is a 100%-coverage footprint with zero
        coverage variance by construction, and gauge_scale's clamp is what makes that safe.
        """
        K = self._intrinsics(video)
        X = [camera_points(self._kf_depth(video, seq[0]), K)]
        if len(seq) > 1:
            from lietorch import SE3
            idx = torch.as_tensor(seq, device=video.poses.device, dtype=torch.long)
            c2w = SE3(video.poses[idx]).inv().matrix().detach().cpu().numpy().astype(np.float64)
            t_from_w = np.linalg.inv(c2w[0])
            for j in range(1, len(seq)):
                X.append(transform_points(camera_points(self._kf_depth(video, seq[j]), K),
                                          t_from_w @ c2w[j]))
        X = np.concatenate(X) if len(X) > 1 else X[0]
        if len(X) < self.cfg.gauge_min_pixels:
            return None
        return gauge_scale(X, self.cfg.gauge_clamp)

    def kf_target(self, video, ix):
        """(depth, mask) at VGGT's input size for EVERY keyframe slot in `ix`, both (N, H, W).

        1/disps_up under the multi-view consistency mask - the same quantity extract/export.py
        writes to depth_slam/ + mask_slam/, taken after LOCAL BA rather than after global BA. That
        is the price of a single stage and is worth remembering when the numbers are read.

        `ix` is a LIST even for one keyframe, and the leading axis is always there; sample()
        squeezes it away when only the target frame is supervised, so that path stays exactly what
        it was. confidence_mask already scores several indices in one call - it reshapes ix to
        (-1) and returns (len(ix), h/8, w/8) - so N frames cost one depth_filter, not N.
        """
        n = video.counter.value
        ix = [int(i) for i in ix]
        ds = []
        for i in ix:
            d = self._kf_depth(video, i)
            # the training side of the far-field ceiling (14.6): clamp the target at the same
            # ratio the serving clamp uses, over its VALID pixels only - the median must not read
            # the zero-filled holes, and min() cannot lift a zero, so the mask below is
            # unaffected. PER FRAME, against that frame's own median, as it always was.
            if self.cfg.ceil_target:
                valid = d > 0
                if valid.any():
                    np.minimum(d, self.cfg.ceil_ratio * np.median(d[valid]), out=d)
            ds.append(d)
        d = np.stack(ds)                                          # (N, H, W)

        low = confidence_mask(video.poses[:n], video.disps[:n], video.intrinsics[0] * 8,
                              self.cfg, ix=ix)
        m = F.interpolate(low[:, None].float(), size=self.hw, mode='nearest')[:, 0]
        m = (m.cpu().numpy() > 0.5) & (d > 0)                     # (N, H, W)

        # PER-FRAME MASK FLOOR, and only when there is more than one frame. Stacked, depth_loss's
        # single `mask.sum() < min_mask_pixels` guard becomes a whole-STACK test, so one dense
        # frame could carry N-1 near-empty ones past it and their few surviving pixels would skew
        # the pooled median that the shared scale is fitted from. Zeroing rather than dropping
        # keeps N fixed, so the stack stays aligned with the images and with gt_enc.
        if len(ix) > 1:
            thin = m.reshape(len(ix), -1).sum(1) < self.cfg.min_mask_pixels
            m[thin] = False
        return torch.from_numpy(d), torch.from_numpy(m)

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
        """(images, gt_depth, mask, gt_enc, seq, gauge) for keyframe `t`, target FIRST.

        First because VGGT predicts in frame 0's coordinate frame - the same invariant
        adapt/data.py:75-78 documents and verifies.

        gt_depth/mask are (H, W) normally and (S, H, W) under depth_all_frames; every consumer -
        depth_loss, median_scale, relative_loss, the pred_med logging - masks and reduces to a
        scalar, so all of them are shape-agnostic as long as the three agree.
        """
        self._intrinsics(video)                      # caches stream_hw on the first call
        seq = [int(t)] + context_keyframes(int(t), self.cfg.context_kf,
                                           self.cfg.context_stride)
        images = torch.stack([self.frame(video, i) for i in seq])
        # depth_all_frames supervises the WHOLE sequence under one shared scale (config.py);
        # otherwise only the target, and the squeeze below leaves that path byte-for-byte what it
        # was. The order is seq's - target first, context ascending - so the stacked targets line
        # up with the frames forward() returns depth for.
        gt_depth, mask = self.kf_target(video, seq if self.cfg.depth_all_frames else [int(t)])
        if not self.cfg.depth_all_frames:
            gt_depth, mask = gt_depth[0], mask[0]
        # normalize_target (config.py): divide the target by the sample's own gauge, which puts it
        # in VGGT's pretrained convention and cancels the map's drift exactly. A sample with no
        # usable gauge is SKIPPED rather than divided by a guess - zeroing the mask is how
        # depth_loss's min_mask_pixels branch already spells 'no depth gradient here'.
        gauge = self._gauge(video, seq) if self.cfg.normalize_target else None
        if self.cfg.normalize_target:
            if gauge is None or not gauge > 0:
                mask = torch.zeros_like(mask)
            else:
                gt_depth = gt_depth / gauge
        # pose_loss returns zeros below 2 frames; skip the encoding entirely at context_kf=0
        gt_enc = self.pose_encoding(video, seq) if len(seq) > 1 else torch.zeros(1, 9)
        # gauge_pose (config.py): put the TRANSLATIONS in the same gauge as the depth target, which
        # is what VGGT's own training does - it divides camera translations and depth maps by one
        # avg_scale. Dims 3:7 are a quaternion and 7:9 a field of view; both are scale-free and
        # must not be touched.
        if self.cfg.gauge_pose and len(seq) > 1 and gauge:
            gt_enc = gt_enc.clone()
            gt_enc[:, :3] = gt_enc[:, :3] / gauge
        return images, gt_depth, mask, gt_enc, seq, gauge
