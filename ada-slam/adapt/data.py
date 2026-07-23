"""Turning an extract stage's export into training samples.

Reads the layout scripts/run_pipeline.py's extract stage writes (ARCHITECTURE.md §7):

    <scene_dir>/depth_<src>/%06d.npy   float32 depth in SLAM units, one per keyframe
    <scene_dir>/mask_<src>/%06d.png    multi-view consistency mask
    <scene_dir>/poses_slam.txt         the exported keyframes, TUM c2w  <- the keyframe list
    <scene_dir>/traj_full.txt          every frame's pose, TUM c2w
    <scene_dir>/intrinsics.npy         fx fy cx cy at the tracker's resolution
"""
import os

import cv2
import numpy as np
import torch

from common import stream_resize


def tum_to_c2w(row):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
    T[:3, 3] = row[1:4]
    return T


def split_keyframes(kf, cfg):
    """Train / val split over the exported keyframe list.

    'stride' holds out every Nth keyframe, so val covers the whole trained region and the val
    curve is not confounded by the scene changing. Its caveat is that a held-out keyframe's
    neighbours are visually near-identical to trained ones, so it measures memorisation more than
    generalisation; 'contiguous' is the opposite trade.
    """
    kf = list(kf)
    if cfg.train_frac >= 1.0 or len(kf) < 5:
        return kf, []
    if cfg.split_mode == 'stride':
        val = kf[::max(2, int(round(1.0 / (1.0 - cfg.train_frac))))]
    elif cfg.split_mode == 'contiguous':
        val = kf[int(round(len(kf) * cfg.train_frac)):]
    else:                                  # 'random'; AdaptConfig already validated the name
        perm = np.random.default_rng(cfg.seed).permutation(len(kf))
        val = sorted(kf[i] for i in perm[int(round(len(kf) * cfg.train_frac)):])
    vset = set(val)
    return [t for t in kf if t not in vset], val


class SceneData:
    """One keyframe = one sample, placed FIRST in the sequence so VGGT's predictions land in that
    keyframe's coordinate frame (verified: extrinsic[0] is identity to 5e-4). Around it we attach
    a random number of neighbouring non-keyframe frames, so the adapter works both monocular - the
    way MotionFilter.prior_extractor calls it - and with a few frames of context."""

    def __init__(self, scene_dir, image_dir, lora, cfg):
        self.scene_dir, self.image_dir, self.cfg = scene_dir, image_dir, cfg
        self.hw = lora.vggt_hw
        self.files = sorted(os.listdir(image_dir))

        self.ddir, self.mdir = f'depth_{cfg.depth_source}', f'mask_{cfg.depth_source}'
        if not os.path.isdir(f'{scene_dir}/{self.ddir}'):
            raise SystemExit(f'{scene_dir}/{self.ddir} not found - re-run the extract stage with '
                             f'depth_source = {cfg.depth_source!r}')

        traj = np.loadtxt(f'{scene_dir}/traj_full.txt')
        self.c2w = {int(r[0]): tum_to_c2w(r) for r in traj}
        self.t_min, self.t_max = int(traj[0, 0]), int(traj[-1, 0])
        self.kf = [int(t) for t in np.loadtxt(f'{scene_dir}/poses_slam.txt')[:, 0]]
        self.train_kf, self.val_kf = split_keyframes(self.kf, cfg)

        # intrinsics: stored at the tracker's resolution, rescale to the VGGT input size
        fx, fy, cx, cy = np.load(f'{scene_dir}/intrinsics.npy')
        probe = stream_resize(cv2.imread(os.path.join(image_dir, self.files[0])), cfg.stream_res)
        self.stream_hw = probe.shape[:2]
        sy, sx = self.hw[0] / probe.shape[0], self.hw[1] / probe.shape[1]
        self.K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], np.float64)

    def aspect_report(self):
        """The stream -> VGGT resize, and a warning when it distorts. frame() resizes without
        letterboxing, so a mismatched aspect ratio squashes the image off VGGT's training
        distribution; the suggested value is the one that would not."""
        h, w = self.stream_hw
        vh, vw = self.hw
        skew = (vw / vh) / (w / h)
        lines = [f'stream {w}x{h} (aspect {w/h:.3f}) -> VGGT {vw}x{vh} '
                 f'(aspect {vw/vh:.3f}), squash {skew:.3f}x']
        if not 0.95 < skew < 1.05:
            lines.append(f'  WARNING: aspect ratios differ by {abs(1-skew)*100:.0f}%. '
                         f'SceneData.frame() resizes without letterboxing, so VGGT sees a '
                         f'distorted image. Consider vggt_hw = ({14*round(518*h/w/14)}, 518)')
        return lines

    def frame(self, t):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, self.files[t])),
                           cv2.COLOR_BGR2RGB)
        img = cv2.resize(stream_resize(img, self.cfg.stream_res), (self.hw[1], self.hw[0]),
                         interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def kf_target(self, t):
        d = np.load(f'{self.scene_dir}/{self.ddir}/{t:06d}.npy')
        m = cv2.imread(f'{self.scene_dir}/{self.mdir}/{t:06d}.png', cv2.IMREAD_GRAYSCALE) > 127
        d = cv2.resize(d, (self.hw[1], self.hw[0]), interpolation=cv2.INTER_NEAREST)
        m = cv2.resize(m.astype(np.uint8), (self.hw[1], self.hw[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
        return torch.from_numpy(d).float(), torch.from_numpy(m & (d > 0))

    def neighbours(self, t, rng, n_left, n_right):
        """Random non-keyframe neighbours within radius; edge keyframes take from the other side."""
        r = self.cfg.radius
        left = [x for x in range(max(t - r, self.t_min), t) if x in self.c2w]
        right = [x for x in range(t + 1, min(t + r, self.t_max) + 1) if x in self.c2w]
        want = n_left + n_right
        n_left, n_right = min(n_left, len(left)), min(n_right, len(right))
        # keyframes at the sequence ends have no frames on one side; make up the shortfall from
        # the other side so the requested context size is still met where the frames exist
        n_right = min(n_right + (want - n_left - n_right), len(right))
        n_left = min(n_left + (want - n_left - n_right), len(left))
        picks = list(rng.choice(left, n_left, replace=False)) + \
            list(rng.choice(right, n_right, replace=False))
        return sorted(int(x) for x in picks)

    def sample(self, rng, t=None, single=None):
        from vggt.utils.pose_enc import extri_intri_to_pose_encoding
        cfg = self.cfg
        t = int(rng.choice(self.train_kf)) if t is None else t
        if single is None:
            single = rng.random() < cfg.p_single_view
        nb = [] if single else self.neighbours(t, rng, rng.integers(1, cfg.max_left + 1),
                                               rng.integers(1, cfg.max_right + 1))
        seq = [t] + nb

        images = torch.stack([self.frame(x) for x in seq])
        gt_depth, mask = self.kf_target(t)

        # rebase every pose so the keyframe is the world origin -> frame 0 is identity
        kf_c2w = self.c2w[t]
        extr = np.stack([(np.linalg.inv(self.c2w[x]) @ kf_c2w)[:3] for x in seq])
        K = np.broadcast_to(self.K, (len(seq), 3, 3))
        gt_enc = extri_intri_to_pose_encoding(
            torch.from_numpy(extr).float()[None], torch.from_numpy(K.copy()).float()[None],
            image_size_hw=self.hw)[0]
        return images, gt_depth, mask, gt_enc, seq
