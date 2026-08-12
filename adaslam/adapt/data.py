"""Turning an extract stage's export into training samples.

`scene_dir` is an extract EXPERIMENT directory - its top level, never full/. Poses come from
traj_full.txt (post-refinement); poses_slam.txt supplies the keyframe LIST only.

`image_dir` is the FULL colour directory, not <scene_dir>/image: frame() indexes it by frame
number, so a keyframes-only folder would silently return the wrong image.
"""
import os

import cv2
import numpy as np
import torch

from ..common import DEPTH_DIR, MASK_DIR, stream_resize

from .config import aspect_lines


def tum_to_c2w(row):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
    T[:3, 3] = row[1:4]
    return T


def evenly(seq, n):
    """`n` items of `seq`, evenly spaced BY INDEX, order preserved. Fewer if indices collide."""
    seq = list(seq)
    if n >= len(seq):
        return seq
    pick = np.linspace(0, len(seq) - 1, n).round().astype(int)
    return [seq[i] for i in sorted(set(pick.tolist()))]


def select_keyframes(kf, cfg):
    """The exported keyframes this run trains on at all: `kf_fraction` of them, equidistant.

    Equidistant over the keyframe LIST, not over frames - keyframes are unevenly spaced in time,
    so this is every Nth keyframe.
    """
    kf = list(kf)                          # already ascending: poses_slam.txt column 0
    if cfg.kf_fraction >= 1.0:
        return kf
    return evenly(kf, max(2, int(round(len(kf) * cfg.kf_fraction))))


def split_keyframes(kf, cfg):
    """Train / val split over a keyframe list: val is the contiguous TAIL.

    So val measures generalisation forward, and the trained region is a strict PREFIX.
    """
    kf = list(kf)
    if cfg.train_frac >= 1.0 or len(kf) < 5:
        return kf, []
    cut = int(round(len(kf) * cfg.train_frac))
    return kf[:cut], kf[cut:]


def training_split(kf, cfg):
    """(train, val) over the exported keyframes - the ONE place both val modes live.

    Select first, then decide what the rest of the export is for: 'rest' validates on every
    keyframe the selection skipped (interleaved through the whole sequence), 'tail' on the
    contiguous end of the selection itself.
    """
    sel = select_keyframes(kf, cfg)
    if cfg.val_source == 'rest':
        return sel, [int(t) for t in kf if t not in set(sel)]
    return split_keyframes(sel, cfg)


class SceneData:
    """One keyframe = one sample, placed FIRST so VGGT predicts in that keyframe's frame.

    Around it, a random number of neighbouring non-keyframes, so the adapter works both monocular
    (the way prior_extractor calls it) and with context.
    """

    def __init__(self, scene_dir, image_dir, lora, cfg):
        self.scene_dir, self.image_dir, self.cfg = scene_dir, image_dir, cfg
        self.hw = lora.vggt_hw
        self.files = sorted(os.listdir(image_dir))

        self.ddir, self.mdir = DEPTH_DIR, MASK_DIR
        if not os.path.isdir(f'{scene_dir}/{self.ddir}'):
            raise SystemExit(f'{scene_dir}/{self.ddir} not found - re-run the extract stage')

        traj = np.loadtxt(f'{scene_dir}/traj_full.txt')
        self.c2w = {int(r[0]): tum_to_c2w(r) for r in traj}
        self.t_min, self.t_max = int(traj[0, 0]), int(traj[-1, 0])
        # self.kf stays the WHOLE export: t_min/t_max and the recorded split_at are about the
        # extract window, not about which of its keyframes this run happens to train on
        self.kf = [int(t) for t in np.loadtxt(f'{scene_dir}/poses_slam.txt')[:, 0]]
        self.train_kf, self.val_kf = training_split(self.kf, cfg)

        # intrinsics: stored at the tracker's resolution, rescale to the VGGT input size
        fx, fy, cx, cy = np.load(f'{scene_dir}/intrinsics.npy')
        probe = stream_resize(cv2.imread(os.path.join(image_dir, self.files[0])), cfg.stream_res)
        self.stream_hw = probe.shape[:2]
        sy, sx = self.hw[0] / probe.shape[0], self.hw[1] / probe.shape[1]
        self.K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], np.float64)

    def aspect_report(self):
        """The stream -> VGGT resize, and a warning when it distorts.

        With vggt_hw derived this should never fire; if it does, the value was pinned by hand or
        read off an adapter trained on another stream.
        """
        return aspect_lines(self.stream_hw, self.hw, 'SceneData.frame()')

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
