"""LoRA-adapt VGGT on HI-SLAM2's own depth + poses (prototype).

One keyframe = one sample, placed FIRST in the sequence so VGGT's predictions land in that
keyframe's coordinate frame (verified: extrinsic[0] is identity to 5e-4). Around it we attach a
random number of neighbouring non-keyframe frames, so the adapter works both monocular - the way
MotionFilter.prior_extractor calls it today - and with a few frames of context.

  depth supervision : frame 0 only (we only have depth for keyframes), from the directory
                      export_slam_depth.py --depth_source wrote - Gaussian-rendered by default,
                      which is both closer to GT and rendered from the same post-refinement
                      trajectory that traj_full.txt below holds
  pose supervision  : all frames, from traj_full.txt, rebased to the keyframe

Only the aggregator (main transformer) is adapted; depth_head, camera_head and patch_embed stay
frozen. Gradients still reach the aggregator *through* the frozen heads.

  python scripts/lora_adapt_vggt.py --scene outputs/replica/room0_p40 \
                                    --images data/Replica/room0/colors
"""
import os    # nopep8
import sys   # nopep8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # nopep8
sys.path.insert(0, os.path.join(_ROOT, 'thirdparty/vggt'))            # nopep8
import argparse
import json
import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from scipy.spatial.transform import Rotation

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import extri_intri_to_pose_encoding

# ==============================================================================
#  PARAMETERS
# ==============================================================================

WEIGHTS = 'pretrained_models/vggt'      # local VGGT-1B snapshot

RANK, ALPHA = 16, 16                    # LoRA rank / scaling
LORA_TARGETS = ('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2')
LORA_PATCH_EMBED = False                # False = adapt only the alternating-attention stack

EPOCHS, STEPS_PER_EPOCH = 3, 256
LR = 1e-4
LAMBDA_POSE = 1.0

MAX_LEFT, MAX_RIGHT, RADIUS = 4, 4, 8   # neighbour count and search radius, in frames
P_SINGLE_VIEW = 1                       # 0 = always multi-view, 1 = always monocular

DEPTH_SPACE = 'depth'                   # 'disparity' (as HI-SLAM2 consumes it) or 'depth'
COUPLED_SCALE = False                   # True = one scale shared by depth and pose
SUPERVISE_FOV = False

VGGT_HW = (294, 518)                    # 21x14 by 37x14; aspect 1.76, suits Replica's 344x616
SEED = 0

# ==============================================================================


def parse_hw(s):
    """'H,W' -> (H, W), validated against VGGT's 14x14 patch grid.

    Worth overriding per dataset: data.frame() resizes straight to VGGT_HW without letterboxing,
    so a mismatched aspect ratio squashes the image off VGGT's training distribution. TUM's
    400x544 stream (aspect 1.36) wants 378,518, not the Replica default.
    """
    h, w = (int(x) for x in s.replace('x', ',').split(','))
    if h % 14 or w % 14:
        raise ValueError(f'--vggt_hw {h},{w}: both dims must be divisible by 14')
    return h, w



# ------------------------------------------------------------------ LoRA

class LoRALinear(nn.Module):
    """y = W0 x + (B A) x * alpha/r, with B zero-initialised so the adapter starts as identity."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scaling


def inject_lora(model, rank, alpha):
    """Wrap the targeted Linears inside the aggregator's frame/global blocks."""
    blocks = list(model.aggregator.frame_blocks) + list(model.aggregator.global_blocks)
    if LORA_PATCH_EMBED and hasattr(model.aggregator.patch_embed, 'blocks'):
        blocks += list(model.aggregator.patch_embed.blocks)

    n = 0
    for blk in blocks:
        for tgt in LORA_TARGETS:
            parent_path, _, leaf = tgt.rpartition('.')
            parent = blk.get_submodule(parent_path) if parent_path else blk
            child = getattr(parent, leaf, None)
            if isinstance(child, nn.Linear):
                setattr(parent, leaf, LoRALinear(child, rank, alpha))
                n += 1
    return n


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith('.A') or k.endswith('.B')}


# ------------------------------------------------------------------ data

def mono_stream_resize(img):
    """Identical to demo.py mono_stream (lines 49-55), so neighbour frames match the exports."""
    RES = 341 * 640
    h0, w0, _ = img.shape
    h1 = int(h0 * np.sqrt(RES / (h0 * w0)))
    w1 = int(w0 * np.sqrt(RES / (h0 * w0)))
    return cv2.resize(img, (w1 - w1 % 8, h1 - h1 % 8))


def tum_to_c2w(row):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
    T[:3, 3] = row[1:4]
    return T


class SceneData:
    def __init__(self, scene_dir, image_dir, depth_source='rendered'):
        self.scene_dir, self.image_dir = scene_dir, image_dir
        self.files = sorted(os.listdir(image_dir))

        # written by export_slam_depth.py --depth_source; both hold float32 .npy in SLAM units
        self.ddir, self.mdir = f'depth_{depth_source}', f'mask_{depth_source}'
        if not os.path.isdir(f'{scene_dir}/{self.ddir}'):
            raise SystemExit(f'{scene_dir}/{self.ddir} not found - re-run export_slam_depth.py '
                             f'with --depth_source {depth_source}')

        traj = np.loadtxt(f'{scene_dir}/traj_full.txt')
        self.c2w = {int(r[0]): tum_to_c2w(r) for r in traj}
        self.t_min, self.t_max = int(traj[0, 0]), int(traj[-1, 0])
        self.kf = [int(t) for t in np.loadtxt(f'{scene_dir}/poses_slam.txt')[:, 0]]

        # intrinsics: stored at the tracker's 344x616, rescale to the VGGT input size
        fx, fy, cx, cy = np.load(f'{scene_dir}/intrinsics.npy')
        probe = mono_stream_resize(cv2.imread(os.path.join(image_dir, self.files[0])))
        self.stream_hw = probe.shape[:2]
        sy, sx = VGGT_HW[0] / probe.shape[0], VGGT_HW[1] / probe.shape[1]
        self.K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], np.float64)

    def frame(self, t):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, self.files[t])), cv2.COLOR_BGR2RGB)
        img = cv2.resize(mono_stream_resize(img), (VGGT_HW[1], VGGT_HW[0]), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def kf_target(self, t):
        d = np.load(f'{self.scene_dir}/{self.ddir}/{t:06d}.npy')
        m = cv2.imread(f'{self.scene_dir}/{self.mdir}/{t:06d}.png', cv2.IMREAD_GRAYSCALE) > 127
        d = cv2.resize(d, (VGGT_HW[1], VGGT_HW[0]), interpolation=cv2.INTER_NEAREST)
        m = cv2.resize(m.astype(np.uint8), (VGGT_HW[1], VGGT_HW[0]), interpolation=cv2.INTER_NEAREST) > 0
        return torch.from_numpy(d).float(), torch.from_numpy(m & (d > 0))

    def neighbours(self, t, rng, n_left, n_right):
        """Random non-keyframe neighbours within RADIUS; edge keyframes take from the other side."""
        left = [x for x in range(max(t - RADIUS, self.t_min), t) if x in self.c2w]
        right = [x for x in range(t + 1, min(t + RADIUS, self.t_max) + 1) if x in self.c2w]
        want = n_left + n_right
        n_left, n_right = min(n_left, len(left)), min(n_right, len(right))
        # keyframes at the sequence ends have no frames on one side; make up the shortfall from
        # the other side so the requested context size is still met where the frames exist
        n_right = min(n_right + (want - n_left - n_right), len(right))
        n_left = min(n_left + (want - n_left - n_right), len(left))
        picks = list(rng.choice(left, n_left, replace=False)) + list(rng.choice(right, n_right, replace=False))
        return sorted(int(x) for x in picks)

    def sample(self, rng, t=None, single=None):
        t = int(rng.choice(self.kf)) if t is None else t
        if single is None:
            single = rng.random() < P_SINGLE_VIEW
        nb = [] if single else self.neighbours(t, rng, rng.integers(1, MAX_LEFT + 1),
                                               rng.integers(1, MAX_RIGHT + 1))
        seq = [t] + nb

        images = torch.stack([self.frame(x) for x in seq])
        gt_depth, mask = self.kf_target(t)

        # rebase every pose so the keyframe is the world origin -> frame 0 is identity
        kf_c2w = self.c2w[t]
        extr = np.stack([(np.linalg.inv(self.c2w[x]) @ kf_c2w)[:3] for x in seq])
        K = np.broadcast_to(self.K, (len(seq), 3, 3))
        gt_enc = extri_intri_to_pose_encoding(
            torch.from_numpy(extr).float()[None], torch.from_numpy(K.copy()).float()[None],
            image_size_hw=VGGT_HW)[0]
        return images, gt_depth, mask, gt_enc, seq


# ------------------------------------------------------------------ losses

def _median_scale(pred, gt, mask):
    """Median ratio. Deliberately NOT detached.

    Detaching makes the loss only *look* scale-invariant: the optimiser then sees a gradient that
    rewards shrinking the prediction, even though rescaling is a no-op once the scale is recomputed
    on the next forward pass. Letting the gradient flow through the ratio makes the loss genuinely
    invariant, so it can only push shape, never overall magnitude.
    """
    return gt[mask].median() / pred[mask].median().clamp(min=1e-6)


def depth_loss(pred_depth, gt_depth, mask, scale=None):
    """Returns (loss, depth-space scale). The returned scale is always expressed in DEPTH terms,
    even when the loss is computed in disparity, so callers can compare it against the pose scale."""
    if mask.sum() < 16:
        return pred_depth.sum() * 0.0, None
    p, g = pred_depth.clamp(min=1e-3), gt_depth.clamp(min=1e-3)
    inv = DEPTH_SPACE == 'disparity'
    if inv:
        p, g = 1.0 / p, 1.0 / g
    # a scale supplied by the caller (COUPLED_SCALE) is a depth scale; in disparity space the
    # equivalent multiplier is its reciprocal
    s = _median_scale(p, g, mask) if scale is None else (1.0 / scale if inv else scale)
    loss = (g[mask] - s * p[mask]).abs().mean()
    return loss, (1.0 / s if inv else s)


def pose_loss(pred_enc, gt_enc):
    """Translation (independently norm'd) + quaternion, over the non-reference frames."""
    if pred_enc.shape[0] < 2:
        z = pred_enc.sum() * 0.0
        return z, z, None
    tp, tg = pred_enc[1:, :3], gt_enc[1:, :3]
    # same reasoning as _median_scale: normalising by a *detached* predicted norm lets the
    # translations collapse toward zero at no loss cost. Keep the gradient in the normaliser.
    np_, ng = tp.norm(dim=-1).mean().clamp(min=1e-6), tg.norm(dim=-1).mean().clamp(min=1e-6)
    l_t = F.huber_loss(tp / np_, tg / ng)

    qp = F.normalize(pred_enc[1:, 3:7], dim=-1)
    qg = F.normalize(gt_enc[1:, 3:7], dim=-1)
    l_r = (1.0 - (qp * qg).sum(-1).abs()).mean()      # abs handles quaternion sign ambiguity
    return l_t, l_r, (ng / np_).detach()


# ------------------------------------------------------------------ model

def load_model(adapter=None):
    model = VGGT.from_pretrained(WEIGHTS)
    model.point_head, model.track_head = None, None       # not supervised
    for p in model.parameters():
        p.requires_grad_(False)
    n = inject_lora(model, RANK, ALPHA)
    if adapter is not None:
        missing = model.load_state_dict(load_file(adapter), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    return model.cuda(), n


def forward(model, images):
    """Aggregator once; depth head on frame 0 only; camera head on everything."""
    tok, ps_idx = model.aggregator(images[None])
    # this build caches only layers 4/11/17/23 and leaves the rest None to save memory
    # (aggregator.py:196) - the frame slice must preserve those Nones
    tok0 = [t[:, :1] if t is not None else None for t in tok]
    depth, _ = model.depth_head(tok0, images[None][:, :1], ps_idx)
    pose_enc = model.camera_head(tok)[-1]
    return depth[0, 0, :, :, 0], pose_enc[0]


# ------------------------------------------------------------------ eval

@torch.no_grad()
def eval_depth(model, data, single_view=True):
    """Scale-aligned masked depth L1 over every keyframe, same metric as export_slam_depth.py."""
    model.eval()
    rng = np.random.default_rng(SEED)
    errs = []
    for t in data.kf:
        images, gt, mask, _, _ = data.sample(rng, t=t, single=single_view)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred, _ = forward(model, images.cuda())
        l, _ = depth_loss(pred.float(), gt.cuda(), mask.cuda())
        errs.append(l.item())
    model.train()
    return float(np.mean(errs))


# ------------------------------------------------------------------ main

def main():
    global VGGT_HW
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='outputs/replica/room0_p40')
    ap.add_argument('--images', default='data/Replica/room0/colors')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--steps', type=int, default=STEPS_PER_EPOCH)
    ap.add_argument('--skip_eval', action='store_true')
    ap.add_argument('--vggt_hw', default=None, metavar='H,W',
                    help=f'VGGT input size, dims divisible by 14 (default {VGGT_HW[0]},{VGGT_HW[1]}). '
                         'Match the stream aspect ratio; TUM wants 378,518.')
    ap.add_argument('--depth_source', choices=('rendered', 'slam'), default='rendered',
                    help='which export_slam_depth.py target to supervise on (default rendered)')
    args = ap.parse_args()

    if args.vggt_hw:
        VGGT_HW = parse_hw(args.vggt_hw)

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    out = f'{args.scene}/lora-vggt'
    os.makedirs(out, exist_ok=True)

    data = SceneData(args.scene, args.images, args.depth_source)
    print(f"scene {args.scene}: {len(data.kf)} keyframes, frames {data.t_min}..{data.t_max}, "
          f"supervised on {data.ddir}/")
    sh, sw = data.stream_hw
    skew = (VGGT_HW[1] / VGGT_HW[0]) / (sw / sh)
    print(f"stream {sw}x{sh} (aspect {sw/sh:.3f}) -> VGGT {VGGT_HW[1]}x{VGGT_HW[0]} "
          f"(aspect {VGGT_HW[1]/VGGT_HW[0]:.3f}), squash {skew:.3f}x")
    if not 0.95 < skew < 1.05:
        print(f"  WARNING: aspect ratios differ by {abs(1-skew)*100:.0f}%. data.frame() resizes "
              f"without letterboxing, so VGGT sees a distorted image. Consider --vggt_hw "
              f"{14*round(518*sh/sw/14)},518")

    model, n_wrapped = load_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA r={RANK} on {n_wrapped} Linears -> {n_train/1e6:.2f}M trainable "
          f"/ {n_total/1e9:.2f}B ({100*n_train/n_total:.2f}%)")

    base_l1 = None
    if not args.skip_eval:
        base_l1 = eval_depth(model, data)
        print(f"base VGGT depth L1 (masked, scale-aligned, {DEPTH_SPACE}): {base_l1:.4f}")

    model.train()                        # enables the aggregator's gradient checkpointing
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
    log, t0 = [], time.time()

    for epoch in range(args.epochs):
        run = []
        for step in range(args.steps):
            images, gt, mask, gt_enc, seq = data.sample(rng)
            images, gt, mask, gt_enc = images.cuda(), gt.cuda(), mask.cuda(), gt_enc.cuda()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_depth, pred_enc = forward(model, images)
            pred_depth, pred_enc = pred_depth.float(), pred_enc.float()

            l_t, l_r, pose_scale = pose_loss(pred_enc, gt_enc)
            l_d, depth_scale = depth_loss(pred_depth, gt, mask,
                                          scale=pose_scale if COUPLED_SCALE else None)
            loss = l_d + LAMBDA_POSE * (l_t + l_r)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            rec = {'epoch': epoch, 'step': step, 'S': len(seq), 'loss': loss.item(),
                   'l_depth': l_d.item(), 'l_trans': l_t.item(), 'l_rot': l_r.item()}
            # depth and pose scale agreed to 1% on the pretrained model; if they diverge during
            # training the adapter is breaking depth/pose consistency
            if pose_scale is not None and depth_scale is not None:
                rec['scale_ratio'] = (depth_scale / pose_scale).item()
            log.append(rec)
            run.append(loss.item())

            if step % 32 == 0:
                print(f"  e{epoch} s{step:4d}  loss {np.mean(run[-32:]):.4f}  "
                      f"(d {l_d.item():.4f} t {l_t.item():.4f} r {l_r.item():.4f})  S={len(seq)}  "
                      f"{torch.cuda.max_memory_allocated()/2**30:.1f}GiB")
        print(f"epoch {epoch}: mean loss {np.mean(run):.4f}  ({time.time()-t0:.0f}s elapsed)")

    save_file(lora_state_dict(model), f'{out}/adapter.safetensors')
    cfg = {'rank': RANK, 'alpha': ALPHA, 'targets': list(LORA_TARGETS),
           'lora_patch_embed': LORA_PATCH_EMBED, 'epochs': args.epochs, 'steps': args.steps,
           'lr': LR, 'lambda_pose': LAMBDA_POSE, 'depth_space': DEPTH_SPACE,
           'depth_source': args.depth_source,
           'coupled_scale': COUPLED_SCALE, 'p_single_view': P_SINGLE_VIEW,
           'max_left': MAX_LEFT, 'max_right': MAX_RIGHT, 'radius': RADIUS,
           'vggt_hw': list(VGGT_HW), 'weights': WEIGHTS, 'scene': args.scene,
           'trainable_params': n_train, 'seed': SEED}
    json.dump(cfg, open(f'{out}/config.json', 'w'), indent=2)
    json.dump(log, open(f'{out}/train_log.json', 'w'))

    if not args.skip_eval:
        ad_l1 = eval_depth(model, data)
        print(f"\ndepth L1 (masked, scale-aligned, {DEPTH_SPACE}, {len(data.kf)} keyframes, "
              f"single-view, TRAIN SET):")
        print(f"  base VGGT    {base_l1:.4f}")
        print(f"  LoRA-adapted {ad_l1:.4f}   ({100*(base_l1-ad_l1)/base_l1:+.1f}%)")
        cfg['base_l1'], cfg['adapted_l1'] = base_l1, ad_l1
        json.dump(cfg, open(f'{out}/config.json', 'w'), indent=2)

    print(f"saved adapter ({n_train/1e6:.1f}M params) to {out}")


if __name__ == '__main__':
    main()
