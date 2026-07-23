"""LoRA-adapt VGGT on HI-SLAM2's own depth + poses - the adapt stage, on its own.

    python scripts/lora_adapt_vggt.py --scene outputs/tum/rgbd_dataset_freiburg1_room_p40 \
                                      --images data/TUM/rgbd_dataset_freiburg1_room/colors

A thin CLI over ada-slam/adapt/: it builds the two config dataclasses and calls
LoRAVGGT.train(), which is exactly what scripts/run_pipeline.py's adapt stage does - the same
code, reached a different way. Use this when a scene has already been extracted and only the
adaptation needs re-running (a sweep over rank, or depth space, or the split); use
run_pipeline.py for the whole extract -> adapt -> test experiment.

--scene must be an extract stage's output directory: depth_<src>/, mask_<src>/, poses_slam.txt,
traj_full.txt and intrinsics.npy (ARCHITECTURE.md §7). Every flag below defaults to the value
run_pipeline.py currently uses, so an unadorned run reproduces its adapt stage.
"""
import os    # nopep8
import sys   # nopep8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # nopep8
sys.path.insert(0, os.path.join(_ROOT, 'ada-slam'))                   # nopep8
import argparse

from adapt import AdaptConfig, LoRAConfig, LoRAVGGT


def parse_hw(s):
    """'H,W' or 'HxW' -> (H, W). LoRAConfig checks it against VGGT's 14x14 patch grid."""
    h, w = (int(x) for x in s.replace('x', ',').split(','))
    return h, w


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = ap.add_argument_group('paths')
    g.add_argument('--scene', required=True, help="an extract stage's output directory")
    g.add_argument('--images', required=True, help='the sequence colours, e.g. data/.../colors')
    g.add_argument('--out', default=None, help='adapter directory (default <scene>/lora-vggt)')
    g.add_argument('--weights', default='pretrained_models/vggt', help='local VGGT-1B snapshot')

    g = ap.add_argument_group('adapter structure (recorded into the adapter, read back at test)')
    g.add_argument('--vggt_hw', type=parse_hw, default=(378, 518), metavar='H,W',
                   help='VGGT input size, dims %%14. MUST match the stream aspect: Replica wants '
                        '294,518 and TUM 378,518')
    g.add_argument('--rank', type=int, default=8)
    g.add_argument('--alpha', type=int, default=16)
    g.add_argument('--targets', nargs='+',
                   default=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
                   help='Linear leaves to wrap in every aggregator block')
    g.add_argument('--patch_embed', action='store_true',
                   help='also adapt patch_embed (default: only the attention stack)')

    g = ap.add_argument_group('data')
    g.add_argument('--depth_source', choices=('slam', 'rendered'), default='slam',
                   help='which export target supervises: depth_<src>/ + mask_<src>/')
    g.add_argument('--stream_res', type=int, default=341 * 640,
                   help='tracking resolution budget the export was produced at')
    g.add_argument('--p_single_view', type=float, default=1,
                   help='0 = always multi-view, 1 = always monocular')
    g.add_argument('--max_left', type=int, default=4)
    g.add_argument('--max_right', type=int, default=4)
    g.add_argument('--radius', type=int, default=8, help='neighbour search radius, in frames')

    g = ap.add_argument_group('optimisation')
    g.add_argument('--epochs', type=int, default=10)
    g.add_argument('--batch_size', type=int, default=2)
    g.add_argument('--lr', type=float, default=1e-4)
    g.add_argument('--weight_decay', type=float, default=0.0)
    g.add_argument('--grad_clip', type=float, default=1.0)
    g.add_argument('--lambda_pose', type=float, default=1.0)
    g.add_argument('--depth_space', choices=('depth', 'disparity'), default='disparity')
    g.add_argument('--no_coupled_scale', action='store_true',
                   help='fit the depth scale independently of the pose scale')
    g.add_argument('--min_mask_pixels', type=int, default=16)
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--log_every', type=int, default=20)

    g = ap.add_argument_group('train/val split and evaluation')
    g.add_argument('--train_frac', type=float, default=0.8, help='1.0 = no val set')
    g.add_argument('--split_mode', choices=('stride', 'contiguous', 'random'), default='stride')
    g.add_argument('--no_eval_train', action='store_true')
    g.add_argument('--no_eval_val', action='store_true')
    g.add_argument('--eval_last_only', action='store_true',
                   help='evaluate before training and after the last epoch only')
    g.add_argument('--eval_max_kf', type=int, default=100, help='0 = no cap')
    g.add_argument('--keep_best', action='store_true',
                   help='save the best-val epoch instead of the last')
    return ap


def main():
    a = build_parser().parse_args()

    # relative paths are repo-root relative, as everywhere else in this repo
    scene, images = os.path.abspath(a.scene), os.path.abspath(a.images)
    out = os.path.abspath(a.out) if a.out else f'{scene}/lora-vggt'
    os.chdir(_ROOT)

    lora_cfg = LoRAConfig(weights=a.weights, vggt_hw=a.vggt_hw, rank=a.rank, alpha=a.alpha,
                          targets=tuple(a.targets), patch_embed=a.patch_embed)
    cfg = AdaptConfig(
        depth_source=a.depth_source, stream_res=a.stream_res, p_single_view=a.p_single_view,
        max_left=a.max_left, max_right=a.max_right, radius=a.radius,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay,
        grad_clip=a.grad_clip, lambda_pose=a.lambda_pose, depth_space=a.depth_space,
        coupled_scale=not a.no_coupled_scale, min_mask_pixels=a.min_mask_pixels, seed=a.seed,
        log_every=a.log_every,
        train_frac=a.train_frac, split_mode=a.split_mode,
        eval_on_train=not a.no_eval_train, eval_on_val=not a.no_eval_val,
        eval_every_epoch=not a.eval_last_only, eval_max_kf=a.eval_max_kf, keep_best=a.keep_best)

    # the seed goes to the CONSTRUCTOR: the adapter's A matrices are initialised when LoRA is
    # injected, so seeding any later does not reproduce a run
    lora = LoRAVGGT(lora_cfg, seed=cfg.seed)
    lora.train(scene, images, out, cfg)


if __name__ == '__main__':
    main()
