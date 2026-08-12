"""run_adapt - stage 2 end to end: check the export, build the model, train, save, release.

The order matters: save() goes through _ensure_live(), so release() is last. Training itself
writes no adapter (trainer.py returns `state` and `run`, which are save()'s arguments), so where
the adapter lands is decided here, at the call site.
"""
import os
import time

import numpy as np

from ..common import ADAPTER_FILE, DEPTH_DIR
from ..print_utils import banner
from ..runtime import free_vram

from .model import LoRAVGGT


def init_adapter_path(init_dir):
    """The adapter file a run STARTS from. None = stock VGGT-1B.

    `init_dir` is an adapt handoff DIRECTORY (or one of its checkpoints), the same thing an
    end2end arm names - so continuing from a previous run and testing it are spelled alike.
    """
    if init_dir is None:
        return None
    path = f'{str(init_dir).rstrip("/")}/{ADAPTER_FILE}'
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found - init_adapter must be an adapt stage handoff '
                         f'directory (or one of its checkpoints), not {init_dir!r}')
    return path


def check_export(in_dir, image_dir):
    """That `in_dir` is an extract export and `image_dir` is the sequence it was produced from.

    Here rather than in SceneData: the two are free to be any pair, and a wrong one otherwise dies
    deep inside the first sample - or worse, silently returns the wrong image.
    """
    for f in (f'{in_dir}/poses_slam.txt', f'{in_dir}/traj_full.txt', f'{in_dir}/intrinsics.npy',
              f'{in_dir}/{DEPTH_DIR}', image_dir):
        if not os.path.exists(f):
            raise SystemExit(f'adapt input missing: {f}   (in_dir={in_dir} must be an extract '
                             f"stage's export directory)")
    # SceneData indexes image_dir by frame number, so a mismatched pair is an IndexError
    last_kf = int(np.loadtxt(f'{in_dir}/poses_slam.txt')[:, 0].max())
    n_img = len(os.listdir(image_dir))
    if last_kf >= n_img:
        raise SystemExit(f'{in_dir} has keyframe {last_kf} but {image_dir} holds {n_img} frames; '
                         f'the export and the images must be the same sequence.')


def run_adapt(lora_cfg, cfg, in_dir, image_dir, out_dir, ckpt_dir,
              init_adapter=None, skip_existing=False):
    """LoRA-adapt on `in_dir`'s export of `image_dir`, into `out_dir`. Returns the adapter path.

    Every path is an argument and none is read out of a global, so this can be pointed at any
    earlier extract's export and write the adapter anywhere. `init_adapter` is the adapt directory
    to CONTINUE from, or None to start from stock VGGT-1B.
    """
    adapter = f'{out_dir}/{ADAPTER_FILE}'
    banner(f'adapt  {in_dir} -> {adapter}')
    if skip_existing and os.path.exists(adapter):
        print(f'{adapter} exists - skipping')
        return adapter

    check_export(in_dir, image_dir)
    init = init_adapter_path(init_adapter)
    print(f'starting from {init if init else "stock VGGT-1B (no adapter)"}')

    t0 = time.time()
    # ONE call for both starts: from_adapter(None, ...) is a plain stock build. The seed reaches
    # the CONSTRUCTOR because the A matrices are initialised when LoRA is injected.
    lora = LoRAVGGT.from_adapter(init, lora_cfg, seed=cfg.seed)
    summary = lora.train(in_dir, image_dir, out_dir, cfg, ckpt_dir=ckpt_dir)
    # the one save - and before release(), which invalidates it
    print(f'saved adapter to {lora.save(out_dir, state=summary["state"], extra=summary["run"])}')
    lora.release()
    free_vram('adapt')
    print(f'=== adapt done in {time.time()-t0:.0f}s')
    return adapter
