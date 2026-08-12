"""What a driver does AROUND the stages - the preamble every entry point repeats otherwise.

Not a stage and not a config: these are the checks and the one-off resolution that must happen
after chdir and before any Process is spawned or any GPU work starts. `scripts/*_pipeline.py` hold
the parameters and the dispatch; this holds what would otherwise be copied between them.
"""
import os

import numpy as np

from .common import probe_stream_hw


def enter(root):
    """Start-method and working directory, once per process, before anything else in main().

    'spawn' must be set before any Process is started; the chdir makes every relative path in a
    PARAMETERS block repo-root relative however the script was invoked. torch is imported here
    rather than at module scope so a report-only consumer of this module does not pay for it.
    """
    import torch
    torch.multiprocessing.set_start_method('spawn', force=True)
    os.chdir(root)


def check_sequence(colors, depths=None, gt_traj=None, required=()):
    """Every `required` path exists and the sequence is self-consistent. Returns the frame count.

    Every consumer indexes GT depth and GT poses by RGB frame number (10.1), so a sequence whose
    directories are not 1:1 by index produces silently misaligned numbers rather than an error.
    Checked here, before any GPU work.
    """
    for f in (*required, colors):
        if not os.path.exists(f):
            raise SystemExit(f'missing input: {f}')

    n_frames = len(os.listdir(colors))
    for name, path in (('depths', depths), ('traj', gt_traj)):
        if path is None:
            continue
        n = len(os.listdir(path)) if os.path.isdir(path) else len(np.loadtxt(path))
        if n != n_frames:
            raise SystemExit(f'{path} has {n} entries but {colors} has {n_frames}; they must be '
                             f'1:1 by index ({name}). Re-run the dataset preprocess script.')
    return n_frames


def warn_runtime_undistort(undistort, crop_border):
    """Undistorting in the reader misaligns every consumer that re-derives a frame (10.1)."""
    if undistort or crop_border:
        print('WARNING: undistorting at runtime - the extract accuracy table, the prior test and '
              'the LoRA data loader all re-derive the frame with a resize only, so predictions '
              'and GT will not line up (ARCHITECTURE.md 10.1)')


def resolve_lora(lora, colors, stream_res):
    """(the LoRAConfig with vggt_hw derived, the stream (H, W)).

    Here rather than in a PARAMETERS block: deriving reads a frame, which that block must not do,
    and it runs before main()'s chdir. Call after chdir, before any Process is spawned.
    """
    stream_hw = probe_stream_hw(colors, stream_res)
    return lora.resolved(stream_hw), stream_hw


def print_arm_dirs(stages, kinds):
    """Where each test arm will land, printed before a multi-hour run.

    `kinds` is (stage name, its config, its output root) per test kind; arm directories are
    inferred from the prior specs, never typed, so this is the only preview of them there is.
    """
    for kind, cfg, root in kinds:
        if kind in stages:
            for spec, d in cfg.arm_dirs(root).items():
                print(f'            {d:<58} <- {spec}')
