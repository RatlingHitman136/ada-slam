"""Process and shared-workstation hygiene: VRAM, subprocesses, fd limits.

What free_vram() cannot reclaim, and why (8): with Tracking.pgba.active, Hi2 hands the DepthVideo
buffers to the PGBA child over CUDA IPC then terminate()s it, so those blocks stay pinned for the
life of the process. Measured on TUM at buffer=500: 1.29 GiB after one run, +1.26 GiB per run
after, versus 0.04 GiB flat with pgba off; only 0.03 GiB of it is reachable from Python. Run one
STAGES entry per process if that does not fit alongside a VGGT arm's ~10 GiB peak.
"""
import gc
import os
import resource
import subprocess
import sys

import torch


# ---------------------------------------------------------------- subprocesses

def sh(cmd, **kw):
    """Shell out, capturing both streams. Used for evo_ape, which is the only CLI left."""
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def ensure_venv_on_path():
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')


def raise_fd_limit():
    resource.setrlimit(resource.RLIMIT_NOFILE,
                       (100000, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))


# ---------------------------------------------------------------- VRAM

def free_vram(tag=''):
    """Drop everything the finished stage held. In-process stages otherwise accumulate."""
    gc.collect()
    torch.cuda.ipc_collect()      # reclaims blocks whose consumer *did* exit cleanly
    torch.cuda.empty_cache()
    if tag:
        held = torch.cuda.memory_allocated() / 2**30
        note = '  <- pgba IPC limbo, not reclaimable in-process' if held > 1.0 else ''
        print(f'  [vram] after {tag}: {held:.2f} GiB allocated, '
              f'{torch.cuda.memory_reserved()/2**30:.2f} GiB reserved{note}')


def gpu_gate(min_free_mb):
    """Refuse to start when someone else's job already holds the card."""
    r = sh('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits')
    used, total = (int(x) for x in r.stdout.splitlines()[0].replace(',', '').split())
    if total - used < min_free_mb:
        raise SystemExit(f'only {total - used} MiB VRAM free (need {min_free_mb}); another '
                         f'job is running. Lower MIN_FREE_VRAM_MB to override.')
    print(f'GPU free  : {total - used} / {total} MiB')
