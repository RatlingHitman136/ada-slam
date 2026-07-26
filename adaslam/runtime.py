"""Process and shared-workstation hygiene: VRAM, subprocesses, fd limits, printing.

Nothing here is specific to a stage - it is what any driver of this repo needs on a box that
several people share, kept in one place so a stage package never has to grow its own.

The one cost free_vram() CANNOT reclaim is worth budgeting for. With `Tracking.pgba.active`
(true for TUM, false for Replica), Hi2 spawns the PGBA process and hands it the DepthVideo buffers
over CUDA IPC, then `terminate()`s it - abruptly, so the producer side never learns the blocks are
free and they stay pinned in IPC limbo for the life of the process. Measured on TUM at buffer=500:
1.29 GiB retained after one SLAM run and +1.26 GiB per run after that, versus 0.04 GiB flat with
pgba off. Only 0.03 GiB of it is reachable from Python, so no gc/empty_cache/ipc_collect call
touches it.

Consequence: extract + two arms strands ~3.8 GiB by the end. That fits alongside a VGGT arm's
~10 GiB peak on this 24 GB card, but if it does not on yours, run one STAGES entry per process -
a fresh process starts from zero.
"""
import contextlib
import gc
import os
import resource
import subprocess
import sys

import torch


# ---------------------------------------------------------------- subprocesses

def sh(cmd, **kw):
    """Shell out, capturing both streams. Used for evo_ape, tsdf_integrate.py, eval_recon.py."""
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def ensure_venv_on_path():
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')


def raise_fd_limit():
    resource.setrlimit(resource.RLIMIT_NOFILE,
                       (100000, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))


# ---------------------------------------------------------------- VRAM

def free_vram(tag=''):
    """Drop everything the finished stage held. In-process stages otherwise accumulate.

    See the module docstring for the part of it that cannot be reclaimed - which is what the
    'pgba IPC limbo' note below is reporting when it fires.
    """
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


# ---------------------------------------------------------------- printing

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def tee(path):
    """Print to stdout and to a file at once - export.txt is read back by other tooling."""
    with open(path, 'w') as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            yield


def banner(title):
    print(f'\n{"=" * 78}\n=== {title}\n{"=" * 78}')
