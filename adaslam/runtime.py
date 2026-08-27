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


def vram_free_mb():
    """(free, total) MiB off nvidia-smi, or None when it cannot be read."""
    r = sh('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits')
    try:
        used, total = (int(x) for x in r.stdout.splitlines()[0].replace(',', '').split())
    except (IndexError, ValueError):
        return None
    return total - used, total


def gpu_gate(min_free_mb, wait_min=0, poll_s=60):
    """Refuse to start when someone else's job already holds the card.

    `wait_min` > 0 WAITS for the card instead of refusing, polling every `poll_s` until enough is
    free or the budget runs out. On a shared box the card is usually free again within the hour,
    and a queued run that dies on arrival wastes the slot it was queued for - but a run that waits
    silently forever is worse, so the budget is finite and stated. 0 keeps the original behaviour.

    Only a STARTING gate. Nothing re-checks mid-run, so a neighbour that allocates after this
    returns will still OOM the run; the budget buys a clean start, not a reservation.
    """
    import time
    deadline = time.time() + wait_min * 60
    waited, polls = 0.0, 0
    while True:
        got = vram_free_mb()
        if got is None:
            # a transient nvidia-smi failure must not kill a long wait; a persistent one will
            # still fall out of the loop when the budget expires
            print('  [gpu] nvidia-smi unreadable, retrying')
        else:
            free, total = got
            if free >= min_free_mb:
                note = f'  (waited {waited/60:.1f} min)' if waited else ''
                print(f'GPU free  : {free} / {total} MiB{note}')
                return
            if time.time() >= deadline:
                # wait_min=0 never waited, so it keeps the original wording the other four
                # drivers' users know
                waited_note = ('' if not wait_min else
                               f' after waiting {waited/60:.1f} of {wait_min} min')
                raise SystemExit(
                    f'only {free} MiB VRAM free (need {min_free_mb}){waited_note}; another job '
                    f'is holding the card. '
                    + ('Lower MIN_FREE_VRAM_MB to override.' if not wait_min else
                       'Raise GPU_WAIT_MIN, or lower MIN_FREE_VRAM_MB to override.'))
            # every poll while waiting is noise in a multi-hour log; say it once, then every ~5 min
            if polls == 0 or (polls * poll_s) % 300 == 0:
                print(f'  [gpu] {free} / {total} MiB free, need {min_free_mb} - waiting '
                      f'(budget {wait_min} min, {waited/60:.1f} used)', flush=True)
        polls += 1
        time.sleep(poll_s)
        waited += poll_s
