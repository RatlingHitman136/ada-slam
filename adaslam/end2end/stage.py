"""run_end2end_test - one full-sequence run per depth-prior generator, then the comparison table.

The arms differ in ONE thing, the depth prior, which is what makes a delta mean anything. Each is
reusable: its directory is named after the adapter it uses, so a baseline is run once.
"""
import json
import os
import time

from ..common import probe_stream_hw
from ..print_utils import banner
from ..runtime import free_vram

from .config import SENTINELS, VGGT_BASE, adapter_path, arm_name
from .metrics import RESULTS, evaluate
from .prior import VggtPrior
from .report import compare, print_report


def make_prior(spec, cfg, stream_hw=None):
    """The one place a prior spec becomes a depth prior. None = stock Omnidata."""
    if spec not in SENTINELS:
        return VggtPrior(cfg, adapter_path(spec), stream_hw)
    if spec == VGGT_BASE:
        return VggtPrior(cfg, None, stream_hw)      # stock VGGT-1B, no adapter
    return None                                     # 'omnidata': upstream's own prior


def cached_results(out, split_at):
    """That arm's results if it already has some at THIS split, else None.

    The SLAM run and the scoring cache separately: arms are reused across comparisons, and
    re-scoring at another split is one evo_ape over a trajectory already on disk.
    """
    path = f'{out}/{RESULTS}'
    if not os.path.exists(path):
        return None
    res = json.load(open(path))
    return res if res.get('split_at') == split_at else None


def run_end2end_test(runner, slam_cfg, cfg, out_root, arm_config, split_at, skip_existing=False):
    """One arm per entry in cfg.priors, then compare(). Returns the per-arm result dicts.

    No `adapter` parameter: each arm carries its own, which is what lets one comparison hold
    several adapters and their checkpoints. `arm_config` is the base tracking YAML, identical for
    every arm.
    """
    banner(f'end2end  -> {out_root}')
    print(f'tracking config for every arm: {arm_config} (unmodified; an extract stage\'s kf_* '
          f'knobs apply to that run only)')

    # here, not in __post_init__: that runs before chdir and must not touch the filesystem, and
    # these adapters legitimately do not exist yet when the whole pipeline runs
    cfg.check_priors_exist()

    # probed once, not per arm: it only depends on the stream, and every arm shares that
    stream_hw = probe_stream_hw(slam_cfg.colors, slam_cfg.stream_res)

    labels, results = [], []
    for spec in cfg.priors:
        name = arm_name(spec)
        out = f'{out_root}/{name}'
        labels.append(name)

        if skip_existing:
            res = cached_results(out, split_at)
            if res is not None:
                print(f'=== {name}: {RESULTS} at split_at={split_at} exists, skipping '
                      f'({res["label"]})')
                results.append(res)
                continue

        banner(f'arm {name}  ({spec})  -> {out}')

        if skip_existing and os.path.exists(f'{out}/traj_full.txt'):
            # the SLAM run is done; only the scoring is absent or was computed at another split
            print(f'{out}/traj_full.txt exists - reusing the SLAM run, re-scoring at '
                  f'split_at={split_at}')
            old = f'{out}/{RESULTS}'
            label = json.load(open(old))['label'] if os.path.exists(old) else name
        else:
            prior = make_prior(spec, cfg, stream_hw)
            label = prior.label if prior else 'Omnidata depth (baseline)'
            try:
                t0 = time.time()
                # SlamRunner installs and restores the prior, so no arm inherits another's.
                # gtdepthdir stays None - nothing scores renders any more.
                n_kf = runner.run(out, arm_config, cfg.length, cfg.buffer,
                                  gtdepthdir=None, prior=prior).n_kf
                print(f'{label}: SLAM done in {time.time()-t0:.0f}s, {n_kf} keyframes')
            finally:
                if prior is not None:
                    prior.release()      # in a finally: a crashed arm otherwise strands ~2.5 GB
            free_vram(f'arm {name}')

        res = evaluate(out, label, split_at, cfg)
        print_report(res)
        results.append(res)
        free_vram(f'arm {name} eval')

    banner('comparison')
    compare(labels, results)
    return results
