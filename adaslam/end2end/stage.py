"""run_end2end_test - one full-sequence run per depth-prior generator, then the comparison table.

The arms differ in ONE thing: the depth prior. Everything else - the stream, the tracking config,
the buffer, the metric code - is shared, which is what makes a delta between them mean anything.

Each arm is a standalone, reusable unit living in a directory named after the adapter it uses
(config.py:arm_name), so a scene's omnidata baseline is run once and every later comparison finds
it rather than repeating it.
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

    Two things are cached separately because they cost differently. The SLAM run is the expensive
    one, keyed on the arm directory holding a finished trajectory. Scoring is cheap - one evo_ape
    over a trajectory already on disk - and is keyed on split_at too, because arms are reused
    across comparisons and every adapter has its own training fraction, so one comparison's split
    is not another's. Re-scoring at a new split therefore costs no SLAM run.
    """
    path = f'{out}/{RESULTS}'
    if not os.path.exists(path):
        return None
    res = json.load(open(path))
    return res if res.get('split_at') == split_at else None


def run_end2end_test(runner, slam_cfg, cfg, out_root, arm_config, split_at, skip_existing=False):
    """One arm per entry in cfg.priors, then compare(). Returns the per-arm result dicts.

    `out_root` is outputs/test/end2end/<scene>; each arm gets a subdirectory of it whose name is
    inferred from its adapter. There is deliberately no `adapter` parameter - each arm carries its
    own, which is what lets one comparison hold several adapters and their checkpoints.
    `arm_config` is the base tracking YAML, identical for every arm - the caller asserts it is not
    the extract run's derived one, where both paths are visible side by side.
    """
    # Here rather than in End2EndConfig.__post_init__: the config is built in a block that must not
    # touch the filesystem and runs before main()'s chdir, and an adapter listed there legitimately
    # does not exist yet when the whole pipeline runs. By now the adapt stage has made it.
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
                # SlamRunner installs the prior and restores the stock one in a finally, so no arm
                # can inherit the previous arm's patch and quietly become a second VGGT arm. That
                # matters more now the arm list is arbitrary rather than three fixed names.
                # gtdepthdir stays None: Hi2 reads it only inside the eval_rendering guard, and
                # nothing scores renders any more (SlamConfig.render_eval)
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
