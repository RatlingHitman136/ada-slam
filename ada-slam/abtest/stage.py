"""run_ab_test - one full-sequence run per arm, then the comparison table.

The arms differ in ONE thing: the depth prior. Everything else - the stream, the tracking config,
the buffer, the metric code - is shared, which is what makes a delta between them mean anything.
"""
import json
import os
import time

from runtime import banner, free_vram

from .config import ARM_NAMES
from .metrics import evaluate
from .prior import VggtPrior
from .report import compare, print_report


def make_prior(arm, cfg, adapter):
    """The one place an arm name becomes a depth prior. None = stock Omnidata."""
    if arm == 'omnidata':
        return None
    if arm == 'vggt_lora':
        if not os.path.exists(adapter):
            raise SystemExit(f'no adapter at {adapter} - run the adapt stage first')
        return VggtPrior(cfg, adapter)
    return VggtPrior(cfg, None)          # 'vggt_base': stock VGGT-1B, no adapter


def run_ab_test(runner, slam_cfg, cfg, arm_config, adapter, split_at, skip_existing=False):
    """One arm per entry in cfg.arms, then compare(). Returns the per-arm result dicts.

    `arm_config` is the base tracking YAML, identical for every arm - the caller asserts it is not
    the extract run's derived one, where both paths are visible side by side.
    """
    labels, results = [], []
    for arm in cfg.arms:
        out = cfg.out_dir(arm)
        labels.append(ARM_NAMES[arm])
        if skip_existing and os.path.exists(f'{out}/ab_results.json'):
            res = json.load(open(f'{out}/ab_results.json'))
            print(f'=== {arm}: ab_results.json exists, skipping ({res["label"]})')
            results.append(res)
            continue

        banner(f'arm {arm} -> {out}')
        prior = make_prior(arm, cfg, adapter)
        label = prior.label if prior else 'Omnidata depth (baseline)'
        try:
            t0 = time.time()
            # SlamRunner installs the prior and restores the stock one in a finally, so no arm can
            # inherit the previous arm's patch and quietly become a second VGGT arm
            n_kf = runner.run(out, arm_config, cfg.length, cfg.buffer,
                              gtdepthdir=cfg.gt_depths, prior=prior).n_kf
            print(f'{label}: SLAM done in {time.time()-t0:.0f}s, {n_kf} keyframes')
        finally:
            if prior is not None:
                prior.release()      # in a finally: a crashed arm otherwise strands ~2.5 GB

        free_vram(f'arm {arm}')
        res = evaluate(out, label, split_at, slam_cfg, cfg)
        print_report(res)
        results.append(res)
        free_vram(f'arm {arm} eval')

    banner('comparison')
    compare(labels, results)
    return results
