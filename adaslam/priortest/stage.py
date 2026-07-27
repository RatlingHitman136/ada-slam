"""run_prior_test - score every depth-prior generator against GT, with no SLAM run.

Answers the question the end2end test cannot: when swapping the prior changes nothing downstream
(ARCHITECTURE.md 9.4's null), was the new prior no better, or was HI-SLAM2 insensitive to the way
it is better? The three alignments in metrics.py separate those.

Minutes per arm rather than forty, because nothing here tracks.
"""
import json
import os

from ..end2end.config import SENTINELS, VGGT_BASE, adapter_path, arm_name
from ..end2end.prior import VggtPrior
from ..print_utils import banner
from ..runtime import free_vram

from .config import resolve_split
from .metrics import aggregate
from .predict import FRAMES, build_rows, read_cached, write_rows
from .report import compare, print_report

RESULTS = 'results.json'


def make_prior(spec, cfg):
    """The one place a prior spec becomes a depth prior. None = stock Omnidata.

    No stream_hw: the aspect warning belongs to a run that will feed BA. Here the probe resizes the
    same way MotionFilter does, so a skew would already have been reported by whichever arm ran.
    """
    if spec not in SENTINELS:
        return VggtPrior(cfg, adapter_path(spec))
    if spec == VGGT_BASE:
        return VggtPrior(cfg, None)
    return None


def run_prior_test(slam_cfg, cfg, out_root, skip_existing=False):
    """Score every entry in cfg.priors into out_root/<arm>, then compare(). Returns the results.

    No `runner` argument, and none needed: the prior is evaluated directly (slam/prior_probe.py),
    never through a SLAM run.
    """
    cfg.check_priors_exist()
    split_at, own = resolve_split(cfg.priors)
    spec_of_split = next((s for s in cfg.priors if own[s] is not None), None)
    print(f'seen/unseen split: {split_at}'
          + (f'  (from {arm_name(spec_of_split)})' if spec_of_split else '  - no adapter listed')
          + '\nevery arm is scored at that split, sentinels included: their seen/unseen rows are '
            'the control for "is the back of the sequence simply harder?"')

    results = []
    for spec in cfg.priors:
        name = arm_name(spec)
        out = f'{out_root}/{name}'
        frames_path = f'{out}/{FRAMES}'
        spec_dict = cfg.eval_spec()

        rows = read_cached(frames_path, spec_dict) if skip_existing else None
        if rows is not None:
            print(f'=== {name}: {FRAMES} matches the eval spec, reusing {len(rows)} frames '
                  f'(re-aggregating at split {split_at})')
            label = json.load(open(f'{out}/{RESULTS}'))['label'] \
                if os.path.exists(f'{out}/{RESULTS}') else name
        else:
            banner(f'prior arm {name}  ({spec})  -> {out}')
            prior = make_prior(spec, cfg)
            label = prior.label if prior else 'Omnidata depth (baseline)'
            try:
                rows, gscale = build_rows(slam_cfg, cfg, prior, name)
            finally:
                free_vram(f'prior arm {name}')
            print(f'  global scale fitted over all {len(rows)} frames: {gscale:.4f}')
            write_rows(frames_path, spec_dict, rows)

        res = {'arm': name, 'spec': spec, 'label': label, 'output': out,
               'split_at': split_at, 'own_split_at': own[spec],
               'split_mismatch': own[spec] is not None and own[spec] != split_at,
               'eval_spec': spec_dict, 'blocks': aggregate(rows, split_at)}
        json.dump(res, open(f'{out}/{RESULTS}', 'w'), indent=2, default=float)
        print_report(res)
        results.append(res)

    banner('prior comparison')
    compare(results)
    return results
