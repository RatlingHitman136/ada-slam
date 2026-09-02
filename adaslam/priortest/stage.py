"""run_prior_test - score every depth-prior generator against GT, with no SLAM run.

Answers what the end2end null cannot (9.4): was the new prior no better, or was HI-SLAM2
insensitive to the way it is better? Minutes per arm, because nothing here tracks.
"""
import json
import os

from ..end2end.config import (OMNIDATA, OMNIDATA_DENSE, SENTINELS, VGGT_BASE, adapter_path,
                              arm_name, split_mods)
from ..end2end.prior import VggtPrior, mods_label, wrap_mods
from ..print_utils import banner
from ..runtime import free_vram

from .config import resolve_split
from .metrics import aggregate
from .predict import FRAMES, build_rows, read_cached, write_rows
from .report import compare, print_report

RESULTS = 'results.json'


def make_prior(spec, cfg):
    """The one place a prior spec becomes a depth prior. None = stock Omnidata.

    No stream_hw: the aspect warning belongs to a run that will feed BA. An '@ceil<tag>' (14) or
    '@ped<tag>' (14.9) modifier wraps the base prior, so a transformed generator scores here under
    the same arm name the end2end stage gives it.
    """
    base, mods = split_mods(spec)
    if base == OMNIDATA_DENSE:
        # Refused for the opposite reason to end2end's: there IS no tracking here, so keyframe
        # density has nothing to act on and this would score a byte-identical duplicate of `omni`
        # into a second directory - a row that looks like evidence and is not.
        raise SystemExit(
            f'{spec!r} has no meaning in the prior test: it differs from {SENTINELS[OMNIDATA]!r} '
            f'only in KEYFRAME DENSITY, and this test runs no SLAM at all (each frame goes through '
            f'PriorProbe). It would duplicate the omni row. Use {SENTINELS[OMNIDATA]!r} here; '
            f'{base!r} belongs in END2END_PRIORS.')
    if base == OMNIDATA:
        if not mods:
            return None
        from ..slam import stock_prior_extractor
        return wrap_mods(stock_prior_extractor(), mods,
                         f'Omnidata depth{mods_label(mods)} / Omnidata normals')
    inner = VggtPrior(cfg, None if base == VGGT_BASE else adapter_path(base))
    if not mods:
        return inner
    return wrap_mods(inner.extractor(), mods, f'{inner.label}{mods_label(mods)}',
                     release=inner.release)


def run_prior_test(slam_cfg, cfg, out_root, skip_existing=False):
    """Score every entry in cfg.priors into out_root/<arm>, then compare(). Returns the results.

    No `runner`: each prior goes through slam.PriorProbe, never a SLAM run.
    """
    banner(f'prior test  -> {out_root}')
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
            if hasattr(prior, 'chain'):
                # a modified arm's audit trail, same files the end2end stage writes
                for tr in prior.chain():
                    json.dump(tr.report(), open(f'{out}/{tr.STATS_FILE}', 'w'), indent=2)

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
