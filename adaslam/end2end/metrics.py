"""Scoring one arm: ATE, split seen/unseen (11).

Reads a finished output directory; nothing re-runs SLAM. evo writes the per-pose error array and
timestamps beside its RMSE, so the split is recoverable without re-running anything.
"""
import json
import os

import numpy as np

from ..runtime import sh

# One arm's scores, recording the split_at they were computed at - arms are reused, and a score
# split at the wrong frame is worse than no score.
RESULTS = 'results.json'


def load_ape(out):
    """The per-pose APE evo saved for a finished arm: (err, frame indices, GT distance).

    ONE definition of the evo/ layout - run_ate writes these files and this reads them back, as
    does scripts/ate_over_time.py. `err` is APE (translation, metres) AFTER the single Sim(3)
    alignment fitted over the whole trajectory, one value per frame of traj_full.txt;
    `dist` is the REFERENCE trajectory's cumulative path length, not the estimate's.
    """
    d = f'{out}/evo'
    if not os.path.exists(f'{d}/error_array.npy'):
        raise SystemExit(f'{d}/error_array.npy not found - {out} has not been scored yet '
                         f'(run the end2end stage, or delete the directory and re-run it)')
    return (np.load(f'{d}/error_array.npy'),
            np.load(f'{d}/timestamps.npy'),
            np.load(f'{d}/distances_from_start.npy'))


def run_ate(out, gt_traj):
    """evo with Sim3 alignment. Returns (overall rmse, per-frame errors, timestamps)."""
    sh(f'cd {out} && evo_ape tum {os.path.abspath(gt_traj)} traj_full.txt -vas '
       f'--save_results evo.zip --no_warnings > ape.txt 2>&1')
    sh(f'rm -rf {out}/evo && unzip -q {out}/evo.zip -d {out}/evo')
    err, ts, _ = load_ape(out)
    return float(np.sqrt((err ** 2).mean())), err, ts


def evaluate(out, label, split_at, cfg):
    """Every metric for one arm, written to out/results.json and returned. Cheap to re-run."""
    res = {'label': label, 'output': out, 'split_at': split_at}

    ate, err, ts = run_ate(out, cfg.gt_traj)
    res['ate_all'] = ate
    for name, sel in (('seen', ts < split_at), ('unseen', ts >= split_at)):
        res[f'ate_{name}'] = float(np.sqrt((err[sel] ** 2).mean())) if sel.sum() else None
        res[f'n_{name}'] = int(sel.sum())
    # two arms that scored a different pose count are not comparable, however close the RMSEs look
    res['n_all'] = int(len(ts))

    json.dump(res, open(f'{out}/{RESULTS}', 'w'), indent=2, default=float)
    return res
