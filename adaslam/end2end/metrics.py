"""Scoring one arm: ATE, split seen/unseen.

Everything here reads an arm's finished output directory; nothing re-runs SLAM. The split is the
point of the whole harness - evo reports one RMSE over the sequence, but it also writes the
per-pose error array and the timestamps beside it, so the seen/unseen breakdown is recoverable
without re-running anything.

ATE is the only metric left. The render metrics that used to sit here - PSNR, SSIM, depth L1 from
renders/, and the TSDF mesh accuracy/completion from tsdf_integrate.py + eval_recon.py - went with
the terminate-time render (SlamConfig.render_eval) when the project target narrowed to pose
estimation. tsdf_integrate.py and scripts/eval_recon.py are still in the repo, unreached from here.
"""
import json
import os

import numpy as np

from ..runtime import sh

# One arm's scores. It records the split_at it was computed at, because arms are reused across
# comparisons and a score split at the wrong frame is worse than no score (stage.py:cached_results).
RESULTS = 'results.json'


def run_ate(out, gt_traj):
    """evo with Sim3 alignment. Returns (overall rmse, per-frame errors, timestamps)."""
    sh(f'cd {out} && evo_ape tum {os.path.abspath(gt_traj)} traj_full.txt -vas '
       f'--save_results evo.zip --no_warnings > ape.txt 2>&1')
    sh(f'rm -rf {out}/evo && unzip -q {out}/evo.zip -d {out}/evo')
    err = np.load(f'{out}/evo/error_array.npy')
    ts = np.load(f'{out}/evo/timestamps.npy')
    return float(np.sqrt((err ** 2).mean())), err, ts


def evaluate(out, label, split_at, cfg):
    """Every metric for one arm, written to out/results.json and returned.

    Reads a finished arm directory and re-runs nothing expensive, so re-scoring an existing arm at
    a different split_at is cheap - which is what makes arms reusable across comparisons.
    """
    res = {'label': label, 'output': out, 'split_at': split_at}

    ate, err, ts = run_ate(out, cfg.gt_traj)
    res['ate_all'] = ate
    for name, sel in (('seen', ts < split_at), ('unseen', ts >= split_at)):
        res[f'ate_{name}'] = float(np.sqrt((err[sel] ** 2).mean())) if sel.sum() else None
        res[f'n_{name}'] = int(sel.sum())
    # how many poses each number was averaged over: two arms that scored a different count are not
    # comparable however close their RMSEs look, and this is the only place that is now visible
    res['n_all'] = int(len(ts))

    json.dump(res, open(f'{out}/{RESULTS}', 'w'), indent=2, default=float)
    return res
