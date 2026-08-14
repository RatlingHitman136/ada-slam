"""Scoring one arm: ATE, split seen/unseen (11).

Reads a finished output directory; nothing re-runs SLAM. evo writes the per-pose error array and
timestamps beside its RMSE, so the split is recoverable without re-running anything.
"""
import json
import os

import numpy as np

from ..common import ONLINE_ARM_SUFFIX, test_dir
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


def load_alignment(out):
    """The Sim(3) evo fitted for a finished arm, as a 4x4 with the SCALE BAKED IN.

    Same layout contract as load_ape - run_ate writes it, this reads it back. Applying it to
    traj_full.txt's translations puts the estimate in the GT frame, and the residual there
    equals error_array.npy to ~6e-14, so anything drawn through it is positioned by the same
    transform the numbers in results.json were measured under.
    """
    p = f'{out}/evo/alignment_transformation_sim3.npy'
    if not os.path.exists(p):
        raise SystemExit(f'{p} not found - {out} was not scored with alignment (run_ate passes '
                         f'evo -vas, which saves it), so its trajectory cannot be put in the GT '
                         f'frame')
    return np.load(p)


def gt_traj_of(out):
    """The GT trajectory `out` was scored against - the absolute path evo recorded at the time.

    A scene id does not determine its dataset directory (`rellis_00000` -> `data/RELLIS/00000`,
    but the TUM scene is named after its own directory), and only each driver's PARAMETERS block
    knows the mapping. The arm itself does, though: evo wrote the reference it was given.
    """
    p = f'{out}/evo/info.json'
    if not os.path.exists(p):
        raise SystemExit(f'{p} not found - {out} has not been scored yet (run the end2end '
                         f'stage), so the GT it should be compared against is not recorded')
    return json.load(open(p))['ref_name']


def no_such_arm(scene, arm, have):
    """The message for a name that is not an arm directory. A scene accumulates dozens of arms,
    so dumping all of them buries the answer - lead with what the name probably meant."""
    lines = [f'no arm {arm!r} in scene {scene!r}.']

    # The one STRUCTURAL near-miss, and the likeliest: an online run (13) names its ADAPTER
    # <name> and its arm <name>_live, so the adapt directory's name is not an arm.
    if f'{arm}{ONLINE_ARM_SUFFIX}' in have:
        lines.append(f'  Did you mean {arm}{ONLINE_ARM_SUFFIX}?')
        lines.append(f'  An online run (13) names its ADAPTER {arm} (under outputs/adapt/) and its '
                     f'ARM {arm}{ONLINE_ARM_SUFFIX}')
        lines.append('  (under outputs/test/end2end/). The trajectory this reads is the second.')
    else:
        near = [h for h in have if arm in h or h in arm] or \
               [h for h in have if h.split('_')[0] == arm.split('_')[0]]
        if near:
            lines.append(f'  Closest: {" ".join(near[:8])}')
        else:
            lines.append(f'  This scene has {len(have)} arms: {" ".join(have)}')
    return '\n'.join(lines)


def arm_dir(root, scene, arm):
    """`<root>/test/end2end/<scene>/<arm>`, checked - the one place a typed arm id is resolved.

    Every read-only view over a scored arm starts here, so the near-miss message above is
    written once and both scripts/ate_over_time.py and scripts/plot_trajectories.py get it.
    """
    out = f'{test_dir(root, "end2end", scene)}/{arm}'
    if not os.path.isdir(out):
        raise SystemExit(no_such_arm(scene, arm,
                                     sorted(os.listdir(test_dir(root, 'end2end', scene)))))
    return out


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
