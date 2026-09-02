"""Re-run one scored arm N times to measure the RUN-TO-RUN FLOOR (14.7).

    python scripts/replicate_arm.py -s kitti_00_fg2a05_f0-1000 -a base_ped1 \
        -c live_fg2a05_pedfine -n 3                                  # untrained, ~6 min each

    python scripts/replicate_arm.py -s kitti_00_fg2a05_f0-1000 -n 1 \
        --adapter live_fg2a05ped1p35_e5_w10_a16_w12_lag5_base        # live, ~95 min each

WHY THIS EXISTS. Every adapter effect this project has claimed is ~0.5 m, and the f0-1000 live
sweep spans 2.3 m of ATE across ratios whose drift correlation is not significant (Spearman -0.60,
p = 0.28). Until the spread of the SAME arm re-run is known, none of those numbers can be read as
an effect. The floor has been "still unmeasured (14.7)" in every run config header since; this is
the thing that closes it.

HOW. Same seed, same data, same config - the only source of spread is GPU/tracking
nondeterminism. Each replicate renames the existing arm out of the way to `<arm>_r<i>`, then
re-invokes the driver, whose SKIP_EXISTING rebuilds exactly the one directory that went missing
and reuses every other reference arm at zero cost. A live arm needs its ADAPTER moved aside too,
or the online stage is skipped and the replicate would re-score the same adapter.

`--adapter` also solves the config problem: a live arm's recipe may no longer exist in
run_configs/ (the ped files were edited in place as the sweep advanced), but the driver copied the
config it actually ran into the adapter directory. That copy is the source of truth, so it is
staged back into run_configs/ under the adapter's own name rather than trusting a hand-edited
file to still say what it said at the time.

Reads nothing it does not need: ATE comes from each arm's results.json and drift from
plot_trajectories.scale_drift (12.4's convention, GT metres per estimated unit over the last tenth
against the first), so this file defines no metric of its own.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from adaslam.common import ONLINE_ARM_SUFFIX, experiment_dir, test_dir   # noqa: E402
from adaslam.end2end.metrics import RESULTS, gt_traj_of                  # noqa: E402
from plot_trajectories import load_gt, scale_drift                       # noqa: E402

DRIVER = 'scripts/online_adapt_pipeline.py'
RUN_CONFIG_DIR = 'run_configs'


def replicate_dirs(base):
    """Every `<base>_r<i>` already on disk, so a second invocation continues the numbering."""
    head, name = os.path.split(base)
    return sorted(d for d in os.listdir(head) if d.startswith(f'{name}_r'))


def next_index(base):
    used = [int(d.rsplit('_r', 1)[1]) for d in replicate_dirs(base) if d.rsplit('_r', 1)[1].isdigit()]
    return max(used, default=0) + 1


def measure(out, gt_xyz):
    """(ATE, drift) for one scored arm directory - the two numbers a replicate contributes."""
    ate = json.load(open(f'{out}/{RESULTS}'))['ate_all']
    est = np.loadtxt(f'{out}/traj_full.txt')
    frames = est[:, 0].astype(int)
    keep = [i for i, f in enumerate(frames.tolist()) if f in gt_xyz]
    e = est[keep, 1:4]
    r = np.array([gt_xyz[frames[i]] for i in keep])
    s_first, s_last, _ = scale_drift(e, r, max(len(e) // 10, 3))
    return ate, s_last / s_first


def stage_config(adapter_dir, name):
    """Copy the config the driver actually ran back into run_configs/, and return its -c name.

    runconfig.py refuses a config whose name does not carry the driver's prefix, and every online
    experiment name already starts with 'live', so the adapter's own name is a legal -c name and
    records which arm the file replicates.
    """
    src = f'{adapter_dir}/run_config.yaml'
    if not os.path.exists(src):
        raise SystemExit(f'{src} not found - this adapter predates the config copy, so pass -c '
                         f'with the run config that produced it')
    dst = os.path.join(_ROOT, RUN_CONFIG_DIR, f'{name}.yaml')
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
        print(f'staged {src}\n    -> {dst}')
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-s', '--scene', required=True, help='scene key, e.g. kitti_00_fg2a05_f0-1000')
    ap.add_argument('-a', '--arm', help='arm directory to rebuild; implied by --adapter')
    ap.add_argument('-c', '--config', help='run config name; implied by --adapter')
    ap.add_argument('--adapter', help='online experiment to replicate - moves the adapter aside '
                                      'too, and takes its config and arm name from it')
    ap.add_argument('-n', '--repeats', type=int, default=3)
    ap.add_argument('-o', '--out-root', default='outputs')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and stop')
    args = ap.parse_args()

    os.chdir(_ROOT)
    arm = args.arm or (args.adapter and f'{args.adapter}{ONLINE_ARM_SUFFIX}')
    if not arm:
        raise SystemExit('pass -a ARM or --adapter NAME')
    arm_path = f'{test_dir(args.out_root, "end2end", args.scene)}/{arm}'
    if not os.path.isdir(arm_path):
        raise SystemExit(f'{arm_path} not found - replicate an arm that has already been scored')

    adapter_path = (experiment_dir(args.out_root, 'adapt', args.scene, args.adapter)
                    if args.adapter else None)
    if adapter_path and not os.path.isdir(adapter_path):
        raise SystemExit(f'{adapter_path} not found')
    if args.config:
        config = args.config
    elif args.adapter:                    # --dry-run must not write into run_configs/
        config = args.adapter if args.dry_run else stage_config(adapter_path, args.adapter)
    else:
        raise SystemExit('pass -c CONFIG (only --adapter can infer it)')

    gt_xyz = load_gt(gt_traj_of(arm_path))
    print(f'replicating {arm_path}')
    if adapter_path:
        print(f'        and {adapter_path}')
    print(f'via         {DRIVER} -c {config}   x{args.repeats}\n')

    for k in range(args.repeats):
        # nothing moves under --dry-run, so the disk cannot supply the increment
        i = next_index(arm_path) + (k if args.dry_run else 0)
        moves = [(arm_path, f'{arm_path}_r{i}')]
        if adapter_path:
            moves.append((adapter_path, f'{adapter_path}_r{i}'))
        for src, dst in moves:
            print(f'  [{k + 1}/{args.repeats}] mv {src} -> {dst}')
            if not args.dry_run:
                os.rename(src, dst)
        if args.dry_run:
            print(f'  [{k + 1}/{args.repeats}] would run {DRIVER} -c {config}')
            continue
        rc = subprocess.call([sys.executable, DRIVER, '-c', config])
        if rc != 0:
            raise SystemExit(f'{DRIVER} exited {rc} after {k} replicate(s); the arm that was '
                             f'moved to {arm_path}_r{i} still holds that draw')

    if args.dry_run:
        return

    draws = [(os.path.basename(d), measure(f'{os.path.dirname(arm_path)}/{d}', gt_xyz))
             for d in replicate_dirs(arm_path)] + [(arm, measure(arm_path, gt_xyz))]
    print(f'\n{len(draws)} draws of {arm}')
    for name, (ate, drift) in draws:
        print(f'  {name:<44} ATE {ate:7.3f}   drift {drift:5.3f}')
    a = np.array([d[1][0] for d in draws])
    v = np.array([d[1][1] for d in draws])
    print(f'\n  ATE   mean {a.mean():7.3f}  std {a.std(ddof=1) if len(a) > 1 else 0:6.3f}  '
          f'range {a.max() - a.min():6.3f}')
    print(f'  drift mean {v.mean():7.3f}  std {v.std(ddof=1) if len(v) > 1 else 0:6.3f}  '
          f'range {v.max() - v.min():6.3f}')
    print('\nAn ATE range at or above ~0.5 m means every adapter effect measured in this tree '
          '(+0.655, +0.698, ~+0.5 m) is inside the noise and cannot be read as an effect.')


if __name__ == '__main__':
    main()
