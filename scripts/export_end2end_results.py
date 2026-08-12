"""One scene's end2end arms as a CSV, for a Notion database (12).

    python scripts/export_end2end_results.py -n "aug10 rellis" -s rellis_00000
    -> outputs/aug10 rellis.csv

Read-only over outputs/: it joins each arm's results.json to the adapt config.json behind it and
the extract that config trained on. Regenerable at any time, never a source of truth.

Two things it deliberately does NOT do:

  * parse the arm name. Every parameter is recorded in the adapter's config.json, and the names
    lie - wonline_r8_e5_w20_p10 records epochs: 3. The name is a label, config.json is the data.
  * report ate_seen / ate_unseen. Each arm's results.json computed them at whatever FRACTION was
    current when it ran (omni at 284, normal_r8_e20 at 1138), so they are not comparable across
    arms. ate_all is split-independent and is the metric this table exists for.

TIME. Wall clock is not comparable on a shared GPU - the four runs that recorded train_seconds
disagree 2x per unit of work. `adapt_cost` replaces it: how many times ANY keyframe was pushed
through VGGT, computed from the schedule in adapt/trainer.py.

    normal   n_train_kf * epochs
    online   n_train_kf * epochs
    wonline  (n_train_kf - window + 1) * epochs * window

A CHECKPOINT arm (<aname>_chkp_NNN) is charged only the work done up to its snapshot. Its
`train_frames` is NOT reduced to match: under 'normal' the adapter did see the whole window, just
fewer times, and under 'online'/'wonline' the keyframe list the run actually reached is not
recorded anywhere - so on those two a checkpoint's train_frames is the run's window, not its own.

NOTION. One database per scene (`arm` is only unique within one).

  1. first time   /database -> Import -> CSV. Then set the property types once: `arm` stays
                  Title, `style` and `extract` -> Select, every numeric column -> Number,
                  `exported_at` -> Date. Merges preserve types, so this is a one-off.
  2. after that   database ... menu -> Merge with CSV. Notion matches rows on the Title, so `arm`
                  makes a re-export UPDATE rows in place instead of duplicating them; new arms
                  are appended.

  Views worth making: Table filtered `ate_all is not empty`, grouped by `style`, sorted
  `adapt_cost` ascending (where more adaptation stops buying ATE); a second grouped by
  `train_pct`, sorted `ate_all` ascending (the best result each frame budget can reach); and a
  small one filtered to `arm is omni or base`.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)          # repo root, so `adaslam` imports

from adaslam.common import ADAPT_CKPT_SUBDIR, experiment_dir, test_dir   # noqa: E402
from adaslam.end2end.config import SENTINELS, arm_name                   # noqa: E402

# arm_name is IMPORTED rather than reimplemented: the pipeline infers an arm's directory from its
# prior spec (7.1), and a second naming rule here would join the wrong rows the day one changes.
OMNI_ARM = SENTINELS['omnidata']

RESULTS, CONFIG = 'results.json', 'config.json'

# `arm` first: Notion's Merge with CSV matches rows on the Title property, which is column 0.
COLUMNS = ('arm', 'style', 'epochs', 'window', 'lr',
           'train_pct', 'train_frames', 'n_train_kf', 'adapt_cost', 'train_seconds',
           'extract', 'extract_kf',
           'ate_all', 'd_ate_vs_omni', 'd_ate_pct', 'exported_at')


def adapt_cost(cfg):
    """How many times any keyframe is pushed through VGGT - adapt/trainer.py:schedule, counted.

    The one cross-style comparable measure of how much adapting a run did, and the reason there is
    no wall-clock column. Counted as units_done * visits_per_unit rather than from `epochs`
    directly, because a CHECKPOINT records the whole run's `epochs` and stopped at `saved_epoch`:

        normal   a unit is an epoch            n_train_kf visits    of `epochs` units
        online   a unit is an arriving kf      epochs visits        of `n_train_kf` units
        wonline  a unit is a sliding window    epochs * window      of `n - window + 1` units

    None if the config is too old to say.
    """
    n, e, style = cfg.get('n_train_kf'), cfg.get('epochs'), cfg.get('adapt_style', 'normal')
    if not n or not e:
        return None
    if style == 'online':
        units, per_unit = n, e
    elif style == 'wonline':
        w = cfg.get('window_size')
        if not w or w > n:
            return None
        units, per_unit = n - w + 1, e * w
    else:
        units, per_unit = e, n

    # saved_epoch is the 0-indexed unit the adapter was written at - the last one for a final
    # save, an earlier one for a checkpoint or a keep_best snapshot
    done = cfg['saved_epoch'] + 1 if cfg.get('saved_epoch') is not None else units
    return min(done, units) * per_unit


def window_of(cfg):
    """window_size, but only for the style that reads it.

    trainer.py:160 records window_size whatever the style, so on a normal or online run it is a
    leftover from the PARAMETERS block rather than a parameter of that run - and sorting on it
    would group runs that have nothing in common.
    """
    return cfg.get('window_size') if cfg.get('adapt_style', 'normal') == 'wonline' else None


def extract_keyframes(extract_dir):
    """The keyframe count off export.txt's first line: `48 keyframes, 368x584, intrinsics ...`."""
    path = f'{extract_dir}/export.txt'
    if not extract_dir or not os.path.exists(path):
        return None
    with open(path) as f:
        m = re.match(r'\s*(\d+)\s+keyframes', f.readline())
    return int(m.group(1)) if m else None


def adapt_dirs(root, scene):
    """{arm name: its adapt directory} - every adapter and checkpoint this scene has on disk.

    Built by running arm_name over the specs, i.e. the map is inverted rather than the naming rule
    re-derived: <aname>_chkp_005 is arm_name's business, not this file's.
    """
    any_experiment = experiment_dir(root, 'adapt', scene, '*')
    specs = sorted(glob.glob(f'{any_experiment}/{CONFIG}'))
    specs += sorted(glob.glob(f'{any_experiment}/{ADAPT_CKPT_SUBDIR}/epoch_*/{CONFIG}'))
    return {arm_name(os.path.dirname(s)): os.path.dirname(s) for s in specs}


def load_json(path):
    """The file's contents, or None - every join here is allowed to be missing."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect(root, scene):
    """One dict per arm, unjoined values still raw. Rows without an adapter keep blank knobs."""
    out_root = test_dir(root, 'end2end', scene)
    tested = {os.path.basename(os.path.dirname(p)): load_json(p)
              for p in sorted(glob.glob(f'{out_root}/*/{RESULTS}'))}
    adapters = adapt_dirs(root, scene)

    # an arm scored but whose adapter was deleted still belongs in the table, and so does an
    # adapter that has not been through end2end yet - the blank ate_all is the point
    rows = []
    for arm in sorted(set(tested) | set(adapters)):
        cfg = load_json(f'{adapters[arm]}/{CONFIG}') if arm in adapters else None
        rows.append({'arm': arm, 'res': tested.get(arm), 'cfg': cfg or {}})
    return rows


def sequence_frames(rows):
    """The sequence's frame count, taken from the arms' pose counts - only train_pct needs it.

    Arms that scored a different count are not comparable at all, so a disagreement is worth a
    word rather than a silently wrong percentage.
    """
    counts = {r['res']['n_all'] for r in rows if r['res'] and r['res'].get('n_all')}
    if len(counts) > 1:
        print(f'WARNING: arms scored different pose counts {sorted(counts)}; their ATEs are not '
              f'comparable and train_pct uses the largest')
    return max(counts) if counts else None


def build(rows, n_frames):
    """The CSV records. Every cell is blank rather than 0 when it was not measured."""
    omni = next((r['res']['ate_all'] for r in rows
                 if r['arm'] == OMNI_ARM and r['res'] and r['res'].get('ate_all') is not None),
                None)
    if omni is None:
        print(f'WARNING: no {OMNI_ARM} arm with an ate_all - the delta columns will be blank')
    today = datetime.date.today().isoformat()

    def num(v, nd=4):
        return '' if v is None else round(v, nd) if isinstance(v, float) else v

    out = []
    for r in rows:
        cfg, res = r['cfg'], r['res'] or {}
        extract = cfg.get('scene')
        ate = res.get('ate_all')
        split = cfg.get('split_at')
        delta = ate - omni if (ate is not None and omni) else None
        out.append({
            'arm': r['arm'],
            'style': cfg.get('adapt_style', 'normal') if cfg else '',
            'epochs': num(cfg.get('epochs')),
            'window': num(window_of(cfg)),
            'lr': num(cfg.get('lr'), 6),
            'train_pct': num(round(split / n_frames * 100, 1) if split and n_frames else None, 1),
            'train_frames': num(split),
            'n_train_kf': num(cfg.get('n_train_kf')),
            'adapt_cost': num(adapt_cost(cfg)),
            'train_seconds': num(cfg.get('train_seconds'), 1),
            'extract': os.path.basename(extract.rstrip('/')) if extract else '',
            'extract_kf': num(extract_keyframes(extract)),
            'ate_all': num(ate),
            'd_ate_vs_omni': num(delta),
            'd_ate_pct': num(delta / omni * 100 if delta is not None else None, 2),
            'exported_at': today,
        })
    return out, omni


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-n', '--name', required=True, help='CSV name, written to <root>/<name>.csv')
    ap.add_argument('-s', '--scene', required=True,
                    help='the ONE scene to export; arm names are unique only within a scene')
    ap.add_argument('--root', default='outputs', help='the outputs tree (default: outputs)')
    args = ap.parse_args()

    os.chdir(_ROOT)                              # --root is repo-root relative, however invoked

    scenes = sorted(os.listdir(f'{args.root}/test/end2end')) \
        if os.path.isdir(f'{args.root}/test/end2end') else []
    if args.scene not in scenes:
        raise SystemExit(f'no end2end results for scene {args.scene!r} under {args.root}/; '
                         f'available: {scenes or "(none)"}')

    rows = collect(args.root, args.scene)
    records, omni = build(rows, sequence_frames(rows))

    path = f'{args.root}/{args.name}.csv'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(records)

    scored = sum(1 for r in records if r['ate_all'] != '')
    print(f'{path}')
    print(f'  scene    {args.scene}')
    print(f'  rows     {len(records)}  ({scored} with an ate_all, '
          f'{len(records) - scored} not yet run end2end)')
    print(f'  baseline {OMNI_ARM} ate_all = {omni:.4f}' if omni is not None else
          f'  baseline {OMNI_ARM} MISSING - deltas blank')
    print(f'\nNotion: first import creates the database; after that use Merge with CSV, which '
          f'matches\non the Title column ({COLUMNS[0]}) and updates rows in place.')


if __name__ == '__main__':
    main()
