"""One scene's end2end arms as a CSV, for a Notion database (12).

    python scripts/export_end2end_results.py -n "aug10 rellis" -s rellis_00000
    -> outputs/aug10 rellis.csv

Read-only over outputs/: it joins each arm's results.json to the adapt config.json behind it and
the extract that config trained on. Regenerable at any time, never a source of truth.

THREE DRIVERS, THREE TABLES. A scene's arms come from three pipelines whose parameters do not
overlap, so one column set cannot serve them - `train_pct` is meaningless for a run that adapted
across the whole sequence, and `warmup_prior` does not exist for one that adapted offline. Exactly
one kind per invocation, argparse-enforced, `--init` when none is given:

    --init   (default)  init_adapt_pipeline.py  - extract a prefix, adapt on it, test frozen (9.1)
    --cont              cont_adapt_pipeline.py  - NOT IMPLEMENTED, see main()             (9.7)
    --live              online_adapt_pipeline.py - one SLAM run that adapts as it tracks  (13)

WHICH ARMS a table holds is decided by the experiment-name PREFIX its driver's names carry -
`live*` for the online driver, `cont*` for the continual one, everything else init. That is the one
and only place this file reads a name (see below), and it is unavoidable: nothing inside an arm's
directory records which driver produced it. For `live` there IS corroborating data,
config.json's `online: true`, so the two are cross-checked and a disagreement is printed rather
than silently resolved.

Two rules sit on top of the prefix:

  * `omni` and `base` are in EVERY table. The delta columns are against `omni`, and 13's online
    driver compares against exactly those two.
  * under --live only, an un-run arm whose name is a live adapter's without ONLINE_ARM_SUFFIX is
    dropped. That is the frozen-replay name (13.4), deliberately not run by default, and until it
    is scored it is a blank duplicate of the `<name>_live` row beside it rather than an arm
    waiting its turn. An init adapter with no arm still gets its blank row - there the blank is
    the point.

Three things it deliberately does NOT do:

  * parse the arm name for PARAMETERS. Every parameter is recorded in the adapter's config.json,
    and the names lie - wonline_r8_e5_w20_p10 records epochs: 3, live_e3_a16_w12_lag4_* records
    alpha: 8. The name is a label, config.json is the data. The pipeline prefix above is the one
    exception, and only because it is the one fact config.json cannot supply.
  * report ate_seen / ate_unseen. Each arm's results.json computed them at whatever FRACTION was
    current when it ran (omni at 284, normal_r8_e20 at 1138), so they are not comparable across
    arms. ate_all is split-independent and is the metric this table exists for.
  * export a constant. Which fields are constant depends on the kind: `alpha` is the same in all
    93 offline configs and takes two values across the live ones, so it is a --live column only.

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
A LIVE arm does not use that table at all: its trainer COUNTED the visits (online/trainer.py:130)
and recorded them as `kf_visits`, so the measured number is exported instead of the derived one.

NOTION. One database per scene AND per kind - `arm` is only unique within one scene, and the three
kinds do not share a column set.

  1. first time   /database -> Import -> CSV. Then set the property types once: `arm` stays
                  Title, `style` / `extract` / `warmup_prior` / `init_adapter` -> Select, every
                  numeric column -> Number, `exported_at` -> Date. Merges preserve types, so this
                  is a one-off.
  2. after that   database ... menu -> Merge with CSV. Notion matches rows on the Title, so `arm`
                  makes a re-export UPDATE rows in place instead of duplicating them; new arms
                  are appended.

  Views worth making. --init: Table filtered `ate_all is not empty`, grouped by `style`, sorted
  `adapt_cost` ascending (where more adaptation stops buying ATE); a second grouped by
  `train_pct`, sorted `ate_all` ascending (the best result each frame budget can reach); and a
  small one filtered to `arm is omni or base`. --live: grouped by `init_adapter` (continuing from
  an offline adapter vs starting stock is the biggest split there), sorted `ate_all` ascending.
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

from adaslam.common import (ADAPT_CKPT_SUBDIR, ONLINE_ARM_SUFFIX,        # noqa: E402
                            experiment_dir, test_dir)
from adaslam.end2end.config import SENTINELS, arm_name                   # noqa: E402

# arm_name is IMPORTED rather than reimplemented: the pipeline infers an arm's directory from its
# prior spec (7.1), and a second naming rule here would join the wrong rows the day one changes.
OMNI_ARM = SENTINELS['omnidata']
SENTINEL_ARMS = frozenset(SENTINELS.values())   # omni, base - the baselines every table carries

RESULTS, CONFIG = 'results.json', 'config.json'

KIND_INIT, KIND_CONT, KIND_LIVE = 'init', 'cont', 'live'

# The experiment-name prefix each non-init driver's names carry. Read the module docstring before
# adding to this: a prefix is a CONVENTION the PARAMETERS block maintains by hand, and it is the
# only pipeline fact an arm's directory does not already record.
PREFIX = {KIND_LIVE: 'live', KIND_CONT: 'cont'}

# `arm` first in every kind: Notion's Merge with CSV matches rows on the Title property, column 0.
INIT_COLUMNS = ('arm', 'style', 'epochs', 'window', 'lr',
                'train_pct', 'train_frames', 'n_train_kf', 'adapt_cost', 'train_seconds',
                'extract', 'extract_kf',
                'ate_all', 'd_ate_vs_omni', 'd_ate_pct', 'exported_at')

LIVE_COLUMNS = ('arm', 'style', 'steps_per_kf', 'window', 'lr', 'alpha', 'lag',
                'warmup_kf', 'handover_kf', 'warmup_prior', 'warmup_end_frame',
                'first_adapted_kf',
                'n_units', 'n_train_kf', 'adapt_cost', 'train_seconds', 'init_adapter',
                'ate_all', 'd_ate_vs_omni', 'd_ate_pct', 'exported_at')


def num(v, nd=4):
    """A numeric cell, blank rather than 0 when it was not measured."""
    return '' if v is None else round(v, nd) if isinstance(v, float) else v


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


def live_cost(cfg):
    """The same quantity for a live run - MEASURED, not derived.

    LiveTrainer increments `kf_visits` once per keyframe actually pushed through VGGT
    (online/trainer.py:130), which is exactly what adapt_cost defines, and a checkpoint's
    config.json records the count at its own snapshot. So the recorded number wins, and adapt_cost
    is the fallback only for a config too old to carry it.

    The two AGREE on every live run on disk, and that is a coincidence worth not relying on. A
    live unit is an ARRIVING KEYFRAME in both styles, so the true count is n_units * epochs *
    window; adapt_cost reconstructs it as (n_train_kf - window + 1) * epochs * window, and those
    match only while every arrival adds exactly one distinct keyframe to a window that is never
    clipped. Both premises can fail - target.py:41 clips the window at the start of the sequence
    (window_size > warmup_kf), and n_train_kf counts distinct FRAME indices, which pruning shifts
    (13.5). Counting is not subject to either.
    """
    visits = cfg.get('kf_visits')
    return visits if visits is not None else adapt_cost(cfg)


def window_of(cfg):
    """window_size, but only for the style that reads it.

    trainer.py:160 records window_size whatever the style, so on a normal or online run it is a
    leftover from the PARAMETERS block rather than a parameter of that run - and sorting on it
    would group runs that have nothing in common. Live runs record it the same way, for the same
    two styles (online/config.py:30).
    """
    return cfg.get('window_size') if cfg.get('adapt_style', 'normal') == 'wonline' else None


def init_adapter_of(cfg):
    """The adapter a run CONTINUED from, as its directory name - `stock` when it started from
    VGGT-1B.

    Blank stays reserved for "not measured", which is what a row with no adapter at all gets;
    starting from stock is a measurement and says so.
    """
    if not cfg:
        return ''
    path = cfg.get('init_adapter')
    return os.path.basename(os.path.dirname(str(path).rstrip('/'))) if path else 'stock'


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
    tops = sorted(glob.glob(f'{any_experiment}/{CONFIG}'))
    specs = tops + sorted(glob.glob(f'{any_experiment}/{ADAPT_CKPT_SUBDIR}/epoch_*/{CONFIG}'))
    dirs = {arm_name(os.path.dirname(s)): os.path.dirname(s) for s in specs}

    # An ONLINE run (13) produces its adapter AND the arm that trained it, and that arm carries
    # ONLINE_ARM_SUFFIX so a later frozen test of the same adapter cannot overwrite it. Both names
    # therefore point at the one config.json; the checkpoints are left alone, since a checkpoint is
    # only ever run frozen and already names itself <aname>_chkp_NNN.
    for s in tops:
        d = os.path.dirname(s)
        if (load_json(s) or {}).get('online'):
            dirs[f'{arm_name(d)}{ONLINE_ARM_SUFFIX}'] = d
    return dirs


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


def kind_of(arm):
    """Which driver produced an arm, off its experiment-name prefix. See PREFIX."""
    for kind, prefix in PREFIX.items():
        if arm.startswith(prefix):
            return kind
    return KIND_INIT


def select(rows, kind):
    """The rows one kind's table holds - its own arms plus the sentinel baselines.

    The prefix decides, and where config.json can corroborate it, disagreement is reported: a live
    run whose name lacks the prefix is INVISIBLE to --live, and a missing row is much harder to
    notice than a wrong one.
    """
    arms = {r['arm'] for r in rows}
    out = []
    for r in rows:
        arm, cfg, named = r['arm'], r['cfg'], kind_of(r['arm'])

        if cfg and bool(cfg.get('online')) != (named == KIND_LIVE):
            does = 'does' if cfg.get('online') else 'does not'
            print(f'WARNING: {arm} is named for the {named!r} driver but its config.json {does} '
                  f"record online: true - it is exported as {named!r}. Rename the experiment "
                  f"({PREFIX[KIND_LIVE]}* for an online run) to fix it")

        if arm in SENTINEL_ARMS:      # omni is what the deltas are against; base is 13's reference
            out.append(r)
            continue
        if named != kind:
            continue
        # the un-run FROZEN REPLAY of a live run (13.4): it shares its config.json with the
        # <arm>_live row sitting right beside it, so until it is actually scored it is a duplicate
        # rather than an arm waiting its turn
        if kind == KIND_LIVE and not r['res'] and f'{arm}{ONLINE_ARM_SUFFIX}' in arms:
            continue
        out.append(r)

    if not any(r['arm'] not in SENTINEL_ARMS for r in out):
        print(f'WARNING: no {kind!r} arms in this scene - only the baselines will be exported. '
              f'That driver names its experiments with the {PREFIX.get(kind, "(no)")!r} prefix; '
              f'an experiment named otherwise is exported as {KIND_INIT!r}')
    return out


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


def init_cells(cfg, n_frames):
    """The columns only an offline adapt run has: a fixed train set, drawn from an extract."""
    extract, split = cfg.get('scene'), cfg.get('split_at')
    return {
        'epochs': num(cfg.get('epochs')),
        'window': num(window_of(cfg)),
        'train_pct': num(round(split / n_frames * 100, 1) if split and n_frames else None, 1),
        'train_frames': num(split),
        'n_train_kf': num(cfg.get('n_train_kf')),
        'adapt_cost': num(adapt_cost(cfg)),
        'train_seconds': num(cfg.get('train_seconds'), 1),
        'extract': os.path.basename(extract.rstrip('/')) if extract else '',
        'extract_kf': num(extract_keyframes(extract)),
    }


def live_cells(cfg, n_frames):
    """The columns only a live run has - warm-up, lag, and a train set that arrived as it ran.

    `epochs` is exported as `steps_per_kf` because that is what it MEANS live: LiveTrainer.stats
    stores steps-per-arrival under the offline key on purpose, so adapt_cost needs no special case
    (online/trainer.py:185), but in a live table the word `epochs` would only mislead.

    No train_pct / train_frames: split_at is the whole sequence for every live run (13.4), so it
    separates nothing. No extract / extract_kf: there is no extract stage, and `scene` records the
    arm's OWN output directory. What replaces them is warmup_end_frame - the one frame index that
    does mean something here, where the fallback prior handed over to VGGT.

    `warmup_kf` and `handover_kf` are TWO gates, not one (online/config.py): the first is when the
    adapter started learning, the second when it started serving. They were one field, so on every
    adapter written before the split `handover_kf` is blank and equal to `warmup_kf` by
    construction - blank because that is what "not measured" looks like everywhere else here.
    """
    return {
        'steps_per_kf': num(cfg.get('epochs')),
        'window': num(window_of(cfg)),
        'alpha': num(cfg.get('alpha')),
        'lag': num(cfg.get('lag')),
        'warmup_kf': num(cfg.get('warmup_kf')),
        # blank on every adapter written before online/config.py split the one warm-up gate in two;
        # there handover_kf WAS warmup_kf, and a blank says "not measured" rather than asserting it
        'handover_kf': num(cfg.get('handover_kf')),
        'warmup_prior': cfg.get('warmup_prior', ''),
        'warmup_end_frame': num(cfg.get('warmup_end_frame')),
        'first_adapted_kf': num(cfg.get('first_adapted_kf')),
        'n_units': num(cfg.get('n_units')),
        'n_train_kf': num(cfg.get('n_train_kf')),
        'adapt_cost': num(live_cost(cfg)),
        'train_seconds': num(cfg.get('train_seconds'), 1),
        'init_adapter': init_adapter_of(cfg),
    }


# columns and the cell builder for the middle of a row, per kind. KIND_CONT is deliberately absent:
# main() rejects it before any work, and an entry here would be a table nobody designed.
KINDS = {KIND_INIT: (INIT_COLUMNS, init_cells),
         KIND_LIVE: (LIVE_COLUMNS, live_cells)}


def build(rows, n_frames, kind):
    """The CSV records. Every cell is blank rather than 0 when it was not measured.

    What every kind shares is the identity of the arm and its result; `cells` supplies the middle.
    """
    omni = next((r['res']['ate_all'] for r in rows
                 if r['arm'] == OMNI_ARM and r['res'] and r['res'].get('ate_all') is not None),
                None)
    if omni is None:
        print(f'WARNING: no {OMNI_ARM} arm with an ate_all - the delta columns will be blank')
    today = datetime.date.today().isoformat()
    cells = KINDS[kind][1]

    out = []
    for r in rows:
        cfg, res = r['cfg'], r['res'] or {}
        ate = res.get('ate_all')
        delta = ate - omni if (ate is not None and omni) else None
        out.append({
            'arm': r['arm'],
            'style': cfg.get('adapt_style', 'normal') if cfg else '',
            'lr': num(cfg.get('lr'), 6),
            **cells(cfg, n_frames),
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

    # one kind per run, enforced by argparse rather than checked: the three tables do not share a
    # column set, so there is no such thing as exporting two of them into one file
    kinds = ap.add_mutually_exclusive_group()
    kinds.add_argument('--init', dest='kind', action='store_const', const=KIND_INIT,
                       help='arms from init_adapt_pipeline.py - anything not named for another '
                            'driver, plus the baselines (default)')
    kinds.add_argument('--cont', dest='kind', action='store_const', const=KIND_CONT,
                       help=f'arms from cont_adapt_pipeline.py ({PREFIX[KIND_CONT]}*) - '
                            f'NOT IMPLEMENTED')
    kinds.add_argument('--live', dest='kind', action='store_const', const=KIND_LIVE,
                       help=f'arms from online_adapt_pipeline.py ({PREFIX[KIND_LIVE]}*), plus '
                            f'the baselines')
    ap.set_defaults(kind=KIND_INIT)
    args = ap.parse_args()

    if args.kind == KIND_CONT:
        raise SystemExit(
            '--cont is not implemented yet: the continual table (9.7) needs a column set of its '
            'own, and none of what separates those runs - kf_fraction, val_source, and the '
            "val_l1 series in config.json's eval_history - is an --init column. Note the name is "
            'the only reliable marker there too: a cont run at KF_FRACTION=1.0 records '
            "val_source: 'tail' exactly like an init run. Use --init or --live.")

    os.chdir(_ROOT)                              # --root is repo-root relative, however invoked

    scenes = sorted(os.listdir(f'{args.root}/test/end2end')) \
        if os.path.isdir(f'{args.root}/test/end2end') else []
    if args.scene not in scenes:
        raise SystemExit(f'no end2end results for scene {args.scene!r} under {args.root}/; '
                         f'available: {scenes or "(none)"}')

    rows = select(collect(args.root, args.scene), args.kind)
    records, omni = build(rows, sequence_frames(rows), args.kind)

    columns = KINDS[args.kind][0]
    path = f'{args.root}/{args.name}.csv'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(records)

    scored = sum(1 for r in records if r['ate_all'] != '')
    print(f'{path}')
    print(f'  scene    {args.scene}')
    print(f'  kind     --{args.kind}')
    print(f'  rows     {len(records)}  ({scored} with an ate_all, '
          f'{len(records) - scored} not yet run end2end)')
    print(f'  baseline {OMNI_ARM} ate_all = {omni:.4f}' if omni is not None else
          f'  baseline {OMNI_ARM} MISSING - deltas blank')
    print(f'\nNotion: one database per scene AND per kind - the column sets differ. First import '
          f'creates it;\nafter that use Merge with CSV, which matches on the Title column '
          f'({columns[0]}) and updates rows in place.')


if __name__ == '__main__':
    main()
