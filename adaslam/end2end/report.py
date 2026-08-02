"""Printing results: one arm at a time, then side by side.

Pure formatting over evaluate()'s dicts, so both re-run against results.json on disk with no GPU.
Column headers are the arms' inferred directory names; the full label stays in results.json.
"""
from ..print_utils import delta_header, delta_row

POPULATIONS = ('all', 'seen', 'unseen')


def _ate(res, pop):
    """One arm's ATE for one population, or None if it has none."""
    return res.get(f'ate_{pop}')


def _n(res, pop):
    """Poses that population's ATE was averaged over. .get(): older files carry no counts."""
    return res.get(f'n_{pop}')


def print_report(res):
    """One arm's numbers, split at the frame its adapter's training data ended."""
    k = res['split_at']
    print(f"\n{'='*66}\n  {res['label']}  ->  {res['output']}\n{'='*66}")
    print(f"  {'metric':<22}{'all':>13}{f'seen <{k}':>13}{f'unseen >={k}':>15}")
    print(f"  {'-'*63}")
    print(f"  {'ATE RMSE (m)':<22}{res['ate_all']:>13.4f}"
          f"{res.get('ate_seen') or float('nan'):>13.4f}"
          f"{res.get('ate_unseen') or float('nan'):>15.4f}")
    cells = ''.join(f'{n if n is not None else "n/a":>{13 if i < 2 else 15}}'
                    for i, n in enumerate(_n(res, p) for p in POPULATIONS))
    print(f"  {'poses evaluated':<22}{cells}")
    print()


def compare(labels, res):
    """Every arm side by side, one table per population - the shape priortest/report.py prints."""
    base = res[0]
    k = base['split_at']

    # an arm run at a different split or pose count is not comparable, however the numbers look
    for lbl, r in zip(labels[1:], res[1:]):
        if r['split_at'] != k:
            raise SystemExit(f"  !! {lbl} used split_at={r['split_at']}, baseline used {k} - "
                             'the arms are not comparable; delete its output and re-run')
        n0, n1 = _n(base, 'all'), _n(r, 'all')
        if n0 is not None and n1 is not None and n0 != n1:
            print(f'  !! {lbl} evaluated {n1} poses, baseline {n0} - arms are not comparable')

    # the column must fit an adapter's name plus a checkpoint suffix
    width = max(12, max(len(lbl) for lbl in labels) + 2)
    n_all = _n(base, 'all')
    over = f'over {n_all} poses, ' if n_all is not None else ''
    print(f'  full-sequence comparison {over}split at frame {k}')

    for pop in POPULATIONS:
        if all(_ate(r, pop) is None for r in res):
            continue
        print(f'\n  [{pop}]')
        delta_header(labels, width=width)
        delta_row('ATE RMSE (m)', [_ate(r, pop) for r in res], True, width=width)

    print("\n  '+' better than the baseline column, '-' worse.")
    print("  'unseen' is the row that matters: it is the only evidence the adaptation")
    print('  generalises rather than having memorised the keyframes it trained on.')
