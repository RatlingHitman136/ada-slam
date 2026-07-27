"""Printing results: one arm at a time, then side by side.

Pure formatting over the dicts evaluate() returns, so both functions can be re-run against
results.json files already on disk without touching a GPU. That includes files written before the
render and mesh metrics were removed - their extra keys are simply not read, and every count is
fetched with .get() so an older file still prints its ATE.

compare()'s column headers are the arms' inferred directory names (omni, base, lr1e4_chkp_005) -
short, and unique by construction since End2EndConfig rejects two priors that infer one name. The
full human label ('VGGT+LoRA depth / Omnidata normals') stays in results.json.
"""
from ..print_utils import delta_header, delta_row

POPULATIONS = ('all', 'seen', 'unseen')


def _ate(res, pop):
    """One arm's ATE for one population, or None if it has none."""
    return res.get(f'ate_{pop}')


def _n(res, pop):
    """How many poses that population's ATE was averaged over, or None if the file predates them.

    results.json files written before the render metrics came out carry no counts. Their ATE is
    unchanged and still worth printing, so a missing count reads n/a rather than a misleading 0.
    """
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
    """Every arm side by side, ONE TABLE PER POPULATION - the shape priortest/report.py prints.

    Metric down the rows, arm across the columns, one block each for all / seen / unseen. With ATE
    the only metric left each block is a single row, but the layout is kept: it is the one the
    prior test prints, and 'how does this arm do on unseen frames' stays a horizontal read.
    """
    base = res[0]
    k = base['split_at']

    # an arm run at a different split or over a different pose count is not comparable, however
    # tempting the numbers look side by side
    for lbl, r in zip(labels[1:], res[1:]):
        if r['split_at'] != k:
            raise SystemExit(f"  !! {lbl} used split_at={r['split_at']}, baseline used {k} - "
                             'the arms are not comparable; delete its output and re-run')
        n0, n1 = _n(base, 'all'), _n(r, 'all')
        if n0 is not None and n1 is not None and n0 != n1:
            print(f'  !! {lbl} evaluated {n1} poses, baseline {n0} - arms are not comparable')

    # arm names are as long as an adapter's name plus a checkpoint suffix, so the column has to fit
    # them; the old hardcoded 12 silently misaligned the table for anything longer
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
