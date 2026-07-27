"""Printing prior-test results: one arm, then all of them side by side.

Pure formatting over the dicts aggregate() returns, so it can be re-run against results.json files
already on disk without a GPU:

    from adaslam.priortest.report import compare, print_report

compare() prints one table per population - metric down the rows, arm across the columns - and
end2end/report.py now prints the same shape, off the same print_utils.delta_header and delta_row.
What stays local to each is what genuinely differs: this one can mark an arm whose split is not its
own with a `*`, and the end2end one has a mesh block with no seen/unseen to split.
"""
from ..print_utils import delta_header, delta_row

# (key, label, lower-is-better)
ROWS = (('l1_perframe', 'L1 per-frame scale (m)', True),
        ('l1_jdsa', 'L1 2x2 JDSA grid (m)', True),
        ('l1_global', 'L1 global scale (m)', True),
        ('consistency_index', 'consistency index', True),
        ('scale_cv', 'scale CV', True),
        ('absrel', 'AbsRel', True),
        ('delta125', 'delta < 1.25', False))

POPULATIONS = ('all', 'seen', 'unseen')


def marked(res):
    """The arm's display name, with a leading * when the table's split is not its own."""
    star = '*' if res.get('split_mismatch') else ''
    return f'{star}{res["arm"]}'


def print_report(res):
    """One arm's numbers."""
    k = res.get('split_at')
    print(f"\n{'='*72}\n  {marked(res)}  ->  {res['output']}\n{'='*72}")
    print(f"  {res['label']}")
    if res.get('split_mismatch'):
        print(f"  * this arm trained to frame {res['own_split_at']}, scored at the table's "
              f"split {k}")
    pops = [p for p in POPULATIONS if res['blocks'].get(p)]
    head = {'all': 'all', 'seen': f'seen <{k}', 'unseen': f'unseen >={k}'}
    print(f"  {'metric':<26}" + ''.join(f'{head[p]:>14}' for p in pops))
    print(f"  {'-' * (26 + 14 * len(pops))}")
    for key, name, _ in ROWS:
        print(f'  {name:<26}' + ''.join(f"{res['blocks'][p][key]:>14.4f}" for p in pops))
    print(f"  {'frames':<26}" + ''.join(f"{res['blocks'][p]['n']:>14}" for p in pops))
    print()


def compare(results):
    """Every arm side by side, one block per population. results[0] is the baseline column."""
    base = results[0]
    split_at = base.get('split_at')
    names = [marked(r) for r in results]
    width = max(12, max(len(n) for n in names) + 2)

    print(f'  prior comparison over {base["blocks"]["all"]["n"]} frames'
          + (f", split at {split_at}" if split_at is not None else ' (no adapter -> no split)'))
    mismatched = [r for r in results if r.get('split_mismatch')]
    if mismatched:
        # the arm that DEFINED the boundary: the first one whose own split is the table's
        owner = next((r['arm'] for r in results if r.get('own_split_at') == split_at), '?')
        print(f'  !! that split is {owner}\'s; these trained to a different frame, so their '
              f'seen/unseen rows are cut somewhere they did train past - marked * below:')
        for r in mismatched:
            print(f'     {r["arm"]}: trained to {r["own_split_at"]}, scored at {split_at}')

    for pop in POPULATIONS:
        if not all(r['blocks'].get(pop) for r in results):
            continue
        print(f'\n  [{pop}]')
        delta_header(names, width=width)
        for key, name, lower_better in ROWS:
            delta_row(name, [r['blocks'][pop][key] for r in results], lower_better, width=width)

    print("\n  '+' better than the baseline column, '-' worse.")
    print('  consistency index = L1 global / L1 per-frame: 1.0 means one scale fits the whole')
    print('  sequence as well as a per-frame one, higher means the prior drifts between frames.')
    print('  That drift is what the adaptation targets (ARCHITECTURE.md 10.2).')
    print('  The JDSA-grid row is a LOWER bound on what JDSA leaves behind - the real solver fits')
    print('  that grid against photometric residuals, not in closed form against GT.')
    if split_at is not None:
        print("  Read 'unseen' against the SENTINELS' unseen, not against an adapter's own 'seen':")
        print('  if every prior is worse there, the back of the sequence is simply harder.')
