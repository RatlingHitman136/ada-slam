"""Printing results: one arm at a time, then side by side.

Pure formatting over the dicts evaluate() returns, so both functions can be re-run against
results.json files already on disk without touching a GPU.

compare()'s column headers are the arms' inferred directory names (omni, base, lr1e4_chkp_005) -
short, and unique by construction since End2EndConfig rejects two priors that infer one name. The
full human label ('VGGT+LoRA depth / Omnidata normals') stays in results.json.
"""
from ..print_utils import delta_header, delta_row

# (key, label, lower-is-better). ate_<pop> sits at the top level of a result dict, the other three
# under res['render'][<pop>] - _cell knows which.
ROWS = (('ate', 'ATE RMSE (m)', True),
        ('psnr', 'PSNR (dB)', False),
        ('ssim', 'SSIM', False),
        ('depth_l1', 'depth L1 (m)', True))

# One TSDF is fused over the whole trajectory, so these have no seen/unseen to be split into and
# get a block of their own after the three population tables.
MESH_ROWS = (('mean precision', 'mesh accuracy (m)', True),
             ('mean recall', 'mesh completion (m)', True),
             ('recall', 'mesh comp-ratio', False),
             ('f-score', 'mesh F-score', False))

POPULATIONS = ('all', 'seen', 'unseen')


def _cell(res, pop, key):
    """One arm's value for one metric in one population, or None if it has none."""
    if key == 'ate':
        return res.get(f'ate_{pop}')
    return (res['render'].get(pop) or {}).get(key)


def print_report(res):
    """One arm's numbers, split at the frame its adapter's training data ended."""
    k = res['split_at']
    print(f"\n{'='*66}\n  {res['label']}  ->  {res['output']}\n{'='*66}")
    print(f"  {'metric':<22}{'all':>13}{f'seen <{k}':>13}{f'unseen >={k}':>15}")
    print(f"  {'-'*63}")
    print(f"  {'ATE RMSE (m)':<22}{res['ate_all']:>13.4f}"
          f"{res.get('ate_seen') or float('nan'):>13.4f}"
          f"{res.get('ate_unseen') or float('nan'):>15.4f}")
    r = res['render']
    for key, name in (('psnr', 'PSNR (dB)'), ('ssim', 'SSIM'), ('depth_l1', 'depth L1 (m)')):
        vals = [(r[s] or {}).get(key) for s in ('all', 'seen', 'unseen')]
        if any(v is not None for v in vals):
            cells = ''.join(f'{v:>13.4f}' if i < 2 else f'{v:>15.4f}'
                            if v is not None else f'{"n/a":>13}' for i, v in enumerate(vals))
            print(f'  {name:<22}{cells}')
    print(f"  {'frames evaluated':<22}{(r['all'] or {}).get('n', 0):>13}"
          f"{(r['seen'] or {}).get('n', 0):>13}{(r['unseen'] or {}).get('n', 0):>15}")
    if res.get('mesh'):
        m = res['mesh']
        print(f"\n  mesh (whole sequence, Sim3-aligned, voxel {m['voxel_size']}): "
              f"acc {m['mean precision']:.4f} m  comp {m['mean recall']:.4f} m  "
              f"comp-ratio {100*m['recall']:.1f}%  F {m['f-score']:.3f}")
    print()


def compare(labels, res):
    """Every arm side by side, ONE TABLE PER POPULATION - the shape priortest/report.py prints.

    Metric down the rows, arm across the columns, one block each for all / seen / unseen. The
    previous layout put the population in the row label and the metrics in groups, which made
    "how does this arm compare on unseen frames" a vertical scan across three separate groups.
    """
    base = res[0]
    k = base['split_at']

    # an arm run at a different split or over a different frame count is not comparable, however
    # tempting the numbers look side by side
    for lbl, r in zip(labels[1:], res[1:]):
        if r['split_at'] != k:
            raise SystemExit(f"  !! {lbl} used split_at={r['split_at']}, baseline used {k} - "
                             'the arms are not comparable; delete its output and re-run')
        n0, n1 = (base['render']['all'] or {}).get('n'), (r['render']['all'] or {}).get('n')
        if n0 != n1:
            print(f'  !! {lbl} evaluated {n1} frames, baseline {n0} - arms are not comparable')

    # arm names are as long as an adapter's name plus a checkpoint suffix, so the column has to fit
    # them; the old hardcoded 12 silently misaligned the table for anything longer
    width = max(12, max(len(lbl) for lbl in labels) + 2)
    n_all = (base['render']['all'] or {}).get('n', 0)
    print(f'  full-sequence comparison over {n_all} frames, split at frame {k}')

    for pop in POPULATIONS:
        if all(_cell(r, pop, key) is None for r in res for key, _, _ in ROWS):
            continue
        print(f'\n  [{pop}]')
        delta_header(labels, width=width)
        for key, name, lower_better in ROWS:
            delta_row(name, [_cell(r, pop, key) for r in res], lower_better, width=width)

    meshes = [r.get('mesh') for r in res]
    if all(meshes):
        voxels = {m['voxel_size'] for m in meshes}
        if len(voxels) > 1:
            print(f'\n  !! voxel sizes differ ({sorted(voxels)}) - mesh numbers are NOT '
                  'comparable; re-run with the same voxel_size')
        else:
            print(f'\n  [mesh]  whole sequence, Sim(3)-aligned, voxel {voxels.pop()} - one TSDF '
                  f'per arm, so this block has no seen/unseen split')
            delta_header(labels, width=width)
            for key, name, low in MESH_ROWS:
                delta_row(name, [m[key] for m in meshes], low, width=width)
    elif any(meshes):
        print('\n  mesh metrics unavailable for at least one arm')

    print("\n  '+' better than the baseline column, '-' worse.")
    print("  'unseen' is the row that matters: it is the only evidence the adaptation")
    print('  generalises rather than having memorised the keyframes it trained on.')
