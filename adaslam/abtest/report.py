"""Printing results: one arm at a time, then side by side.

Pure formatting over the dicts evaluate() returns, so both functions can be re-run against
ab_results.json files already on disk without touching a GPU.
"""


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
    """Side-by-side table: baseline absolute, then absolute + delta for every other arm."""
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

    print(f'  full-sequence comparison, split at frame {k}')
    print(f"  {'metric':<26}{labels[0]:>12}" +
          ''.join(f'{l:>12}{"delta":>11}' for l in labels[1:]))
    print('  ' + '-' * (26 + 12 + 23 * (len(labels) - 1)))

    def row(name, vals, better_low=True):
        if vals[0] is None:
            return
        line = f'  {name:<26}{vals[0]:>12.4f}'
        for v in vals[1:]:
            if v is None:
                line += f'{"n/a":>12}{"":>11}'
                continue
            d = v - vals[0]
            mark = ' ' if abs(d) < 1e-9 else ('+' if (d < 0) == better_low else '-')
            line += f'{v:>12.4f}{d:>+9.4f} {mark}'
        print(line)

    for s in ('all', 'seen', 'unseen'):
        row(f'ATE RMSE ({s})', [r.get(f'ate_{s}') for r in res])
    print()
    for s in ('all', 'seen', 'unseen'):
        for m, low in (('psnr', False), ('ssim', False), ('depth_l1', True)):
            row(f'{m} ({s})', [(r['render'].get(s) or {}).get(m) for r in res], low)
        print()

    meshes = [r.get('mesh') for r in res]
    if all(meshes):
        voxels = {m['voxel_size'] for m in meshes}
        if len(voxels) > 1:
            print(f'  !! voxel sizes differ ({sorted(voxels)}) - mesh numbers are NOT comparable; '
                  're-run with the same voxel_size')
        else:
            for key, name, low in (('mean precision', 'mesh accuracy (m)', True),
                                   ('mean recall', 'mesh completion (m)', True),
                                   ('recall', 'mesh comp-ratio', False),
                                   ('f-score', 'mesh F-score', False)):
                row(name, [m[key] for m in meshes], low)
    elif any(meshes):
        print('  mesh metrics unavailable for at least one arm')

    print("\n  '+' better than baseline, '-' worse.")
    print("  'unseen' is the row that matters: it is the only evidence the adaptation")
    print('  generalises rather than having memorised the keyframes it trained on.')
