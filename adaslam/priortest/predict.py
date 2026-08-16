"""Producing one arm's frames.csv - the expensive artifact, and the only one inference touches.

Split-independent, so it is cached on the EVAL SPEC alone, written into the file's `#` header line
so a stale cache identifies itself instead of being trusted because the path exists.
"""
import csv
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from ..slam import PriorProbe

from .metrics import FRAME_FIELDS, finish_global, score_frame

FRAMES = 'frames.csv'
_SPEC_PREFIX = '# eval_spec '


def frame_paths(slam_cfg):
    """Every colour frame the tracker would stream, in order. idx is the position in this list.

    The tracker's WINDOW, [start, stop) - the same slice mono_stream takes, so a windowed run is
    scored on the frames it actually tracked and no others.
    """
    files = sorted(os.listdir(slam_cfg.colors))[slam_cfg.start:slam_cfg.stop]
    return [os.path.join(slam_cfg.colors, f) for f in files]


def gt_paths(cfg, slam_cfg, n_frames):
    """GT depth for those same frames. Every consumer indexes it by frame number (10.1).

    Sliced by the SAME window as frame_paths, which is the whole point: taking files[:n_frames]
    from index 0 instead would pair colour frame start+i with GT depth i - invisible at start=0
    and silently wrong at any other start.
    """
    files = sorted(os.listdir(cfg.gt_depths))
    if len(files) < slam_cfg.start + n_frames:
        raise SystemExit(f'{cfg.gt_depths} has {len(files)} depths but the stream needs frames '
                         f'{slam_cfg.start}..{slam_cfg.start + n_frames - 1}; they must be 1:1 by '
                         f'index with {slam_cfg.colors}. Re-run the dataset preprocess script.')
    window = files[slam_cfg.start:slam_cfg.stop]
    return [os.path.join(cfg.gt_depths, f) for f in window[:n_frames]]


def read_cached(path, spec):
    """The rows in `path` if it was built with `spec`, else None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        head = f.readline()
        if not head.startswith(_SPEC_PREFIX) or json.loads(head[len(_SPEC_PREFIX):]) != spec:
            print(f'{path} was built with a different eval spec - re-running inference')
            return None
        return [{k: (int(v) if k in ('idx', 'n_valid') else float(v)) for k, v in row.items()}
                for row in csv.DictReader(f)]


def write_rows(path, spec, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        f.write(_SPEC_PREFIX + json.dumps(spec, sort_keys=True) + '\n')
        w = csv.DictWriter(f, fieldnames=list(FRAME_FIELDS))
        w.writeheader()
        w.writerows(rows)


def build_rows(slam_cfg, cfg, prior, label):
    """Run `prior` over every frame and score it. Returns (rows, global_scale).

    One pass, and the global scale is fitted from the same samples - so the consistency index is a
    ratio of two numbers measured on identical pixels.
    """
    paths = frame_paths(slam_cfg)
    gts = gt_paths(cfg, slam_cfg, len(paths))
    rng = np.random.default_rng(cfg.seed)
    probe = PriorProbe(slam_cfg, prior)
    rows, samples, skipped = [], [], 0

    try:
        for idx, (impath, gtpath) in enumerate(tqdm(list(zip(paths, gts)), desc=label)):
            pred = probe.depth(impath)
            gt = cv2.imread(gtpath, cv2.IMREAD_ANYDEPTH)
            if gt is None:
                raise SystemExit(f'could not read {gtpath}')
            gt = cv2.resize(gt.astype(np.float32) / cfg.depth_png_scale,
                            (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            row, sample = score_frame(idx, pred, gt, cfg, rng)
            if row is None:
                skipped += 1
                continue
            rows.append(row)
            samples.append(sample)
    finally:
        probe.release()      # in a finally: a crashed arm otherwise strands the model

    if not rows:
        raise SystemExit(f'{label}: no frame had usable GT depth in '
                         f'[{cfg.eval_min_depth}, {cfg.eval_max_depth}] m')
    if skipped:
        print(f'  {skipped} of {len(paths)} frames had too little valid GT and were skipped')
    return finish_global(rows, samples)
