"""TMP DIAGNOSTIC - what each depth prior does where the lidar GT cannot see it.

    python scripts/tmp_sky_probe.py                  # every arm in ARMS, one at a time
    python scripts/tmp_sky_probe.py omni base        # a subset, when the GPU is contended
    python scripts/tmp_sky_probe.py --stride 100     # fewer frames

WHY THIS EXISTS. The prior stage scores a generator only where the velodyne returned, and on KITTI
the HDL-64E's upper field of view ends about a third of the way down the frame. At the tracking
resolution that leaves the top quarter-to-third of the rows with no GT at all - and BA does not
skip them: depth_video.py:70-73 hands the solver `disps_prior[3::8, 3::8]`, and a large share of
those points falls in that band. The exact figures depend on which frames are sampled (the band is
the UNION of every sampled frame's GT), so they are measured and printed at run time rather than
quoted here - on this window it is ~25% of the rows and ~39% of BA's points. Either way every
depth number the prior test reports is blind to a large fraction of what BA is actually fed, and
"the adapters predict better depth but every arm loses ATE to Omnidata" could be decided entirely
in the unscored region.

WHAT IT MEASURES. Per arm, over a stride of frames, split into the two regions:

    band  the rows where GT exists       (what the prior test scores)
    top   the rows above the topmost beam (what it cannot)

    top/band     median depth in the top region over median depth in the band. THE HEADLINE.
                 A prior that puts the sky barely further away than the road reads ~1.5; one that
                 predicts a genuinely distant sky reads 10x or more.
    disp spread  p95/p5 of INVERSE depth over exactly the [3::8,3::8] grid BA consumes - how much
                 geometry the prior asserts at all. The regulariser hypothesis says a prior that
                 asserts little is a better damping term and biases the solver less (ba.py:206
                 has alpha REPLACE the network's per-pixel damping eta wherever the prior is
                 valid, which for every prior here is everywhere).
    top CV       spread of the per-frame top median across frames - is the unscored region even
                 stable, or is it noise the tracker has to absorb?

READ RATIOS, NOT ABSOLUTE DEPTHS. Omnidata's output is relative depth scaled x50 in
motion_filter.py and VGGT's is in its own units; only within-frame ratios and cross-frame spreads
compare across arms. That is why no metre value is printed.

It goes through slam.PriorProbe, so it calls the very extractor a real arm calls (9.3) rather than
a re-implementation - a probe that resized differently would report numbers no arm produces.

Arms run in SEPARATE PROCESSES by default: VGGT plus the Omnidata normal branch is ~6.6 GiB and
the allocator does not hand it all back between arms, so on a shared box a single process OOMs
partway through and loses the arms it had already done. Each arm writes its row to OUT as it
finishes, so an interrupted run keeps what it got. --inproc forces the old single-process
behaviour if you have the card to yourself.

NOT A STAGE. Nothing under adaslam/ imports this; it answers one question and can be deleted.
"""
import os    # nopep8
import sys   # nopep8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # nopep8
# repo root, so `adaslam` imports; its __init__ adds hislam2/ and thirdparty/vggt
sys.path.insert(0, _ROOT)                                                 # nopep8
import argparse
import json
import subprocess

import cv2
import numpy as np

from adaslam.adapt import LoRAConfig
from adaslam.common import experiment_dir
from adaslam.end2end import End2EndConfig
from adaslam.pipeline import enter, resolve_lora
from adaslam.runtime import ensure_venv_on_path, raise_fd_limit
from adaslam.slam import SlamConfig

# ==============================================================================
#  PARAMETERS
# ==============================================================================

SCENE_KEY = 'kitti_00_f0-1000'         # names the outputs/ tree the adapters live in
DATA      = 'data/KITTI/00'
COLORS    = f'{DATA}/colors'
DEPTHS    = f'{DATA}/depths'           # only to locate the GT/no-GT boundary, never scored against
CALIB     = f'{DATA}/calib.txt'
START, STOP = 0, 1000                  # the window the arms were run over
STREAM_RES  = 341 * 640
STRIDE      = 25                       # every Nth frame of the window; 40 frames at 25
DEPTH_PNG_SCALE = 256.0

OUT = f'outputs/test/prior/{SCENE_KEY}/tmp_sky_probe.json'   # rows appended as arms finish


def _a(name):
    return experiment_dir('outputs', 'adapt', SCENE_KEY, name)


# label -> prior spec, exactly the vocabulary END2END_PRIORS uses. Order is the print order;
# 'omni' first because it is the arm every VGGT one has to be read against.
ARMS = {
    'omni':          'omnidata',
    'base':          'vggt_base',
    'live_e15_lag5': _a('live_e15_w10_a16_w12_lag5_low045_raw_base'),
    'live_e40_lag3': _a('live_e40_a16_w12_lag3_low045_raw_base'),
    'live_e3_lag3':  _a('live_e3_w10_a16_w12_lag3_low045_raw_base'),
    'wonline_e12':   _a('wonline_a16_e12_w10_p10'),
}

SLAM = SlamConfig(
    weights='pretrained_models/droid.pth', colors=COLORS, calib=CALIB,
    start=START, stop=STOP, undistort=False, crop_border=0,
    stream_res=STREAM_RES, render_eval=False)

LORA = LoRAConfig(
    weights='pretrained_models/vggt', vggt_hw=None,      # None -> derived from the stream
    rank=8, alpha=16,
    targets=('attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'), patch_embed=False)

# ==============================================================================

raise_fd_limit()
ensure_venv_on_path()


def regions(gt_paths, hw):
    """(band, top) boolean masks at tracking resolution, plus the per-frame GT masks.

    `band` is every pixel that has GT in ANY of the sampled frames, `top` the rows above all of
    them. Taking the union over frames rather than per-frame keeps the two regions fixed, so a
    per-arm median is over the same pixels every time and the arms stay comparable.
    """
    h, w = hw
    any_gt = np.zeros((h, w), bool)
    for p in gt_paths:
        g = cv2.imread(p, cv2.IMREAD_ANYDEPTH)
        if g is None:
            raise SystemExit(f'could not read {p}')
        any_gt |= cv2.resize(g.astype(np.float32) / DEPTH_PNG_SCALE, (w, h),
                             interpolation=cv2.INTER_NEAREST) > 0
    rows = np.where(any_gt.any(1))[0]
    top = np.zeros((h, w), bool)
    top[:rows.min()] = True                       # strictly above the topmost beam anywhere
    return any_gt, top, int(rows.min()), int(rows.max())


def probe_arm(label, spec, frames, files, band, top):
    """One arm -> its row. Builds the prior, walks the frames, releases everything."""
    from adaslam.end2end.stage import make_prior
    from adaslam.slam import PriorProbe

    e2e = End2EndConfig(
        priors=(spec,), length=1, buffer=8, gt_traj=f'{DATA}/traj_tum.txt', lora=LORA,
        omni_normal_ckpt='pretrained_models/omnidata_dpt_normal_v2.ckpt',
        omni_normal_hw=(512, 512))
    prior = make_prior(spec, e2e, band.shape) if spec != 'omnidata' else None
    probe = PriorProbe(SLAM, prior)

    t_med, b_med, t_p95, t_min, spread = [], [], [], [], []
    try:
        for t in frames:
            d = probe.depth(os.path.join(COLORS, files[t]))
            t_med.append(np.median(d[top]))
            b_med.append(np.median(d[band]))
            t_p95.append(np.percentile(d[top], 95))
            t_min.append(d[top].min())
            # exactly the grid BA is handed (depth_video.py:72), in INVERSE depth, which is the
            # quantity JDSA's residual is written in
            sub = 1.0 / np.clip(d[3::8, 3::8], 1e-6, None)
            spread.append(np.percentile(sub, 95) / max(np.percentile(sub, 5), 1e-9))
    finally:
        probe.release()

    t_med, b_med = np.array(t_med), np.array(b_med)
    return dict(arm=label, spec=spec, n_frames=len(frames),
                top_med=float(np.median(t_med)), band_med=float(np.median(b_med)),
                top_over_band=float(np.median(t_med) / np.median(b_med)),
                top_p95=float(np.median(t_p95)), top_min=float(np.median(t_min)),
                disp_spread=float(np.median(spread)),
                top_cv=float(t_med.std() / t_med.mean()))


HEAD = (f'{"arm":<16}{"top/band":>10}{"disp p95/p5":>13}{"top CV":>9}'
        f'{"top med":>10}{"band med":>10}{"top p95":>10}{"top min":>10}')


def row_line(r):
    return (f'{r["arm"]:<16}{r["top_over_band"]:>10.2f}{r["disp_spread"]:>13.1f}'
            f'{r["top_cv"]:>9.3f}{r["top_med"]:>10.3f}{r["band_med"]:>10.3f}'
            f'{r["top_p95"]:>10.3f}{r["top_min"]:>10.4f}')


def load_rows():
    return json.load(open(OUT)) if os.path.exists(OUT) else []


def save_row(r):
    rows = [x for x in load_rows() if x['arm'] != r['arm']] + [r]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rows, open(OUT, 'w'), indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arms', nargs='*', default=None,
                    help=f'subset of {list(ARMS)}; default all of them')
    ap.add_argument('--stride', type=int, default=STRIDE, help='every Nth frame of the window')
    ap.add_argument('--inproc', action='store_true',
                    help='run every arm in THIS process - faster, but one OOM loses the run')
    ap.add_argument('--child', default=None, help=argparse.SUPPRESS)   # internal: one arm, in-proc
    args = ap.parse_args()

    enter(_ROOT)
    for p in (COLORS, DEPTHS, CALIB):
        if not os.path.exists(p):
            raise SystemExit(f'missing input: {p}  (run scripts/preprocess_kitti.py --with-depth)')

    want = args.arms or list(ARMS)
    bad = [a for a in want if a not in ARMS]
    if bad:
        raise SystemExit(f'unknown arm(s) {bad}; choose from {list(ARMS)}')

    global LORA
    LORA, stream_hw = resolve_lora(LORA, COLORS, STREAM_RES)
    files = sorted(os.listdir(COLORS))[START:STOP]
    frames = list(range(0, len(files), args.stride))
    band, top, r0, r1 = regions([f'{DEPTHS}/{START + t:06d}.png' for t in frames], stream_hw)

    if args.child:
        r = probe_arm(args.child, ARMS[args.child], frames, files, band, top)
        save_row(r)
        print(row_line(r), flush=True)
        return

    sub = np.zeros(stream_hw, bool)[3::8, 3::8]
    ba_band = band[3::8, 3::8]
    print(f'stream    : {stream_hw[1]}x{stream_hw[0]}   VGGT input {LORA.vggt_hw[1]}x'
          f'{LORA.vggt_hw[0]}   {len(frames)} frames, stride {args.stride}')
    print(f'GT rows   : {r0}..{r1} of {stream_hw[0]}   -> top {r0} rows ({100*r0/stream_hw[0]:.0f}%'
          f' of the frame) never have GT')
    print(f'BA subsample ([3::8,3::8] = {sub.size} points, depth_video.py:72): '
          f'{100*(~ba_band).mean():.0f}% of them fall where the prior test cannot score\n')
    print('top/band  = median depth above the beams / median depth where GT exists.')
    print('            Ratios only - Omnidata is relative depth x50, VGGT is in its own units.')
    print('disp p95/p5 = spread of INVERSE depth over the BA grid: how much geometry the prior')
    print('            asserts at all. Low = closer to a flat damping term than to a scene.\n')
    print(HEAD)
    # children write straight to the terminal; without this the header lands after their rows
    sys.stdout.flush()

    done = {}
    for a in want:
        if args.inproc:
            r = probe_arm(a, ARMS[a], frames, files, band, top)
            save_row(r)
            print(row_line(r), flush=True)
            done[a] = r
            continue
        # a fresh process per arm: the allocator keeps ~6.6 GiB of VGGT + Omnidata-normal cached
        # and only process exit reliably returns it on a shared card
        p = subprocess.run([sys.executable, os.path.abspath(__file__), '--child', a,
                            '--stride', str(args.stride)], cwd=_ROOT)
        if p.returncode != 0:
            print(f'{a:<16}  FAILED (rc={p.returncode}) - likely OOM; retry this arm alone',
                  flush=True)

    rows = {r['arm']: r for r in load_rows()}
    got = [rows[a] for a in want if a in rows]
    if got:
        print(f'\n=== every requested arm on disk in {OUT} ===')
        print(HEAD)
        for r in got:
            print(row_line(r))
        base = got[0]
        if len(got) > 1:
            print(f'\ntop/band against {base["arm"]}: '
                  + ',  '.join(f'{r["arm"]} {r["top_over_band"]/base["top_over_band"]:.1f}x'
                               for r in got[1:]))
    missing = [a for a in want if a not in rows]
    if missing:
        print(f'\nno row for {missing} - rerun those alone, the card was probably full')
    print(f'\nwrote {OUT}')
    print('\nreading it: the hypothesis is that the prior BA rewards is the one that asserts the')
    print('LEAST in the unscored band. If omni sits near 1.5 while the VGGT arms sit an order of')
    print(f'magnitude above it, the ATE gap is decided on the {100*(~ba_band).mean():.0f}% of BA'
          ' points no depth metric on this')
    print('scene can see, and mono_depth_alpha is the knob that controls how much it costs.')


if __name__ == '__main__':
    main()
