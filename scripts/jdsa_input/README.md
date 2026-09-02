# What JDSA hears from a depth prior — the 2026-08-31 analysis

CPU-only. Every number in the report was produced by the scripts below, in this order, from three
sources and nothing else: the seven `full/slam_depth.npz` dumps under `outputs/extract/`, the raw
`traj_full.txt` of every arm under `outputs/test/end2end/`, and `data/KITTI/00/{depths,traj_tum.txt}`.
No `results.json` or `export.txt` is read anywhere — the point was to re-derive, not to re-quote.

Report (charts + tables): https://claude.ai/code/artifact/82aad0e7-b9bf-45e5-b034-6cd65ca82afc
A copy of the page and its data is in `report/` (`report.html` is the published file; `arms.json`
is the 72-arm table it plots).

Run everything **from the repo root** with the project venv:

    P=/usr/stud/treh/envs/adaslam/bin/python

## The cache

`01_build_cache.py` reduces one 500 MB dump to a ~13 MB per-keyframe cache at the 1/8 grid JDSA
actually runs on: `dp` (served prior disparity), `db` (tracker disparity), `dscales`, `poses`,
`intrinsics`, `tstamp`, and `gt` — the lidar depth resized nearest to the tracking stream and
sampled at the same `[3::8, 3::8]` offsets `depth_video.py:82` uses.

    C=/storage/user/treh/adaslam_analysis/jdsa_input          # already built; $JDSA_CACHE overrides
    $P scripts/jdsa_input/01_build_cache.py \
        outputs/extract/kitti_00_fg2a05_f0-1000/normal/full/slam_depth.npz $C/omni_fg2a05.npz

Seven caches are on disk: `omni_a01`, `vggt_a01` (α=0.01 tracking) and `omni_fg2a05`,
`vggt_fg2a05`, `adapt_fg2a05`, `omni_ceil15_fg2a05`, `omni_fg5` (α=0.05). Scripts 14–16 find them
by name through `$JDSA_CACHE`; 02–12 take cache paths on the command line.

## What each script established

| script | what it measures | the finding it produced |
|---|---|---|
| `02_served_distributions.py` | disparity quantiles of prior / tracker / lidar per frame, fitted `dscales` | the served tail (omni floor 0.145× median, VGGT 0.269×, adapted 0.376×); grid tilt |
| `03_bands_vs_step_scale.py` | keyframe-step scale vs map depth error per range band | every band tracks the step scale at slope ≈1 — the map breathes as a whole |
| `04_aligned_assertion_by_band.py` | JDSA's own weighted scale fit solved against lidar | what each prior *asserts* far away, and how `far_gain` tilts the fit |
| `05_where_the_fusion_binds.py` | aligned prior vs tracker disparity by band, + fresh ATE | the prior is overruled in the far field (ratio 1.5–2.4×) |
| `06_interframe_repeatability.py` | same 3D points across consecutive keyframes, per-frame scale removed | **falsified**: VGGT is 2× more repeatable than Omnidata and tracks worse |
| `07_gauge_series.py` | the per-step gauge change on shared points | it is contaminated by the range-dependence of the error — read 08 instead |
| `08_local_scale_and_segments.py` | local Sim(3) scale in a window, KITTI segment ratios | the wobble, and its −0.97 tie to the map's own 8–20 m depth ratio |
| `09_scale_blindness_and_affine.py` | `dscales × median(disps_prior)` vs the tracker's median; affine fit to lidar | **rescale is a no-op**; every prior's accuracy-optimal disparity offset is *negative* |
| `10_info_redistribution.py` | pixel share vs scale-fit information share, whole frame | 82% of the alignment's leverage sits nearer than the frame median |
| `11_farfield_contamination.py` | 2×2 grid fitted on all pixels vs near pixels only | **falsified** as the mechanism (ranks the clamp below the raw prior); predicted the clamped arm's realised grid tilt to 0.002 |
| `12_row_band_profile.py` | prior and tracker disparity by image row band | the clamp barely moves the fused field — the effect is not a redrawn map |
| `13_measure_all_arms.py` | ATE, wobble, 100 m chord for every arm on disk | **ATE = 66 × wobble**, corr +0.988 over 71 arms |
| `14_distribution_stats_vs_wobble.py` | eight summary statistics of the served disparity vs realised wobble | no scalar of the served distribution predicts across priors — it is two factors |
| `15_assertion_vs_ate.py` | asserted depth / true depth per transform, against measured ATE | the accuracy paradox, and that the far-field profile alone does not separate the arms |
| `16_placement_table.py` | the same for candidate transforms, plus near-field cost | placed the `@soft` / `@mask` sweep in `run_configs/live_fg2a05_softmask.yaml` |
| `17_dump_arms_json.py` | writes `report/arms.json` | the scatter in the report |
| `18_gtfree_profile.py` | the same anatomy with NO GT: served tail, pixel/info share, and the aligned prior against the tracker, binned by the prior's own depth | RELLIS's tail is short (p99 2.4× median) where KITTI's is long (5.8×) |
| `19_infer_the_tag.py` | can the tag be **inferred**? locates each tag-curve's optimum, reads nine statistics there, inverts the winner per prior, and scores it over all 30 frozen arms | the **push** rule: aligned prior / tracker disparity on the tracker's own far pixels is 1.89/1.89/1.95 at the three optima; inverting it gives 1.42 vs 1.435 measured (omni@ceil) and 1.79 vs 1.862 (omni@ped); R² 0.53 over the whole sweep, so necessary, not sufficient |

Typical invocations:

    $P scripts/jdsa_input/09_scale_blindness_and_affine.py $C/omni_fg2a05.npz $C/vggt_fg2a05.npz
    $P scripts/jdsa_input/13_measure_all_arms.py kitti_00_fg2a05_f0-1000 kitti_00_f0-1000
    $P scripts/jdsa_input/16_placement_table.py
    $P scripts/jdsa_input/18_gtfree_profile.py $C/omni_fg2a05.npz $C/rellis_omni.npz
    $P scripts/jdsa_input/19_infer_the_tag.py            # ~4 min, the bisections are the slow part

Another scene needs its own GT: `01_build_cache.py <npz> <out> <gt_depth_dir> <png_scale>` and
`JDSA_GT=data/RELLIS/00000/traj_tum.txt` for `13_measure_all_arms.py`. That script also drops
windows covering less than `$JDSA_MIN_WIN_M` (3 m) of GT path — without the guard a scene with
stops reports noise as wobble (RELLIS: corr −0.33 without, +0.83 with).

## What came out of it, in the tree

* `@soft<tag>` and `@mask<tag>` prior-spec modifiers — `adaslam/end2end/{config,prior}.py`.
  `@ceil`, `@soft`, `@ped` are `q' = (q^k + b^k)^(1/k)` at k = ∞, 2, 1; `@mask` serves 0 so JDSA's
  `m` gates the pixel out entirely.
* `run_configs/live_fg2a05_softmask.yaml` — nine arms (~54 min), six references reused. Its header
  carries the rule the tags come from and what each arm would falsify. First result in:
  `omni@soft1p7` → 4.21 m (against 11.40 untransformed).

## Limits to carry forward

The lidar covers 56% of the frame, none of the sky, ~45 pixels per keyframe past 40 m — every
far-field ratio rests on those. Scripts 14–16 apply a candidate transform to a prior recorded under
a *different* run's keyframes and tracker field, so they are counterfactual screens, not runs.
Every ATE is a single draw; run-to-run spread is ~0.08 m (`base_ped1` seeds: 3.62 / 3.64 / 3.70).
