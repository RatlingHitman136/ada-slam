# Archived run configs

Moved out of `run_configs/` to keep the active set readable. Nothing here was
deleted: `mv _archive/<name>.yaml ..` puts any of it back, and every one of these
still has its outputs under `outputs/`.

## pedestal / ceiling family - explicitly scoped out as a separate axis from adaptation

- `init_fg2a05_f0k1k_ped1_insitu.yaml`
- `init_fg2a05_f1k2k_ped0p8_insitu.yaml`
- `live_fg2a05_f1k2k_ped1p35_e5.yaml`
- `live_fg2a05_f1k2k_pedfine.yaml`
- `live_fg2a05_omniped.yaml`
- `live_fg2a05_ped1_e5.yaml`
- `live_fg2a05_ped1p4_e5.yaml`
- `live_fg2a05_pedfine.yaml`

## full-sequence runs - excluded from the report

- `live_full_adapt.yaml`
- `live_full_omni.yaml`
- `live_full_omni_tail.yaml`
- `live_full_vggt.yaml`

## superseded objectives - every one landed at baseline, and normalize_target replaced them

- `live_anchor_ctx0.yaml`
- `live_b10_allframes.yaml`
- `live_b10_allframes_r1.yaml`
- `live_b10_allframes_r2.yaml`
- `live_b10_allframes_r3.yaml`
- `live_b10_bscale.yaml`
- `live_b10_bscale_r2.yaml`
- `live_b10_bscale_r3.yaml`
- `live_b10_cons.yaml`
- `live_b10_frozen.yaml`
- `live_b10_frozengauge.yaml`
- `live_b10_frozengauge_r2.yaml`
- `live_b10_frozengauge_r3.yaml`
- `live_b10_nocs.yaml`
- `live_ctx2_cscale.yaml`
- `live_fg2a05_softmask.yaml`
- `live_one_target.yaml`

## lambda_pose family - measured a dead axis (term saturates in 12 steps)

- `live_ctx2_lp2.yaml`
- `live_ctx2_lp3.yaml`
- `live_ctx2_lp4.yaml`
- `live_ctx2_r16_lp2.yaml`
- `live_ctx2_r16_lp2_lr1e5.yaml`
- `live_ctx2_r16_lp2_lr5e5.yaml`
- `live_ctx2_r16_lp2_r1.yaml`
- `live_ctx2_r16_lp2_r2.yaml`
- `live_ctx2_r16_lp3.yaml`
- `live_ctx2_r16_lp4.yaml`

## pre-gauge ctx2 seed/variant sweep

- `live_ctx2_b10.yaml`
- `live_ctx2_b10_r3.yaml`
- `live_ctx2_b10_r4.yaml`
- `live_ctx2_s3.yaml`
- `live_ctx2_s5.yaml`
- `live_ctx3.yaml`

## one-off dumps

- `init_fg2a05_f1k2k_basedump.yaml`


## M2DGR configs prepared but never run (`m2dgr_not_run/`, archived 2026-09-27)

Each is runnable as `-c _archive/m2dgr_not_run/<name>`; none has outputs.

- `live_m2dgr_street02_full_gauge_ctx0_warmvggt_gate1_s2`, `_s3` - seeds 2 and 3 of the per-sample loss gate; seed 0
  showed it refuses the right samples but changes nothing (121.5 vs 121.4 m), so the other seeds were not worth ~11 h.
- `live_m2dgr_street02_full_gauge_ctx2_warmvggt_s3` - the third ctx2 seed on the full street_02 route; never launched,
  so the ctx2 family has 2 seeds.
- `live_m2dgr_street04_full_gauge_ctx0_warmvggt_pgba`, `_s2`, `_s3` - adaptation with online loop closure on street_04;
  not run because frozen VGGT with loop closure logged zero loop closures there (the drifted map hides the closing loop).
