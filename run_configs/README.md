# Run configs

`-c` takes a name relative to this folder, so subfolders are addressed by path:

    python scripts/online_adapt_pipeline.py -c m2dgr/street02/live_m2dgr_street02_full_gauge_ctx0_warmvggt

Each config states every parameter; its header says what the run is, why it exists and what to read in its outputs.

| folder | contents |
| --- | --- |
| `init_default.yaml`, `live_default.yaml` | templates - copy these to start a new config |
| `kitti/` | KITTI 00 (frames 0-999 / 1000-1999) gauge family: ctx0 / ctx2, pose-loss variants, rank 16, the init (`init_*`) configs of the in-situ driver, and the no-prior control `live_nomono` |
| `m2dgr/gate01/` | gate_01 (frames 0-2059): gauge ctx0 with Omnidata and VGGT-base warm-up (3 seeds each), batch 10 x 10 passes |
| `m2dgr/street01/` | street_01 full route: plain adaptation, breaker + serve VGGT-base, frozen VGGT repeat and the `keyframe_thresh` sweep (5 / 6 / 8, tracking configs `config/m2dgr_kf*_config.yaml`) |
| `m2dgr/street02/` | street_02: the 660 m window (Omnidata / VGGT warm-up), and the full route - ctx0, ctx2, per-sample gate, breaker, breaker + serve VGGT-base, frozen VGGT repeats |
| `m2dgr/street03/` | street_03: plain adaptation, and plain adaptation with online loop closure (`config/m2dgr_pgba_config.yaml`; seed 3 crashed and is a rerun candidate) |
| `m2dgr/street04/` | street_04: plain adaptation, breaker + serve VGGT-base, frozen VGGT repeats, frozen VGGT with loop closure |
| `_archive/` | superseded or never-run configs, kept for reference - see `_archive/README.md` |

The results of every M2DGR config here are written up in the "M2DGR online adaptation report".
Seed configs sit next to their seed-0 config (`_s2`, `_s3`). Run seed 0 of a new scene first: it also builds that
scene's Omnidata and frozen-VGGT reference arms, which the other seeds reuse through `SKIP_EXISTING`.
