# ada-slam / HI-SLAM2 — code map

A file-by-file description of this repository. HI-SLAM2 ([arXiv:2411.17982](https://arxiv.org/pdf/2411.17982))
is a **monocular** SLAM system that produces both a camera trajectory and a 3D Gaussian
Splatting map. It is built from three lineages, and the folder layout mirrors them:

| Lineage | What it contributes | Where it lives |
|---|---|---|
| **DROID-SLAM** | dense flow-based tracking, factor graph, dense BA | `hislam2/{factor_graph,track_*,depth_video,motion_filter}.py`, `hislam2/geom`, `hislam2/modules`, `src/` |
| **Omnidata / DPT (MiDaS)** | monocular depth + normal priors | `hislam2/midas/` |
| **MonoGS / 3DGS / RaDe-GS** | Gaussian map, rasterizer, GUI | `hislam2/gaussian/`, `thirdparty/diff-gaussian-rasterization` |
| **VGGT** *(this fork, §9)* | alternative depth prior, LoRA-adapted on SLAM depth | `thirdparty/vggt`, `adaslam/`, `scripts/run_pipeline.py` |

The HI-SLAM2-specific contributions on top of those are: **JDSA** (joint depth–scale
adjustment, `geom/ba.py`), **PGBA** (Sim(3) pose-graph + bundle adjustment for loop closure,
`pgo_buffer.py` + `droid_backends.pgba`), and the geometry-aware Gaussian losses
(`gaussian/utils/slam_utils.py`).

This fork adds one research track on top: replacing the Omnidata depth prior with a VGGT model
LoRA-adapted on HI-SLAM2's *own* SLAM depth. That is entirely additive — it lives in
`adaslam/`, `scripts/` and `thirdparty/vggt`, and touches the core only through a 17-line
optional dump hook in `hi2.py`. See **§9**.

---

## 1. Pipeline at a glance

```
demo.py
  └─ reader Process ──queue──▶ main loop
                                 │
                                 ▼
                          Hi2.track(t, image)          hislam2/hi2.py
                                 │
        ┌────────────────────────┼──────────────────────────────┐
        ▼                        ▼                              ▼
  MotionFilter            TrackFrontend                    GSBackEnd
  motion_filter.py        track_frontend.py                gs_backend.py
  • flow magnitude        • FactorGraph edges              • new Gaussians from
    → keyframe?           • dense BA (droid_backends.ba)     RGB + SLAM depth
  • Omnidata depth        • JDSA depth/scale alignment     • 10 map iters / kf
    + normal prior        • keyframe pruning               • densify / prune
        │                        │                              │
        └──────▶ DepthVideo ◀────┘                              │
                 depth_video.py  (shared-memory state)          │
                        ▲                                       │
                        │  PGOBuffer.spin()  (separate process) │
                        │  pgo_buffer.py — loop-closure search  │
                        │  + Sim(3) PGBA, pushes pose/scale ────┘
                        │  corrections into the Gaussian map
                        ▼
                 Hi2.terminate()
                   • extra keyframes in low-covisibility gaps
                   • global BA (TrackBackend, 4 then 8 steps)
                   • GS colour refinement + pose/exposure opt
                   • fill non-keyframe poses (PoseTrajectoryFiller)
                   • render + evaluate
                        │
                        ▼
              outputs/<seq>/  →  tsdf_integrate.py  →  mesh
```

Two resolutions are used throughout: images are resized so `H*W ≈ 341*640` with both
dims divisible by 8; **tracking runs at 1/8 resolution** (`disps`, `fmaps`, correlation),
while the priors and the Gaussian map are *produced* at full resolution (`disps_up`).

Produced, not necessarily consumed: the **depth prior is consumed by BA at 1/8**
(`disps_prior`), and the full-res `disps_prior_up` reaches nothing. Only the normals and the
Gaussian map are genuinely full-resolution end to end. §9.6 has the whole chain with numbers —
it is the difference that decides what a depth-prior model's output resolution can buy.

---

## 2. Repository root

| File | Purpose |
|---|---|
| `demo.py` | **Upstream's entry point**, and **deliberately frozen**. Spawns a reader `Process` that decodes/resizes/undistorts images into a `Queue`, constructs `Hi2` lazily on the first frame (needs the image size), loops `hi2.track(...)` until the last frame, then `hi2.terminate()` and writes `traj_kf.txt`, `traj_full.txt`, `intrinsics.npy`. CLI: `--imagedir --calib --config --output --gtdepthdir --buffer --undistort --cropborder --start --length --droidvis --gsvis --dump_slam_depth`. `run_replica.py` / `run_scannet.py` drive that CLI as a subprocess, which is why it is left alone. The fork's own copy of this loop is `adaslam/slam/runner.py`, which every stage of §9 goes through; `demo.py` is the only other place in the repo that touches `Hi2`. |
| `tsdf_integrate.py` | Post-process: fuses the **rendered** depth/colour images from `outputs/<seq>/renders/*_after_opt` with `traj_full.txt` into an Open3D `VoxelBlockGrid` and extracts a triangle mesh (`tsdf_mesh_w<weight>.ply`). Depth PNGs are 16-bit scaled by 6553.5. |
| `setup.py` | Builds two CUDA extensions: `droid_backends` (from `src/`) and `lietorch` (from `thirdparty/lietorch`). Fork change: gencode fixed to `compute_89/sm_89` (CUDA 13 rejects Pascal/Volta arches). Note it calls `setup()` **twice**, so `pip install .` would only build the first — use `python setup.py install`. |
| `setup_env.sh` | One-shot environment bootstrap: installs `uv`, syncs submodules (`--force`), applies `patches/lietorch.patch`, loads the `cuda/13.0.1` lmod module, creates `.venv`, installs `requirements.txt`, builds all four extensions, optionally downloads the Omnidata weights (`--with-weights`), then verifies real kernels execute. Idempotent; `--force-rebuild` recompiles. |
| `requirements.in` | Hand-edited dependency source. Only load-bearing pins: `torch==2.9.0+cu130`, `torchvision==0.24.0+cu130`, `torch-scatter==2.1.2+pt29cu130` (newest torch with a prebuilt cu130 torch-scatter wheel), `opencv-python<5`, `numpy<3`. Documents why `--index-strategy unsafe-best-match` is mandatory. |
| `requirements.txt` | Fully pinned lockfile compiled from `requirements.in`. |
| `README.md` | Upstream README with the Getting Started section rewritten for this fork's CUDA 13 setup. |
| `LICENSE` | Upstream licence. |
| `.gitmodules` | Submodules: `lietorch`, `eigen`, `simple-knn`, `glm` (inside diff-gaussian-rasterization). |
| `data` → `/storage/user/treh/data` | Symlink to datasets on big storage. |
| `outputs` → `/storage/user/treh/adaslam_outputs` | Symlink to results on big storage. |
| `.venv` → `/usr/stud/treh/envs/adaslam` | Symlink to the venv. |
| `build/`, `droid_backends.egg-info/` | Compilation artifacts (ninja objects, built `.so`s). Not source. |

Directories:

| Dir | Contents |
|---|---|
| `adaslam/` | The VGGT track's own code — every stage of the §9 pipeline, as importable packages (§9.5). `slam/` is the **single interface to HI-SLAM2** and the only package here that imports `Hi2` or `MotionFilter`; `extract/`, `adapt/`, `end2end/` and `priortest/` are the four stages; `common.py`, `runtime.py` and `paths.py` hold what more than one of them needs. It is a **real package** — `from adaslam.adapt import ...`, `from adaslam.common import ...`. Unlike `hislam2/`, which has no top-level `__init__.py` and so can only be a `sys.path` entry, this one has one, and that `__init__` is where `hislam2/` and `thirdparty/vggt` get put on `sys.path` (§9.5). It was `ada-slam/` until the hyphen — illegal in an identifier — was the only thing forcing the same treatment as `hislam2/`. |
| `calib/` | Plain-text intrinsics, one line: `fx fy cx cy [k1 k2 p1 p2 ...]`. `replica.txt` (600 600 599.5 339.5), `scannet.txt`, `euroc.txt` (with distortion). Loaded by `demo.py:mono_stream`. |
| `config/` | Per-dataset YAML (see §6). |
| `media/` | README images (`logo.png`, `teaser.jpg`, `owndata.gif`). |
| `patches/` | `lietorch.patch` — CUDA 13 fixes for the lietorch submodule. Because lietorch is a submodule, git can only record a commit pointer, so the fixes ship as a patch applied by `setup_env.sh`. **Edit the patch, not `thirdparty/lietorch/` — the setup script force-resets that tree.** |
| `pretrained_models/` | `droid.pth` (16 MB, DROID-SLAM tracking net, in git), `omnidata_dpt_depth_v2.ckpt` + `omnidata_dpt_normal_v2.ckpt` (1.9 GB each, downloaded), `vggt/` (VGGT-1B, ~4.7 GB, §9). `.gitignore` covers `*.pth`, `*.ckpt` and `*.safetensors` so none of these are committable — `droid.pth` is tracked only because it predates the rule. |
| `scripts/` | Dataset prep, evaluation drivers (§5) and the VGGT track (§9). |
| `src/` | CUDA/C++ sources for `droid_backends` (§4). |
| `thirdparty/` | Submodules: `eigen`, `lietorch`, `simple-knn` (kNN for initial Gaussian scales), `diff-gaussian-rasterization` (RaDe-GS rasterizer), `vggt` (pinned at `a288dd0`, §9). |

---

## 3. `hislam2/` — core logic

### 3.1 Orchestration

**`hi2.py` — the `Hi2` class, the system's spine.**
Constructs and wires every component: loads `droid.pth` into `DroidNet` (truncating the
`update.weight/delta` heads from 3 to 2 channels — this fork's net predicts 2D flow only),
creates the shared `DepthVideo`, `MotionFilter`, `TrackFrontend`, `TrackBackend`,
`GSBackEnd`, `PoseTrajectoryFiller`, and optionally the Open3D visualiser process and the
`PGOBuffer` background process.

- `track()` — per frame: motion filter → frontend local BA → if PGBA is active, apply any
  pending loop-closure correction and push the resulting pose/scale deltas to the Gaussian
  map → push newly optimised keyframes to the Gaussian map via `call_gs`.
- `call_gs()` — packages keyframe state (poses, images, normals, `1/disps_up` as depth,
  intrinsics ×8 back to full res, optional pose/scale updates) and calls the Gaussian
  backend **synchronously**.
- `terminate()` — the long tail of the pipeline: stops PGBA; finds keyframe gaps where
  covisibility exceeds `covis_thresh` and **inserts new keyframes** there (poses from the
  trajectory filler, priors from Omnidata, then `video.shift` to make room); runs global BA
  twice (4 and 8 steps); computes `dposes`/`dscale` and rigidly corrects the Gaussian map;
  runs the final Gaussian colour refinement (which also refines camera poses, and writes
  them back into `video.poses`); fills all non-keyframe poses; renders and evaluates.
  Between global BA and the Gaussian refinement sits the **only core change this fork's VGGT
  track makes**: a 17-line block guarded by `--dump_slam_depth` that writes `slam_depth.npz`
  (§9). It sits exactly there because that is the *only* instant where `disps`, `disps_up` and
  `poses` are mutually consistent — `PoseTrajectoryFiller` later re-upsamples every keyframe's
  `disps_up` (via `factor_graph.py:231`, which runs unconditionally), and the Gaussian
  refinement overwrites `video.poses`.

**`depth_video.py` — `DepthVideo`, the shared state store.**
Every buffer is preallocated to `--buffer` keyframes and `share_memory_()`d so the frontend,
the PGBA process and the visualiser process see the same tensors. Guarded by
`counter.get_lock()`.

State per keyframe: `tstamp`, `images` (CPU uint8), `poses` (SE3 world→cam, 7-vec),
`poses_sim3` (8-vec, used only during PGBA), `disps` (1/8 res inverse depth),
`disps_up` (full res, CPU), `disps_prior` / `disps_prior_up` (Omnidata inverse depth),
`intrinsics` (÷8), `normals` (full res, CPU), `dscales` (**2×2 grid** of prior-depth scale
factors per keyframe), `doffset`. Features: `fmaps` (correlation), `nets`/`inps` (GRU
hidden state and context), all fp16 at 1/8 res.

Methods: `append` / `__setitem__` / `shift` (insert or delete a slot, shifting everything),
`upsample` (convex upsampling of `disps` → `disps_up`), `normalize` (rescale the whole map
so mean disparity × `scale_multiplier` = 1 — this fixes the arbitrary monocular scale after
initialisation), `reproject`, `distance` (flow-based frame distance, used to pick edges),
`distance_covis` (covisibility ratio, used for keyframe decisions), `cuda_ba` (calls
`droid_backends.ba`, then optionally **JDSA**), `cuda_pgba` (Sim(3) pose-graph BA with
relative-pose constraints, calls `droid_backends.pgba`).

**`motion_filter.py` — `MotionFilter`.**
Decides which incoming frames become keyframes. Encodes the frame with `fnet`, builds a
correlation volume against the last keyframe, runs **one** GRU update, and thresholds the
mean flow magnitude against `thresh` (or `init_thresh` before initialisation). When a
keyframe is accepted it also runs `prior_extractor` — the two Omnidata DPT models at
512×512, depth scaled ×50 and interpolated back — and the context encoder, then appends to
`DepthVideo`. `skip_blur` keeps a rolling 5-frame cache scored by Laplacian variance and
substitutes the sharpest frame. `self.deltas` accumulates per-frame flow, later used by
`Hi2.terminate` to place new keyframes at flow-midpoints.

**`track_frontend.py` — `TrackFrontend`.** Local windowed optimisation, one call per frame.
- `__initialize()` — fires once `warmup` (12) keyframes exist: neighbourhood factors (r=3),
  8 update iterations, then proximity factors and 8 more (JDSA on from iteration 3),
  drops keyframes with too little motion, re-optimises, and calls `video.normalize()`.
- `__update()` — ages out factors older than `max_age` (25), adds proximity factors over the
  last `frontend_window` (25) keyframes, seeds `dscales` from the median disparity ratio,
  runs `iters1`(4) updates with JDSA, then decides whether the second-to-last keyframe is
  redundant (both flow distance < `keyframe_thresh` **and** covisibility < 0.1 → remove it),
  otherwise `iters2`(2) more updates. Returns the indices whose depth/pose changed, which
  `Hi2` forwards to the Gaussian backend.

**`track_backend.py` — `TrackBackend`.** Global BA, called only from `terminate()`.
Builds a fresh `FactorGraph` with the memory-efficient `"alt"` correlation implementation
over all keyframes (up to `20*t` factors) and runs `update_lowmem`.

**`factor_graph.py` — `FactorGraph`.** The optimisation graph shared by frontend, backend
and PGBA. Holds edge lists `ii`/`jj`, per-edge `target` (predicted correspondence) and
`weight` (confidence), edge `age`, per-pixel `damping`, plus an *inactive* edge set
(`ii_inac`, …) that keeps old measurements contributing to BA without recomputing features.
- `add_factors` / `rm_factors` / `rm_keyframe` — edge bookkeeping; `rm_factors(store=True)`
  also hands the retired edges to `PGOBuffer.add_rel_poses` as pose-graph constraints.
- `add_neighborhood_factors` — all pairs within index radius r.
- `add_proximity_factors` — the interesting one: scores all candidate pairs by
  `video.distance`, applies non-maximum suppression in edge space, forces edges between
  temporal neighbours, and (in backend mode) rejects pairs whose relative rotation exceeds
  150°. This is what discovers loop closures.
- `update` — one iteration: reproject → motion features → GRU update op → new
  `target`/`weight`/`damping`/`upmask` → dense BA (`video.cuda_ba`) → convex upsample.
- `update_lowmem` — same, but streams the correlation volume in chunks of 2 keyframes with
  `AltCorrBlock`, so global BA over hundreds of keyframes fits in VRAM.
- `update_pgba` — the Sim(3) variant: reprojects with `poses_sim3`, calls `video.cuda_pgba`,
  and afterwards **rescales** poses, disparities, prior scales and relative poses by the
  recovered per-keyframe scale, then writes Sim(3) back into SE(3).

**`pgo_buffer.py` — `PGOBuffer` + loop closure.** Runs `spin()` in its own process.
- `spin()` — for each new keyframe (once ≥60 exist), searches for loop candidates ~55
  keyframes back via `search_lc_candidate` (flow distance < `pgba_thresh` and relative
  rotation < 120°). Once ≥24 candidate edges accumulate (or one has waited >3 keyframes),
  it pushes them through `LC_data_queue` to the main process.
- `add_rel_poses()` — converts retired BA edges into pose-graph constraints: 4 Gauss-Newton
  iterations on the relative pose, plus a **covariance** from the residual and `H⁻¹`, stored
  in shared `rel_*` buffers.
- `_pgba()` — the main-process handler: builds a graph from the loop edges plus the current
  frontend edges, copies SE(3) into Sim(3), runs `update_pgba`, then re-runs 6 frontend
  updates to re-settle the local window.
- `global_relative_posesim3_constraints()` — assembles the H/v blocks for the relative-pose
  residual `log(Gij · Gi · Gj⁻¹)` using **numerical** Jacobians (`num_jacobi`, central
  differences in float64), weighted by the inverse covariances.

**`gs_backend.py` — `GSBackEnd`, the Gaussian mapping backend.**
Despite subclassing `mp.Process`, it is **never started as a process** — `Hi2` calls it
inline, so mapping is synchronous with tracking. The optional GUI *is* a separate process.
- `process_track_data(packet)` — builds the projection matrix on first call; applies any
  pose/scale correction from PGBA or global BA directly to the Gaussians (rotate+translate
  `_xyz`, divide `_scaling`, compose `_rotation`); creates a `Camera` per keyframe; seeds
  new Gaussians from the keyframe's RGB + SLAM depth; then runs 10 mapping iterations over
  a sliding window of the last ~11 keyframes plus 2 random earlier ones.
- Losses: L1 RGB + inverse-depth L1 against the SLAM depth (`get_loss_mapping_rgbd`, α=0.95),
  a **normal consistency** term against the Omnidata normals (`get_loss_normal`, weighted by
  `lambda_dnormal`), and an isotropy regulariser on the scales. Note the constructor forces
  `config["Training"]["monocular"] = False`, i.e. the SLAM depth is always used as pseudo-GT.
- Densification/pruning follow 3DGS, with periodic opacity resets.
- `finalize()` — `color_refinement` over `position_lr_max_steps` iterations, jointly
  optimising Gaussians, **per-camera pose deltas** and (optionally) per-camera exposure
  a/b; saves `3dgs_final.ply`; returns the refined poses so `Hi2` can write them back.
- `eval_rendering()` — renders and scores against ground truth.

### 3.2 `hislam2/geom/` — projective geometry and solvers

| File | Purpose |
|---|---|
| `projective_ops.py` | `projective_transform` — the core ii→jj reprojection with analytic Jacobians w.r.t. pose i, pose j and inverse depth; `actp` handles both SE(3) (6-DoF) and Sim(3) (7-DoF) point actions. `MIN_DEPTH = 0.2` culls points behind/too near the camera. |
| `pinhole.py` | `iproj_pinhole` / `proj_pinhole` — pinhole (un)projection and their Jacobians. |
| `ba.py` | Python-side bundle adjustment. `BA` (full, Schur complement over depths), `MoBA` (motion-only, used by the trajectory filler), `get_prior_depth_aligned` (bilinearly interpolates the 2×2 `dscales` grid up to the prior's own resolution — 1/8, not full, §9.6 — via `droid_backends.bi_inter`, giving a **spatially varying** scale for the mono depth prior), and **`JDSA`** — HI-SLAM2's joint depth–scale adjustment: solves for inverse depths and the scale grid together, so the learned prior is fused into the BA rather than applied as a fixed rescaling. `alpha` (`mono_depth_alpha`) sets how strongly the prior pulls. |
| `chol.py` | `CholeskySolver` (differentiable, fails soft), `block_solve`, `schur_solve` (returns depth covariances too), `schur_solve_mono_prior` (the JDSA variant). |
| `graph_utils.py` | Small helpers converting dict-graphs to edge lists; used by `DroidNet.forward` (the training path, not exercised at inference). |

### 3.3 `hislam2/modules/` — the DROID network

| File | Purpose |
|---|---|
| `droid_net.py` | `DroidNet` = `fnet` (correlation features) + `cnet` (context) + `UpdateModule`. `UpdateModule` is the recurrent update operator: encodes correlation and flow, runs a `ConvGRU`, and emits flow `delta`, `weight`, and via `GraphAgg` the per-pixel damping `eta` and the 8×8×9 convex-upsampling mask. Also `cvx_upsample` (with border masking so upsampling never mixes in out-of-image pixels) and the full training-time `forward`. |
| `extractor.py` | `BasicEncoder` (residual/bottleneck blocks), the 1/8-resolution feature backbone for both `fnet` and `cnet`. |
| `gru.py` | `ConvGRU` with an extra global-context pathway (`convz_glo`/`convr_glo`/`convq_glo`). |
| `corr.py` | `CorrBlock` — precomputed 4-level all-pairs correlation pyramid, indexed on GPU by `droid_backends.corr_index_*`. `AltCorrBlock` — recomputes correlation on the fly from feature pyramids (`altcorr_*`), far less memory, used by global BA. |
| `clipping.py` | `GradientClip` — zeroes gradients above 0.01 and NaNs; stabilises training. |

### 3.4 `hislam2/midas/` — monocular depth & normal priors

Vendored DPT/MiDaS code, used **only** for inference of the Omnidata checkpoints.

| File | Purpose |
|---|---|
| `omnidata.py` | `OmnidataModel` — thin wrapper loading `omnidata_dpt_{depth,normal}_v2.ckpt` into `DPTDepthModel` (backbone `vitb_rn50_384`, 1 or 3 output channels). **Fork change:** `torch.load(..., weights_only=False)`, required since torch 2.6 flipped the default and these are full Lightning pickles. |
| `dpt_depth.py` | `DPT` / `DPTDepthModel` — ViT encoder + RefineNet-style fusion decoder. |
| `vit.py` | ViT and ViT-hybrid (ResNet50) backbones, positional-embedding resizing, readout ops. |
| `blocks.py` | Encoder/decoder building blocks (`_make_encoder`, `FeatureFusionBlock*`, `ResidualConvUnit*`, `Interpolate`). |
| `transforms.py` | `Resize` / `NormalizeImage` / `PrepareForNet` (only `Resize` is used at runtime). |
| `base_model.py` | Checkpoint-loading mixin. |
| `midas_net.py`, `midas_net_custom.py` | Legacy MiDaS v2 architectures — **unused** by HI-SLAM2, kept from the vendored tree. |

### 3.5 `hislam2/util/`

| File | Purpose |
|---|---|
| `trajectory_filler.py` | `PoseTrajectoryFiller` — recovers poses for *non-keyframe* frames: linearly interpolates between bracketing keyframes in SE(3), appends them temporarily to the video, then runs 6 **motion-only** BA iterations against their two anchors. Batched 16 frames at a time. Also used by `Hi2.terminate` to pose newly inserted keyframes. |
| `utils.py` | `load_config` (YAML with `inherit_from` support), `Log` (colour-tagged rich printing), `colorize_np`, `clone_obj` (deep copy detaching tensors, for the GUI queue). |
| `droid_visualization.py` | The `--droidvis` Open3D window: per-keyframe camera frusta and back-projected point clouds, filtered by multi-view depth consistency (`droid_backends.depth_filter`). Keys `S`/`A` tighten/loosen the filter. Runs in its own process, reads the shared `DepthVideo` and the `dirty` flags. |

### 3.6 `hislam2/gaussian/` — the 3DGS map

| Path | Purpose |
|---|---|
| `scene/gaussian_model.py` | `GaussianModel` — the map itself: `_xyz`, `_features_dc/_rest` (SH, degree 0 here), `_scaling`, `_rotation`, `_opacity`, plus `unique_kfIDs` (which keyframe spawned each Gaussian — this is what lets pose corrections be applied per-Gaussian). `create_pcd_from_image_and_depth` back-projects an RGB-D keyframe with Open3D, downsamples (`pcd_downsample` / `..._init`), sets initial scales from `simple_knn.distCUDA2` × `point_size`. Also densify/split/clone/prune, opacity reset, optimiser tensor surgery, `save_ply`. |
| `renderer/__init__.py` | `render()` — wraps the RaDe-GS `diff_gaussian_rasterization`. Note the extra arguments beyond stock 3DGS: `projmatrix_raw`, and `theta`/`rho` (the camera's `cam_rot_delta` / `cam_trans_delta`), which is how gradients flow to camera poses during refinement. Returns render, **expected depth**, radii, `n_touched`. |
| `utils/camera_utils.py` | `Camera` — an `nn.Module` viewpoint holding image, SLAM depth, prior normal, R/T, intrinsics/FoV, and the learnable `cam_rot_delta`, `cam_trans_delta`, `exposure_a`, `exposure_b`. |
| `utils/slam_utils.py` | SE(3)/SO(3) exp maps and `update_pose` (applies and zeroes the pose deltas), `depth_to_normal` (finite differences on back-projected depth), image gradients, and the loss functions: `get_loss_mapping_rgbd`, `get_loss_tracking_*`, **`get_loss_normal`** (1 − cosine between rendered-depth normals and the Omnidata prior, masked where the prior is weak). |
| `utils/loss_utils.py` | `l1_loss`, `l2_loss`, `ssim`, `psnr`. |
| `utils/graphics_utils.py` | `getWorld2View2`, `getProjectionMatrix2` (from fx/fy/cx/cy rather than FoV), `fov2focal`/`focal2fov`, `BasicPointCloud`. |
| `utils/general_utils.py` | Quaternion→rotation, covariance strip/build, `inverse_sigmoid`, LR schedule `helper`. |
| `utils/sh_utils.py` | Spherical-harmonics evaluation and `RGB2SH`/`SH2RGB`. |
| `utils/eval_utils.py` | `eval_rendering` — renders every 5th frame plus all keyframes, writes JPEG renders and 16-bit depth PNGs (×6553.5) into `renders/{image,depth}_after_opt/`, scores PSNR/SSIM/LPIPS (and depth L1 if `--gtdepthdir`), dumps `psnr/after_opt/final_result.json`. `eval_rendering_kf` does the keyframe-only variant with exposure compensation. `save_gaussians`. |
| `gui/slam_gui.py` | The `--gsvis` Open3D-GUI window: live Gaussian rendering, keyframe frusta, depth/normal/opacity view modes, screenshots. Runs in its own process, fed by a queue. |
| `gui/gui_utils.py` | `GaussianPacket` (the queue payload), `ParamsGUI`, camera-frustum geometry helpers. |
| `gui/gl_render/` | A raw-OpenGL Gaussian renderer used by the GUI when not rasterising through CUDA: `render_ogl.py`, `util_gau.py`, `util.py`, and GLSL shaders `shaders/gau_{vert,frag}.glsl`. |

---

## 4. `src/` — the `droid_backends` CUDA extension

| File | Purpose |
|---|---|
| `droid.cpp` | pybind11 bindings and host-side wrappers. Exports `ba`, `pgba`, `proj_trans`, `frame_distance`, `covis_distance`, `depth_filter`, `iproj`, `bi_inter`, `corr_index_{forward,backward}`, `altcorr_{forward,backward}`. |
| `droid_kernels.cu` | The heavy lifting (~2200 lines): `projective_transform_kernel` and its Sim(3) variant build the reduced camera system on-GPU; `EEt6x6`/`Ev6x1`/`EvT6x1` implement the Schur complement; `pose_retr`/`disp_retr` apply the update; plus `frame_distance`, `covis_distance`, `depth_filter`, `iproj`, and `bi_inter` (bilinear interpolation of the 2×2 scale grid, with Jacobians, for JDSA). |
| `correlation_kernels.cu` | `corr_index_{forward,backward}` — gathers a radius-r window from a precomputed correlation volume. |
| `altcorr_kernel.cu` | `corr_{forward,backward}` — computes correlation on the fly from feature maps (the low-memory path). |

---

## 5. `scripts/`

| File | Purpose |
|---|---|
| `download_replica.sh` | Downloads the **NICE-SLAM rendered** Replica sequences + culled GT meshes into `data/Replica`. (Not the raw Facebook Replica release.) |
| `preprocess_replica.py` | Symlinks `results/frame*` → `colors/`, `results/depth*` → `depths/`, converts `traj.txt` (4×4) to TUM format `traj_tum.txt`. |
| `preprocess_scannet.py` | Writes `calib.txt` from `intrinsic_color.txt` and converts per-frame pose files to a single TUM `traj.txt` (NaN poses zeroed). |
| `preprocess_tum.py` | Turns a raw TUM RGB-D sequence into the Replica layout (§10): associates rgb↔depth↔groundtruth by timestamp, undistorts both images and depth, crops the undistortion border, rescales depth 5000→6553.5, and writes `colors/%06d.png depths/%06d.png traj_tum.txt calib.txt`. Undistortion happens **here** rather than via `demo.py --undistort`, because `common.py:stream_resize` does not undistort when re-deriving GT frames. |
| `preprocess_owndata.py` | For casual video: extracts every frame to `images/`, every 10th (max 100) to `images_colmap/`, runs the full COLMAP pipeline (OPENCV camera model) to estimate intrinsics, writes `calib.txt`. |
| `run_replica.py` | Full Replica benchmark: runs `demo.py` per sequence, `evo_ape` for ATE, reads the render metrics, runs TSDF fusion at 6 mm, aligns the mesh with the evo Sim(3) transform, then `eval_recon.py`; averages everything. |
| `run_scannet.py` | Same for the 8 selected ScanNet scenes (`--cropborder 12`, 15 mm voxels, keyframe-only render metrics, no 3D recon eval). |
| `eval_recon.py` | Mesh evaluation: accuracy / completion / completion-ratio via KD-trees and `evaluate_3d_reconstruction`, plus an optional 2D depth-L1 metric that renders random in-room views of GT vs. reconstruction with Open3D. **Note:** its `trimesh` import was never satisfied before this fork installed it, so this script (and `run_replica.py`'s recon metrics) could not run at all. |

The VGGT track adds exactly one more — see §9: `run_pipeline.py`, the single driver
(extract → adapt → end2end comparison). The standalone single-stage tools `export_slam_depth.py` and
`lora_adapt_vggt.py` are **deleted**: once every stage became an importable package under
`adaslam/` (§9.5) they were thin argparse wrappers over code reachable in three lines from a
REPL, and a second way to invoke a stage is a second place for its defaults to drift.

---

## 6. `config/` — what the knobs mean

Five files: `replica_config.yaml`, `scannet_config.yaml`, `owndata_config.yaml`, `euroc_config.yaml`,
`tum_config.yaml` (§10 — ScanNet's real-sensor preset with `skip_blur`, `pgba.active`, exposure
compensation, and `mono_depth_alpha: 0.01`). `run_pipeline.py` generates a sixth at runtime,
`OUT_EXTRACT/full/extract_config.yaml`, which `inherit_from`s one of these and overrides the
keyframe thresholds for the extract run only (§9.2.1).

- **`Dataset`** — `pcd_downsample(_init)`: how aggressively new Gaussians are subsampled from a
  keyframe; `point_size` + `adaptive_pointsize`: initial Gaussian scale; `scale_multiplier`:
  the global scale fixed at initialisation.
- **`Tracking.motion_filter`** — `thresh` / `init_thresh`: flow magnitude needed for a new
  keyframe; `skip_blur`: prefer the sharpest of the last 5 frames.
- **`Tracking.frontend`** — `keyframe_thresh` (redundant-keyframe removal), `frontend_thresh`
  (max distance for an edge), `frontend_window`, `frontend_radius`, `frontend_nms`, and
  `mono_depth_alpha` (**JDSA prior weight** — 0.001 on Replica, 0.01 on casual video, where
  the prior matters more).
- **`Tracking.backend`** — `backend_thresh` / `radius` / `nms` for global BA edges;
  `covis_thresh` for inserting extra keyframes in `terminate()`.
- **`Tracking.pgba`** — `active` (off for Replica/ScanNet, **on** for own data, where loops
  and drift are real), `pgba_thresh`.
- **`Training`** — Gaussian densification schedule, `window_size`, `lambda_dnormal`
  (normal-loss weight: 0.1 Replica, 0.5 own data), `compensate_exposure`.
- **`opt_params`** — 3DGS learning rates; `position_lr_max_steps` doubles as the **number of
  final colour-refinement iterations** (2000 for Replica, 26000 for own data).

---

## 7. Outputs

`outputs/<seq>/` after a run:

```
intrinsics.npy            fx fy cx cy at the tracking-stream resolution (the 1/8 grid x8)
traj_kf.txt               keyframe poses, TUM format (tstamp tx ty tz qx qy qz qw), cam→world
traj_full.txt             every frame, same format
3dgs_final.ply            the Gaussian map
renders/image_after_opt/  rendered RGB (jpg)
renders/depth_after_opt/  rendered depth (16-bit png, ×6553.5)
psnr/after_opt/final_result.json      PSNR/SSIM/LPIPS/depth-L1 over all evaluated frames
psnr/after_opt/final_result_kf.json   the keyframe-only variant
tsdf_mesh_w<W>.ply        written later by tsdf_integrate.py
```

Caveat on `final_result.json`'s `mean_l1`: `eval_utils.py:62` compares metric GT depth against
**unscaled** SLAM depth, so on a monocular run it is dominated by the arbitrary global scale
(~0.7 m on Replica, where the Sim(3) scale is ~1.3) and does not measure depth shape. The §9
harness recomputes it after a global median-ratio scale fit (~0.02 m) — that is the meaningful
number.

### 7.1 The §9 pipeline's tree — stage, then scene, then experiment

The fan-out is real: a scene has several extracts, each has several adapts, each has several
tests. So `outputs/` is keyed by **stage, then scene**, and an experiment directory holds what the
*next* stage consumes and nothing else. The raw HI-SLAM2 run goes one level down in `full/`, which
means it can be deleted to reclaim the Gaussian map and the renders (~400 MB a run) without
breaking the stage after it.

```
outputs/
  extract/<SCENE>/<EXTRACT_NAME>/ the handoff to adapt, and nothing else
    depth_slam/%06d.npy           per-keyframe training depth, float32, SLAM units. BOTH sources
    mask_slam/%06d.png            are always written (EXTRACT.depth_sources) — choosing one at
    depth_rendered/%06d.npy       extract time would cost another SLAM run the day you want the
    mask_rendered/%06d.png        other. `slam` is 1/disps_up, `rendered` the Gaussian map's
                                  expected depth; masks are droid_backends.depth_filter & depth>0
    image/%06d.jpg                the matching keyframe RGB — a record, NOT read by adapt, which
                                  indexes the full colour dir by frame number (ADAPT_IMAGES)
    poses_slam.txt                the exported keyframes, TUM c2w. adapt reads column 0 (the
                                  keyframe list) and takes its poses from traj_full.txt; the list
                                  is the INTERSECTION over the sources written, so either choice
                                  of ADAPT.depth_source has a file for every keyframe named here
    traj_full.txt                 copied up from full/ — every pose adapt actually uses
    intrinsics.npy                copied up from full/
    export.txt                    the depth-source accuracy table (read this first)
    full/                         the untouched SLAM run: a normal outputs/<seq>/ as above, plus
                                  extract_config.yaml (generated; inherit_from CONFIG + EXTRACT's
                                  kf_* knobs, this run only) and slam_depth.npz (post-global-BA
                                  state: tstamp, disps 1/8, disps_up full, poses w2c, images,
                                  intrinsics, dscales, disps_prior)

  adapt/<SCENE>/<ADAPT_NAME>/     the handoff to end2end, and nothing else
    adapter.safetensors           ~48 MB
    config.json                   the structure from_adapter rebuilds: rank, alpha, targets,
                                  lora_patch_embed, vggt_hw — plus `scene`, the extract directory
                                  it trained on, which is where lineage is recorded
    train_log.json
    checkpoints/epoch_NNN/        every ADAPT.checkpoint_every epochs, each a COMPLETE adapter
                                  dir that from_adapter loads and an arm can run

  test/
    end2end/<SCENE>/<arm>/        ONE DIRECTORY PER DEPTH-PRIOR GENERATOR, <arm> inferred (below).
                                  A normal outputs/<seq>/ as above, plus results.json (that arm's
                                  metrics and the split_at they were computed at), evo/, ape.txt,
                                  tsdf_mesh_w<W>[_aligned].ply, eval_recon.txt
    prior/<SCENE>/<arm>/          the SAME generators scored against GT depth with no SLAM run
      frames.csv                  one row per frame — the expensive artifact, split-INDEPENDENT,
                                  cached on the eval spec written into its `#` header line
      results.json                aggregates at one split_at, recomputed from frames.csv in ms
```

**Naming.** `EXTRACT_NAME` and `ADAPT_NAME` in `run_pipeline.py` are **required** and are checked by
`common.py:require_name` before any GPU work. The scene is a directory of its own, so a name only
has to be unique within its scene — it does not carry the scene and does not chain the run before
it. Lineage is recorded as *data* instead: an adapter's `config.json` holds the extract directory it
trained on (`adapt/trainer.py:113`). `FRACTION` is not in the name either; put it there yourself if
you vary it, or two fractions overwrite each other.

**Arm names are inferred, never typed** (`end2end/config.py:arm_name`):

| entry in `END2END_PRIORS` | directory under `test/end2end/<SCENE>/` |
|---|---|
| `'omnidata'` | `omni` |
| `'vggt_base'` | `base` |
| `outputs/adapt/<scene>/<aname>` | `<aname>` |
| `outputs/adapt/<scene>/<aname>/checkpoints/epoch_005` | `<aname>_chkp_005` |

That is what makes an arm **reusable**: one adapter always scores into one directory, so a scene's
`omni` baseline is run once and every later comparison finds it instead of repeating it. Two entries
that infer the same name are a hard error — they would write into one directory and the comparison
would silently score an arm against itself.

---

## 8. Fork-specific notes (CUDA 13 / PyTorch 2.9 / Python 3.12)

Everything this fork changed relative to upstream HI-SLAM2:

1. **`setup.py`** — gencode narrowed to `compute_89`/`sm_89` + PTX. CUDA 13 dropped Pascal
   and Volta, so upstream's `compute_60/61/70` lines are now hard nvcc errors. On a
   non-Ada GPU, change this **and** the matching lines in `patches/lietorch.patch`.
2. **`patches/lietorch.patch`** — the same arch fix plus CUDA 13 source fixes for the
   lietorch submodule, applied by `setup_env.sh` because a submodule's contents cannot be
   committed here.
3. **`thirdparty/diff-gaussian-rasterization`** — added missing `#include`s in
   `backward.h` / `forward.h` / `rasterizer_impl.h` for newer nvcc/gcc.
4. **`hislam2/midas/omnidata.py`** — `torch.load(..., weights_only=False)` for torch ≥ 2.6.
5. **Dependency management** — conda `environment.yaml` replaced by `requirements.in` /
   `requirements.txt` (uv), plus `setup_env.sh`.
6. **Three added packages**, all numpy-only so none can perturb the torch/numpy resolution:
   `einops` (VGGT's one runtime dependency we lacked), `trimesh` and
   `evaluate_3d_reconstruction` (pinned git, for `eval_recon.py`). VGGT itself is **vendored**
   as a submodule rather than pip-installed, because its own requirements pin `torch==2.3.1` /
   `numpy==1.26.1` and would try to downgrade torch.

Known rough edges:

- `README.md` and `setup_env.sh` both reference `new-udpate-env.md`, which was **deleted**
  in commit `f4029db` when its content moved into `setup_env.sh`. `../CLAUDE.md` also still
  points at it. Dead references.
- `hislam2/gaussian/gui/gl_render/`, `midas/midas_net*.py`, `geom/graph_utils.py` and
  `DroidNet.forward` are vendored/training-path code not exercised by `demo.py`.
- `demo.py` still carries its own copy of the stream/track loop, deliberately: it is upstream's
  CLI and `run_replica.py` / `run_scannet.py` drive it as a subprocess. Nothing else duplicates
  any more — the pipeline's loop, export and LoRA all live in `adaslam/` and every entry point
  imports them (§9.5).
- `GSBackEnd` inherits `mp.Process` but is driven inline; mapping and tracking are
  serialised in the main process. The GUI, the DROID visualiser and `PGOBuffer.spin` are
  the only genuine extra processes (plus `demo.py`'s image reader).
- Several `torch.cuda.amp.autocast` call sites use the deprecated API and emit warnings on
  torch 2.9 (functionally fine).
- `Hi2.terminate()` `terminate()`s the PGBA child rather than letting it exit, so the
  `DepthVideo` buffers it held over CUDA IPC stay pinned for the life of the process. Invisible
  to `demo.py` (the process ends anyway), but a driver that runs several sequences in **one**
  process strands ~1.26 GiB per run — measured on TUM at `buffer: 500`; nothing reclaims it, and
  it is exactly 0 when `pgba.active` is false. See `adaslam/runtime.py`, whose module docstring
  carries the measurement.

---

## 9. VGGT depth-prior track (this fork)

**Premise.** Omnidata degrades under unfamiliar lighting. Its measured weakness on Replica is
specifically **cross-frame scale inconsistency**: it is the only depth source that gets *worse*
under a single global scale fit (0.0611 → 0.0836 m), while SLAM and Gaussian-rendered depth both
improve. The idea is to LoRA-adapt VGGT on HI-SLAM2's own depth for a scene, then swap it in
for Omnidata on the rest of that scene. The better supervision target is the **Gaussian-rendered**
depth (`DEPTH_SOURCE = 'rendered'`): it is 2.4× closer to GT than `1/disps_up` (0.0133 vs
0.0324 m global-scale L1 on room0) and, unlike the npz dump, it is produced from the same
post-refinement trajectory that `traj_full.txt` supplies for the pose loss. `run_pipeline.py`
nevertheless currently ships `DEPTH_SOURCE = 'slam'`, matching the §9.4 Replica runs. Since the
§7.1 restructure that is a *selection*, not a production decision: extract writes both
(`EXTRACT.depth_sources`), so switching costs an adapt run rather than another SLAM run.

### 9.1 Pipeline

**`scripts/run_pipeline.py` is the only way to run it.** Four stages in one process, every
parameter in the block at its top — a handful of CAPITAL constants and the five config dataclass
literals `SLAM` / `EXTRACT` / `LORA` / `ADAPT` / `TEST` they feed (§9.5). The file is the knob
panel, not the implementation: every stage is a package under `adaslam/`. No CLI, no environment:

```
python scripts/run_pipeline.py        # from the repo root, adaslam venv active

  1 extract   SLAM over the first FRACTION% of the sequence (generated extract_config.yaml)
              into OUT_EXTRACT/full/ → slam_depth.npz → depth_{slam,rendered}/ mask_*/ image/
              poses_slam.txt + export.txt one level up, with traj_full.txt / intrinsics.npy
              copied up beside them
  2 adapt     LoRA-adapt VGGT on a TRAIN subset of those keyframes, depth L1 reported on a
              held-out VAL subset  → ADAPT_OUT/adapter.safetensors, plus a snapshot every
              ADAPT.checkpoint_every epochs into ADAPT_CKPT/epoch_NNN/. Its four paths are
              arguments (ADAPT_IN / ADAPT_IMAGES / ADAPT_OUT / ADAPT_CKPT), so it can be run
              against any earlier extract's export
  3 end2end   one full-sequence arm per entry in END2END_PRIORS, differing ONLY in the depth
              prior, each into OUT_END2END/<inferred name>/ → evo ATE → TSDF → Sim(3) align →
              eval_recon + render metrics, and a comparison table split at the frame the
              adapter's training data ended
  4 prior     the SAME generators in PRIOR_PRIORS scored against GT depth directly, with NO SLAM
              run — minutes an arm rather than forty — decomposed by alignment so an end2end null
              can be attributed (§9.2.2)
```

Every stage takes its input and output paths as arguments (§9.5's fourth rule) — `main()` names
them from `EXTRACT_NAME` / `ADAPT_NAME` and hands them down, so which directory a stage writes to
is visible at the call site and never read out of a global inside it.

`END2END_PRIORS` lists the depth-prior generators to compare, `[0]` being the baseline column:
`'omnidata'` (upstream's), `'vggt_base'` (stock VGGT-1B, the §10.2 sanity arm), and any number of
adapt handoff directories or their checkpoints. `STAGES` skips stages, `SKIP_EXISTING` reuses
finished ones — `STAGES = ('end2end',)` re-runs just the comparison against adapters already on
disk, and any arm already scored at the same `split_at` is skipped outright.

The earlier route — a shell/CLI chain of `run_slam_depth_batch.sh`, `run_full_{omnidata,vggt}.py`,
`_full_run_common.py`, `temp_run_ab_comparison.sh` and `run_tum_experiment.sh` — has been
**deleted**. Everything it did lives in `run_pipeline.py`; batching several scenes now means
editing `SCENE` and re-running, or driving the file from a loop.

Dataset preprocessing is deliberately *not* part of it: run `scripts/preprocess_tum.py` (or
`preprocess_replica.py`) once by hand first.

### 9.2 The scripts

| File | Purpose |
|---|---|
| `run_pipeline.py` | **The single entry point** — the PARAMETERS block, three thin stage wrappers and `main()`, ~320 lines. See §9.2.1 for what each stage does and where it lives. It imports nothing from `demo.py` or from `scripts/`; the stages are the packages in `adaslam/`, and the only other dependencies are three CLIs those packages drive as subprocesses (`evo_ape`, `tsdf_integrate.py`, `scripts/eval_recon.py`). |
| ~~`export_slam_depth.py`~~ / ~~`lora_adapt_vggt.py`~~ | **Deleted.** Both were argparse wrappers over one stage. Since the stages became packages, the same thing is `from adaslam.extract.export import load_export, write_keyframes` or `LoRAVGGT(cfg).train(...)` — reachable from a REPL, with no second set of defaults to drift out of step with the PARAMETERS block. Re-exporting an existing `slam_depth.npz` without re-running SLAM is still possible that way; the accuracy table without the files is `load_export` then `from adaslam.extract.accuracy import report_accuracy`, skipping `write_keyframes`. Neither name is in `extract`'s `__all__` — the fifth rule in §9.5 says why, and why that costs nothing. |

Every stage is importable — `adaslam.extract.run_extract`, `adaslam.adapt.LoRAVGGT.train`,
`adaslam.end2end.run_end2end_test`, `adaslam.priortest.run_prior_test`,
`adaslam.slam.SlamRunner.run` — so `run_pipeline.py` is a convenience, not a
gate. Nothing duplicates anything (§9.5).

#### 9.2.1 The stage packages

`run_pipeline.py` holds the parameters and four thin wrappers; this is where the work actually is.

| Entry point | What it does |
|---|---|
| `adaslam.slam.SlamRunner.run` | **The single interface to HI-SLAM2**; `adaslam/slam/` is the only package importing `Hi2` or `MotionFilter` — an invariant one grep checks (§9.3). Three or more call sites reach it: the extract run and one per end2end arm. One `SlamRunner` is built in `main()` from `SLAM`, so the arms cannot disagree about the stream, calibration or resolution; everything that legitimately differs (tracking YAML, output dir, length, buffer, `gtdepthdir`, `dump_slam_depth`, **depth prior**) is an argument, visible at the call site. It asserts the 9-field `Hi2` args contract (`HI2_ARGS`) before construction. |
| `adaslam.extract.run_extract` | Writes a generated `extract_config.yaml` that `inherit_from`s `CONFIG` and applies `EXTRACT`'s four `kf_*` knobs, runs SLAM over the first `FRACTION`% with the `hi2.py` depth dump enabled **into `<exp>/full/`**, copies `traj_full.txt` and `intrinsics.npy` up to the experiment level, then exports **every** source in `EXTRACT.depth_sources` (§7.1). Its two halves skip independently: an existing `full/slam_depth.npz` reuses the SLAM run but the export still re-runs when a handoff artifact is missing, so a run made before a source existed can gain it without re-tracking. The generated config is given to **this run only**; `main()` asserts the arms get the unmodified `CONFIG`, so a denser training set can never masquerade as a tracking change in the comparison. Note the binding gate is `kf_redundant_thresh` (`frontend.keyframe_thresh`), not the motion filter: over 204 TUM frames `(motion, redundant) = (2.4, 4.0)` gave 43 keyframes, `(1.2, 4.0)` only 45, `(1.2, 1.5)` 83, because `track_frontend.py:49-52` prunes back whatever the motion filter proposes. GT depth reaches `ExtractConfig.gt_depths` (the accuracy table) and never `Hi2` (§9.3). |
| `adaslam.adapt.LoRAVGGT.train` | The **first stage that was wired to explicit I/O**, and the pattern the other two now follow: `stage_adapt(in_dir, image_dir, out_dir, ckpt_dir)` reads no path global, so it can be pointed at any earlier extract run without moving `OUT_EXTRACT`. It checks its four paths (including that the export's highest keyframe index exists in `image_dir`, since the two are now free to be any pair), then `LoRAVGGT(LORA, seed=ADAPT.seed).train(...)`, **the one `lora.save()`**, and `release()` — in that order, because `save()` goes through `_ensure_live()`. Training itself writes no adapter: `run_training` returns `state` and `run`, which are exactly `save()`'s `state=` and `extra=`, so where the adapter lands is a decision at the call site. Includes a **train/val split** of the exported keyframes (`train_frac`, `split_mode`) so depth L1 is reported on held-out keyframes rather than on the adapter's own training set; `keep_best` optionally snapshots on val improvement instead of keeping the last epoch, and `checkpoint_every` drops a full loadable adapter dir into `ckpt_dir` every N epochs. |
| `adaslam.end2end.run_end2end_test` | One full-sequence run per entry in `END2END.priors` into `out_root/<arm_name(spec)>` — the directory is an argument, so `End2EndConfig` holds knobs and never a location — then `evaluate` + `print_report` per arm and `compare` at the end. There is deliberately **no `adapter` parameter**: each arm carries its own, which is what lets one comparison hold several adapters and their checkpoints. Caching splits in two, because arms are reused across comparisons: the SLAM run is skipped when `{out}/traj_full.txt` exists, and scoring is skipped only when `results.json` records the *same* `split_at` — every adapter has its own training fraction, so one comparison's split is not another's, and re-scoring reads the saved renders at no SLAM cost. The prior is passed **into** `SlamRunner.run`, which snapshots `MotionFilter.prior_extractor`, installs, and restores it in a `finally` — so a VGGT arm's patch cannot leak into a later Omnidata arm and silently make it a second VGGT arm, and cannot leak out of a run that raised either. The restore is deliberately **after** `hi2.terminate()`: `hi2.py:143` calls the extractor again for the keyframes `terminate()` inserts into low-covisibility gaps. Each arm's prior is `release()`d in its own `finally`, so a crashed arm no longer strands ~2.5 GB. |
| `adaslam.end2end.VggtPrior` | The depth-prior swap. Normals stay Omnidata, so **depth is the only variable between arms**. Undoes `motion_filter.py`'s ImageNet normalisation (§9.3), and reports the stream→VGGT aspect skew, warning above 5 % — the inference-side half of the `vggt_hw` guard (§9.3). The model comes from `LoRAVGGT.from_adapter`, which rebuilds the **whole** structure — rank, alpha, targets, `patch_embed` and `vggt_hw` — from what the adapter recorded, so an arm cannot run the adapter in a shape or at a resolution it was never trained in. `extractor()` returns a **plain function, never a bound method**: functions are descriptors, so the `MotionFilter` binds as the first argument while the `VggtPrior` arrives through the closure cell. A bound method or `functools.partial` is not a descriptor — `mf` would never be passed and `mf.MEAN` / `mf.STDV` / the cached normal model would be lost. |
| `adaslam/end2end/metrics.py` — `evaluate`, and the `run_ate` / `run_mesh` / `split_render_metrics` it calls | The metrics harness: evo ATE (Sim(3)-aligned), TSDF fuse → Sim(3)-align → `eval_recon.py` (skipped when `END2END.gt_mesh is None`, as on TUM), and PSNR/SSIM/depth-L1 recomputed per frame from the saved renders so every number can be **split seen/unseen** at `split_at`. `run_mesh` retries down `voxel_fallbacks` when marching cubes OOMs on the shared GPU and records which size won. Reached through `run_end2end_test`; `from adaslam.end2end.metrics import evaluate` re-scores a finished output dir. |
| `adaslam/end2end/report.py` — `compare` | Prints the baseline in absolutes and every other arm as absolute + delta; refuses to print mesh rows when the arms' `voxel_size` disagree. Pure formatting over `results.json`, so `from adaslam.end2end.report import compare, print_report` re-runs on finished output without a GPU. |
| `adaslam.priortest.run_prior_test` | The same generators, scored against GT depth with **no SLAM run** — see §9.2.2 for what it measures and why. Takes no `runner`: each frame goes through `slam.PriorProbe`, which calls the very extractor a real arm would (that is the point — a probe that resized differently would report a number no arm ever produces). Caching is split in two: `frames.csv` is the expensive artifact and is keyed on the **eval spec** written into its header, while `results.json` is a cheap aggregate at one `split_at`. Every per-frame value is split-independent, so changing which adapter defines the boundary re-aggregates in milliseconds instead of re-running inference. |
| `adaslam.runtime.free_vram` / `gpu_gate` | Shared-workstation hygiene: `MIN_FREE_VRAM_MB` is checked once at the top of `main()`, and VRAM is force-released between stages (§8's last rough edge is why this is needed at all). |

#### 9.2.2 The prior test — what it measures, and why alignment is the axis

The end2end test cannot attribute its own null (§9.4). "Swapping the prior changed nothing" is
either *the new prior is no better* or *HI-SLAM2 is insensitive to the way it is better*, and
telling those apart needs the priors measured before SLAM touches them.

The design axis is **alignment**, not more error functions, because HI-SLAM2 never consumes the
prior raw: JDSA fits a 2×2 bilinear scale grid per keyframe and multiplies
(`geom/ba.py:get_prior_depth_aligned` is `depth_prior * mscales_bi` — **scale-only, no shift**). So
one prediction yields three very different errors depending on how much of that freedom you grant,
and the differences are the diagnosis:

| alignment | residual means |
|---|---|
| per-frame scale (median ratio) | pure shape error, scale-free |
| per-frame 2×2 JDSA grid (least squares, fitted in **disparity** — where JDSA fits it) | the error the pipeline **cannot** absorb |
| one global scale for the whole sequence | shape error **plus** cross-frame scale drift |

Two derived numbers carry the story. **`consistency_index` = L1_global / L1_per-frame** is §10.2's
"first number to check" generalised to every arm — Omnidata on Replica room0 reads 0.0735 / 0.2078,
a 2.8× blow-up, and *that* is the cross-frame inconsistency this whole track targets; an adapter
trained on SLAM depth should drive it toward 1.0. **`scale_cv`**, the spread of the per-frame fitted
scales, measures the same drift directly. Alongside them: L1 (m), AbsRel and δ<1.25 per alignment.

Scale**+shift** alignment is deliberately not offered — it is the MiDaS convention and would
flatter Omnidata with a degree of freedom the pipeline can never exploit.

**Two caveats to carry with the numbers.** The JDSA-grid row is a *lower* bound on what JDSA leaves
behind: the real solver fits that grid jointly with poses and depths against photometric residuals,
not in closed form against GT. And the seen/unseen rows are only readable against the **sentinels'**
seen/unseen — one split, taken from the first adapter in `priors`, is applied to every arm precisely
so `omni` and `base` act as the control. If every prior degrades past the split, the back of the
sequence is simply harder and the adapter has not generalised worse at all. Adapters whose own
training boundary differs from the table's are warned about and carry a `*`.

### 9.3 Traps that silently corrupt results

- **Only `adaslam/slam/` may import `hi2` or `motion_filter`.** That is what makes "every invocation
  goes through one interface" a checkable property rather than a convention, and it is what lets the
  prior swap be a run *parameter*. An importer elsewhere would be free to patch the class around a
  call again, reintroducing the leak the `finally` exists to prevent:
  `grep -rn 'from hi2 import\|from motion_filter import' adaslam/` must return **only paths under
  `adaslam/slam/`**. Two files match today: `runner.py`, which runs SLAM, and `prior_probe.py`,
  which needs `MotionFilter.prior_extractor` because the stock depth prior *is* that method and the
  prior test must call it rather than a re-implementation. The rule is the package, not the file —
  it was one file until the prior test needed the same import for a different reason.
- **`prior_extractor` receives an ImageNet-normalised tensor** (`motion_filter.py:88-89`), but
  VGGT expects `[0,1]` and normalises internally (`aggregator.py:205`). `vggt_prior_extractor`
  undoes it; forgetting to would just quietly make VGGT worse.
- **The TSDF mesh must be Sim(3)-aligned before scoring.** SLAM scale is arbitrary (~1.3× here)
  and `eval_recon`'s ICP is rigid-only, so skipping the alignment gives ~0.7 m accuracy instead of
  ~0.03 m. `run_replica.py:46` does this; `end2end/metrics.py:run_mesh` does too.
- **Scale estimates in the losses must not be detached.** Detaching makes a loss only *look*
  scale-invariant — the optimiser then sees a gradient rewarding a shrinking prediction, and
  translations collapse toward zero. This diverged the first overfit test 12×.
- **Every end2end arm must use the same TSDF voxel size.** Marching-cubes allocation fails at 0.006
  when the shared GPU is busy; `run_mesh` has a fallback ladder (`VOXEL_FALLBACKS`) and records
  `voxel_size`, and `compare` refuses to print mesh rows if the arms disagree.
- VGGT's aggregator returns `None` for uncached layers (only 4/11/17/23 are kept, deliberately,
  so layer indices stay stable — `aggregator.py:196`). Any per-frame slicing of the token list
  must preserve those `None`s.
- **`vggt_hw` must match the tracking stream's aspect ratio — so it is derived, not typed.**
  Nothing letterboxes anywhere (`SceneData.frame()` and `VggtPrior` both resize straight to it),
  so a mismatched aspect squashes the image off VGGT's training distribution: `(294, 518)` suits
  Replica's 344×616, `(378, 518)` suits TUM's 400×544, and each distorts the other by ~30 %.
  `VGGT_HW = None` (the default) makes `main()` derive it from the stream via
  `LoRAConfig.resolved` → `adapt/config.py:vggt_hw_for`, which pins W to 518 and rounds H to a
  multiple of 14 — exactly VGGT's trained shape (§9.6). **Precedence, highest first:** an
  adapter's recorded `vggt_hw` (`LoRAVGGT.from_adapter`, so an adapter always runs in the shape
  it was trained in) → an explicitly pinned `VGGT_HW` → the derived value. On both paths a >5 %
  skew still prints a warning — `SceneData.aspect_report()` when adapting, `VggtPrior` when
  running an arm — which after derivation means only two things, both worth hearing: someone
  pinned a value, or an adapter trained on a different stream is being reused here.
  The `vggt_base` arm, which has no adapter to read a shape back from, is the case that was
  silently unguarded before and is now covered by both the derivation and the `VggtPrior` check.
- **The depth prior reaches BA at 1/8 resolution, through a point subsample.**
  `depth_video.py:70-73` keeps the full-res prior in `disps_prior_up` but feeds BA
  `item[4][3::8, 3::8]` — one pixel per 8×8 block, no averaging — as `disps_prior`, which is what
  `geom/ba.py:JDSA` and `track_frontend.py:42,88` read. That is **3400 values on TUM**
  (50×68), 1/64 of the full-res prior. `disps_prior_up` reaches nothing else: not BA, not the
  Gaussian mapper (`hi2.py:73-82` passes `1./disps_up`, the tracker's own depth), only the
  `--droidvis` window (hardcoded `False` at `slam/runner.py:110`) and index bookkeeping.
  Consequence, and it is easy to waste time on: **a depth-prior model's output resolution above
  `stream_res/64` is discarded before it can affect anything.** `vggt_hw` is an aspect knob, not
  a quality knob. The only lever that would raise the information reaching BA is that subsample
  itself — considered and declined, since changing it moves both arms and invalidates every
  number already collected.
- **Omnidata runs at 512×512 square, VGGT does not — an uncontrolled difference between the
  arms.** `motion_filter.py:62` passes `transforms.Resize` a 2-tuple, which ignores aspect, and
  the aspect-preserving resamplers in `midas/omnidata.py:139-155` are commented out. That is a
  26 % horizontal squash on TUM and 44 % on Replica, with no letterboxing; the VGGT arms are
  aspect-matched by construction. So part of any VGGT-over-Omnidata delta is simply *undistorted
  input*, and "VGGT is the better prior" is not the only reading of a win. Left as-is
  deliberately — the baseline should stay upstream's — but do not report a win without this
  caveat. Note the asymmetry it creates in the tripwires: the repo warns above 5 % skew on the
  VGGT path and says nothing about Omnidata's 26 %.
- **Never pass GT depth to `Hi2` on a run whose renders will become training data.**
  `eval_utils.py:50-52` zeroes the rendered depth wherever the GT depth is invalid. Replica's GT is
  dense so it is a no-op there, but TUM's Kinect GT is ~24 % holes sitting on exactly the hard
  surfaces (shiny, dark, far) — supervising on that discards a quarter of the pixels and ties the
  training mask to where the sensor happened to work. `stage_extract` therefore calls `run_slam`
  with `gtdepthdir=None` and hands `DEPTHS` to the export only; the same applies to a manual
  `demo.py --dump_slam_depth` run, which must be given no `--gtdepthdir`. Nothing is lost: the
  export's accuracy table masks
  on `(gt > 0) & mask` regardless, and the only casualty is `final_result.json`'s `mean_l1`, which
  §7 already documents as meaningless on a monocular run. The end2end arms *do* keep it — there the
  masking is correct, since depth cannot be scored where there is no GT.

### 9.4 Status

The plumbing is verified end to end: frame-0 convention confirmed, adapter identity-at-init and
save/load round-trip both exactly `0.0`, overfit test converges, ~7.8 GiB peak.

The **result so far is a null**. On the training keyframes the adapter improves depth L1 a lot
(0.0080 → 0.0032), but a full-sequence comparison on room0/1/2 shows essentially no downstream
difference — ATE identical to four decimals, PSNR within ±0.2 dB, depth L1 within ±0.0005, on
both the trained and the unseen halves. Two plausible reasons, and they are not exclusive:
base VGGT is already very strong on Replica (0.04° rotation, ~1 % relative disparity), leaving
almost no headroom; and JDSA re-solves the prior's scale per keyframe anyway
(`track_frontend.py:42`), which is exactly the failure mode the adaptation targets — so HI-SLAM2
may already be robust to the weakness being fixed. Replica is well-lit synthetic data and does
not test the difficult-lighting premise that motivated the work.

Those Replica experiment outputs lived in `outputs/ab_{depth,disp}_p{40,100}/` (the `depth`/`disp`
suffix is `ADAPT.depth_space`, the `p` number the sequence fraction the adapter trained on) and the
TUM ones in `outputs/tum*/`, both predating §7.1. Everything from before that restructure was moved
to `outputs/old/` untouched — nothing was migrated into the new shape, so `SKIP_EXISTING` does not
see it.

Note those runs predate the choice of supervision target and used `slam` depth (`1/disps_up`),
which is the less accurate target and is dumped *before* the refinement that produces the poses
they were trained against. `DEPTH_SOURCE = 'slam'` reproduces them exactly.

§10 moves the same experiment onto real data, which is what the Replica null asks for next.

### 9.5 `adaslam/` — the pipeline as packages

`run_pipeline.py` once carried all the stages in 1366 lines. The adapt stage came out first,
then extract and test; the file is now ~320 lines of parameters and dispatch, and every stage is
an importable package, under one parent package `adaslam`.

That parent is what makes the `sys.path` setup a single line. Python imports a parent package
before any of its children, so `adaslam/__init__.py` running `bootstrap(HISLAM2, VGGT)` covers
every `adaslam.*` import there is — no per-package preamble, and from the repo root a REPL needs
no setup at all (`sys.path[0]` is the cwd, so `from adaslam.adapt import LoRAVGGT` just works).
The directory was `ada-slam/` until then; a hyphen is illegal in an identifier, so it could only
be a `sys.path` entry like `hislam2/`, and each of the four packages had to carry the same
four-line manual insert to reach `paths` — a sibling it could not import until the directory it
lived in was already on the path.

**`adaslam/` itself must never go on `sys.path`.** It is a package now, so a directory entry as
well would make `adapt` and `adaslam.adapt` two distinct module objects with separate globals —
two `LoRAConfig` classes that fail `isinstance` against each other. `bootstrap()` therefore adds
exactly the directories it is given and nothing implicit.

| Path | Contents |
|---|---|
| `__init__.py` | Six lines: `bootstrap(HISLAM2, VGGT)`. The **only** place in the repo besides `demo.py` that touches `sys.path`. |
| `paths.py` | `ROOT` / `ADA_SLAM` / `HISLAM2` / `VGGT` and `bootstrap()`, whose one caller is the `__init__` above. Stdlib-only and side-effect-free beyond the `sys.path` inserts — every `adaslam.*` import goes through it, so it must cost nothing. Note what it is *not* needed for: **a spawned child inherits the parent's `sys.path` verbatim** (`multiprocessing/spawn.py:173` copies it, `:228-229` installs it, both before `__main__` is re-imported and before the target is unpickled), so nothing here has to run again in the reader process. |
| `common.py` | `stream_resize` — ONE definition, used by the reader, the LoRA data loader and the render metrics; they must agree or renders and GT stop lining up pixel for pixel. Also `DEPTH_SOURCES`, because `extract` writes `depth_<src>/` and `adapt` reads it, and the §7.1 layout vocabulary (`EXTRACT_RUN_SUBDIR`, `ADAPT_CKPT_SUBDIR`, `TEST_KINDS`, `HANDOFF_UP`, `extract_run_dir`, `experiment_dir`, `test_dir`, `require_name`) for the same reason: more than one stage — and `run_pipeline.py` itself — has to agree on it. |
| `runtime.py` | `sh`, `free_vram`, `gpu_gate`, `raise_fd_limit`, `ensure_venv_on_path` — shared-workstation hygiene, nothing stage-specific. Its module docstring is where the pgba CUDA-IPC measurement in §8 is written down. `free_vram` and `gpu_gate` print, but they do work and report on it; anything that only *formats* lives next door. |
| `print_utils.py` | `banner`, `tee` (+`_Tee`), `delta_row` — formatting output for a human, and nothing else. **Stdlib only**, no torch or cv2, so a finished comparison can be reprinted on a machine with neither. `delta_row` is the one comparison-row formatter both `report.py` modules use: baseline absolute, later columns absolute + signed delta + a `+`/`-` mark, `n/a` for `None`, blank mark on an exact tie. Their table *layouts* stay separate — the prior test repeats its table per population and can star an arm whose split is not its own, which `end2end`'s `compare()` has no notion of. |
| `slam/` | `SlamConfig`; `mono_stream` (the reader `Process` target) and the `load_frame` it is built on — ONE definition of what the tracker is shown, because `PriorProbe` scores priors on exactly those pixels; `write_tracking_config`; `SlamRunner`, **the single interface to HI-SLAM2** (§9.2.1); and `PriorProbe`, which runs a prior over frames with no SLAM run. `PriorProbe` lives here rather than in `priortest/` because the stock prior **is** a `MotionFilter` method and this is the only package allowed to import it. |
| `priortest/` | `PriorTestConfig` + `arm_split_at` / `resolve_split`; `predict.py` (inference → `frames.csv`); `metrics.py` (the three alignments, and `aggregate`, the only place the split is used); `report.py`; `stage.py` (`run_prior_test`). Imports `arm_name` from `end2end` rather than restating it, so both test kinds name a scene's arms identically. |
| `extract/` | `ExtractConfig`; `export.py` (`confidence_mask`, `load_export`, `write_keyframes`, `export_slam_depth`); `accuracy.py` (the depth-source table, §10.2's first number); `stage.py` (`run_extract`, `handoff_paths`). Loading is split from writing so `--no_export` can have the table without the files. Every function here takes either the **experiment** directory or the **run** directory (`<exp>/full`) and its parameter name says which — the npz and the renders are in the run, the handoff artifacts in the experiment above it. |
| `adapt/` | `LoRAConfig` / `AdaptConfig`; `lora.py` (`LoRALinear`, `inject_lora`, `lora_state_dict` — hand-rolled, no `peft`: rank 8 on `attn.{qkv,proj}` + `mlp.{fc1,fc2}` across the aggregator's 24+24 blocks → 6.29 M trainable of 1.17 B; `B` starts at zero so the adapter is identity at step 0, `A` is kaiming — **which is why the seed goes to the constructor**); `model.py` (`LoRAVGGT` — build/inject/load, `forward`, `predict_depth`, `save`, `release`, `from_adapter`); `data.py` (`SceneData`, one keyframe = one sample placed **first** so VGGT predicts in that keyframe's frame — verified: `extrinsic[0]` is identity to 5e-4, rebased poses match SLAM GT to 0.04°); `losses.py` (the two undetached scale estimates §9.3 warns about); `trainer.py`, which **reports and returns but does not write the adapter** — `model.py:save` writes, the caller decides when and where. The only weights `trainer.py` writes are `checkpoint_every`'s periodic snapshots. |
| `end2end/` | `End2EndConfig` + the prior-spec vocabulary (`SENTINELS`, `arm_name`, `adapter_path`); `prior.py` (`VggtPrior`); `metrics.py`; `report.py`; `stage.py` (`run_end2end_test`). Was `abtest/` until an arm stopped being one of three fixed names. Note `End2EndConfig.check_priors_exist` is called by the **stage**, not `__post_init__`: the config is built in a block that must not touch the filesystem and that runs before `main()`'s `chdir`, and an adapter listed there legitimately does not exist yet when the whole pipeline runs — the adapt stage is about to create it. |

Five rules hold across all of it — the fourth still being rolled out:

- **No config field carries a default.** Five frozen dataclasses now — `SlamConfig`,
  `ExtractConfig`, `LoRAConfig`, `AdaptConfig`, `End2EndConfig` — and every one of them is stated in
  full in `run_pipeline.py`'s PARAMETERS block. A knob is therefore written down in exactly one
  place per entry point, and none can be inherited silently from a package. The one field that
  may be left `None` is `LoRAConfig.vggt_hw`, and that is not a default: `None` is a stated
  instruction meaning *derive this from the stream*, honoured by `LoRAConfig.resolved(stream_hw)`
  in `main()` and refused by `LoRAVGGT` if it ever reaches the model unresolved (§9.3).
- **A package that receives another's config does not re-declare its fields.** `extract` and
  `end2end` are both handed the `SlamConfig`, so `colors` / `calib` / `stream_res` / `start` exist
  once. Two configs naming the same *value* (`DEPTH_SOURCE` reaching both `EXTRACT` and `ADAPT`)
  is not divergence — that is the block above feeding both, which is the point.
- **The PARAMETERS block is re-executed in every spawned child**, because `spawn` re-imports the
  driver for the reader process. No config field may be a computed path or an open handle — and no
  config's `__post_init__` may touch the filesystem. That block also runs *before* `main()`'s
  `os.chdir(_ROOT)`, so a relative path checked there would resolve against whatever directory the
  script was invoked from. `End2EndConfig.check_priors_exist` is a method the stage calls for
  exactly this reason (and because the adapters it names may not exist yet when the full pipeline
  runs — the adapt stage is about to create them).
- **A stage receives its input and output paths; it reads no path global.** Configs still arrive
  as globals — they are knobs, not locations. This is what lets one stage be pointed at another
  run's results (adapt on one extract's export, write the adapter elsewhere) without moving
  `OUT_EXTRACT`, which every other stage also keys off. **All three hold now**: the `stage I/O`
  block names every path (`OUT_EXTRACT`, `ADAPT_IN` / `ADAPT_IMAGES` / `ADAPT_OUT` / `ADAPT_CKPT`,
  `OUT_END2END`) and `main()` passes them down. The last two conversions came with the §7.1
  restructure and took `out_root` out of the test config with them — a config that carried its own
  output directory could not be pointed anywhere else. The check:
  `grep -n 'OUT_EXTRACT\|OUT_END2END\|COLORS' scripts/run_pipeline.py` returns nothing inside a
  stage body.
- **`__all__` lists only what another package or an entry point imports** — 13 names across the
  four packages, down from 44. Everything else is reachable by its module path
  (`adaslam.extract.export.load_export`, `adaslam.end2end.report.compare`), which is the difference between not
  advertising a function and hiding it. A package's `__init__` is therefore a statement of its
  interface, and re-exporting `inject_lora` or `save_trajectory` would be advertising a second
  door into a stage that the four rules above exist to close. The two exceptions are types in a
  cross-package signature: `SlamResult` and `VggtPrior` are exported because `SlamRunner.run`
  returns one and takes the other, even though neither name is imported anywhere. The check is
  that every other `__all__` entry appears in an `import` outside its own package. Dropping the
  re-exports also stops `import adapt` from eagerly loading `data.py` / `losses.py` / `trainer.py`
  in every spawned child; they now arrive only when `LoRAVGGT.train()` defers to `run_training`.
  That saves the module bodies, not the third-party imports — `cv2`/`numpy`/`torch` still come in
  through `common` and `model.py` — so it is the same *kind* of care as deferring `vggt`,
  `safetensors` and `scipy` inside functions, an order of magnitude smaller.

Two things are easy to get wrong and are worth stating:

- **The seed belongs to the constructor, not the trainer.** `LoRALinear.A` is kaiming-initialised
  when LoRA is injected, so `LoRAVGGT(cfg, seed=...)` seeds `torch` immediately before the model
  is built. Seeding inside `run_training` would be too late: the adapter is identity at step 0
  either way (`B` is zero), but `A`'s values steer the whole trajectory, so a run seeded
  afterwards is not reproducible.
- **`from_adapter` overrides the config, deliberately.** An adapter only means anything inside
  the structure it was trained in, so the `rank`, `alpha`, `targets`, `lora_patch_embed` and
  `vggt_hw` in its `config.json` win over whatever `LORA` says, and each override is printed. Only
  `weights` (where the VGGT-1B snapshot lives on this machine) still comes from the caller. Those
  key names predate the package and must not be renamed — adapters already on disk are read
  through them.

### 9.6 Resolutions, end to end

Every resize between a raw frame and the number BA actually sees. Measured, not nominal.

| stage | TUM fr1 | Replica | code |
|---|---|---|---|
| raw (H, W) | (444, 604) a=1.360 | (680, 1200) a=1.765 | `preprocess_*.py` output |
| → tracking stream | **(400, 544)** a=1.360 | **(344, 616)** a=1.791 | `common.py:stream_resize` |
| → depth-prior model in | (378, 518) VGGT · (512, 512) Omnidata | (294, 518) · (512, 512) | `VggtPrior` · `motion_filter.py:62` |
| → back to stream | (400, 544) | (344, 616) | `F.interpolate(..., input_size)` |
| → **into BA** | **(50, 68) = 3400** | **(43, 77) = 3311** | `depth_video.py:72`, `[3::8, 3::8]` |

Three things this table is here to make unmissable:

- **`stream_res` is a pixel budget, not a shape.** `341*640` is the scalar `218240`; both dims
  are scaled by `sqrt(res / h₀w₀)` and floored to a multiple of 8. The floor is not
  aspect-preserving — Replica drifts 1.765 → 1.791 (+1.5 %) — but `slam/stream.py:40-41` rescales
  the intrinsics with the *actual* ratios, so that is image shear, not a calibration error.
- **The last row is 1/64 of the row above it**, taken by point subsample with no averaging. That
  is the §9.3 trap, and it is why `vggt_hw` is chosen for aspect and nothing else.
- **518 is not arbitrary.** VGGT trained with width pinned to exactly 518 and height a multiple
  of 14 in [168, 518], aspect 0.33–1.0 (`thirdparty/vggt/training/config/default.yaml:5`,
  `training/data/base_dataset.py:95-113`) — landscape-or-square only, never above 518 on any
  axis. `vggt_hw_for` reproduces exactly that shape. Going above 518 does not crash: DINOv2's
  `pos_embed` interpolates bicubically (`vision_transformer.py:180-212`), but the aggregator's
  48 alternating-attention blocks use 2D RoPE with `scaling_factor` unused, so they extrapolate
  to relative offsets never trained on, at ~1.85× tokens and up to ~3.4× global-attention cost
  for (518, 700). And it would buy nothing anyway — see the last row. Worth knowing too: the DPT
  depth head's deepest features sit at 4/7 of input (`dpt_head.py:261-291`), so its "full
  resolution" output is already partly a bilinear upsample.

---

## 10. TUM RGB-D track

Replica cannot test the premise: it is synthetic, well-lit and slow-moving. TUM RGB-D
`freiburg1_room` is the opposite — real Kinect, handheld, fast rotation, motion blur,
auto-exposure, a genuine loop, sparse and holed GT depth. Nothing needs downloading; the sequence
is mirrored at `/storage/group/dataset_mirrors/01_incoming/TUM_RGBD_Dataset/`.

**Two commands run everything:** `python scripts/preprocess_tum.py` once (§10.1), then
`python scripts/run_pipeline.py` (~2 h 15 for extract → adapt → end2end). Every parameter is a
constant at the top of `run_pipeline.py`; `STAGES` re-runs a single stage and `SKIP_EXISTING`
reuses what is already on disk. `SCENE` selects the sequence — `rgbd_dataset_freiburg1_desk` at
the moment, `rgbd_dataset_freiburg1_room` for the loop-closure case described below.

### 10.1 Making TUM look like Replica

`scripts/preprocess_tum.py` exists so the rest of the toolchain needs no dataset-specific code.
Two properties of its output are load-bearing:

- **Sequential `%06d` filenames.** `slam/runner.py:save_trajectory` derives each trajectory timestamp from the
  filename, so index names make timestamps frame indices — matching `preprocess_replica.py:28` and
  letting `evo_ape` associate exactly. Real TUM names would break more than that:
  `adapt/data.py:SceneData` keys poses by `int(timestamp)`, and `1305031910.765238` truncates to
  the same integer for ~30 consecutive frames, silently collapsing the pose dict.
- **`colors/` and `depths/` 1:1 by index.** `eval_utils.py:47`, `extract/export.py` and
  `end2end/metrics.py:split_render_metrics` all index GT depth by RGB frame number.
  `run_pipeline.py:main` asserts this before any GPU work.

**Undistortion happens offline, not via `UNDISTORT`/`CROP_BORDER` (or `demo.py
--undistort/--cropborder`).** `split_render_metrics` re-derives the GT frame with `stream_resize`
only, so the runtime flags would compare undistorted renders against distorted GT — `main()` warns
if either is set. Preprocessing undistorts colour
(bilinear) and depth (nearest — interpolating across a depth discontinuity invents surfaces, and
blending with an invalid 0 drags real depths down), crops the measured black border, rescales depth
5000 → 6553.5, and writes a distortion-free `calib.txt`. Measured for fr1: 18 px border →
604×444 → **400×544** at tracking resolution, from which `vggt_hw` is derived as (378, 518) to
match that aspect (§9.6; nothing needs setting by hand).

Verified end to end on the real data: 0.000 % black pixels after the crop, processed depth median
within 0.4 % of the raw file, and warping frame 0 forward with the written calib + GT poses + GT
depth beats the unwarped baseline ~7× photometrically — which only holds if intrinsics,
undistortion, depth scale and pose convention are all mutually consistent.

### 10.2 What to read from the result

`config/tum_config.yaml` sets `mono_depth_alpha: 0.01`, not Replica's 0.001 — at 0.001 the prior
barely enters BA and a null would be guaranteed by construction. If the comparison is null anyway, sweep
that knob (0.001 / 0.01 / 0.05) before concluding: JDSA re-solving the prior's scale per keyframe
(`track_frontend.py:42`) is exactly the failure mode the adaptation targets, so HI-SLAM2 may be
structurally insensitive to it, and α is the only knob that changes how much the prior can matter.

The first number to check is `export.txt`'s per-frame vs global depth-L1 columns for the Omnidata
row. On Replica room0 they read 0.0735 / 0.2078 — that 2.8× blow-up under a single global scale
*is* the cross-frame inconsistency this whole track targets. If TUM shows it too, there is headroom;
if it does not, expect another null, and the honest reading is that the prior was never the
bottleneck. Adding `'vggt_base'` to `END2END_PRIORS` (stock VGGT-1B, no adapter) separates that from
"VGGT is simply the better prior"; it is off by default because it costs a third full run.

**The `prior` stage (§9.2.2) is the direct form of that check**, and it is the one to run first now:
it reports exactly this ratio as `consistency_index`, for every generator rather than only for the
Omnidata prior of one extract run, over every frame rather than that run's keyframes, and without a
SLAM run. Run it before committing forty minutes an arm to `end2end` — if no generator's index
moves, an end2end null is predictable rather than surprising.

Second, check `export.txt`'s Gaussian-rendered row: it is the training target, so its L1 bounds
what the adapter can learn. On Replica it was 0.0133 m; if TUM's is much worse, the supervision
itself is the limiting factor, not the adaptation.

Note `compensate_exposure: true` means `split_render_metrics` reads *uncompensated* renders, so its
PSNR is comparable between arms but not against published numbers; `final_result_kf.json` has the
compensated variant.
