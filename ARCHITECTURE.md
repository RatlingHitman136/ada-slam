# ada-slam / HI-SLAM2 — code map

A file-by-file description of this repository. HI-SLAM2 ([arXiv:2411.17982](https://arxiv.org/pdf/2411.17982))
is a **monocular** SLAM system that produces both a camera trajectory and a 3D Gaussian
Splatting map. It is built from three lineages, and the folder layout mirrors them:

| Lineage | What it contributes | Where it lives |
|---|---|---|
| **DROID-SLAM** | dense flow-based tracking, factor graph, dense BA | `hislam2/{factor_graph,track_*,depth_video,motion_filter}.py`, `hislam2/geom`, `hislam2/modules`, `src/` |
| **Omnidata / DPT (MiDaS)** | monocular depth + normal priors | `hislam2/midas/` |
| **MonoGS / 3DGS / RaDe-GS** | Gaussian map, rasterizer, GUI | `hislam2/gaussian/`, `thirdparty/diff-gaussian-rasterization` |
| **VGGT** *(this fork, §9)* | alternative depth prior, LoRA-adapted on SLAM depth | `thirdparty/vggt`, `adaslam/`, `scripts/*_pipeline.py` |

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
                   • render + evaluate   ← optional, off by default (§11)
                        │
                        ▼
              outputs/<seq>/  →  tsdf_integrate.py  →  mesh
```

**§11 first.** The project target is now **pose estimation alone**, so the render-and-evaluate step
above is off by default and every metric derived from it has been removed from the pipeline. What
is described below as producing PSNR/SSIM/depth-L1/mesh numbers still exists in the code but is no
longer reached; §11 says exactly what changed and what stayed.

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
| `tsdf_integrate.py` | Post-process: fuses the **rendered** depth/colour images from `outputs/<seq>/renders/*_after_opt` with `traj_full.txt` into an Open3D `VoxelBlockGrid` and extracts a triangle mesh (`tsdf_mesh_w<weight>.ply`). Depth PNGs are 16-bit scaled by 6553.5. **Not reached from `adaslam/` any more** (§11): it needs `renders/`, which the §9 pipeline no longer produces. Still driven by `run_replica.py` / `run_scannet.py`. |
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
| `adaslam/` | The VGGT track's own code — every stage of the §9 pipeline, as importable packages (§9.5). `slam/` is the **single interface to HI-SLAM2** and the only package here that imports `Hi2` or `MotionFilter`; `extract/`, `adapt/`, `end2end/` and `priortest/` are the four stages of the offline track, and `online/` is §13's single-stage alternative to the first three; `common.py`, `runtime.py` and `paths.py` hold what more than one of them needs. It is a **real package** — `from adaslam.adapt import ...`, `from adaslam.common import ...`. Unlike `hislam2/`, which has no top-level `__init__.py` and so can only be a `sys.path` entry, this one has one, and that `__init__` is where `hislam2/` and `thirdparty/vggt` get put on `sys.path` (§9.5). It was `ada-slam/` until the hyphen — illegal in an identifier — was the only thing forcing the same treatment as `hislam2/`. |
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
  them back into `video.poses` — **this is why it is kept even though the map is not a
  deliverable**, §11); fills all non-keyframe poses; then renders and evaluates **only if
  `args.render_eval`** (read through a `getattr(..., True)` so `demo.py` is unaffected).
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
| `utils/eval_utils.py` | `eval_rendering` — renders every 5th frame plus all keyframes, writes JPEG renders and 16-bit depth PNGs (×6553.5) into `renders/{image,depth}_after_opt/`, scores PSNR/SSIM/LPIPS (and depth L1 if `--gtdepthdir`), dumps `psnr/after_opt/final_result.json`. `eval_rendering_kf` does the keyframe-only variant with exposure compensation. `save_gaussians` was never called by anything. **Reached only from `demo.py` now** — the §9 pipeline runs with `render_eval=False` (§11). |
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
| `init_adapt_pipeline.py` | **The VGGT track's initial-adaptation driver** — extract a densified *prefix*, adapt on all of it, compare arms. §9.1. |
| `cont_adapt_pipeline.py` | **Its continual-adaptation sibling** — extract the *whole* sequence at stock keyframe density, adapt on a thin equidistant sample of it (optionally continuing from an earlier adapter), compare arms. §9.7. |
| `online_adapt_pipeline.py` | **The single-stage driver** — §13. No extract and no frozen second pass: ONE SLAM run whose depth prior LoRA-adapts on each keyframe local BA settles, then the reference arms and one comparison table. Same PARAMETERS-block shape as the two above; the stage is `adaslam/online/`. |
| `export_end2end_results.py` | **One scene's end2end arms as a CSV**, for a Notion database — §12. `-n <name> -s <scene> [--init\|--cont\|--live]` → `outputs/<name>.csv`. Read-only over `outputs/`: it joins each arm's `results.json` to the adapt `config.json` behind it and the extract that trained it, and decomposes the un-sortable experiment name into columns. One kind per run, `--init` by default; each driver gets its own column set and its own Notion database, and which arms belong to which is read off the experiment-name prefix (`live*`, `cont*`). Its computed numbers are `adapt_cost`, `train_pct` / `train_span_pct` and the `--cont` table's `regime`. |
| `ate_over_time.py` | **Where in the sequence an arm's ATE lives** — §12.3. `-s <scene> <arm> [<arm> …]` prints the per-pose APE evo already saved, one row per frame (`--keyframes` / `--bins N` for the other two granularities, `--csv` to dump). Read-only, no GPU, nothing recomputed. Its docstring is the how-to-read-it, and it is needed: the value is a residual after one *global* Sim(3) fit, so it neither starts at zero nor rises monotonically. |
| `plot_trajectories.py` | **Where in *space* it lives** — §12.4. `-s <scene> -o <name> <arm> …` draws the estimated paths from above into `outputs/plots/<name>.png`, every pose coloured green→red by its APE on one scale shared by the whole image. Read-only, no GPU: it applies the Sim(3) evo already saved (`evo/alignment_transformation_sim3.npy`) to `traj_full.txt` and colours by `evo/error_array.npy`. |

The VGGT track adds five more — see §9 and §12: **two drivers**, `init_adapt_pipeline.py` and
`cont_adapt_pipeline.py` (extract → adapt → end2end comparison, differing only in *which*
keyframes the adapter is trained on — §9.1, §9.7), and **three read-only views** over what they
wrote, `export_end2end_results.py` (arms as CSV rows), `ate_over_time.py` (one arm's error
along the sequence) and `plot_trajectories.py` (the paths in space). The standalone single-stage tools `export_slam_depth.py` and
`lora_adapt_vggt.py` are **deleted**: once every stage became an importable package under
`adaslam/` (§9.5) they were thin argparse wrappers over code reachable in three lines from a
REPL, and a second way to invoke a stage is a second place for its defaults to drift.

---

## 6. `config/` — what the knobs mean

Five files: `replica_config.yaml`, `scannet_config.yaml`, `owndata_config.yaml`, `euroc_config.yaml`,
`tum_config.yaml` (§10 — ScanNet's real-sensor preset with `skip_blur`, `pgba.active`, exposure
compensation, and `mono_depth_alpha: 0.01`). Either driver generates a sixth at runtime,
`OUT_EXTRACT/full/extract_config.yaml`, which `inherit_from`s one of these and overrides the
keyframe thresholds for the extract run only (§9.2.1). `cont_adapt_pipeline.py` leaves all four
`None`, so its generated file carries nothing but the `inherit_from` — stock keyframing (§9.7).

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
3dgs_final.ply            the Gaussian map — always written, gs.finalize() saves it

--- only when render_eval is on (§11); the §9 pipeline runs with it off ---
renders/image_after_opt/  rendered RGB (jpg)
renders/depth_after_opt/  rendered depth (16-bit png, ×6553.5)
psnr/after_opt/final_result.json      PSNR/SSIM/LPIPS/depth-L1 over all evaluated frames
psnr/after_opt/final_result_kf.json   the keyframe-only variant
tsdf_mesh_w<W>.ply        written later by tsdf_integrate.py, which reads renders/
```

Caveat on `final_result.json`'s `mean_l1`, for the `demo.py` runs that still produce it:
`eval_utils.py:62` compares metric GT depth against **unscaled** SLAM depth, so on a monocular run
it is dominated by the arbitrary global scale (~0.7 m on Replica, where the Sim(3) scale is ~1.3)
and does not measure depth shape.

### 7.1 The §9 pipeline's tree — stage, then scene, then experiment

The fan-out is real: a scene has several extracts, each has several adapts, each has several
tests. So `outputs/` is keyed by **stage, then scene**, and an experiment directory holds what the
*next* stage consumes and nothing else. The raw HI-SLAM2 run goes one level down in `full/`, which
means it can be deleted to reclaim the Gaussian map without breaking the stage after it.

```
outputs/
  extract/<SCENE>/<EXTRACT_NAME>/ the handoff to adapt, and nothing else
    depth_slam/%06d.npy           per-keyframe training depth, float32, SLAM units: 1/disps_up.
    mask_slam/%06d.png            mask is droid_backends.depth_filter & depth>0. ONE source —
                                  `rendered`, the Gaussian map's expected depth, went with the
                                  terminate-time render (§11). common.py:DEPTH_DIR / MASK_DIR
    image/%06d.jpg                the matching keyframe RGB — a record, NOT read by adapt, which
                                  indexes the full colour dir by frame number (ADAPT_IMAGES)
    poses_slam.txt                the exported keyframes, TUM c2w. adapt reads column 0 (the
                                  keyframe list) and takes its poses from traj_full.txt; every
                                  keyframe in the npz gets a depth file, so the list is all of them
    traj_full.txt                 copied up from full/ — every pose adapt actually uses
    intrinsics.npy                copied up from full/
    export.txt                    the depth accuracy table (read this first)
    full/                         the untouched SLAM run: a normal outputs/<seq>/ as above, plus
                                  extract_config.yaml (generated; inherit_from CONFIG + EXTRACT's
                                  kf_* knobs, this run only) and slam_depth.npz (post-global-BA
                                  state: tstamp, disps 1/8, disps_up full, poses w2c, images,
                                  intrinsics, dscales, disps_prior)

  adapt/<SCENE>/<ADAPT_NAME>/     the handoff to end2end, and nothing else
    adapter.safetensors           ~48 MB
    config.json                   the structure from_adapter rebuilds: rank, alpha, targets,
                                  lora_patch_embed, vggt_hw — plus `scene`, the extract directory
                                  it trained on, which is where lineage is recorded, and
                                  `train_seconds`, the wall time of the training loop at this save
    train_log.json
    checkpoints/epoch_NNN/        every ADAPT.checkpoint_every epochs, each a COMPLETE adapter
                                  dir that from_adapter loads and an arm can run

  test/
    end2end/<SCENE>/<arm>/        ONE DIRECTORY PER DEPTH-PRIOR GENERATOR, <arm> inferred (below).
                                  An ONLINE run (13) writes one of these too, named <NAME>_live so
                                  a later frozen test of its adapter cannot overwrite the live run.
                                  A normal outputs/<seq>/ as above, plus results.json (ate_all /
                                  ate_seen / ate_unseen, the n_* pose counts each was averaged
                                  over, and the split_at they were computed at), evo/, ape.txt
    prior/<SCENE>/<arm>/          the SAME generators scored against GT depth with no SLAM run
      frames.csv                  one row per frame — the expensive artifact, split-INDEPENDENT,
                                  cached on the eval spec written into its `#` header line
      results.json                aggregates at one split_at, recomputed from frames.csv in ms
```

**Naming.** `EXTRACT_NAME` and `ADAPT_NAME` in either driver are **required** and are checked by
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
for Omnidata on the rest of that scene.

The supervision target is `1/disps_up`, written to `depth_slam/`. There was a second and better
one — the **Gaussian-rendered** depth, 2.4× closer to GT (0.0133 vs 0.0324 m global-scale L1 on
room0) and produced from the same post-refinement trajectory `traj_full.txt` supplies for the pose
loss. It came out with the terminate-time render (§11); it was never what the runs in §9.4 used.

### 9.1 Pipeline

**A driver under `scripts/` is the only way to run it.** There are two, and they run the *same*
four stages over the same packages — they differ only in **which keyframes the adapter is trained
on**:

| driver | extract | adapt trains on | val is | §
|---|---|---|---|---|
| `init_adapt_pipeline.py` | the first `FRACTION`%, keyframes **densified** by `EXTRACT`'s `kf_*` | every keyframe of it | the contiguous tail (`val_source='tail'`) | 9.1 |
| `cont_adapt_pipeline.py` | the **whole sequence**, **stock** keyframing (all `kf_*` `None`) | `KF_FRACTION` of them, **equidistant** | every keyframe skipped (`val_source='rest'`) | 9.7 |

Both take an `ADAPT_INIT` — the adapter to **continue** from, or `None` for stock VGGT-1B.

There is a **third** driver, `online_adapt_pipeline.py`, and it is not a row in that table: it has
no extract stage and no frozen second pass at all. One SLAM run adapts the prior on each keyframe
local BA settles and is itself the arm — §13.

Four stages in one process, every parameter in the block at the driver's top — a handful of
CAPITAL constants and the config dataclass literals `SLAM` / `EXTRACT` / `LORA` / `ADAPT` /
`END2END` / `PRIOR` they feed (§9.5). The file is the knob panel, not the implementation: every
stage is a package under `adaslam/`. No CLI, no environment:

```
python scripts/init_adapt_pipeline.py        # from the repo root, adaslam venv active

  1 extract   SLAM over the first FRACTION% of the sequence (generated extract_config.yaml)
              into OUT_EXTRACT/full/ → slam_depth.npz → depth_slam/ mask_slam/ image/
              poses_slam.txt + export.txt one level up, with traj_full.txt / intrinsics.npy
              copied up beside them
  2 adapt     LoRA-adapt VGGT on a TRAIN subset of those keyframes, starting from ADAPT_INIT's
              adapter or from stock VGGT-1B, depth L1 reported on a held-out VAL subset
              → ADAPT_OUT/adapter.safetensors, plus a snapshot every ADAPT.checkpoint_every
              epochs into ADAPT_CKPT/epoch_NNN/. Its five paths are arguments (ADAPT_IN /
              ADAPT_IMAGES / ADAPT_OUT / ADAPT_CKPT / ADAPT_INIT), so it can be run against any
              earlier extract's export and continue from any earlier adapter
  3 end2end   one full-sequence arm per entry in END2END_PRIORS, differing ONLY in the depth
              prior, each into OUT_END2END/<inferred name>/ → evo ATE, and a comparison table
              split at the frame the adapter's training data ended
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
**deleted**. Everything it did lives in the drivers; batching several scenes now means
editing `SCENE` and re-running, or driving the file from a loop.

Dataset preprocessing is deliberately *not* part of it: run `scripts/preprocess_tum.py` (or
`preprocess_replica.py`) once by hand first.

### 9.2 The scripts

| File | Purpose |
|---|---|
| `init_adapt_pipeline.py` | **The initial-adaptation driver** — the PARAMETERS block and `main()`, which is a flat dispatch over the four stage functions. See §9.2.1 for what each stage does and where it lives. It imports nothing from `demo.py` or from `scripts/`; the stages are the packages in `adaslam/`, and the only other dependency is the one CLI those packages drive as a subprocess, `evo_ape` (it was three until the mesh metric went, §11). |
| `cont_adapt_pipeline.py` | **The continual-adaptation driver** — §9.7. Same shape, same stages, same `main()` dispatch; what differs is the PARAMETERS block. There are **no stage wrappers in either**: the `banner()` and the explanatory prints that used to sit in `stage_extract` / `stage_adapt` / `stage_end2end` / `stage_prior` moved into the stage functions themselves, so the second driver inherited them instead of copying them. |
| ~~`export_slam_depth.py`~~ / ~~`lora_adapt_vggt.py`~~ | **Deleted.** Both were argparse wrappers over one stage. Since the stages became packages, the same thing is `from adaslam.extract.export import load_export, write_keyframes` or `LoRAVGGT(cfg).train(...)` — reachable from a REPL, with no second set of defaults to drift out of step with the PARAMETERS block. Re-exporting an existing `slam_depth.npz` without re-running SLAM is still possible that way; the accuracy table without the files is `load_export` then `from adaslam.extract.accuracy import report_accuracy`, skipping `write_keyframes`. Neither name is in `extract`'s `__all__` — the fifth rule in §9.5 says why, and why that costs nothing. |

Every stage is importable — `adaslam.extract.run_extract`, `adaslam.adapt.run_adapt`,
`adaslam.end2end.run_end2end_test`, `adaslam.priortest.run_prior_test`,
`adaslam.slam.SlamRunner.run` — so a driver is a convenience, not a
gate. Nothing duplicates anything (§9.5).

#### 9.2.1 The stage packages

A driver holds the parameters and a flat dispatch; this is where the work actually is.

| Entry point | What it does |
|---|---|
| `adaslam.slam.SlamRunner.run` | **The single interface to HI-SLAM2**; `adaslam/slam/` is the only package importing `Hi2` or `MotionFilter` — an invariant one grep checks (§9.3). Three or more call sites reach it: the extract run and one per end2end arm. One `SlamRunner` is built in `main()` from `SLAM`, so the arms cannot disagree about the stream, calibration or resolution; everything that legitimately differs (tracking YAML, output dir, length, buffer, `gtdepthdir`, `dump_slam_depth`, **depth prior**) is an argument, visible at the call site. It asserts the 9-field `Hi2` args contract (`HI2_ARGS`) before construction. |
| `adaslam.extract.run_extract` | Writes a generated `extract_config.yaml` that `inherit_from`s `CONFIG` and applies `EXTRACT`'s four `kf_*` knobs, runs SLAM over the first `FRACTION`% with the `hi2.py` depth dump enabled **into `<exp>/full/`**, copies `traj_full.txt` and `intrinsics.npy` up to the experiment level, then exports `depth_slam/` + `mask_slam/` (§7.1). Its two halves skip independently: an existing `full/slam_depth.npz` reuses the SLAM run but the export still re-runs when a handoff artifact is missing, so a run whose export was interrupted can finish it without re-tracking. The generated config is given to **this run only**; `main()` asserts the arms get the unmodified `CONFIG`, so a denser training set can never masquerade as a tracking change in the comparison. Note the binding gate is `kf_redundant_thresh` (`frontend.keyframe_thresh`), not the motion filter: over 204 TUM frames `(motion, redundant) = (2.4, 4.0)` gave 43 keyframes, `(1.2, 4.0)` only 45, `(1.2, 1.5)` 83, because `track_frontend.py:49-52` prunes back whatever the motion filter proposes. GT depth reaches `ExtractConfig.gt_depths` (the accuracy table) and never `Hi2` (§9.3). |
| `adaslam.adapt.run_adapt` | The **first stage that was wired to explicit I/O**, and the pattern the other two now follow: `run_adapt(lora_cfg, cfg, in_dir, image_dir, out_dir, ckpt_dir, init_adapter=…)` reads no path global, so it can be pointed at any earlier extract run without moving `OUT_EXTRACT`. It checks its paths (including that the export's highest keyframe index exists in `image_dir`, since the two are now free to be any pair), then `LoRAVGGT.from_adapter(init, lora_cfg, seed=cfg.seed).train(...)`, **the one `lora.save()`**, and `release()` — in that order, because `save()` goes through `_ensure_live()`. Training itself writes no adapter: `run_training` returns `state` and `run`, which are exactly `save()`'s `state=` and `extra=`, so where the adapter lands is a decision at the call site. **`init_adapter` is the adapter this run continues from**, named the same way an `END2END_PRIORS` entry names one (a handoff directory or one of its checkpoints), or `None` for stock VGGT-1B — and because `from_adapter(None, …)` is exactly a stock build, warm and cold start are *one call with no branch*. The adapter it started from is recorded into the new `config.json` as `init_adapter`, read off `LoRAVGGT.adapter` rather than passed down, so checkpoints carry it too. Which keyframes are trained on is `kf_fraction` + `val_source` (§9.7); `AdaptConfig.adapt_style` picks how they reach the loop: `'normal'` is `epochs` shuffled passes over the train set, `'online'` walks them in **arrival (frame) order**, adapting `epochs` consecutive steps on each before the next — a simulation of adapting as keyframes come in, one continuous run (the optimiser state carries across). `'wonline'` is the same simulation with a **sliding window**: the arriving keyframe plus the `window_size`−1 before it, shuffled, `epochs` batched passes over that window, then the window slides on by one — so a keyframe is revisited for `window_size` arrivals instead of being seen once and dropped. There is deliberately **no partial warm-up window**: the first unit is the first *full* one, windows running `[0..w−1]` … `[n−w..n−1]`, which is `n − w + 1` of them. `window_size` is read by this style alone. Note all three change only the training *order*: an end2end arm still loads a frozen adapter. Includes a **train/val split** of the exported keyframes, **select then split** (`adapt/data.py:training_split`, the one place both modes live): `kf_fraction` picks which keyframes are trained on at all — equidistant over the keyframe *list*, `1.0` = every one — and `val_source` says what the remainder is for. `'tail'` validates on the contiguous last `1 - train_frac` of the selection, so val measures generalising **forward** and the trained region is a strict prefix; `'rest'` validates on **every keyframe the selection skipped**, interleaved through the whole sequence, which is the mode a sparse sample needs (§9.7). Either way depth L1 is reported on held-out keyframes rather than on the adapter's own training set. `keep_best` optionally snapshots on val improvement instead of keeping the last epoch, and `checkpoint_every` drops a full loadable adapter dir into `ckpt_dir` every N epochs. |
| `adaslam.end2end.run_end2end_test` | One full-sequence run per entry in `END2END.priors` into `out_root/<arm_name(spec)>` — the directory is an argument, so `End2EndConfig` holds knobs and never a location — then `evaluate` + `print_report` per arm and `compare` at the end. There is deliberately **no `adapter` parameter**: each arm carries its own, which is what lets one comparison hold several adapters and their checkpoints. Caching splits in two, because arms are reused across comparisons: the SLAM run is skipped when `{out}/traj_full.txt` exists, and scoring is skipped only when `results.json` records the *same* `split_at` — every adapter has its own training fraction, so one comparison's split is not another's, and re-scoring is one `evo_ape` over a trajectory already on disk, at no SLAM cost. The prior is passed **into** `SlamRunner.run`, which snapshots `MotionFilter.prior_extractor`, installs, and restores it in a `finally` — so a VGGT arm's patch cannot leak into a later Omnidata arm and silently make it a second VGGT arm, and cannot leak out of a run that raised either. The restore is deliberately **after** `hi2.terminate()`: `hi2.py:143` calls the extractor again for the keyframes `terminate()` inserts into low-covisibility gaps. Each arm's prior is `release()`d in its own `finally`, so a crashed arm no longer strands ~2.5 GB. |
| `adaslam.end2end.VggtPrior` | The depth-prior swap. Normals stay Omnidata, so **depth is the only variable between arms**. Undoes `motion_filter.py`'s ImageNet normalisation (§9.3), and reports the stream→VGGT aspect skew, warning above 5 % — the inference-side half of the `vggt_hw` guard (§9.3). The model comes from `LoRAVGGT.from_adapter`, which rebuilds the **whole** structure — rank, alpha, targets, `patch_embed` and `vggt_hw` — from what the adapter recorded, so an arm cannot run the adapter in a shape or at a resolution it was never trained in. `extractor()` returns a **plain function, never a bound method**: functions are descriptors, so the `MotionFilter` binds as the first argument while the `VggtPrior` arrives through the closure cell. A bound method or `functools.partial` is not a descriptor — `mf` would never be passed and `mf.MEAN` / `mf.STDV` / the cached normal model would be lost. |
| `adaslam/end2end/metrics.py` — `evaluate`, and the `run_ate` it calls | The metrics harness, now **ATE and nothing else**: evo with Sim(3) alignment, then the per-pose error array evo saves is **split seen/unseen** at `split_at`, and the pose count behind each number is recorded as `n_all` / `n_seen` / `n_unseen` so two arms that scored different counts can be caught. `run_mesh` (TSDF → Sim(3)-align → `eval_recon.py`) and `split_render_metrics` (PSNR/SSIM/depth-L1 from the saved renders) were both here and both went with the render (§11). Reached through `run_end2end_test`; `from adaslam.end2end.metrics import evaluate` re-scores a finished output dir. |
| `adaslam/end2end/report.py` — `compare` | **One row per arm**, one table per population: its ATE, the signed delta against `priors[0]`, and that delta as a percent. Transposed relative to `priortest`'s table on purpose — see `print_utils.py` below for why the two shapes differ. Pure formatting over `results.json`, so `from adaslam.end2end.report import compare, print_report` re-runs on finished output without a GPU — including on files written before §11, whose extra `render`/`mesh` keys are simply not read and whose absent pose counts print `n/a` rather than a misleading 0. A population every arm scored `None` on is skipped entirely, which is the normal case for `[unseen]` under §9.7's driver. |
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
  ~0.03 m. `run_replica.py:46` does this. *(`end2end/metrics.py:run_mesh` did too and is gone —
  §11. Kept here because anything that reinstates a mesh metric has to reinstate this with it.)*
- **Scale estimates in the losses must not be detached.** Detaching makes a loss only *look*
  scale-invariant — the optimiser then sees a gradient rewarding a shrinking prediction, and
  translations collapse toward zero. This diverged the first overfit test 12×.
- ~~**Every end2end arm must use the same TSDF voxel size.**~~ Marching-cubes allocation failed at
  0.006 when the shared GPU was busy, so `run_mesh` had a fallback ladder and `compare` refused to
  print mesh rows if the arms disagreed. **Vacuous since §11** — there is no mesh metric. The
  surviving form of the same idea is the `n_all` pose-count check in `report.py:compare`.
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
  training mask to where the sensor happened to work. `run_extract` therefore calls `runner.run`
  with `gtdepthdir=None` and hands `DEPTHS` to the export only; the same applies to a manual
  `demo.py --dump_slam_depth` run, which must be given no `--gtdepthdir`. Nothing is lost: the
  export's accuracy table masks
  on `(gt > 0) & mask` regardless, and the only casualty is `final_result.json`'s `mean_l1`, which
  §7 already documents as meaningless on a monocular run. **Dormant since §11** — `Hi2` reads
  `gtdepthdir` only inside the `render_eval` guard, so every caller now passes `None`, the end2end
  arms included. The argument is kept, and this note with it, precisely so the trap stays disarmed
  by construction if the render ever goes back on.

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
suffix was `ADAPT.depth_space`, a knob that **no longer exists** — adaptation is always in depth
space now, §9.5 — and the `p` number the sequence fraction the adapter trained on) and the
TUM ones in `outputs/tum*/`, both predating §7.1. Everything from before that restructure was moved
to `outputs/old/` untouched — nothing was migrated into the new shape, so `SKIP_EXISTING` does not
see it.

Note those runs predate the choice of supervision target and used `slam` depth (`1/disps_up`),
which is the less accurate target and is dumped *before* the refinement that produces the poses
they were trained against. `DEPTH_SOURCE = 'slam'` reproduces them exactly.

§10 moves the same experiment onto real data, which is what the Replica null asks for next.

### 9.5 `adaslam/` — the pipeline as packages

The driver once carried all the stages in 1366 lines. The adapt stage came out first,
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
| `common.py` | `stream_resize` — ONE definition, used by the reader, the LoRA data loader and the prior probe; they must agree or predictions and GT stop lining up pixel for pixel. Also `DEPTH_DIR` / `MASK_DIR`, because `extract` writes them and `adapt` reads them, and the §7.1 layout vocabulary (`EXTRACT_RUN_SUBDIR`, `ADAPT_CKPT_SUBDIR`, `TEST_KINDS`, `HANDOFF_UP`, `extract_run_dir`, `experiment_dir`, `test_dir`, `require_name`, `ADAPTER_FILE`) for the same reason: more than one stage — and both drivers — have to agree on it. `ADAPTER_FILE` lives here rather than in `end2end/config.py` because `adapt/stage.py` needs the same name to warm-start from an adapter and **cannot import `end2end`** — `end2end` imports `adapt`, so that would be a cycle. |
| `pipeline.py` | What a driver does **around** the stages, so the second one did not restate it: `enter` (spawn start-method + `chdir`, once per process), `check_sequence` (every required path exists, returns the frame count, and asserts `colors`/`depths`/`traj` are 1:1 by index — §10.1), `warn_runtime_undistort`, `resolve_lora` (`probe_stream_hw` + `LoRAConfig.resolved`, which must run after `chdir` and before any spawn), and `print_arm_dirs`. Not a stage and not a config; `torch` is imported inside `enter` so a report-only consumer does not pay for it. |
| `runtime.py` | `sh`, `free_vram`, `gpu_gate`, `raise_fd_limit`, `ensure_venv_on_path` — shared-workstation hygiene, nothing stage-specific. Its module docstring is where the pgba CUDA-IPC measurement in §8 is written down. `free_vram` and `gpu_gate` print, but they do work and report on it; anything that only *formats* lives next door. |
| `print_utils.py` | `banner`, `tee` (+`_Tee`), and the two comparison formatters — formatting output for a human, and nothing else. **Stdlib only**, no torch or cv2, so a finished comparison can be reprinted on a machine with neither. They are **transposes of each other**, and which one a report wants follows from its shape: `delta_header`/`delta_row` put an *entity per column* and a metric per row, which suits `priortest` (seven metrics, a handful of arms); `delta_table` puts an **entity per row** with `value` / `vs <baseline>` / `%` columns, which suits `end2end` (one metric, as many arms as a scene has accumulated — an arm per column runs off the terminal well before eighty of them). Both read `values[0]` as the baseline, mark a later value `+` when it beats it and `-` when it loses, print `n/a` for `None`, and leave the mark blank on an exact tie. The table *layouts* around them stay separate — the prior test can star an arm whose split is not its own, which `end2end`'s `compare()` has no notion of. |
| `slam/` | `SlamConfig` (whose `render_eval` is §11's toggle); `mono_stream` (the reader `Process` target) and the `load_frame` it is built on — ONE definition of what the tracker is shown, because `PriorProbe` scores priors on exactly those pixels; `write_tracking_config`; `SlamRunner`, **the single interface to HI-SLAM2** (§9.2.1); and `PriorProbe`, which runs a prior over frames with no SLAM run. `PriorProbe` lives here rather than in `priortest/` because the stock prior **is** a `MotionFilter` method and this is the only package allowed to import it. |
| `priortest/` | `PriorTestConfig` + `arm_split_at` / `resolve_split`; `predict.py` (inference → `frames.csv`); `metrics.py` (the three alignments, and `aggregate`, the only place the split is used); `report.py`; `stage.py` (`run_prior_test`). Imports `arm_name` from `end2end` rather than restating it, so both test kinds name a scene's arms identically. |
| `extract/` | `ExtractConfig`; `export.py` (`confidence_mask`, `load_export`, `write_keyframes`, `export_slam_depth`); `accuracy.py` (the depth table, §10.2's first number); `stage.py` (`run_extract`, `handoff_paths`). Loading is split from writing so the table can be had without the files. `load_export` takes the **run** directory (`<exp>/full`), where the npz is; `write_keyframes` takes the **experiment** directory above it, where the handoff artifacts go. Since §11 nothing here reads the run's `renders/`, so those are the only two paths involved. |
| `adapt/` | `LoRAConfig` / `AdaptConfig`; `stage.py` (`run_adapt` — the whole stage, plus `init_adapter_path` and `check_export`; the last package to get one, which is why its body lived in the driver until there were two drivers to copy it between); `lora.py` (`LoRALinear`, `inject_lora`, `lora_state_dict` — hand-rolled, no `peft`: rank 8 on `attn.{qkv,proj}` + `mlp.{fc1,fc2}` across the aggregator's 24+24 blocks → 6.29 M trainable of 1.17 B; `B` starts at zero so the adapter is identity at step 0, `A` is kaiming — **which is why the seed goes to the constructor**); `model.py` (`LoRAVGGT` — build/inject/load, `forward`, `predict_depth`, `save`, `release`, `from_adapter`); `data.py` (`SceneData`, one keyframe = one sample placed **first** so VGGT predicts in that keyframe's frame — verified: `extrinsic[0]` is identity to 5e-4, rebased poses match SLAM GT to 0.04° — plus `evenly` / `select_keyframes` / `split_keyframes` / `training_split`, the select-then-split chain both val modes go through, and `evenly` is the one definition of "n items spaced by index" that `trainer.py`'s `eval_max_kf` cap also uses); `losses.py` (the two undetached scale estimates §9.3 warns about; both losses are **depth-space, not a choice** — VGGT's head emits depth, the prior extractor hands depth to HI-SLAM2, and `depth_video.py:70-73` does the inversion itself for every prior, so a disparity-space loss only changed which end of the range the error weighted); `trainer.py`, whose `schedule` is the **one place the three adaptation styles differ** — `'normal'` yields an epoch of shuffled batches, `'online'` yields one arriving keyframe at a time with `epochs` consecutive steps on each, `'wonline'` yields a sliding window of `window_size` keyframes ending at the arrival with `epochs` shuffled batched passes over it, and everything downstream (eval, `keep_best`, checkpoints, the log) treats an epoch, an arriving keyframe and a window alike as a *unit* — the styles meet again at `batches_of`, which cuts a keyframe order into batches for both `'normal'` and `'wonline'`. It **reports and returns but does not write the adapter** — `model.py:save` writes, the caller decides when and where. The only weights `trainer.py` writes are `checkpoint_every`'s periodic snapshots. |
| `end2end/` | `End2EndConfig` + the prior-spec vocabulary (`SENTINELS`, `arm_name`, `adapter_path`); `prior.py` (`VggtPrior`); `metrics.py`; `report.py`; `stage.py` (`run_end2end_test`). Was `abtest/` until an arm stopped being one of three fixed names. Note `End2EndConfig.check_priors_exist` is called by the **stage**, not `__post_init__`: the config is built in a block that must not touch the filesystem and that runs before `main()`'s `chdir`, and an adapter listed there legitimately does not exist yet when the whole pipeline runs — the adapt stage is about to create it. |
| `online/` | **§13's single-stage alternative to `extract`+`adapt`+`end2end`.** `OnlineConfig`; `target.py` (`LiveSampler` and the window helpers — the live counterpart of `adapt/data.py`, returning **exactly** `SceneData.sample`'s 5-tuple so `losses.py` is reused unchanged); `trainer.py` (`LiveTrainer` — ONE AdamW for the whole run, which is what makes it continual, and `batches_of` imported from `adapt/trainer.py` rather than restated); `prior.py` (`OnlineVggtPrior`, a `VggtPrior` whose extractor adapts before it predicts); `stage.py` (`run_online_adapt`). It imports `adapt` and `end2end` and neither imports it, so there is no cycle. |

Five rules hold across all of it — the fourth still being rolled out:

- **No config field carries a default.** Five frozen dataclasses now — `SlamConfig`,
  `ExtractConfig`, `LoRAConfig`, `AdaptConfig`, `End2EndConfig` — and every one of them is stated in
  full in **each driver's** PARAMETERS block. A knob is therefore written down in exactly one
  place per entry point, and none can be inherited silently from a package. That the two drivers
  restate the same fields is the rule working, not duplication to remove: it is what lets
  `cont_adapt_pipeline.py` set `kf_fraction=0.10, val_source='rest'` while its sibling sets
  `1.0, 'tail'`, with neither reading the other's value or a package default. The one field that
  may be left `None` is `LoRAConfig.vggt_hw`, and that is not a default: `None` is a stated
  instruction meaning *derive this from the stream*, honoured by `LoRAConfig.resolved(stream_hw)`
  in `main()` and refused by `LoRAVGGT` if it ever reaches the model unresolved (§9.3).
- **A package that receives another's config does not re-declare its fields.** `extract` and
  `end2end` are both handed the `SlamConfig`, so `colors` / `calib` / `stream_res` / `start` exist
  once. Two configs naming the same *value* (`DEPTH_PNG_SCALE` reaching both `EXTRACT` and `PRIOR`)
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
  `grep -n 'OUT_EXTRACT\|OUT_END2END\|COLORS' scripts/*_pipeline.py` returns nothing inside a
  stage body — there are no stage bodies left in either driver, only a `main()` that passes every
  path to `run_extract` / `run_adapt` / `run_end2end_test` / `run_prior_test`.
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

### 9.7 Continual adaptation — `scripts/cont_adapt_pipeline.py`

§9.1's driver adapts on a **densified prefix**: lower the `kf_*` thresholds, track the first
`FRACTION`% of the sequence, train on every keyframe it produced. That answers "can the adapter
learn this scene's depth from a short, richly sampled window?" This driver asks the complementary
question — **can it learn the scene from a thin sample of the whole of it, and can it keep
learning from where a previous adapter stopped?** Same four stages, same packages; three
parameters differ.

**1. The extract is the whole sequence at stock keyframe density.** All four `kf_*` are `None`,
which `write_tracking_config` already treats as "inherit", so the generated
`extract_config.yaml` carries nothing but its `inherit_from` and the run keyframes exactly as
upstream HI-SLAM2 would. Hence `EXTRACT_NAME = 'low_dense_kf'`. Two practical consequences: it is
a **full-length SLAM run**, roughly one end2end arm's cost rather than the few minutes a 9%
extract took; and `EXTRACT.buffer` must clear the whole sequence's keyframe count, not a prefix's
(no overflow guard — `run_extract` warns when the count reaches it). It is written once and every
later adapt run reuses it through `SKIP_EXISTING`.

**2. The adapter trains on `KF_FRACTION` of those keyframes, and validates on the rest.**
`adapt/data.py:training_split` is select-then-split: `select_keyframes` takes 10% of the export
**equidistant over the keyframe list** (every 10th *keyframe* — keyframes are unevenly spaced in
time, so this is not every 10th frame), and `val_source='rest'` makes val the complement. On the
226-keyframe `dense_kf_p40` export that is 23 train / 203 val, no overlap, full coverage.

That val set is the point of the design. The tail split its sibling uses measures generalising
*forward in time*; the complement measures generalising **between the samples**, which is the only
thing a sparse selection can be judged on. It is also ~90% of the export, so `eval_max_kf` is what
keeps the eval cheap — it subsamples evenly, so the capped sample still spans the sequence.

**3. There is no seen/unseen frontier, and `SPLIT_AT` says so.** Training keyframes span the whole
sequence, so no frame index separates trained from untrained; `SPLIT_AT = None` sets the end2end
split to the sequence length, everything counts as "seen", and `[unseen]` prints nothing.
**`ate_all` is the row the comparison reduces to** — which §12.2 already argues is the only ATE
column comparable across arms, since each adapter computed its `ate_seen`/`ate_unseen` at whatever
split was current when it ran. Set `SPLIT_AT` to a frame index to override.

**Warm starting** is `ADAPT_INIT`, and it is not specific to this driver — both have it, defaulting
to `None` = stock VGGT-1B. It names an adapt handoff directory (or one of its checkpoints), the
same vocabulary an `END2END_PRIORS` entry uses, so continuing from a run and testing it are spelled
alike. `LoRAVGGT.from_adapter` rebuilds the **recorded** structure, so a continued run cannot
silently change rank, targets or `vggt_hw` mid-lineage; the new adapter's `config.json` records
`init_adapter`, and so does every checkpoint, which is where the chain is readable.

Verified on `dense_kf_p40`: cold start reads base depth L1 0.1029 train / 0.0896 val, warm start
from `normal_r8_e15_p10` reads 0.0440 / 0.0421 at the same first evaluation — i.e. the `base` row
of a continued run is the *incoming* adapter's error, not stock VGGT's, which is the check worth
repeating if the loading path is ever touched.

---

## 10. TUM RGB-D track

Replica cannot test the premise: it is synthetic, well-lit and slow-moving. TUM RGB-D
`freiburg1_room` is the opposite — real Kinect, handheld, fast rotation, motion blur,
auto-exposure, a genuine loop, sparse and holed GT depth. Nothing needs downloading; the sequence
is mirrored at `/storage/group/dataset_mirrors/01_incoming/TUM_RGBD_Dataset/`.

**Two commands run everything:** `python scripts/preprocess_tum.py` once (§10.1), then
`python scripts/init_adapt_pipeline.py` (~2 h 15 for extract → adapt → end2end). Every parameter is a
constant at the top of the driver; `STAGES` re-runs a single stage and `SKIP_EXISTING`
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
  `pipeline.py:check_sequence`, which both drivers' `main()` calls, asserts this before any GPU work.

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

Second, check `export.txt`'s SLAM-depth row: it is the training target, so its L1 bounds what the
adapter can learn. On Replica it was 0.0324 m; if TUM's is much worse, the supervision itself is
the limiting factor, not the adaptation. (The Gaussian-rendered row was the better bound at
0.0133 m and is no longer produced — §11.)

Third — and this is the row the whole comparison now reduces to — read `compare()`'s **unseen**
ATE. Everything else it used to print was render- or mesh-derived and is gone (§11).

---

## 11. Pose estimation only — the terminate-time render is off

The project target narrowed: the deliverable is a **correct trajectory**, not a trajectory *and* a
3DGS map. So `Hi2.terminate()`'s render-and-evaluate step is now optional and off, and everything
that existed to consume its output has been removed from the pipeline.

### 11.1 The toggle

`hi2.py`'s last act before returning was `self.gs.eval_rendering(...)` — render every 5th frame plus
every keyframe, load AlexNet twice for LPIPS, write `renders/{image,depth}_after_opt/` and
`psnr/after_opt/final_result{,_kf}.json`. It is now guarded:

```python
if getattr(self.args, 'render_eval', True):
    self.gs.eval_rendering(...)
```

`getattr(..., True)` mirrors `dump_slam_depth` at `hi2.py:155`: `demo.py` hands `Hi2` a raw argparse
namespace with no such attribute and keeps upstream behaviour unchanged. **This is the only edit
inside `hislam2/`.** The flag reaches `Hi2` through `SlamConfig.render_eval` → `HI2_ARGS` → the
`SimpleNamespace` in `slam/runner.py`, and is set once in each driver's PARAMETERS block as
`RENDER_EVAL = False`.

It is a `SlamConfig` field rather than a `SlamRunner.run()` keyword because it is identical across
every run of an experiment — the extract run and every arm — which is exactly the split
`slam/config.py`'s docstring draws.

### 11.2 What was deliberately NOT switched off, and why

**`gs.finalize()` stays.** It runs `color_refinement`, which jointly optimises the Gaussians **and
per-camera pose deltas**, and `hi2.py:177` writes those refined poses back into `video.poses`;
`traj_filler` at `:179` then builds the whole `traj_full.txt` from them. The Gaussian stage is
therefore part of the *pose* pipeline, not only the map. Cutting it would change every trajectory
the repo has ever produced.

`eval_rendering` sits after all of that, is `@torch.no_grad()`, and writes only files — so
**the toggle cannot move a pose**, and runs made before and after it stay comparable. That is the
whole reason the cut was made at this line and not a few lines earlier. The per-keyframe Gaussian
mapping in `track()` and `3dgs_final.ply` are likewise untouched; the saving is the render pass and
the two AlexNet loads, not the map.

**Measured, and worth knowing for its own sake: the tracker is not bit-reproducible.** Three
150-frame extract runs of `freiburg1_room` — two with the toggle off, one with it on — give
max per-pose deviations in `traj_full.txt` of:

| | max abs difference |
|---|---|
| off vs off (same settings, two runs) | 1.7e-3 |
| off vs on | 2.3e-3 |

The toggle's effect is inside the noise the pipeline already has (CUDA atomics in the BA kernels),
so `diff`ing two trajectories proves nothing either way — the argument above is the structural one,
and it is the one to rely on. The practical consequence is broader than this change: **no ATE
comparison in §9 is repeatable to better than ~1e-3**, so a delta of that size between arms is
noise, not a result.

**And ~1e-3 is a floor, not a constant.** The same measurement on rellis_00000 at 500 frames reads
**8.3e-3** between two identical `vggt_base` runs (§13.6) — 5× the figure above — because the
Gaussian map and the colour refinement that writes poses back are much larger there. The number to
compare a delta against is the one measured at *your* scene and length, not this row.

### 11.3 What was removed

| gone | was in |
|---|---|
| `run_mesh` — TSDF fuse → Sim(3) align → `eval_recon.py` | `end2end/metrics.py` |
| `split_render_metrics` — per-frame PSNR / SSIM / depth-L1 from the saved renders | `end2end/metrics.py` |
| the `hislam2_eval` passthrough of `psnr/after_opt/final_result.json` | `end2end/evaluate` |
| the PSNR / SSIM / depth-L1 / four mesh rows | `end2end/report.py` |
| `gt_mesh`, `gt_depths`, `depth_png_scale`, `voxel_size`, `voxel_fallbacks`, `mesh_weight` | `End2EndConfig` |
| the `'rendered'` depth source: `depth_rendered/`, `mask_rendered/`, the accuracy table's Gaussian row | `common.py`, `extract/`, `adapt/` |
| `ExtractConfig.depth_sources`, `AdaptConfig.depth_source`, `DEPTH_SOURCE`, `GT_MESH` | the configs and the PARAMETERS block |

`results.json` now holds `label`, `output`, `split_at`, `ate_{all,seen,unseen}` and
`n_{all,seen,unseen}`. The pose counts are new and replace the frame-count comparability check the
render block used to provide. `report.py` reads them with `.get()`, so a `results.json` written
before this change still prints its ATE — its counts read `n/a`.

**Nothing upstream was deleted.** `hislam2/gaussian/utils/eval_utils.py`, `tsdf_integrate.py`,
`scripts/eval_recon.py`, `scripts/run_replica.py`, `scripts/run_scannet.py` and `demo.py` are all
still here and still work; they are simply no longer reached from `adaslam/`. Turning
`RENDER_EVAL = True` brings the renders back — what would not come back on its own is the code that
scored them.

### 11.4 What survives untouched

The entire `priortest/` package: it scores depth priors against GT with no SLAM run and never
touched a render. `run_ate`, the `ate_*` keys, `end2end/prior.py`, `arm_name` / `adapter_path` /
`SENTINELS`, the whole `slam/` package bar one field, and every §9.3 trap that is about the depth
prior rather than about rendering.

---

## 12. Results export — `outputs/` as one table

Sixty adapt runs and sixty-odd arms per scene cannot be compared by opening `results.json` files
one at a time. `scripts/export_end2end_results.py -n <name> -s <scene>` writes
`outputs/<name>.csv`, one row per arm, for a Notion database.

The CSV is a **view**, never a source of truth: it is read-only over `outputs/` and regenerable at
any time, so deleting it loses nothing.

**Three drivers, three tables.** The parameters of the three pipelines do not overlap, so one
column set cannot serve them — `train_pct` says nothing about a run that adapted across the whole
sequence, and `warmup_prior` does not exist for one that adapted offline. Exactly **one kind per
invocation**, enforced by an argparse mutually-exclusive group, `--init` when none is given:

| flag | driver | columns |
|---|---|---|
| `--init` *(default)* | `init_adapt_pipeline.py` (§9.1) | `epochs window lr train_pct train_frames n_train_kf adapt_cost train_seconds extract extract_kf` |
| `--cont` | `cont_adapt_pipeline.py` (§9.7) | `regime epochs window lr kf_fraction val_source train_frac train_span_pct n_train_kf n_val_kf adapt_cost train_seconds init_adapter extract extract_kf base_train_l1 train_l1 base_val_l1 val_l1 d_val_l1` |
| `--live` | `online_adapt_pipeline.py` (§13) | `steps_per_kf window lr alpha lag warmup_kf warmup_prior warmup_end_frame first_adapted_kf n_units n_train_kf adapt_cost train_seconds init_adapter` |

all three sharing `arm style ate_all d_ate_vs_omni d_ate_pct exported_at`.

**Which arms** a table holds is decided by the **experiment-name prefix** the driver's names carry
— `live*` online, `cont*` continual, everything else init. Two rules sit on top of it: `omni` and
`base` are in **every** table (the deltas are against `omni`, and §13's driver compares against
exactly those two), and under `--live` an *un-run* arm named for a live adapter without
`ONLINE_ARM_SUFFIX` is dropped — that is the frozen-replay name (§13.4), a blank duplicate of the
`<name>_live` row beside it rather than an arm waiting its turn. On rellis_00000 that is 24 rows.

The `--live` table deviates from §12.1 in one place, on purpose: `adapt_cost` is
`config.json:kf_visits`, the number `LiveTrainer` **counted** (`online/trainer.py:130`), not the
formula. The two agree on every live run on disk, and that is a coincidence — the formula
reconstructs the count only while every arrival adds exactly one distinct keyframe to a window that
is never clipped, and `target.py:41` clips it when `window_size > warmup_kf` while pruning shifts
the frame indices `n_train_kf` counts (§13.5). It also drops `train_pct` / `train_frames`
(`split_at` is the whole sequence for every live run) and `extract` / `extract_kf` (no extract
stage, and `scene` records the arm's own directory) — which retires §13.4's caveat about that
column.

The `--cont` table deviates in two places, both forced by §9.7's design. **It exports a metric
other than ATE**: `base_train_l1 / train_l1 / base_val_l1 / val_l1 / d_val_l1`, the `eval_history`
rows for the base model and for the unit the adapter was *saved* at — matched by **tag**
(`e24`, `k22`, `w224`), not taken as `history[-1]`, since under `keep_best` the trainer records
the whole history beside a smaller `saved_epoch` (`trainer.py:277`) and the last row can be an
evaluation of weights the arm never ran. §12.2 refuses `ate_seen`/`ate_unseen` for being computed
at incomparable splits and the same objection applies here — the **val set differs by regime and
by the knob under it** (a tail of 176 keyframes at `train_frac` 0.25 and 117 at 0.5, the 211 the
selection skipped at `kf_fraction` 0.1, none at all under `full`), and `eval_max_kf` caps the
evaluated subset without recording the cap. It is answered rather than dodged: there is no
split-independent substitute
(val L1 on the keyframes a thin sample skipped *is* what §9.7 set out to measure), the grouping
keys are columns, and on a warm start `base_val_l1` is the **incoming adapter's** error, so
`d_val_l1` — what this run itself added — is the column that compares across lineages. Second,
`regime` (`prefix` / `sample` / `full`) is derived from `val_source` + `train_frac`, and
`train_span_pct` from `train_end`, because init's `train_pct` reads 100 for every run here — the
extract *is* the whole sequence. It is not `kf_fraction` restated either: 25% of the keyframes
reaches frame 672 of 2847 (23.6%), since keyframes are unevenly spaced.

The prefix cross-check runs the other way too, and **only one way**: `val_source: 'rest'` or
`kf_fraction < 1` can only have come from `cont_adapt_pipeline.py` (`init_adapt_pipeline.py` pins
both, and `AdaptConfig` rejects `'rest'` at `kf_fraction 1.0`), so a run carrying either mark
without the `cont` prefix is reported. The converse proves nothing — a cont run at
`KF_FRACTION=1.0` records exactly what an init run records — which is why the prefix still decides.

### 12.1 `adapt_cost` — the time reference

Wall clock is not comparable on a shared workstation. Only 4 of 57 adapt runs even recorded
`train_seconds`, and those four disagree by 2× per unit of work (0.42–0.85 s per keyframe visit)
purely from other users' jobs. So the cost axis is a **deterministic count**: how many times any
keyframe was pushed through VGGT, read straight off `adapt/trainer.py:schedule`.

| style | a unit is | visits per unit | units |
|---|---|---|---|
| `normal` | an epoch | `n_train_kf` | `epochs` |
| `online` | an arriving keyframe | `epochs` | `n_train_kf` |
| `wonline` | a sliding window | `epochs * window_size` | `n_train_kf - window_size + 1` |

`adapt_cost = units_done * visits_per_unit`, where `units_done` is `saved_epoch + 1` — not
`epochs` — because a **checkpoint records the whole run's `epochs` and stopped early**:
`normal_r8_e10_chkp_004` costs 1015, half of `normal_r8_e10`'s 2030.

Note the wonline row: counting *window passes* (`(n - w + 1) * epochs`) instead of visits would
make a wonline arm look 10–20× cheaper than a normal arm doing the same work.

### 12.2 What it deliberately does not do

- **It never parses the arm name for a parameter.** Every parameter is recorded in the adapter's
  `config.json`, and **the names lie**: `outputs/adapt/rellis_00000/wonline_r8_e5_w20_p10/config.json`
  records `epochs: 3`, and `live_e3_a16_w12_lag4_normal_r8_e20_p10` records `alpha: 8`. The name is
  a label; `config.json` is the data. `arm_name` is *imported* from `end2end/config.py` and the
  adapter→arm map inverted, so the export names an arm exactly as the pipeline did. The **one**
  exception is the pipeline prefix above, and only because it is the one fact no file in an arm's
  directory records — which driver ran it. Where `config.json` *can* corroborate it (`online: true`
  for a live run) the two are cross-checked and a disagreement is printed, because a live run whose
  name lacks the prefix is invisible to `--live`, and a missing row is far harder to notice than a
  wrong one.
- **It exports no `ate_seen` / `ate_unseen`.** Each arm's `results.json` computed them at whatever
  `FRACTION` was current when that arm ran — `omni` at `split_at: 284`, `normal_r8_e20` at `1138` —
  so they are **not comparable across arms**. `ate_all` is split-independent and is the metric the
  table exists for. (`evo/error_array.npy` + `timestamps.npy` are saved per arm, so any split is
  recomputable for free if that ever changes.) §9.7's driver takes the same position from the other
  direction: its arms have no seen/unseen split to export at all.
- **It exports no constant** — and *which* fields are constant depends on the kind. `rank`,
  `batch_size`, `seed`, `lambda_pose`, `weight_decay` and `grad_clip` are identical across every
  config of both kinds; `alpha` is identical across all 93 offline ones but takes 8 and 16 across
  the live ones, so it is a `--live` column only. `lr` varies everywhere and is a column in both.
- **It requires `-s`.** `scene` is not a column, and two scenes in one file would collide on `arm`
  — both have an `omni` — and silently mix baselines. One database per scene **and per kind**,
  since the column sets differ.

`omni` is the only baseline: `d_ate_vs_omni` and `d_ate_pct` are against it, and `base` (stock
VGGT) is just another arm scored the same way.

### 12.3 `ate_over_time.py` — where in the sequence the error is

The CSV above answers *which arm*; this answers *where*. `evo_ape --save_results` already writes
the per-pose series, so `<arm>/evo/` carries `error_array.npy` (APE, translation, metres — one
value per **frame** of `traj_full.txt`), `timestamps.npy` (frame indices, because §10.1's `%06d`
names make timestamps frame indices) and `distances_from_start.npy` (the **reference**
trajectory's cumulative path length). `end2end/metrics.py:load_ape` is the one reader of that
layout; `run_ate` writes it and returns through the same function.

```
python scripts/ate_over_time.py -s rellis_00000 omni normal_r8_e20_p10 --bins 12
```

One arm prints value / running RMSE / an ASCII profile; two or more print `print_utils`'
`delta_row`s indexed by frame instead of by metric. `--bins N` aggregates to N windows and is the
better first look — a row is then a window's RMSE, so one outlier frame cannot look like a trend.
`--keyframes` restricts to `traj_kf.txt`, and there the arms **disagree about which frames are
keyframes** (`omni` 233, `base` 228, `normal_r8_e20_p10` 250, sharing ~130): rows come from
`arms[0]` and every arm is sampled at those same frames, which is the only row-aligned comparison
available. The tool prints the counts rather than resolving the disagreement silently.

**Three things about the number, all of which the script restates in its header**, because the
table is easy to misread without them:

- **It is a residual after ONE global Sim(3) fit** (`run_ate` passes `-vas`). So it does not start
  at zero — `omni` frame 0 reads 37.6 m — and it is not monotonic: 35 → 2.5 (frame 504) → 56
  (frame 2742). A dip is where the estimated path crosses the globally-fitted GT path, not a
  recovery.
- **The absolute level is not attributable to a frame.** A better-*shaped* trajectory earns a
  different alignment and its whole curve shifts. Read shape and relative change; never "arm B was
  6 m better at frame 0".
- **Sign consistency across the sequence is the signal.** `normal_r8_e20_p10` beats `omni` in
  every window (−5.9 early, −10.5 mid, −13.7 late) — a real win. A win in one stretch and a loss
  elsewhere is one good segment. A *widening* delta means the baseline drifts faster; a constant
  offset usually means the two differ through the alignment rather than through drift. §11.2's
  ~1.7e-3 non-reproducibility is the floor under all of it.

Two columns are worth knowing. `dist(m)` is GT path length so far, so error rising while it is
**flat** means the rig has stopped and the estimate is wandering — rellis_00000's last ~240 frames
sit at 332.2 m while the error climbs to 56 m. And `cumRMSE` accumulates the rows' member
*frames*, not the row values, so its last entry equals `results.json:ate_all` exactly
(31.278976) — a free check on the whole table. (An equal-weighted RMSE of per-bin RMSEs is not
the RMSE when the bins differ in size, which is what that column got wrong first.)

### 12.4 `plot_trajectories.py` — where in *space* the error is

§12.3 answers *when*; this answers *where*. Same input, drawn instead of tabulated:

```
python scripts/plot_trajectories.py -s rellis_00000 -o omni_vs_lora omni base normal_r8_e20_p10
```

→ `outputs/plots/omni_vs_lora.png`. A column of 2847 residuals cannot show that a run cut a
corner or wandered while the rig was parked; the path can.

**Nothing is recomputed.** `evo/` already holds both halves — `error_array.npy` is the per-pose
APE and `alignment_transformation_sim3.npy` is the 4×4 Sim(3) `evo_ape -vas` fitted, scale baked
into the rotation block. Applying the second to `traj_full.txt`'s translations puts the estimate
in GT coordinates, and the residual there **equals the first to ~6e-14** — so geometry and colour
come from one transform, the same one `results.json:ate_all` was measured under. The script
checks that identity per arm and refuses to draw if it exceeds 1e-6, because a stale `evo/`
beside a re-run `traj_full.txt` would still produce a plausible picture. `metrics.py` grew
`load_alignment` and `gt_traj_of` for it, beside `load_ape`, so the `evo/` layout stays defined
once; `arm_dir` + `no_such_arm` moved there from `ate_over_time.py` for the same reason.

Three things decided by the data rather than typed:

- **The GT** comes from `evo/info.json:ref_name`, the absolute path evo recorded. A scene id does
  not determine its dataset directory (`rellis_00000` → `data/RELLIS/00000`, but the TUM scene is
  named after its own), and only a driver's PARAMETERS block knows the mapping. Arms disagreeing
  on `ref_name` are a hard error — they are not in one frame.
- **The plane** drops the GT axis with the least spread, which is the vertical one for a ground
  vehicle (rellis_00000: x 154 m, y 202 m, z 1.5 m → x/y, *not* the x/z a camera-convention
  dataset would want). `--plane` overrides.
- **The layout**: ≤3 arms overlay on one panel, 4+ facet into small multiples on shared axes.
  Colour is spent on error, so arm identity is carried by marker shape and the legend — never by
  colour, and not by dash pattern either, since a path is one `LineCollection` of ~2850
  centimetre-long segments and a linestyle on it renders solid.

**Reading it.** The same three caveats as §12.3 apply, and one more that only the picture raises:
**each arm carries its own Sim(3)**, so two paths here are each individually best-fitted to GT.
That is what makes their *shapes* comparable and their absolute offsets not. Green is not
"correct" either — the ramp spans the arms you asked for, and the best pose of the best arm on
rellis_00000 still sits 1.9 m from GT. A stretch that reddens while the path barely moves is
§12.3's flat-`dist(m)` case seen directly: the rig stopped and the estimate drifted.

---

## 13. Continuous adaptation — one stage, `scripts/online_adapt_pipeline.py`

§9's track is **offline and three-staged**: `extract` runs SLAM and dumps `slam_depth.npz`, `adapt`
LoRA-trains VGGT on that dump, `end2end` runs a *second* SLAM pass with the adapter frozen. Both
drivers there are variations on *which* keyframes the offline adapt trains on (§9.1, §9.7).

This is the complementary experiment: **one SLAM run in which the prior learns as the map is
built.** Every frame with enough motion becomes a keyframe, local BA settles it, and the adapter
takes an optimiser step on that keyframe's SLAM depth — so the prior serving keyframe *N* has
already been adapted on keyframes up to *N−lag*. The finished run is scored exactly the way an
end2end arm is, and its adapter is saved in the normal handoff shape.

```
python scripts/online_adapt_pipeline.py     # from the repo root, adaslam venv active

  1 online   HI-SLAM2 over the sequence with a SELF-ADAPTING VGGT prior
             -> outputs/adapt/<SCENE>/<ONLINE_NAME>/            the adapter it ended with
             -> outputs/test/end2end/<SCENE>/<ONLINE_NAME>_live/  the arm it was trained by
  2 end2end  the REFERENCE arms (omni, base), reused from disk, then one compare() table holding
             both of them plus the live arm
```

There is **no extract stage and no frozen second pass**. The run that produces the supervision is
the run being evaluated: `<NAME>_live`'s ATE is the trajectory the online run produced *itself*,
with the prior still moving under BA while it was tracked.

Replaying the final adapter frozen — adding `ADAPT_OUT` to `END2END_PRIORS`, which would score into
`<NAME>` and is collision-free by construction (§13.4) — is deliberately **not** the default. It is
a second full SLAM run, and it answers a different question than this driver is asking.

### 13.1 The hook — `hislam2/` is untouched

`SlamRunner.run` installs an arm's prior as a **plain function** on the class
(`slam/runner.py:81-83`), so `prior_extractor(mf, im_tensor)` receives the `MotionFilter` as
argument 0 — and `mf.video` **is** the shared `DepthVideo`. Everything the offline export reads out
of `slam_depth.npz` is therefore reachable live, from inside the extractor: `poses`, `disps`,
`disps_up`, `images`, `intrinsics`, `tstamp`, `counter`.

Three facts make that a *correct* place to train and not merely a reachable one:

- **`disps_up` is current.** `factor_graph.py:231` calls `video.upsample` on every BA iteration for
  every keyframe touched by an active edge, so full-res depth is fresh the moment `frontend()`
  returns.
- **The ordering works out.** In `Hi2.track`, `filterx.track()` — which calls `prior_extractor` for
  the arriving keyframe *N* — runs **before** `frontend()` optimises *N*, but **after** the previous
  call's BA pass that included *N−1*. At prior time, keyframes `0..N−1` are settled targets.
- **`terminate()` is detectable.** `video.ready.value` is set to 1 only at `hi2.py:107`, its first
  line. Past that the extractor is still called — for the keyframes `terminate()` inserts into
  low-covisibility gaps (`hi2.py:143`) — while `video.shift` is rewriting every index. The guard
  `video.ready.value == 0` is what keeps adaptation out of that window.

So this track adds **zero** lines to `hislam2/` (the §11 render toggle remains the only edit there),
and §9.3's invariant still holds: `grep -rn 'from hi2 import\|from motion_filter import' adaslam/`
returns only paths under `adaslam/slam/` — now three files, the third being `stock_prior.py`.

### 13.2 What one arriving keyframe does

```
prior_extractor(mf, im_tensor)                    n = mf.video.counter.value
  |
  +- n > warmup_kf and not in terminate() ?  ->   LiveTrainer.on_keyframe(video)
  |                                                 target  = counter - 1 - lag   (target.py)
  |                                                 sample  = image + 1/disps_up + depth_filter mask
  |                                                 steps   = steps_per_kf, through adapt/losses.py
  |
  +- n <  warmup_kf ?  ->  the FALLBACK prior's depth
     n >= warmup_kf ?  ->  VGGT's depth, from the weights the step above just produced
```

Normals stay Omnidata on **both** branches (`end2end/prior.py`'s job, inherited unchanged), so depth
remains the only variable between this arm and the baselines.

`lag` defaults to 2 because that is the repo's own line: `track_frontend.__update` returns
`arange(graph.ii.min(), t1-1)` (`track_frontend.py:65`), so `counter-2` is the newest index
HI-SLAM2 already treats as settled enough to hand to the Gaussian backend.

**Warm-up** is two things at once, deliberately: `warmup_kf` keyframes are served by the fallback
prior, *and* no optimiser step runs until keyframe `warmup_kf + 1`. `warmup_prior` picks the
fallback — `'omnidata'` is upstream's own prior (a genuinely different model, reached through
`slam/stock_prior.py`), `'self'` is the same VGGT this run adapts, frozen until handover, which with
`ADAPT_INIT` set is the "start from a pretrained adapter" case. The handover frame is printed and
recorded as `warmup_end_frame`.

**The two schedules are §9.5's, live.** `adapt_style='online'` trains on the arriving keyframe alone,
`steps_per_kf` steps; `'wonline'` trains on a sliding window of it plus the `window_size−1` before
it, `steps_per_kf` shuffled batched passes. Same names, same meanings, and `batches_of` is imported
from `adapt/trainer.py` rather than restated — so §12.1's `adapt_cost` table applies with no change.
`'normal'` is rejected: an epoch over a fixed train set has no meaning while the set is still
arriving.

**One AdamW is built for the whole run.** That is what makes this continual rather than a sequence
of independent fits — the moments carry from the first keyframe to the last.

### 13.3 The package

| File | Contents |
|---|---|
| `adaslam/online/config.py` | `OnlineConfig` — frozen, no field carries a default (§9.5 rule 1). Its own config rather than `AdaptConfig`: that one carries a dozen fields this run never reads (`kf_fraction`, `val_source`, `train_frac`, `eval_*`, `keep_best`) whose `__post_init__` would force meaningless choices. Names that mean the same thing are spelled the same. |
| `adaslam/online/target.py` | `settled` / `unit_keyframes` / `context_keyframes` / `LiveSampler` — the online counterpart of `adapt/data.py:SceneData`, returning **exactly** its 5-tuple so `adapt/losses.py` is reused unchanged. |
| `adaslam/online/trainer.py` | `LiveTrainer` — the optimiser, `on_keyframe`, the checkpoint cadence, and `stats()`, whose key names are the offline ones wherever they mean the same thing. |
| `adaslam/online/prior.py` | `OnlineVggtPrior(VggtPrior)` — the parent's extractor with the warm-up branch and the adaptation step around it. Still a plain function, never a bound method (§9.3's descriptor reasoning). |
| `adaslam/online/stage.py` | `run_online_adapt` — build → run SLAM → save adapter → `evaluate` → release, with the same two-level cache `end2end/stage.py` uses. |
| `adaslam/slam/stock_prior.py` | `stock_prior_extractor()`. Only `adaslam/slam/` may import `motion_filter`, and the stock prior *is* a `MotionFilter` method — the same reason `prior_probe.py` lives there. |

Three edits elsewhere, all backward compatible: `extract/export.py:confidence_mask` takes an
optional `ix=` so the live path masks one keyframe instead of all K; `common.py` gains
`ONLINE_ARM_SUFFIX`; `scripts/export_end2end_results.py:adapt_dirs` also maps `<name>_live` to an
online run's adapt directory, so it is one CSV row rather than two half-rows.

### 13.4 Naming — why the arm carries `_live`

An online run produces **both** an adapter and the arm that trained it. `end2end/config.py:arm_name`
returns an adapt directory's basename, so without a suffix, later testing this run's *frozen* final
adapter as an ordinary `END2END_PRIORS` entry would infer the same directory and overwrite the live
run's trajectory with a different experiment's. Hence:

```
outputs/adapt/<SCENE>/<NAME>/               adapter.safetensors, config.json, train_log.json
                             checkpoints/epoch_NNN/   full adapter dirs -> arm <NAME>_chkp_NNN
outputs/test/end2end/<SCENE>/<NAME>_live/   traj_full.txt, results.json, evo/, ape.txt
```

Everything downstream reads both unchanged: `compare()`, `ate_over_time.py`, and
`export_end2end_results.py`, whose `adapt_cost` comes out right because `stats()` records
`adapt_style` / `epochs` / `n_train_kf` / `window_size` with the meanings §12.1 already assumes.
That CSV needs `--live` to be worth reading, though: under `--init` an online row is scored into
columns built for an offline run, and `extract` then shows the **arm's own** output directory,
because `scene` records the run that produced the supervision and there is no extract to point at.
§12's `--live` table drops that column and adds the ones that actually vary here — `warmup_*`,
`lag`, `alpha`, `init_adapter` — and takes `adapt_cost` from the counted `kf_visits`.

`SPLIT_AT = None` → the whole sequence, so everything counts as "seen" and `[unseen]` prints
nothing — the same position §9.7 takes, and for the same reason: the adapter learned across the
whole sequence and no frame index separates trained from untrained. The one meaningful boundary,
`warmup_end_frame`, is recorded in the adapter's `config.json`; §12.3 notes any split re-scores for
free from `evo/error_array.npy`.

### 13.5 Traps specific to this stage

- **The supervision is LOCAL-BA depth.** The offline export dumps after **global** BA
  (`hi2.py:155`), the only instant where `disps` / `disps_up` / `poses` are mutually consistent
  (§3.1). Live targets are noisier by construction. This is the price of a single stage, not a bug.
- **It is a self-training loop.** The target is produced by BA that was itself pulled toward the
  prior being adapted, via JDSA. `mono_depth_alpha` is small (0.001 Replica, 0.01 TUM) so the target
  is mostly photometric, but the failure mode is real and the offline pipeline does not have it. The
  signature to watch for is `train_log.json`'s loss collapsing while ATE gets **worse**.
- **The extractor is called more often than there are keyframes, and the same target comes back.**
  `track_frontend.py:52` prunes a redundant keyframe and **decrements** `counter`, so the next
  acceptance lands on the same index. Measured on rellis_00000: **88 extractor calls over 29
  distinct keyframes.** Without a guard the same target is re-trained and `n_units` no longer means
  what §12.1's `adapt_cost` assumes. `LiveTrainer.on_keyframe` therefore keys a unit on the target's
  **frame timestamp**, not its index — indices shift under that pruning, timestamps do not. The same
  applies to `n_train_kf`.
- **Seeding the adapter inside an arm moves the trajectory, and the RNG state must be restored.**
  §9.5's rule is that the seed belongs to the constructor, because `LoRALinear.A` is
  kaiming-initialised at injection. Offline that is harmless: `adapt` is its own stage and no SLAM
  run is in progress. Here the model is built *inside* an arm, and `torch.manual_seed` resets the
  **global** stream that the Gaussian backend then draws from all run long — the mapping window's
  random earlier keyframes, densification, `pcd_downsample`. A `vggt_base` arm never seeds, so
  leaving the stream reset builds a different map, and `gs.finalize()`'s pose deltas are written
  back into `video.poses` (§11.2) — so the whole trajectory moves. **Measured: 2.2e-2 max pose
  difference against the base arm, 13× the 1.7e-3 floor**, on a run that took no optimiser step at
  all. `OnlineVggtPrior.__init__` therefore snapshots `torch.get_rng_state()` +
  `torch.cuda.get_rng_state_all()`, seeds, builds, and restores in a `finally`: `A` is seeded and
  the SLAM run still sees the exact stream a base arm would have. This is what the null-op arm
  (`steps_per_kf=0`, `warmup_prior='self'`) exists to catch, and it caught it.
- **`stock_prior_extractor()` must be captured BEFORE `SlamRunner.run` installs the prior.**
  `run()` overwrites `MotionFilter.prior_extractor`, so fetching the stock one lazily from inside
  our own extractor fetches *itself* and recurses forever. `OnlineVggtPrior.__init__` takes it, and
  the prior is always constructed before `runner.run`.
- **The training step runs under two contexts that must be undone.** `Hi2.track` (`hi2.py:88`),
  `MotionFilter.track` (`motion_filter.py:76`) and the extractor are all `@torch.no_grad()`, and
  `MotionFilter.track` is also under fp16 autocast. The prior opens `enable_grad`, and
  `on_keyframe` disables the ambient autocast so only the explicit bfloat16 block around the forward
  is in effect — the conditions `adapt/trainer.py` trains under.
- **`train()`/`eval()` toggling is not cosmetic.** VGGT's aggregator gates gradient checkpointing on
  `self.training` (`aggregator.py:282,305`), which is what keeps the backward pass inside VRAM. The
  trainer sets train mode for the step and eval mode back before the extractor predicts.
- **The image resize must be the second half only.** `video.images[i]` is already
  `common.py:stream_resize`'d by the reader and already RGB, so `LiveSampler.frame` applies only the
  resize to `vggt_hw`, with the same `cv2.INTER_AREA` `adapt/data.py:116-117` uses. Doing the full
  chain again, or using a different interpolation, would make live samples disagree with every other
  consumer pixel for pixel.

### 13.6 Status

Verified end to end on rellis_00000, 500 frames, `warmup_kf=14`, `online`, `steps_per_kf=1`:
Omnidata served keyframes 0-13, handover at frame 187, 29 adaptation units over 29 distinct
keyframes, 5 checkpoints written, final adapter 6.29 M parameters in 384 tensors with all 192 `B`
matrices non-zero (they start at zero, so that is the proof steps landed), ATE scored through the
ordinary `evaluate` path. **Peak VRAM 9.1 GiB**, 222 s — a VGGT arm already keeps the model resident
for the whole run, so the incremental cost is optimiser state (6.29 M trainable) plus backward
activations, not a second model. `MIN_FREE_VRAM_MB` is nevertheless 12000 in this driver rather than
the offline ones' 7000.

**The null-op arm passes, and measuring it produced a number worth carrying.** On rellis_00000 at
500 frames:

| | max abs pose difference |
|---|---|
| `vggt_base` vs `vggt_base` (two identical runs — **the floor**) | 8.29e-3 |
| null-op online arm vs `vggt_base` run a | 6.23e-3 |
| null-op online arm vs `vggt_base` run b | 5.68e-3 |

The hook sits *below* the pipeline's own run-to-run spread, which is the strongest statement
available. Note the floor is **5× §11.2's 1.7e-3** — that was TUM at 150 frames, and this scene's
Gaussian map and colour refinement are far larger. **Do not reuse 1.7e-3 as a universal
threshold**; measure the floor at the scene and length you are actually comparing at.

Two things worth knowing before reading any result:

- **The loss trace is not a learning curve.** Every step has a different target, so it tracks how
  hard the scene got as much as how well the model fits. `summary()` says so in the log.
- **The null-op arm is the regression test for this stage.** `steps_per_kf=0` +
  `warmup_prior='self'` is stock VGGT-1B throughout with no optimiser step, so it must reproduce a
  plain `vggt_base` arm. It is what caught the RNG-state trap above, and it is the first thing to
  re-run after touching `prior.py` or `stage.py`. Compare it against a **second** `vggt_base` run
  rather than against §11.2's 1.7e-3: that figure was measured on TUM at 150 frames, and this
  scene's map and colour refinement are much larger, so the honest floor is the one measured at the
  same scene and length.

Not yet run: a full-sequence comparison against `omni` / `base`, which is what the driver's default
`STAGES` produces.
