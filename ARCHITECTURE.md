# ada-slam / HI-SLAM2 — code map

A file-by-file description of this repository. HI-SLAM2 ([arXiv:2411.17982](https://arxiv.org/pdf/2411.17982))
is a **monocular** SLAM system that produces both a camera trajectory and a 3D Gaussian
Splatting map. It is built from three lineages, and the folder layout mirrors them:

| Lineage | What it contributes | Where it lives |
|---|---|---|
| **DROID-SLAM** | dense flow-based tracking, factor graph, dense BA | `hislam2/{factor_graph,track_*,depth_video,motion_filter}.py`, `hislam2/geom`, `hislam2/modules`, `src/` |
| **Omnidata / DPT (MiDaS)** | monocular depth + normal priors | `hislam2/midas/` |
| **MonoGS / 3DGS / RaDe-GS** | Gaussian map, rasterizer, GUI | `hislam2/gaussian/`, `thirdparty/diff-gaussian-rasterization` |
| **VGGT** *(this fork, §9)* | alternative depth prior, LoRA-adapted on SLAM depth | `thirdparty/vggt`, `scripts/lora_adapt_vggt.py`, `scripts/run_full_vggt.py` |

The HI-SLAM2-specific contributions on top of those are: **JDSA** (joint depth–scale
adjustment, `geom/ba.py`), **PGBA** (Sim(3) pose-graph + bundle adjustment for loop closure,
`pgo_buffer.py` + `droid_backends.pgba`), and the geometry-aware Gaussian losses
(`gaussian/utils/slam_utils.py`).

This fork adds one research track on top: replacing the Omnidata depth prior with a VGGT model
LoRA-adapted on HI-SLAM2's *own* SLAM depth. That is entirely additive — it lives in `scripts/`
and `thirdparty/vggt`, and touches the core only through a 17-line optional dump hook in
`hi2.py`. See **§9**.

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
and depth/normal priors and the Gaussian map run at full resolution (`disps_up`).

---

## 2. Repository root

| File | Purpose |
|---|---|
| `demo.py` | **Entry point.** Spawns a reader `Process` that decodes/resizes/undistorts images into a `Queue`, constructs `Hi2` lazily on the first frame (needs the image size), loops `hi2.track(...)` until the last frame, then `hi2.terminate()` and writes `traj_kf.txt`, `traj_full.txt`, `intrinsics.npy`. CLI: `--imagedir --calib --config --output --gtdepthdir --buffer --undistort --cropborder --start --length --droidvis --gsvis --dump_slam_depth`. `mono_stream` and `save_trajectory` are module-level and reused by the §9 scripts rather than duplicated. |
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
| `ba.py` | Python-side bundle adjustment. `BA` (full, Schur complement over depths), `MoBA` (motion-only, used by the trajectory filler), `get_prior_depth_aligned` (bilinearly interpolates the 2×2 `dscales` grid to full resolution via `droid_backends.bi_inter`, giving a **spatially varying** scale for the mono depth prior), and **`JDSA`** — HI-SLAM2's joint depth–scale adjustment: solves for inverse depths and the scale grid together, so the learned prior is fused into the BA rather than applied as a fixed rescaling. `alpha` (`mono_depth_alpha`) sets how strongly the prior pulls. |
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
| `preprocess_owndata.py` | For casual video: extracts every frame to `images/`, every 10th (max 100) to `images_colmap/`, runs the full COLMAP pipeline (OPENCV camera model) to estimate intrinsics, writes `calib.txt`. |
| `run_replica.py` | Full Replica benchmark: runs `demo.py` per sequence, `evo_ape` for ATE, reads the render metrics, runs TSDF fusion at 6 mm, aligns the mesh with the evo Sim(3) transform, then `eval_recon.py`; averages everything. |
| `run_scannet.py` | Same for the 8 selected ScanNet scenes (`--cropborder 12`, 15 mm voxels, keyframe-only render metrics, no 3D recon eval). |
| `eval_recon.py` | Mesh evaluation: accuracy / completion / completion-ratio via KD-trees and `evaluate_3d_reconstruction`, plus an optional 2D depth-L1 metric that renders random in-room views of GT vs. reconstruction with Open3D. **Note:** its `trimesh` import was never satisfied before this fork installed it, so this script (and `run_replica.py`'s recon metrics) could not run at all. |

The VGGT track adds six more — see §9: `export_slam_depth.py`, `run_slam_depth_batch.sh`,
`lora_adapt_vggt.py`, `_full_run_common.py`, `run_full_{omnidata,vggt}.py`,
`temp_run_ab_comparison.sh`.

---

## 6. `config/` — what the knobs mean

Four files: `replica_config.yaml`, `scannet_config.yaml`, `owndata_config.yaml`, `euroc_config.yaml`.

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
intrinsics.npy            full-resolution fx fy cx cy
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

With `--dump_slam_depth` and the §9 scripts, a run additionally produces:

```
slam_depth.npz            post-global-BA SLAM state: tstamp, disps (1/8), disps_up (full),
                          poses (w2c), images, intrinsics, dscales, disps_prior
depth_slam/%06d.npy       per-keyframe SLAM depth, float32, SLAM units
mask_slam/%06d.png        multi-view consistency mask (droid_backends.depth_filter)
image/%06d.jpg            the matching keyframe RGB
poses_slam.txt            keyframe poses, TUM c2w, same convention as traj_kf.txt
export.txt                the depth-source accuracy table
lora-vggt/                adapter.safetensors (~48 MB), config.json, train_log.json
ab_results.json           A/B metrics, split seen/unseen
```

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
- `GSBackEnd` inherits `mp.Process` but is driven inline; mapping and tracking are
  serialised in the main process. The GUI, the DROID visualiser and `PGOBuffer.spin` are
  the only genuine extra processes (plus `demo.py`'s image reader).
- Several `torch.cuda.amp.autocast` call sites use the deprecated API and emit warnings on
  torch 2.9 (functionally fine).

---

## 9. VGGT depth-prior track (this fork)

**Premise.** Omnidata degrades under unfamiliar lighting. Its measured weakness on Replica is
specifically **cross-frame scale inconsistency**: it is the only depth source that gets *worse*
under a single global scale fit (0.0611 → 0.0836 m), while SLAM and Gaussian-rendered depth both
improve. The idea is to LoRA-adapt VGGT on HI-SLAM2's own SLAM depth for a scene, then swap it in
for Omnidata on the rest of that scene.

### 9.1 Pipeline

```
demo.py --dump_slam_depth                        →  slam_depth.npz  (post-global-BA state)
  └─ scripts/export_slam_depth.py                →  depth_slam/ mask_slam/ image/ poses_slam.txt
       └─ scripts/run_slam_depth_batch.sh        (batch the above over scenes / sequence fractions)
            └─ scripts/lora_adapt_vggt.py        →  <scene>/lora-vggt/adapter.safetensors
                 └─ scripts/temp_run_ab_comparison.sh
                      ├─ scripts/run_full_omnidata.py   (arm A, stock prior)
                      └─ scripts/run_full_vggt.py       (arm B, adapted VGGT depth)
                           └─ both via scripts/_full_run_common.py
```

### 9.2 The scripts

| File | Purpose |
|---|---|
| `export_slam_depth.py` | Turns `slam_depth.npz` into training-ready per-keyframe files. Builds the confidence mask with `droid_backends.depth_filter` (≥2 of 6 temporal neighbours agree — the recipe in `util/droid_visualization.py:104-110`; arrays **must** be sliced to the real keyframe count or trailing frames match unused buffer slots). Also reports scale-aligned depth L1 for SLAM vs Gaussian-rendered vs JDSA-aligned-Omnidata depth, in both per-frame and global-scale columns — the gap between those columns *is* the cross-frame-consistency diagnostic. |
| `run_slam_depth_batch.sh` | Batches SLAM + export over scenes at a chosen sequence fraction (`FRACTION`), with a shared-GPU VRAM gate and skip-if-done. Params at top. |
| `lora_adapt_vggt.py` | The adaptation. One keyframe = one sample, placed **first** in the sequence so VGGT predicts in that keyframe's frame (verified: `extrinsic[0]` is identity to 5e-4, and rebased poses match SLAM GT to 0.04°). Depth supervises frame 0 only; poses supervise all frames, rebased to the keyframe. A random number of neighbouring non-keyframes ride along, so the adapter works monocular *and* with context. LoRA is hand-rolled (~40 lines, no `peft`): rank 16 on `attn.{qkv,proj}` + `mlp.{fc1,fc2}` across the aggregator's 24+24 blocks → 12.58 M trainable, 1.07 % of 1.17 B. Heads and `patch_embed` stay frozen; gradients reach the aggregator *through* them. |
| `_full_run_common.py` | Shared A/B harness: replicates `demo.py`'s loop, then evo ATE → TSDF → **Sim(3) align** → `eval_recon.py`, and recomputes PSNR/SSIM/depth-L1 per frame from the saved renders so everything can be split seen/unseen. |
| `run_full_{omnidata,vggt}.py` | The two arms. Thin by design — sharing the harness is what keeps the comparison honest. |
| `temp_run_ab_comparison.sh` | Runs arm A, then arm B, then prints the side-by-side. Temporary/experimental. |

### 9.3 Traps that silently corrupt results

- **`prior_extractor` receives an ImageNet-normalised tensor** (`motion_filter.py:88-89`), but
  VGGT expects `[0,1]` and normalises internally (`aggregator.py:205`). `run_full_vggt.py` undoes
  it; forgetting to would just quietly make VGGT worse.
- **The TSDF mesh must be Sim(3)-aligned before scoring.** SLAM scale is arbitrary (~1.3× here)
  and `eval_recon`'s ICP is rigid-only, so skipping the alignment gives ~0.7 m accuracy instead of
  ~0.03 m. `run_replica.py:46` does this; the harness does too.
- **Scale estimates in the losses must not be detached.** Detaching makes a loss only *look*
  scale-invariant — the optimiser then sees a gradient rewarding a shrinking prediction, and
  translations collapse toward zero. This diverged the first overfit test 12×.
- **Both A/B arms must use the same TSDF voxel size.** Marching-cubes allocation fails at 0.006
  when the shared GPU is busy; the harness has a fallback ladder and records `voxel_size`, and the
  comparison refuses to print mesh rows if the two arms disagree.
- VGGT's aggregator returns `None` for uncached layers (only 4/11/17/23 are kept, deliberately,
  so layer indices stay stable — `aggregator.py:196`). Any per-frame slicing of the token list
  must preserve those `None`s.

### 9.4 Status

The plumbing is verified end to end: frame-0 convention confirmed, adapter identity-at-init and
save/load round-trip both exactly `0.0`, overfit test converges, ~7.8 GiB peak.

The **result so far is a null**. On the training keyframes the adapter improves depth L1 a lot
(0.0080 → 0.0032), but a full-sequence A/B on room0/1/2 shows essentially no downstream
difference — ATE identical to four decimals, PSNR within ±0.2 dB, depth L1 within ±0.0005, on
both the trained and the unseen halves. Two plausible reasons, and they are not exclusive:
base VGGT is already very strong on Replica (0.04° rotation, ~1 % relative disparity), leaving
almost no headroom; and JDSA re-solves the prior's scale per keyframe anyway
(`track_frontend.py:42`), which is exactly the failure mode the adaptation targets — so HI-SLAM2
may already be robust to the weakness being fixed. Replica is well-lit synthetic data and does
not test the difficult-lighting premise that motivated the work.

Experiment outputs live in `outputs/ab_{depth,disp}_p{40,100}/` (the `depth`/`disp` suffix is
`DEPTH_SPACE`, the `p` number the sequence fraction the adapter trained on).
