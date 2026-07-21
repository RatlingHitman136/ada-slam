# HI-SLAM2 on CUDA 13 / PyTorch 2.9 (RTX 4090, shared workstation)

README targets conda + CUDA 11.8 + torch 2.1.2. `legacy-update-env.md` covers the CUDA 12.8 /
torch 2.8 port using system Python. This is the CUDA 13 port, in a **user-local venv** (shared
machine, no root), with nvcc from lmod.

Verified: Replica room0 → **PSNR 35.12 / SSIM 0.958 / LPIPS 0.042** (keyframes: 35.71 / 0.962),
outputs `3dgs_final.ply` + `tsdf_mesh_w2.0.ply`. Tracking ran ~14 it/s, GS refinement ~86 it/s,
whole run ~5 min on an otherwise-idle 4090.

## What changed vs. the CUDA 12.8 port

| | CUDA 12.8 (legacy) | CUDA 13 (this) |
|---|---|---|
| torch | 2.8.0+cu128 | **2.9.0+cu130** |
| Python | system (root) | **venv at `/usr/stud/treh/envs/hislam2`**, managed by `uv` |
| nvcc | system CUDA 12.8 | **`module load cuda/13.0.1`** (no system nvcc on this box) |
| gencode patch | 2 blocks, an optimisation | **4 blocks across 2 files, a hard requirement** |
| `blinker` workaround | needed | not needed (clean venv, not Debian system Python) |
| THC include | untouched | untouched — **still ships in torch 2.9** |
| CUB / CCCL | untouched | untouched — **checked, genuinely fine** |
| deps | loose | **pinned** in `requirements.txt` |

**torch is pinned to 2.9.0 because of `torch-scatter`, not because of torch.** torch itself
publishes cu130 wheels up to 2.13.0, but `torch-scatter` only publishes `+pt29cu130`. Bumping
torch means compiling torch-scatter from sdist. If you want a newer torch, that is the constraint
to attack first — everything else in the stack is flexible.

## 1. Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # → ~/.local/bin/uv
uv venv /usr/stud/treh/envs/hislam2 --python 3.12
source /usr/stud/treh/envs/hislam2/bin/activate
module load cuda/13.0.1                                  # CUDA_HOME=/storage/software/cuda/cuda-13.0.1
export TORCH_CUDA_ARCH_LIST="8.9+PTX"                    # Ada; change for other GPUs

uv pip install --index-strategy unsafe-best-match -r requirements.txt
```

**`--index-strategy unsafe-best-match` is mandatory, not optional.** uv gives `--extra-index-url`
*priority* over `--index-url`, so PyPI wins for `torch` and the `+cu130` builds are never
considered. Without the flag you get a flat `no version of torch==2.9.0+cu130` even though the
wheel plainly exists. Both indexes involved are official, so the dependency-confusion risk the
flag normally guards against does not apply here.

Edit `requirements.in` (the load-bearing pins) rather than `requirements.txt` (the full
resolution); regenerate with
`uv pip compile --index-strategy unsafe-best-match requirements.in -o requirements.txt`.

`PyGLM`, not `glm`. `pytorch-lightning` is only needed to unpickle the Omnidata checkpoint.
numpy stays 2.x — open3d 0.19 declares no upper bound, so the usual open3d/numpy2 clash is absent.

**`opencv-python` must be capped below 5.** OpenCV 5.x is now `latest` on PyPI; an unpinned
install silently pulls a new major against 4.x-era code. The lockfile holds it at 4.13.0.92.

## 2. Submodules

```bash
git submodule update --init --recursive --force   # eigen, lietorch, simple-knn, glm
```

## 3. Code patches

**a. gencode → sm_89. Four blocks, two files.** CUDA 13 dropped Pascal/Volta, so nvcc now
*hard-errors* (`nvcc fatal : Unsupported gpu architecture 'compute_60'`) instead of just wasting
build time as on 12.8. Replace every `compute_60…86` list with:

```python
'-gencode=arch=compute_89,code=sm_89',
'-gencode=arch=compute_89,code=compute_89',
```

in **both** blocks of `setup.py` (droid_backends + lietorch) **and** both blocks of
`thirdparty/lietorch/setup.py`.

Three traps here, all of which cost time on this port:
- The two blocks in the root `setup.py` are **not** byte-identical — the second has trailing
  whitespace on each line, so a naive find-replace silently patches only the first.
- `thirdparty/lietorch/setup.py` has its own pair. The root build does not use it, but a
  standalone lietorch build hits the same nvcc error.
- **Ninja caches the full compile command line.** After editing `setup.py` you must
  `rm -rf build/temp.*`, or the rebuild reuses the old flags and fails identically.

Verify the result rather than trusting the edit:
```bash
grep -c 'compute_89' setup.py thirdparty/lietorch/setup.py      # expect 4 and 4
cuobjdump --list-elf <built.so> | grep -oE 'sm_[0-9]+' | sort -u  # expect sm_89
```

**b. `thirdparty/diff-gaussian-rasterization/cuda_rasterizer/{rasterizer_impl,forward,backward}.h`**
— add `#include <cstdint>` near the top of each (gcc 13 + CUDA 13 drop the transitive include →
`uint32_t` undefined). Same as the 12.8 port.

**c. `thirdparty/lietorch/lietorch/`** — `.type()` returns `DeprecatedTypeProperties`, which no
longer converts to `ScalarType`:
```bash
sed -i 's/\.type()/.scalar_type()/g' thirdparty/lietorch/lietorch/src/lietorch_gpu.cu \
                                     thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp
```
and in `include/dispatch.h`: `at::ScalarType _st = ::detail::scalar_type(the_type);` →
`at::ScalarType _st = the_type;`, plus `const auto& the_type` → `const auto the_type`.

**d. `hislam2/midas/omnidata.py`** — torch ≥ 2.6 defaults `weights_only=True`:
```python
checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
```

### Checked and deliberately NOT changed

Recorded so nobody re-investigates these:

- **CUB / CCCL 3.x.** CUDA 13 ships CCCL 3.x, which removed the `debug_synchronous` overloads —
  a classic 3DGS-rasterizer breakage. The four call sites in `rasterizer_impl.cu` (lines 173,
  196, 288, 313) use the 5-arg `DeviceScan::InclusiveSum` and the 7-/10-arg
  `DeviceRadixSort::SortPairs` forms, all still valid. It compiles unmodified.
- **`src/altcorr_kernel.cu:2` → `#include <THC/THCAtomics.cuh>`.** THC was removed from PyTorch
  proper long ago, but torch 2.9 **still ships the compatibility shim**
  (`torch/include/THC/THCAtomics.cuh`), so this builds as-is. If a future torch drops it, the fix
  is `#include <ATen/cuda/Atomic.cuh>` — the file only uses plain `atomicAdd` on float accessors.

Harmless noise you will see at runtime: `FutureWarning: torch.cuda.amp.autocast(args...) is
deprecated` from `hislam2/factor_graph.py` and `pgo_buffer.py`. Cosmetic under torch 2.9.

## 4. Build extensions

```bash
uv pip install --no-build-isolation ./thirdparty/simple-knn
uv pip install --no-build-isolation ./thirdparty/diff-gaussian-rasterization
rm -rf build/temp.*                              # if you patched setup.py after a failed build
python setup.py install                          # droid_backends + lietorch
```

`--no-build-isolation` is required (the build needs the installed torch). `import torch` must
precede importing the extensions, else `libc10.so` errors. `gs_backend.py` imports the GUI
unconditionally, so glfw/PyOpenGL/PyGLM/open3d are needed even headless.

Note `python setup.py install` is deprecated but **necessary here** — `setup.py` calls `setup()`
twice, and `pip install .` would build only the first.

## 5. Weights, data, run

Storage split on this workstation (see `~/CLAUDE.md`): weights in home, data and outputs on
`/storage/user/treh` via symlinks.

```bash
wget -c https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt  -P pretrained_models
wget -c https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt -P pretrained_models

mkdir -p /storage/user/treh/data /storage/user/treh/hislam2_outputs
ln -s /storage/user/treh/data data
ln -s /storage/user/treh/hislam2_outputs outputs
bash scripts/download_replica.sh && python scripts/preprocess_replica.py   # 12 GB

python demo.py --imagedir data/Replica/room0/colors --calib calib/replica.txt \
    --config config/replica_config.yaml --output outputs/room0     # headless: no --gsvis/--droidvis
python tsdf_integrate.py --result outputs/room0 --voxel_size 0.01 --weight 2
```

**The dataset mirror does not help.** `/storage/group/dataset_mirrors/01_incoming/Replica` is the
raw Facebook Replica release (meshes, textures, `room_0` naming). HI-SLAM2 needs the NICE-SLAM
rendered RGB-D sequences (`room0/results/frame*.jpg`), which is a separate ~12 GB download.

## 6. Reusing this on another machine

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match -r requirements.txt
# then sections 2-4 to rebuild the CUDA extensions
export TORCH_CUDA_ARCH_LIST="<your arch>"   # 8.9+PTX here; 8.6 for A6000/3090, 9.0 for H100
```

`requirements.txt` pins Python packages only. The four CUDA extensions always recompile against
the local toolkit, and the gencode flags in both `setup.py` files must match the target GPU —
on non-Ada hardware, change `compute_89` accordingly.
