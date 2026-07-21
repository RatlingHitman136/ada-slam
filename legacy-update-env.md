# HI-SLAM2 on CUDA 12.8 / PyTorch 2.8 (RTX 40xx)

README targets conda + CUDA 11.8 + torch 2.1.2. This is the minimal delta to run on
CUDA 12.8 + torch 2.8 + Ada GPU (sm_89), using system Python (no conda). Assumes torch
2.8.0+cu128 already installed, nvcc 12.8, gcc 13, root.

## 1. Submodules
```bash
git submodule update --init --recursive --force   # eigen, lietorch, simple-knn, glm
# if simple-knn ends up empty: git submodule update --init --force thirdparty/simple-knn
```

## 2. Code patches (4 files)

**setup.py** — in BOTH `nvcc` gencode lists (droid_backends + lietorch), replace the whole
`compute_60..86` block with Ada + PTX (else 4090 → "no kernel image is available"):
```python
'-gencode=arch=compute_89,code=sm_89',
'-gencode=arch=compute_89,code=compute_89',
```

**thirdparty/diff-gaussian-rasterization/cuda_rasterizer/{rasterizer_impl,forward,backward}.h**
— add near the top of each (gcc13/CUDA12.8 drop transitive include → `uint32_t` undefined):
```cpp
#include <cstdint>
```

**thirdparty/lietorch/lietorch/{src/lietorch_gpu.cu, src/lietorch_cpu.cpp, include/dispatch.h}**
— `.type()` returns `DeprecatedTypeProperties` which no longer converts to `ScalarType`:
```bash
sed -i 's/\.type()/.scalar_type()/g' thirdparty/lietorch/lietorch/src/lietorch_gpu.cu thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp
```
and in `dispatch.h` change `at::ScalarType _st = ::detail::scalar_type(the_type);` →
`at::ScalarType _st = the_type;` (and `const auto& the_type` → `const auto the_type`).

**hislam2/midas/omnidata.py** (~line 76) — torch 2.6+ defaults `weights_only=True`:
```python
checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
```

## 3. Install (system Python)
```bash
pip install ninja
pip install --ignore-installed blinker          # Debian blinker 1.7 blocks open3d->flask
pip install -f https://data.pyg.org/whl/torch-2.8.0+cu128.html torch-scatter \
    scipy opencv-python matplotlib pyyaml tqdm rich munch plyfile imgviz timm \
    torchmetrics evo open3d PyGLM glfw PyOpenGL pytorch-lightning
#   PyGLM (NOT `glm`); pytorch-lightning needed only to unpickle Omnidata ckpt; numpy stays 2.x

export TORCH_CUDA_ARCH_LIST="8.9+PTX"
pip install --no-build-isolation ./thirdparty/simple-knn
pip install --no-build-isolation ./thirdparty/diff-gaussian-rasterization
python setup.py install                          # droid_backends + lietorch
```
Note: `import torch` before the extensions (else libc10.so error). `gs_backend.py` imports
the GUI unconditionally, so glfw/PyOpenGL/PyGLM/open3d are needed even headless.

## 4. Weights + data + run
```bash
wget -c https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt  -P pretrained_models
wget -c https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt -P pretrained_models
bash scripts/download_replica.sh && python scripts/preprocess_replica.py   # 12.4GB; keep on big disk

python demo.py --imagedir data/Replica/room0/colors --calib calib/replica.txt \
    --config config/replica_config.yaml --output outputs/room0        # headless: no --gsvis/--droidvis
python tsdf_integrate.py --result outputs/room0 --voxel_size 0.01 --weight 2
```
Verified: room0 → PSNR 35.3 / SSIM 0.96, outputs `3dgs_final.ply` + `tsdf_mesh_w2.0.ply`.