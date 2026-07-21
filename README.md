<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"><img src="media/logo.png" width="70">-SLAM2: Geometry-Aware Gaussian SLAM for Fast Monocular Scene Reconstruction</h1>
  <p align="center">
    <a href="https://www.ifp.uni-stuttgart.de/en/institute/team/Zhang-00004/" target="_blank"><strong>Wei Zhang</strong></a>
    ·
    <a href="https://cvg.cit.tum.de/members/cheq" target="_blank"><strong>Qing Cheng</strong></a>
    ·
    <a href="https://www.ifp.uni-stuttgart.de/en/institute/team/Skuddis/" target="_blank"><strong>David Skuddis</strong></a>
    ·
    <a href="https://www.niclas-zeller.de/" target="_blank"><strong>Niclas Zeller</strong></a>
    ·
    <a href="https://cvg.cit.tum.de/members/cremers" target="_blank"><strong>Daniel Cremers</strong></a>
    ·
    <a href="https://www.ifp.uni-stuttgart.de/en/institute/team/Haala-00001/" target="_blank"><strong>Norbert Haala</strong></a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2411.17982">Paper</a> | <a href="https://hi-slam2.github.io/">Project Page</a></h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="./media/teaser.jpg" alt="Logo" width="100%">
  </a>
</p>
<p align="center">
<strong>HI-SLAM2</strong> constructs a 3DGS map (a) from <strong>monocular input</strong>, achieving accurate mesh reconstructions (b) and high-quality renderings (c). It surpasses existing monocular SLAM methods in both <strong>geometric accuracy</strong> and <strong>rendering quality</strong> while achieving <strong>faster runtime</strong>.
</p>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#run-demo">Run Demo</a>
    </li>
    <li>
      <a href="#run-evaluation">Run Evaluation</a>
    </li>
    <li>
      <a href="#semantic-reconstruction">Semantic Reconstruction</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## Getting Started

This fork targets **CUDA 13 / PyTorch 2.9 / Python 3.12** — the upstream conda + CUDA 11.8
instructions no longer apply. Setup is a single script.

1. Clone the repo with submodules
```Bash
git clone --recursive git@github.com:RatlingHitman136/ada-slam.git
cd ada-slam
```

2. Run the setup script
```Bash
./setup_env.sh --with-weights
```

That one command:
- installs [uv](https://docs.astral.sh/uv/) into `~/.local/bin` if it is not already there
- fetches the submodules and applies `patches/lietorch.patch` — lietorch is a git submodule, so
  its CUDA 13 fixes cannot be committed to this repo directly and ship as a patch instead
- creates a venv at `.venv` and installs the pinned dependencies from `requirements.txt`
- compiles the four CUDA extensions (~10 minutes)
- verifies that real CUDA kernels actually execute

`--with-weights` additionally downloads the 3.9 GB Omnidata depth/normal checkpoints into
`pretrained_models/`; drop the flag if you already have them. The script is safe to re-run —
every step detects work that is already done and skips it. Add `--force-rebuild` to recompile
the extensions.

Useful knobs:

| Variable | Default | Purpose |
|---|---|---|
| `TORCH_CUDA_ARCH_LIST` | `8.9+PTX` | Target GPU arch — `8.6+PTX` for A6000/3090, `9.0+PTX` for H100 |
| `ADASLAM_VENV` | `<repo>/.venv` | Where the venv is created |
| `CUDA_MODULE` | `cuda/13.0.1` | lmod module to load when `nvcc` is not on `PATH` |

On non-Ada GPUs, also change `compute_89` in `setup.py` and `patches/lietorch.patch`.

A CUDA toolkit (`nvcc`) is required to **build** the extensions, but not to run them — PyTorch
ships its own CUDA runtime.

## Data Preparation
### Replica
Download and prepare the Replica dataset by running
```Bash
bash scripts/download_replica.sh
python scripts/preprocess_replica.py
```
where the data is converted to the expected format and put to `data/Replica` folder.

### ScanNet
Please follow the instructions in [ScanNet](https://github.com/ScanNet/ScanNet) to download the data and put the extracted color/pose/intrinsic from the .sens files to `data/ScanNet` folder as following:
<details>
  <summary>[Folder structure (click to expand)]</summary>

```
  scene0000_00
  ├── color
  │   ├── 000000.jpg
  │   └── ...
  ├── intrinsic
  │   └── intrinsic_color.txt
  └── pose
  │   ├── 000000.txt
  │   └── ...
```
</details>

Then run the following script to convert the data to the expected input format
```bash
python scripts/preprocess_scannet.py
```
We take the following sequences for evaluation: `scene0000_00`, `scene0054_00`, `scene0059_00`, `scene0106_00`, `scene0169_00`, `scene0181_00`, `scene0207_00`, `scene0233_00`.

## Run Demo
After preparing the Replica dataset, you can run HI-SLAM2 for a demo. It takes about 2 minutes to run the demo on an Nvidia RTX 4090 GPU. The result will be saved in the `outputs/room0` folder including the estimated camera poses, the Gaussian map, and the renderings. To visualize the constructing process of the Gaussian map, using the `--gsvis` flag. To visualize the intermediate results e.g. estimated depth and point cloud, using the `--droidvis` flag.
```bash
python demo.py \
--imagedir data/Replica/room0/colors \
--calib calib/replica.txt \
--config config/replica_config.yaml \
--output outputs/room0 \
[--gsvis] # Optional: Enable Gaussian map display
[--droidvis] # Optional: Enable point cloud display
```
To generate the TSDF mesh from the reconstructed Gaussian map, you can run
```bash
python tsdf_integrate.py --result outputs/room0 --voxel_size 0.01 --weight 2
```

## Run Evaluation
### Replica
Run the following script to automate the evaluation process on all sequences of the Replica dataset. It will evaluate the tracking error, rendering quality, and reconstruction accuracy.
```bash
python scripts/run_replica.py
```

### ScanNet
Run the following script to automate the evaluation process on the selected 8 sequences of the ScanNet dataset. It will evaluate the tracking error and rendering quality.
```bash
python scripts/run_scannet.py
```

## Run your own data
<p align="center">
  <img src="./media/owndata.gif" width="70%" />
</p>

HI-SLAM2 supports casual video recordings from smartphone or camera (demo above with iPhone 15). To use your own video data, we provide a preprocessing script that extracts individual frames from your video and runs COLMAP to automatically estimate camera intrinsics. Run the preprocessing with:
```bash
python scripts/preprocess_owndata.py PATH_TO_YOUR_VIDEO PATH_TO_OUTPUT_DIR
```
once the intrinsics are obtained, you can run HI-SLAM2 by using the following command:
```bash
python demo.py \
--imagedir PATH_TO_OUTPUT_DIR/images \
--calib PATH_TO_OUTPUT_DIR/calib.txt \
--config config/owndata_config.yaml \
--output outputs/owndata \
--undistort --droidvis --gsvis
```
there are some other command line arguments you can use:
- `--undistort` undistort the image if distortion parameters are provided in the calib file
- `--droidvis` visualize the point cloud map and the intermediate results
- `--gsvis` visualize the Gaussian map
- `--buffer` max number of keyframes to pre-allocate memory for (default: 10% of total frames). 
  Increase this if you encounter the error: `IndexError: index X is out of bounds for dimension 0 with size X`. 
- `--start` start frame index (default: from the first frame)
- `--length` number of frames to process (default: all frames)

## Semantic Reconstruction
For semantic reconstruction capabilities, please check the [Semantic](https://github.com/Willyzw/HI-SLAM2/tree/semantic) branch. This branch extends HI-SLAM2 with additional features for semantic understanding and reconstruction.

## Acknowledgement
We build this project based on [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [MonoGS](https://github.com/muskie82/MonoGS), [RaDe-GS](https://github.com/BaowenZ/RaDe-GS) and [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). The reconstruction evaluation is based on [evaluate_3d_reconstruction_lib](https://github.com/eriksandstroem/evaluate_3d_reconstruction_lib). We thank the authors for their great works and hope this open-source code can be useful for your research. 

## Citation
Our paper is available on [arXiv](https://arxiv.org/abs/2411.17982). If you find this code useful in your research, please cite our paper.
```
@article{zhang2024hi2,
  title={HI-SLAM2: Geometry-Aware Gaussian SLAM for Fast Monocular Scene Reconstruction},
  author={Zhang, Wei and Cheng, Qing and Skuddis, David and Zeller, Niclas and Cremers, Daniel and Haala, Norbert},
  journal={arXiv preprint arXiv:2411.17982},
  year={2024}
}
```
