#!/bin/bash
#SBATCH --job-name="ada_slam_baprior"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --constraint="GPU_CC:12.0|GPU_CC:9.0|GPU_CC:8.9|GPU_CC:8.6"
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# Usage: sbatch scripts/train.sbatch <run config, relative to run_configs/, no .yaml>
# Needs a droid_backends build with ba_prior (droid.cpp): the guard below refuses to start otherwise. The one used here
# is built in place in the repo root, which the pipeline puts first on sys.path (the venv's copy predates ba_prior).
# street_01, depth prior inside BA (aligned, alpha 0.05) - three independent jobs, one outputs tree:
#   sbatch scripts/train.sbatch m2dgr/street01/live_m2dgr_street01_full_omni_baprior_aligned_a0p05                  # Omnidata
#   sbatch scripts/train.sbatch m2dgr/street01/live_m2dgr_street01_full_base_baprior_aligned_a0p05                  # frozen VGGT
#   sbatch scripts/train.sbatch m2dgr/street01/live_m2dgr_street01_full_gauge_ctx0_warmvggt_baprior_aligned_a0p05   # adapting VGGT
set -e
CFG=${1:?usage: sbatch scripts/train.sbatch <run config>}
cd /home/stud/treh/ada-slam
/usr/stud/treh/envs/adaslam/bin/python -c "import torch, droid_backends as d; assert hasattr(d, 'ba_prior'), d.__file__; print('droid_backends', d.__file__)"
/usr/stud/treh/envs/adaslam/bin/python scripts/online_adapt_pipeline.py -c "$CFG"
