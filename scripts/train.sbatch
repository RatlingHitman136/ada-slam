#!/bin/bash
#SBATCH --job-name="ada_slam"
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
#   e.g. sbatch scripts/train.sbatch m2dgr/street01/live_m2dgr_street01_full_gauge_ctx0_warmvggt
set -e
CFG=${1:?usage: sbatch scripts/train.sbatch <run config>}
cd /home/stud/treh/ada-slam
/usr/stud/treh/envs/adaslam/bin/python scripts/online_adapt_pipeline.py -c "$CFG"
