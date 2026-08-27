#!/bin/bash
#SBATCH --job-name="ada_slam_e40_online"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --constraint="GPU_CC:12.0|GPU_CC:9.0|GPU_CC:8.9"
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

cd /home/stud/treh/ada-slam
uv run python scripts/online_adapt_pipeline.py -c live_crop976_e40_online