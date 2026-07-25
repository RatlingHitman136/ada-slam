#!/bin/bash
#SBATCH --job-name="adaslam"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/<user>/slurm/logs/slurm-%j.out

export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
pwd; hostname; date
nvidia-smi

uv run python scripts/run_pipeline.py
