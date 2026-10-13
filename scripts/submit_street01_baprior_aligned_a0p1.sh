#!/bin/bash
# TEST: depth prior inside BA (ba_prior), ALIGNED mode, alpha 0.1 - double the 0.05 that won on street_01 (0.01 lost).
# Submits the five street_01 arms (Omnidata, frozen VGGT, adapting VGGT seeds 0/2/3) as independent SLURM jobs via
# scripts/train.sbatch; job names say what each one tests, so squeue reads e.g. s01_bap_aligned_a0p1_base.
# Outputs: outputs/test/end2end/m2dgr_street01_baprior_aligned_a0p1/<arm>/
#
#   bash scripts/submit_street01_baprior_aligned_a0p1.sh             # all five on the default 48 GB GPUs
#   BIG="base" bash scripts/submit_street01_baprior_aligned_a0p1.sh  # these arms on a 96 GB GPU instead
#
# Arms named in BIG (space-separated: omni base s0 s2 s3) go to an H100 NVL / RTX PRO 6000 (>= 80 GB): use it for a
# rerun after an out-of-memory failure - the frozen arm OOMed at aligned 0.05 when its tracking broke down.
set -e
cd /home/stud/treh/ada-slam
TEST=baprior_aligned_a0p1
BIG=${BIG:-}
declare -A ARM=( [omni]=omni [base]=base [s0]=gauge_ctx0_warmvggt [s2]=gauge_ctx0_warmvggt_s2 [s3]=gauge_ctx0_warmvggt_s3 )
for a in omni base s0 s2 s3; do
  cfg=m2dgr/street01/live_m2dgr_street01_full_${ARM[$a]}_${TEST}
  test -f run_configs/$cfg.yaml || { echo "missing run_configs/$cfg.yaml" >&2; exit 1; }
  extra=()
  [[ " $BIG " == *" $a "* ]] && extra=(--gres=gpu:1,VRAM:80G --constraint="GPU_CC:9.0|GPU_CC:12.0")
  echo -n "s01_bap_aligned_a0p1_$a: "
  sbatch --job-name="s01_bap_aligned_a0p1_$a" "${extra[@]}" scripts/train.sbatch "$cfg"
done
