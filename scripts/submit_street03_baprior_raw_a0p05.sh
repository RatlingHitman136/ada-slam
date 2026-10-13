#!/bin/bash
# TEST (street03): depth prior inside BA (ba_prior), RAW mode, alpha 0.05 - the mode that won on street_01 (frozen VGGT 37.65 -> 5.59 m, adapting
# 35.3 -> 9.0 m): BA pulls towards disps_prior in its own units, not prior x dscales.
# Submits the five street03 arms (Omnidata, frozen VGGT, adapting VGGT seeds 0/2/3) as independent SLURM jobs via
# scripts/train.sbatch; job names say what each one tests, so squeue reads e.g. s03_bap_raw_a0p05_base.
# Outputs: outputs/test/end2end/m2dgr_street03_baprior_raw_a0p05/<arm>/
#
#   bash scripts/submit_street03_baprior_raw_a0p05.sh             # all five on the default 48 GB GPUs, 96 GB host RAM
#   BIG="base" bash scripts/submit_street03_baprior_raw_a0p05.sh  # these arms on a 96 GB GPU instead
#   MEM=128G bash scripts/submit_street03_baprior_raw_a0p05.sh    # more host RAM per job
#
# Arms named in BIG (space-separated: omni base s0 s2 s3) go to an H100 NVL / RTX PRO 6000 (>= 80 GB): use it for a
# rerun after a GPU out-of-memory failure. MEM overrides train.sbatch's --mem=64G: BUFFER 3000 preallocates ~15 GB of
# CPU buffers, and the street_01 aligned-0.1 frozen arm was killed by the 64 GB host limit at 97 %.
# Finished arms are skipped (SKIP_EXISTING), so a rerun only redoes the ones without results.json.
set -e
cd /home/stud/treh/ada-slam
TEST=baprior_raw_a0p05
BIG=${BIG:-}
MEM=${MEM:-96G}
declare -A ARM=( [omni]=omni [base]=base [s0]=gauge_ctx0_warmvggt [s2]=gauge_ctx0_warmvggt_s2 [s3]=gauge_ctx0_warmvggt_s3 )
for a in omni base s0 s2 s3; do
  cfg=m2dgr/street03/live_m2dgr_street03_full_${ARM[$a]}_${TEST}
  test -f run_configs/$cfg.yaml || { echo "missing run_configs/$cfg.yaml" >&2; exit 1; }
  extra=(--mem=$MEM)
  [[ " $BIG " == *" $a "* ]] && extra+=(--gres=gpu:1,VRAM:80G --constraint="GPU_CC:9.0|GPU_CC:12.0")
  echo -n "s03_bap_raw_a0p05_$a: "
  sbatch --job-name="s03_bap_raw_a0p05_$a" "${extra[@]}" scripts/train.sbatch "$cfg"
done
