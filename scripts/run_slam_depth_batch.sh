#!/usr/bin/env bash
#
# Batch-generate SLAM depth training data: run HI-SLAM2 on a fraction of each scene with
# --dump_slam_depth, then export per-keyframe depth/mask/image + the accuracy table.
#
#   ./scripts/run_slam_depth_batch.sh
#
# Results land in $OUT_ROOT/<scene>_p<FRACTION>/ with depth_slam/ mask_slam/ image/
# poses_slam.txt slam_depth.npz, plus export.txt holding that scene's accuracy table.

# ==============================================================================
#  PARAMETERS
#
#  Every value below can be overridden from the environment, so the same script serves other
#  datasets without a fork. With nothing set it behaves exactly as it always has (Replica).
#    SCENES=rgbd_dataset_freiburg1_room DATA_ROOT=data/TUM ... ./scripts/run_slam_depth_batch.sh
# ==============================================================================

# Scenes to process, space-separated. Replica has:
#   room0 room1 room2 office0 office1 office2 office3 office4
read -ra SCENES <<< "${SCENES:-room0 room1 room2 office0 office1 office2 office3 office4}"

FRACTION=${FRACTION:-100}   # percent of each sequence to process (frames = N * FRACTION / 100)
START=${START:-0}      # first frame index; use with FRACTION to take a middle/late slice

DATA_ROOT=${DATA_ROOT:-data/Replica}       # expects <DATA_ROOT>/<scene>/colors and optionally /depths
OUT_ROOT=${OUT_ROOT:-outputs/replica}      # symlinked to /storage/user/treh/adaslam_outputs
CONFIG=${CONFIG:-config/replica_config.yaml}
CALIB=${CALIB:-calib/replica.txt}
BUFFER=${BUFFER:-}     # keyframe buffer; empty = demo.py's default of N/10 + 150. Real handheld
                       # data keyframes far more densely than Replica's 89-per-2000 - set it.

DEPTH_SOURCE=${DEPTH_SOURCE:-rendered}  # training target: rendered (Gaussian) | slam (1/disps_up)

FILTER_THRESH=${FILTER_THRESH:-0.005}  # depth_filter disparity agreement threshold (larger = looser mask)
MIN_COUNT=${MIN_COUNT:-2}   # min agreeing neighbours out of 6 (lower = looser mask, more pixels)

SKIP_EXISTING=${SKIP_EXISTING:-1}      # 1 = skip scenes that already have slam_depth.npz
MIN_FREE_VRAM_MB=${MIN_FREE_VRAM_MB:-8000}  # abort if the shared GPU has less than this free

VENV=${VENV:-/usr/stud/treh/envs/adaslam}
CUDA_MODULE=${CUDA_MODULE:-cuda/13.0.1}
TORCH_ARCH=${TORCH_ARCH:-8.9+PTX}

# ==============================================================================

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

# ---- environment ----
# shellcheck disable=SC1091
source "$VENV/bin/activate" || { echo "cannot activate $VENV"; exit 1; }
if ! command -v nvcc >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash 2>/dev/null
    module load "$CUDA_MODULE" 2>/dev/null || { echo "cannot load $CUDA_MODULE"; exit 1; }
fi
export TORCH_CUDA_ARCH_LIST="$TORCH_ARCH"

# ---- shared GPU: do not stomp on another user's job ----
read -r USED TOTAL < <(nvidia-smi --query-gpu=memory.used,memory.total \
                       --format=csv,noheader,nounits | head -1 | tr -d ',')
FREE=$((TOTAL - USED))
if [ "$FREE" -lt "$MIN_FREE_VRAM_MB" ]; then
    echo "only ${FREE} MiB VRAM free (need ${MIN_FREE_VRAM_MB}); another job is probably running."
    echo "lower MIN_FREE_VRAM_MB to override."
    exit 1
fi

echo "scenes    : ${SCENES[*]}"
echo "data      : ${DATA_ROOT}  (config ${CONFIG}, calib ${CALIB})"
echo "fraction  : ${FRACTION}% from frame ${START}${BUFFER:+, buffer ${BUFFER}}"
echo "output    : ${OUT_ROOT}/<scene>_p${FRACTION}"
echo "target    : depth_${DEPTH_SOURCE}/  (mask filter_thresh=${FILTER_THRESH} min_count=${MIN_COUNT})"
echo "GPU free  : ${FREE} / ${TOTAL} MiB"
echo

DONE=(); FAILED=()

for SCENE in "${SCENES[@]}"; do
    SEQ="$DATA_ROOT/$SCENE"
    OUT="$OUT_ROOT/${SCENE}_p${FRACTION}"

    if [ ! -d "$SEQ/colors" ]; then
        echo "[$SCENE] no $SEQ/colors - skipping"; FAILED+=("$SCENE(missing)"); continue
    fi

    # -L is required: data/ is a symlink into /storage, and plain find will not descend it
    N=$(find -L "$SEQ/colors" -maxdepth 1 -type f | wc -l)
    LEN=$((N * FRACTION / 100))
    if [ "$LEN" -lt 20 ]; then
        echo "[$SCENE] found $N frames -> LEN=$LEN, too few to track. Check $SEQ/colors."
        FAILED+=("$SCENE(frames)"); continue
    fi
    mkdir -p "$OUT"

    # GT depths are optional - absent for own-data captures, which just disables the table.
    # They go to the EXPORT only, never to demo.py: eval_utils.py:50-52 zeroes the rendered depth
    # wherever GT is invalid, and on real sensors (TUM: 24% holes, on exactly the hard surfaces)
    # that would both shrink the training set and tie its mask to where the Kinect happened to
    # work. The export table is unaffected - it masks on (gt > 0) & mask anyway. All that is lost
    # is final_result.json's mean_l1, which is meaningless on a monocular run regardless.
    GT=()
    [ -d "$SEQ/depths" ] && GT=(--gtdepthdir "$SEQ/depths")

    echo "=============================================================="
    echo "[$SCENE] $N frames -> processing $LEN (from $START)"

    if [ "$SKIP_EXISTING" = 1 ] && [ -f "$OUT/slam_depth.npz" ]; then
        echo "[$SCENE] slam_depth.npz exists - skipping SLAM run"
    else
        T0=$SECONDS
        BUF=(); [ -n "$BUFFER" ] && BUF=(--buffer "$BUFFER")
        python demo.py \
            --imagedir "$SEQ/colors" "${BUF[@]}" \
            --config "$CONFIG" --calib "$CALIB" --output "$OUT" \
            --start "$START" --length "$LEN" \
            --dump_slam_depth > "$OUT/log.txt" 2>&1
        if [ $? -ne 0 ] || [ ! -f "$OUT/slam_depth.npz" ]; then
            echo "[$SCENE] FAILED - last lines of $OUT/log.txt:"
            tr '\r' '\n' < "$OUT/log.txt" | tail -5 | sed 's/^/    /'
            FAILED+=("$SCENE(run)"); continue
        fi
        echo "[$SCENE] SLAM run done in $((SECONDS - T0))s"
    fi

    python scripts/export_slam_depth.py \
        --result "$OUT" "${GT[@]}" --depth_source "$DEPTH_SOURCE" \
        --filter_thresh "$FILTER_THRESH" --min_count "$MIN_COUNT" \
        2>&1 | grep -v Warning | tee "$OUT/export.txt"
    if [ ! -f "$OUT/poses_slam.txt" ]; then
        echo "[$SCENE] export FAILED"; FAILED+=("$SCENE(export)"); continue
    fi

    DONE+=("$SCENE")
    echo
done

# ---- summary ----
echo "=============================================================="
echo "SUMMARY"
printf '  %-12s %7s %7s %9s   %s\n' scene kfs rendered size "depth L1 (global scale)"
for SCENE in "${DONE[@]}"; do
    OUT="$OUT_ROOT/${SCENE}_p${FRACTION}"
    KFS=$(find -L "$OUT/depth_$DEPTH_SOURCE" -name '*.npy' 2>/dev/null | wc -l)
    REND=$(find -L "$OUT/renders/depth_after_opt" -type f 2>/dev/null | wc -l)
    SIZE=$(du -sh "$OUT" | cut -f1)
    L1=$(awk '/SLAM depth/{s=$NF} /Gaussian-rendered/{r=$NF} /Omnidata/{o=$NF}
              END{if(s!="")printf "slam %s | rendered %s | omnidata %s", s, r, o}' "$OUT/export.txt")
    printf '  %-12s %7s %7s %9s   %s\n' "$SCENE" "$KFS" "$REND" "$SIZE" "${L1:-no GT}"
done
[ ${#FAILED[@]} -gt 0 ] && echo "  failed: ${FAILED[*]}"
echo
echo "done: ${#DONE[@]}/${#SCENES[@]} scenes"
