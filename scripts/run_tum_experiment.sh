#!/usr/bin/env bash
#
# End-to-end depth-prior experiment on TUM RGB-D. Run this and nothing else:
#
#   ./scripts/run_tum_experiment.sh
#
# Four stages, each skipped if its output already exists, so an interrupted run resumes:
#
#   1 preprocess  raw TUM -> Replica-shaped colors/ depths/ traj_tum.txt calib.txt   ~2 min, CPU
#   2 slam        HI-SLAM2 on the first FRACTION%, export its own depth              ~10 min
#   3 lora        LoRA-adapt VGGT on that depth                                      ~40 min
#   4 ab          two-arm full-sequence A/B + comparison table                       ~1 h 20
#
# Stage 4's two arms differ only in where the depth prior comes from (normals are Omnidata in
# both): stock Omnidata, and LoRA-adapted VGGT.
#
# Sequential by design: the VGGT arm holds VGGT (~2.5 GB) + Omnidata normals (~1.9 GB) on top of
# SLAM and the Gaussian map, and this GPU is shared with other users.

# ==============================================================================
#  PARAMETERS
# ==============================================================================

SEQ=${SEQ:-rgbd_dataset_freiburg1_room}
SRC=${SRC:-/storage/group/dataset_mirrors/01_incoming/TUM_RGBD_Dataset/$SEQ}
DST=${DST:-data/TUM/$SEQ}                  # data/ is symlinked to /storage/user/treh/data
CAMERA=${CAMERA:-auto}                     # auto | fr1 | fr2 | fr3

FRACTION=${FRACTION:-40}    # percent of the sequence the adapter trains on. SPLIT_AT is derived
                            # from the real frame count after stage 1, never hardcoded.
BUFFER=${BUFFER:-500}       # keyframe buffer. demo.py's default (N/10+150 = 286 here) is sized
                            # for Replica's sparse keyframing; handheld TUM keyframes far denser.
VGGT_HW=${VGGT_HW:-378,518} # VGGT input size, dims divisible by 14. 378x518 matches the 544x400
                            # tracking stream's aspect; the 294x518 default would squash it ~30%.
DEPTH_SOURCE=${DEPTH_SOURCE:-rendered}  # supervision target: rendered (Gaussian map's expected
                            # depth, ~2.4x closer to GT and consistent with traj_full.txt's
                            # post-refinement poses) | slam (1/disps_up, pre-refinement)

SLAM_OUT=${SLAM_OUT:-outputs/tum}
AB_OUT=${AB_OUT:-outputs/tum_ab_p${FRACTION}}

STAGES=${STAGES:-all}       # all, or a space-separated subset of: preprocess slam lora ab
MIN_FREE_VRAM_MB=${MIN_FREE_VRAM_MB:-10000}

VENV=${VENV:-/usr/stud/treh/envs/adaslam}
CUDA_MODULE=${CUDA_MODULE:-cuda/13.0.1}
TORCH_ARCH=${TORCH_ARCH:-8.9+PTX}

# ==============================================================================

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

# ---- environment (same bootstrap as run_slam_depth_batch.sh) ----
# shellcheck disable=SC1091
source "$VENV/bin/activate" || { echo "cannot activate $VENV"; exit 1; }
if ! command -v nvcc >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash 2>/dev/null
    module load "$CUDA_MODULE" 2>/dev/null || { echo "cannot load $CUDA_MODULE"; exit 1; }
fi
export TORCH_CUDA_ARCH_LIST="$TORCH_ARCH"

want () { [ "$STAGES" = all ] || [[ " $STAGES " == *" $1 "* ]]; }

# The GPU is shared and the stages are hours apart, so re-check before each one rather than
# only at startup - another user's job may have landed in the meantime.
gpu_gate () {
    read -r USED TOTAL < <(nvidia-smi --query-gpu=memory.used,memory.total \
                           --format=csv,noheader,nounits | head -1 | tr -d ',')
    local free=$((TOTAL - USED))
    if [ "$free" -lt "$MIN_FREE_VRAM_MB" ]; then
        echo "only ${free} MiB VRAM free (need ${MIN_FREE_VRAM_MB}); another job is running."
        echo "lower MIN_FREE_VRAM_MB to override."
        exit 1
    fi
    echo "GPU free  : ${free} / ${TOTAL} MiB"
}

stage () {
    echo
    echo "=============================================================="
    echo "=== stage $1"
    echo "=============================================================="
}

T_ALL=$SECONDS

# ------------------------------------------------------------------ 1. preprocess
if want preprocess && [ ! -f "$DST/calib.txt" ]; then
    stage "1/4 preprocess  $SRC -> $DST"
    [ -d "$SRC" ] || { echo "missing source sequence: $SRC"; exit 1; }
    T0=$SECONDS
    python scripts/preprocess_tum.py --src "$SRC" --dst "$DST" --camera "$CAMERA" || exit 1
    echo "=== preprocess done in $((SECONDS - T0))s"
else
    echo "stage 1/4 preprocess: $DST/calib.txt exists, skipping"
fi

for f in "$DST/colors" "$DST/depths" "$DST/traj_tum.txt" "$DST/calib.txt"; do
    [ -e "$f" ] || { echo "stage 1 did not produce $f"; exit 1; }
done

# -L is required: data/ is a symlink into /storage, and plain find will not descend it
N=$(find -L "$DST/colors" -maxdepth 1 -type f | wc -l)
ND=$(find -L "$DST/depths" -maxdepth 1 -type f | wc -l)
NT=$(wc -l < "$DST/traj_tum.txt")
[ "$N" = "$ND" ] && [ "$N" = "$NT" ] \
    || { echo "colors/depths/traj_tum.txt disagree ($N/$ND/$NT); every consumer indexes GT depth"
         echo "by RGB frame number, so they must be 1:1. Re-run stage 1."; exit 1; }

SPLIT_AT=$((N * FRACTION / 100))
# the batch drivers address a sequence as <DATA_ROOT>/<SCENE>, so the scene name has to come from
# DST rather than SEQ - they diverge as soon as anyone overrides DST
DATA_ROOT=$(dirname "$DST")
SCENE_NAME=$(basename "$DST")
SCENE_DIR=$SLAM_OUT/${SCENE_NAME}_p${FRACTION}

echo
echo "sequence  : $SCENE_NAME  ($N frames)"
echo "adapter   : trains on frames 0..$((SPLIT_AT - 1)) (${FRACTION}%), evaluated on 0..$((N - 1))"
echo "target    : depth_${DEPTH_SOURCE}/"
echo "split at  : $SPLIT_AT"

# ------------------------------------------------------------------ 2. SLAM + export
if want slam; then
    stage "2/4 slam + export  -> $SCENE_DIR"
    gpu_gate
    T0=$SECONDS
    SCENES="$SCENE_NAME" FRACTION="$FRACTION" DATA_ROOT="$DATA_ROOT" OUT_ROOT="$SLAM_OUT" \
    CONFIG=config/tum_config.yaml CALIB="$DST/calib.txt" BUFFER="$BUFFER" \
    DEPTH_SOURCE="$DEPTH_SOURCE" MIN_FREE_VRAM_MB="$MIN_FREE_VRAM_MB" \
        ./scripts/run_slam_depth_batch.sh || exit 1
    [ -f "$SCENE_DIR/poses_slam.txt" ] || { echo "stage 2 produced no poses_slam.txt"; exit 1; }
    echo "=== slam + export done in $((SECONDS - T0))s"
fi

# ------------------------------------------------------------------ 3. LoRA adaptation
ADAPTER=$SCENE_DIR/lora-vggt/adapter.safetensors
if want lora && [ ! -f "$ADAPTER" ]; then
    stage "3/4 lora  -> $ADAPTER"
    gpu_gate
    T0=$SECONDS
    # DEPTH_SOURCE is passed to both stages, so the adapter cannot be trained on a target
    # different from the one that was exported
    python scripts/lora_adapt_vggt.py \
        --scene "$SCENE_DIR" --images "$DST/colors" --vggt_hw "$VGGT_HW" \
        --depth_source "$DEPTH_SOURCE" \
        2>&1 | tee "$SCENE_DIR/lora.log"
    [ -f "$ADAPTER" ] || { echo "stage 3 FAILED - last lines of $SCENE_DIR/lora.log:"
                           tr '\r' '\n' < "$SCENE_DIR/lora.log" | tail -8 | sed 's/^/    /'
                           exit 1; }
    echo "=== lora done in $((SECONDS - T0))s"
elif want lora; then
    echo "stage 3/4 lora: $ADAPTER exists, skipping"
fi

# ------------------------------------------------------------------ 4. two-arm A/B
if want ab; then
    stage "4/4 two-arm A/B  -> $AB_OUT"
    [ -f "$ADAPTER" ] || { echo "no adapter at $ADAPTER; run stage 3 first"; exit 1; }
    gpu_gate
    T0=$SECONDS
    # GTMESH= (empty, not unset) is what tells the driver TUM has no GT mesh to score against
    SCENE="$SCENE_NAME" DATA_ROOT="$DATA_ROOT" OUT_ROOT="$AB_OUT" \
    CONFIG=config/tum_config.yaml CALIB="$DST/calib.txt" \
    GTMESH= ADAPTER="$ADAPTER" SPLIT_AT="$SPLIT_AT" BUFFER="$BUFFER" \
    VGGT_HW="$VGGT_HW" MIN_FREE_VRAM_MB="$MIN_FREE_VRAM_MB" \
        ./scripts/temp_run_ab_comparison.sh || exit 1
    echo "=== A/B done in $((SECONDS - T0))s"
fi

echo
echo "=============================================================="
echo "all stages done in $((SECONDS - T_ALL))s"
echo
echo "read first:"
echo "  $SCENE_DIR/export.txt   per-frame vs global depth L1 columns. The gap on the Omnidata"
echo "                          row is the cross-frame scale inconsistency this whole track"
echo "                          targets - if it is small here, there was no headroom to win."
echo "  the table above         'unseen' rows only; 'seen' is the adapter's training set."
