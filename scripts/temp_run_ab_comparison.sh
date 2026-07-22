#!/usr/bin/env bash
#
# TEMPORARY driver for the full-sequence depth-prior A/B.
#
# Runs HI-SLAM2 twice over the same sequence - once with the stock Omnidata depth prior, once with
# the LoRA-adapted VGGT prior (normals stay Omnidata in both, so depth is the only variable) -
# then prints a side-by-side comparison split at frame SPLIT_AT.
#
#   ./scripts/temp_run_ab_comparison.sh
#
# Every parameter below can be overridden from the environment, so the same script serves other
# datasets without a fork. With nothing set it behaves exactly as it always has (Replica room0).
#
# Sequential by design: the VGGT arms hold VGGT (~2.5 GB) + Omnidata normals (~1.9 GB) on top of
# SLAM and the Gaussian map, and this GPU is shared. Roughly 30 min per arm.

# ==============================================================================
#  PARAMETERS
# ==============================================================================

SCENE=${SCENE:-room0}
DATA_ROOT=${DATA_ROOT:-data/Replica}     # expects <DATA_ROOT>/<SCENE>/{colors,depths,traj_tum.txt}
OUT_ROOT=${OUT_ROOT:-outputs/ab_depth_p100}
CONFIG=${CONFIG:-config/replica_config.yaml}
CALIB=${CALIB:-calib/replica.txt}
ADAPTER=${ADAPTER:-outputs/replica/room0_p100/lora-vggt/adapter.safetensors}
GTMESH=${GTMESH-data/Replica/gt_mesh_culled/$SCENE.ply}   # empty = no GT mesh (TUM), skip meshing
VGGT_HW=${VGGT_HW:-}        # VGGT input size; only a fallback, since the adapter's config.json
                            # records the size it was trained at and that always wins

SPLIT_AT=${SPLIT_AT:-2000}  # frames < this are what the adapter trained on; >= this are unseen
LENGTH=${LENGTH:-100000}    # 100000 = whole sequence; lower it for a quick shake-out run
VOXEL=${VOXEL:-0.01}        # pinned for ALL arms - see note below
BUFFER=${BUFFER:-}          # keyframe buffer; empty = demo.py's default of N/10 + 150

SKIP_EXISTING=${SKIP_EXISTING:-1}   # 1 = reuse an arm's output if ab_results.json already exists
MIN_FREE_VRAM_MB=${MIN_FREE_VRAM_MB:-10000}

VENV=${VENV:-/usr/stud/treh/envs/adaslam}

# VOXEL is deliberately 0.01, not tsdf_integrate's 0.006 default: marching-cubes allocation fails
# at 0.006 when the shared GPU is busy. The harness has a fallback ladder, but if it triggers on
# one arm and not the other the mesh numbers stop being comparable, so both are pinned here.
# Consequence: mesh figures will not line up with published HI-SLAM2 numbers (which use 0.006).
# The A/B between the two arms is unaffected.

# ==============================================================================

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

# shellcheck disable=SC1091
source "$VENV/bin/activate" || { echo "cannot activate $VENV"; exit 1; }

DATA=$DATA_ROOT/$SCENE
for f in "$DATA/colors" "$DATA/depths" "$DATA/traj_tum.txt" ${GTMESH:+"$GTMESH"} "$ADAPTER"; do
    [ -e "$f" ] || { echo "missing input: $f"; exit 1; }
done

read -r USED TOTAL < <(nvidia-smi --query-gpu=memory.used,memory.total \
                       --format=csv,noheader,nounits | head -1 | tr -d ',')
FREE=$((TOTAL - USED))
if [ "$FREE" -lt "$MIN_FREE_VRAM_MB" ]; then
    echo "only ${FREE} MiB VRAM free (need ${MIN_FREE_VRAM_MB}); the box is shared - another job is running."
    echo "lower MIN_FREE_VRAM_MB to override."
    exit 1
fi

mkdir -p "$OUT_ROOT"

COMMON=(--imagedir "$DATA/colors" --gtdepthdir "$DATA/depths" --gttraj "$DATA/traj_tum.txt"
        --config "$CONFIG" --calib "$CALIB"
        --split_at "$SPLIT_AT" --length "$LENGTH" --voxel_size "$VOXEL")
if [ -n "$GTMESH" ]; then
    COMMON+=(--gtmesh "$GTMESH")
else
    COMMON+=(--skip_mesh)          # TUM RGB-D ships no GT mesh
fi
[ -n "$BUFFER" ] && COMMON+=(--buffer "$BUFFER")

# Arms, in report order. The first is the baseline every delta is measured against. Only the depth
# prior differs; normals are Omnidata in both.
HW=${VGGT_HW:+--vggt_hw $VGGT_HW}
ARM_LABELS=(Omnidata "VGGT+LoRA")
ARM_DIRS=("$OUT_ROOT/${SCENE}_omnidata" "$OUT_ROOT/${SCENE}_vggt")
ARM_SCRIPTS=(scripts/run_full_omnidata.py scripts/run_full_vggt.py)
ARM_EXTRA=('' "--adapter $ADAPTER $HW")

echo "scene     : $SCENE  ($DATA)"
echo "config    : $CONFIG  calib $CALIB"
echo "arms      : ${ARM_LABELS[*]}${VGGT_HW:+  (VGGT at $VGGT_HW)}"
echo "split at  : frame $SPLIT_AT (below = adapter-trained, above = unseen)"
echo "mesh      : ${GTMESH:-none (skipped)}${GTMESH:+ at voxel $VOXEL, pinned for all arms}"
echo "GPU free  : ${FREE} / ${TOTAL} MiB"
echo

run_arm () {                       # $1 = script, $2 = output dir, $3 = label, $4.. = extra args
    local script=$1 out=$2 label=$3; shift 3
    if [ "$SKIP_EXISTING" = 1 ] && [ -f "$out/ab_results.json" ]; then
        echo "=== $label: ab_results.json exists, skipping ==="
        return 0
    fi
    echo "=============================================================="
    echo "=== $label -> $out"
    local t0=$SECONDS
    python "$script" --output "$out" "${COMMON[@]}" "$@" 2>&1 | tee "$out.log"
    if [ ! -f "$out/ab_results.json" ]; then
        echo "$label FAILED - last lines of $out.log:"
        tr '\r' '\n' < "$out.log" | grep -v '^\s*$' | tail -8 | sed 's/^/    /'
        return 1
    fi
    echo "=== $label done in $((SECONDS - t0))s"
    return 0
}

for i in "${!ARM_LABELS[@]}"; do
    # shellcheck disable=SC2086  -- ARM_EXTRA deliberately holds pre-split flags
    if ! run_arm "${ARM_SCRIPTS[$i]}" "${ARM_DIRS[$i]}" \
                 "ARM $i: ${ARM_LABELS[$i]} depth" ${ARM_EXTRA[$i]}; then
        [ "$i" = 0 ] && echo "aborting: baseline arm failed, nothing to compare against"
        echo "aborting after ${ARM_LABELS[$i]}"
        exit 1
    fi
done

# ------------------------------------------------------------------ compare
echo
echo "=============================================================="
ARM_LABELS_S=$(IFS=$'\n'; echo "${ARM_LABELS[*]}")
ARM_DIRS_S=$(IFS=$'\n'; echo "${ARM_DIRS[*]}")
ARM_LABELS="$ARM_LABELS_S" ARM_DIRS="$ARM_DIRS_S" python - <<'EOF'
import json, os

labels = os.environ['ARM_LABELS'].split('\n')
dirs = os.environ['ARM_DIRS'].split('\n')
res = [json.load(open(d + '/ab_results.json')) for d in dirs]
base = res[0]
k = base['split_at']

# an arm run at a different split or over a different frame count is not comparable, however
# tempting the numbers look side by side
for lbl, r in zip(labels[1:], res[1:]):
    if r['split_at'] != k:
        raise SystemExit(f"  !! {lbl} used split_at={r['split_at']}, baseline used {k} - "
                         "the arms are not comparable; delete its output and re-run")
    n0, n1 = (base['render']['all'] or {}).get('n'), (r['render']['all'] or {}).get('n')
    if n0 != n1:
        print(f"  !! {lbl} evaluated {n1} frames, baseline {n0} - arms are not comparable")

print(f"  full-sequence comparison, split at frame {k}")
print(f"  {'metric':<26}{labels[0]:>12}" + ''.join(f'{l:>12}{"delta":>11}' for l in labels[1:]))
print('  ' + '-' * (26 + 12 + 23 * (len(labels) - 1)))


def row(name, vals, better_low=True):
    """One metric across every arm: baseline absolute, then absolute + delta vs baseline."""
    if vals[0] is None:
        return
    line = f"  {name:<26}{vals[0]:>12.4f}"
    for v in vals[1:]:
        if v is None:
            line += f"{'n/a':>12}{'':>11}"
            continue
        d = v - vals[0]
        mark = ' ' if abs(d) < 1e-9 else ('+' if (d < 0) == better_low else '-')
        line += f"{v:>12.4f}{d:>+9.4f} {mark}"
    print(line)


for s in ('all', 'seen', 'unseen'):
    row(f'ATE RMSE ({s})', [r.get(f'ate_{s}') for r in res])
print()
for s in ('all', 'seen', 'unseen'):
    for m, low in (('psnr', False), ('ssim', False), ('depth_l1', True)):
        row(f'{m} ({s})', [(r['render'].get(s) or {}).get(m) for r in res], low)
    print()

meshes = [r.get('mesh') for r in res]
if all(meshes):
    voxels = {m['voxel_size'] for m in meshes}
    if len(voxels) > 1:
        print(f"  !! voxel sizes differ ({sorted(voxels)}) - mesh numbers are NOT comparable; "
              "re-run with the same --voxel_size")
    else:
        for key, name, low in (('mean precision', 'mesh accuracy (m)', True),
                               ('mean recall', 'mesh completion (m)', True),
                               ('recall', 'mesh comp-ratio', False),
                               ('f-score', 'mesh F-score', False)):
            row(name, [m[key] for m in meshes], low)
elif any(meshes):
    print('  mesh metrics unavailable for at least one arm')

print("\n  '+' better than baseline, '-' worse.")
print("  'unseen' is the row that matters: it is the only evidence the adaptation")
print("  generalises rather than having memorised the keyframes it trained on.")
EOF
