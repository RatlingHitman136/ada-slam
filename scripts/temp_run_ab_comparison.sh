#!/usr/bin/env bash
#
# TEMPORARY driver for the full-sequence depth-prior A/B.
#
# Runs HI-SLAM2 twice on the same sequence - once with the stock Omnidata depth prior, once with
# the LoRA-adapted VGGT prior (normals stay Omnidata in both, so depth is the only variable) -
# then prints a side-by-side comparison split at frame SPLIT_AT.
#
#   ./scripts/temp_run_ab_comparison.sh
#
# Sequential by design: the VGGT arm holds VGGT (~2.5 GB) + Omnidata normals (~1.9 GB) on top of
# SLAM and the Gaussian map, and this GPU is shared. Roughly 30 min per arm.

# ==============================================================================
#  PARAMETERS
# ==============================================================================

SCENE=room0
OUT_ROOT=outputs/ab_depth_p100
ADAPTER=outputs/replica/room0_p100/lora-vggt/adapter.safetensors

SPLIT_AT=2000           # frames < this are what the adapter trained on; >= this are unseen
LENGTH=100000          # 100000 = whole sequence; lower it for a quick shake-out run
VOXEL=0.01             # pinned for BOTH arms - see note below

SKIP_EXISTING=1        # 1 = reuse an arm's output if ab_results.json already exists
MIN_FREE_VRAM_MB=10000

VENV=/usr/stud/treh/envs/adaslam

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

DATA=data/Replica/$SCENE
for f in "$DATA/colors" "$DATA/depths" "$DATA/traj_tum.txt" \
         "data/Replica/gt_mesh_culled/$SCENE.ply" "$ADAPTER"; do
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

OUT_OMNI=$OUT_ROOT/${SCENE}_omnidata
OUT_VGGT=$OUT_ROOT/${SCENE}_vggt
mkdir -p "$OUT_ROOT"

COMMON=(--imagedir "$DATA/colors" --gtdepthdir "$DATA/depths" --gttraj "$DATA/traj_tum.txt"
        --gtmesh "data/Replica/gt_mesh_culled/$SCENE.ply"
        --split_at "$SPLIT_AT" --length "$LENGTH" --voxel_size "$VOXEL")

echo "scene     : $SCENE"
echo "split at  : frame $SPLIT_AT (below = adapter-trained, above = unseen)"
echo "voxel     : $VOXEL (pinned for both arms)"
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

run_arm scripts/run_full_omnidata.py "$OUT_OMNI" "ARM A: Omnidata depth (baseline)" \
    || { echo "aborting: baseline arm failed, nothing to compare against"; exit 1; }

run_arm scripts/run_full_vggt.py "$OUT_VGGT" "ARM B: VGGT+LoRA depth" --adapter "$ADAPTER" \
    || { echo "aborting: VGGT arm failed"; exit 1; }

# ------------------------------------------------------------------ compare
echo
echo "=============================================================="
OMNI="$OUT_OMNI" VGGT="$OUT_VGGT" python - <<'EOF'
import json, os
a = json.load(open(os.environ['OMNI'] + '/ab_results.json'))
b = json.load(open(os.environ['VGGT'] + '/ab_results.json'))
k = a['split_at']

print(f"  full-sequence A/B, split at frame {k}")
print(f"  {'metric':<26}{'Omnidata':>12}{'VGGT+LoRA':>12}{'delta':>12}")
print('  ' + '-' * 62)

def row(name, x, y, better_low=True):
    if x is None or y is None:
        return
    d = y - x
    flag = '' if abs(d) < 1e-9 else ('  better' if (d < 0) == better_low else '  worse')
    print(f"  {name:<26}{x:>12.4f}{y:>12.4f}{d:>+12.4f}{flag}")

for s in ('all', 'seen', 'unseen'):
    row(f'ATE RMSE ({s})', a.get(f'ate_{s}'), b.get(f'ate_{s}'))
print()
for s in ('all', 'seen', 'unseen'):
    for m, low in (('psnr', False), ('ssim', False), ('depth_l1', True)):
        row(f'{m} ({s})', (a['render'].get(s) or {}).get(m), (b['render'].get(s) or {}).get(m), low)
    print()

am, bm = a.get('mesh'), b.get('mesh')
if am and bm:
    if am['voxel_size'] != bm['voxel_size']:
        print(f"  !! voxel sizes differ ({am['voxel_size']} vs {bm['voxel_size']}) "
              "- mesh numbers are NOT comparable; re-run both with the same --voxel_size")
    else:
        for key, name, low in (('mean precision', 'mesh accuracy (m)', True),
                               ('mean recall', 'mesh completion (m)', True),
                               ('recall', 'mesh comp-ratio', False),
                               ('f-score', 'mesh F-score', False)):
            row(name, am[key], bm[key], low)
else:
    print('  mesh metrics unavailable for at least one arm')

print("\n  'unseen' is the row that matters: it is the only evidence the adaptation")
print("  generalises rather than having memorised the keyframes it trained on.")
EOF
