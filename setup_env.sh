#!/usr/bin/env bash
#
# One-shot environment setup for ada-slam (HI-SLAM2 fork) on CUDA 13 / PyTorch 2.9.
#
#   ./setup_env.sh                  # full setup
#   ./setup_env.sh --with-weights   # also download the 3.9 GB Omnidata checkpoints
#   ./setup_env.sh --force-rebuild  # recompile the CUDA extensions even if they import
#
# Safe to re-run: every step detects work that is already done and skips it.
#
# Knobs (environment variables):
#   ADASLAM_VENV            venv location            (default: <repo>/.venv)
#   TORCH_CUDA_ARCH_LIST    target GPU arch          (default: 8.9+PTX, i.e. RTX 40xx / Ada)
#   CUDA_MODULE             lmod module to load      (default: cuda/13.0.1)
#
# See new-udpate-env.md for why each piece is needed.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

VENV="${ADASLAM_VENV:-$REPO/.venv}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9+PTX}"
CUDA_MODULE="${CUDA_MODULE:-cuda/13.0.1}"

WITH_WEIGHTS=0
FORCE_REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --with-weights)  WITH_WEIGHTS=1 ;;
    --force-rebuild) FORCE_REBUILD=1 ;;
    -h|--help)       sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
ok()   { printf '    \033[32m%s\033[0m\n' "$1"; }
warn() { printf '    \033[33m%s\033[0m\n' "$1"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- 0. preflight
step "Preflight"
[ -f setup.py ] && [ -d hislam2 ] || die "run this from the ada-slam repo root"
git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository (submodules cannot be fetched)"
ok "repo: $REPO"
ok "venv: $VENV"
ok "arch: TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# ---------------------------------------------------------------------- 1. uv
step "uv"
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  ok "already installed: $(uv --version)"
else
  warn "not found - installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv install failed; add ~/.local/bin to PATH"
  ok "installed: $(uv --version)"
fi

# -------------------------------------------------------------- 2. submodules
step "Submodules"
# --force resets submodule working trees to their pinned commits. That is deliberate: it
# guarantees a clean base for the patches below, so every run converges to the same state.
# Consequence: DO NOT edit thirdparty/lietorch directly - those edits are wiped on re-run.
# Change patches/lietorch.patch instead (regenerate with `cd thirdparty/lietorch && git diff`).
git submodule update --init --recursive --force
for d in thirdparty/eigen thirdparty/lietorch thirdparty/simple-knn \
         thirdparty/diff-gaussian-rasterization/third_party/glm; do
  [ -n "$(ls -A "$d" 2>/dev/null)" ] || die "submodule still empty: $d"
done
ok "all four populated"

# ------------------------------------------------------------------ 3. patches
# lietorch is a submodule, so its CUDA-13 fixes cannot be committed to this repo -
# git records only a commit pointer. They ship as a patch and are applied here.
step "Patches (submodule fixes that git cannot carry)"
apply_patch() {
  local patch="$1" dir="$2" name="$3"
  [ -f "$REPO/$patch" ] || die "missing patch file: $REPO/$patch"
  # NOTE: git -C <dir> resolves relative paths against <dir>, so the patch must be absolute.
  local abs="$REPO/$patch"
  if git -C "$dir" apply --reverse --check "$abs" 2>/dev/null; then
    ok "$name: already applied"
  elif git -C "$dir" apply --check "$abs" 2>/dev/null; then
    git -C "$dir" apply "$abs"
    ok "$name: applied"
  else
    die "$name: patch neither applies cleanly nor is already applied.
       The submodule pointer probably moved. Regenerate with:
         cd $dir && git diff > $abs"
  fi
}
apply_patch "patches/lietorch.patch" "thirdparty/lietorch" "lietorch"

# ---------------------------------------------------------- 4. CUDA toolchain
step "CUDA toolchain (nvcc, needed to compile the extensions)"
if ! command -v nvcc >/dev/null 2>&1; then
  if [ -f /usr/share/lmod/lmod/init/bash ]; then
    warn "nvcc not on PATH - trying: module load $CUDA_MODULE"
    set +u; source /usr/share/lmod/lmod/init/bash; module load "$CUDA_MODULE" 2>/dev/null || true; set -u
  fi
fi
command -v nvcc >/dev/null 2>&1 || die "nvcc not found.
       On a module-based cluster:  module load $CUDA_MODULE
       Otherwise install a CUDA toolkit matching your PyTorch build."
ok "$(nvcc --version | tail -2 | head -1)"
ok "CUDA_HOME=${CUDA_HOME:-<unset>}"

# --------------------------------------------------------------- 5. venv+deps
step "Virtualenv and dependencies"
if [ -e "$VENV/bin/python" ]; then
  ok "reusing existing venv ($("$VENV/bin/python" --version 2>&1))"
else
  uv venv "$VENV" --python 3.12
  ok "created"
fi
# --index-strategy unsafe-best-match is REQUIRED: uv prioritises --extra-index-url over
# --index-url, so PyPI otherwise shadows the +cu130 torch builds. Both indexes are official.
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match -r requirements.txt
ok "requirements.txt installed"

# ----------------------------------------------------------- 6. CUDA extensions
step "CUDA extensions (droid_backends, lietorch, simple_knn, diff_gaussian_rasterization)"
ext_ok() {
  "$VENV/bin/python" - <<'PY' >/dev/null 2>&1
import torch  # must precede the extensions (libc10.so)
import droid_backends, lietorch, simple_knn, diff_gaussian_rasterization
PY
}
if [ "$FORCE_REBUILD" -eq 0 ] && ext_ok; then
  ok "already built and importable (use --force-rebuild to recompile)"
else
  # ninja caches the full compile command line, so a stale temp dir would silently
  # reuse old gencode flags after setup.py changes.
  rm -rf build/temp.*
  uv pip install --python "$VENV/bin/python" --no-build-isolation ./thirdparty/simple-knn
  uv pip install --python "$VENV/bin/python" --no-build-isolation ./thirdparty/diff-gaussian-rasterization
  # `python setup.py install` (not `pip install .`): setup.py calls setup() twice and pip
  # would build only the first extension.
  "$VENV/bin/python" setup.py install
  ok "built"
fi

# ---------------------------------------------------------------- 7. weights
step "Pretrained weights"
if [ "$WITH_WEIGHTS" -eq 1 ]; then
  mkdir -p pretrained_models
  for f in omnidata_dpt_depth_v2.ckpt omnidata_dpt_normal_v2.ckpt; do
    if [ -s "pretrained_models/$f" ]; then
      ok "$f present"
    else
      wget -c -q --show-progress "https://zenodo.org/records/10447888/files/$f" -P pretrained_models
      ok "$f downloaded"
    fi
  done
else
  missing=0
  for f in omnidata_dpt_depth_v2.ckpt omnidata_dpt_normal_v2.ckpt; do
    [ -s "pretrained_models/$f" ] || { warn "missing: pretrained_models/$f"; missing=1; }
  done
  [ "$missing" -eq 0 ] && ok "Omnidata checkpoints present" \
                       || warn "re-run with --with-weights to fetch them (3.9 GB)"
fi
[ -s pretrained_models/droid.pth ] || warn "missing pretrained_models/droid.pth (tracked in git - try: git checkout pretrained_models/droid.pth)"

# ----------------------------------------------------------------- 8. verify
step "Verification"
"$VENV/bin/python" - <<'PY'
import sys, torch
import droid_backends, lietorch, simple_knn, diff_gaussian_rasterization, torch_scatter
print(f"    python  {sys.version.split()[0]}  ({sys.prefix})")
print(f"    torch   {torch.__version__}   cuda {torch.version.cuda}")
if not torch.cuda.is_available():
    print("    WARNING: torch.cuda.is_available() is False - no GPU visible right now.")
    print("             The build is fine; extensions were compiled for the target arch.")
else:
    name = torch.cuda.get_device_name(0)
    cap  = "".join(map(str, torch.cuda.get_device_capability(0)))
    print(f"    gpu     {name} (sm_{cap})")
    from lietorch import SE3
    SE3.Random(2, device="cuda").log()
    from simple_knn._C import distCUDA2
    distCUDA2(torch.rand(256, 3, device="cuda"))
    torch_scatter.scatter_mean(torch.ones(4, device="cuda"),
                               torch.tensor([0, 0, 1, 1], device="cuda"))
    print("    kernels lietorch / simple_knn / torch_scatter all executed OK")
PY

cat <<EOF

$(printf '\033[1;32mSetup complete.\033[0m')

Run the demo (no 'source', no 'module load' needed at runtime):

    uv run python demo.py --imagedir data/Replica/room0/colors \\
        --calib calib/replica.txt --config config/replica_config.yaml \\
        --output outputs/room0

Data is not fetched by this script - it is site-specific. For Replica:

    mkdir -p /path/to/big/storage/data && ln -s /path/to/big/storage/data data
    bash scripts/download_replica.sh && python scripts/preprocess_replica.py   # 12 GB

EOF
