#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/environment_mjlab.yml"
ENV_NAME="${MJLAB_ENV_NAME:-mjlab}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"

if [[ -z "$CONDA_BIN" ]]; then
  echo "[ERROR] conda was not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] Missing $ENV_FILE" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/setup.py" || ! -f "$ROOT_DIR/rsl_rl/pyproject.toml" ]]; then
  echo "[ERROR] Run this script from the complete AMP_mjlab repository." >&2
  exit 1
fi

if "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[INFO] Updating existing conda environment: $ENV_NAME"
  "$CONDA_BIN" env update --name "$ENV_NAME" --file "$ENV_FILE"
else
  echo "[INFO] Creating conda environment: $ENV_NAME"
  "$CONDA_BIN" env create --name "$ENV_NAME" --file "$ENV_FILE"
fi

run_in_env() {
  "$CONDA_BIN" run --no-capture-output --name "$ENV_NAME" "$@"
}

# The repository contains the AMP-compatible rsl_rl fork. Install it after
# mjlab so it intentionally replaces mjlab's upstream rsl-rl dependency.
run_in_env python -m pip install --no-deps --editable "$ROOT_DIR/rsl_rl"
run_in_env python -m pip install --no-deps --editable "$ROOT_DIR"

PATCH_SOURCE="$ROOT_DIR/mjlab_patch/mjlab/managers/observation_manager.py"
if [[ -f "$PATCH_SOURCE" ]]; then
  MJLAB_DIR="$(run_in_env python -c 'import pathlib, mjlab; print(pathlib.Path(mjlab.__file__).resolve().parent)')"
  install -m 0644 "$PATCH_SOURCE" "$MJLAB_DIR/managers/observation_manager.py"
  echo "[INFO] Applied mjlab history-ordering patch."
fi

run_in_env python -c 'import mjlab, mujoco, torch, warp; print(f"[OK] torch={torch.__version__}, CUDA={torch.version.cuda}, available={torch.cuda.is_available()}"); print(f"[OK] mujoco={mujoco.__version__}, warp={warp.__version__}")'
run_in_env python "$ROOT_DIR/scripts/list_envs.py" --keyword Casbot02

echo "[DONE] Environment is ready. Activate it with: conda activate $ENV_NAME"
