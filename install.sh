#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
ENV_NAME=Optimizer
APPROVED_REF=""
RELEASE_DIR=""
usage() {
  printf '%s\n' 'Usage: ./install.sh --approved-ref FULL_COMMIT_SHA [--env-name Optimizer] [--release-dir PATH]' 'Build, test and install the selected committed source; never installs working-tree edits.'
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --approved-ref) APPROVED_REF="${2:?Missing commit}"; shift 2;;
    --env-name) ENV_NAME="${2:?Missing name}"; shift 2;;
    --release-dir) RELEASE_DIR="${2:?Missing path}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) usage >&2; exit 2;;
  esac
done
[[ "$APPROVED_REF" =~ ^[0-9a-f]{40}$ ]] || { usage >&2; exit 2; }
[[ -x "$CONDA_BIN" ]] || { printf '%s\n' 'Set CONDA_EXE to your conda executable.' >&2; exit 1; }
if ! "$CONDA_BIN" run -n "$ENV_NAME" python -I -c 'import sys' >/dev/null 2>&1; then
  "$CONDA_BIN" env create -n "$ENV_NAME" -f "$ROOT_DIR/environment.yml"
fi
RELEASE_DIR="${RELEASE_DIR:-$HOME/.local/share/optimizer/releases/$APPROVED_REF}"
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -I "$ROOT_DIR/tools/release.py" build --approved-ref "$APPROVED_REF" --output "$RELEASE_DIR"
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -I "$ROOT_DIR/tools/release.py" install --release "$RELEASE_DIR"
printf '%s\n' 'Ready: conda activate Optimizer. Add ~/.local/bin to PATH for use without activation.'
