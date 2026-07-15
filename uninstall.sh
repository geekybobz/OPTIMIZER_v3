#!/usr/bin/env bash
set -euo pipefail

# Remove the editable OPTIMIZER v3 package from the conda environment.  The conda
# environment itself is kept by default.  Pass --remove-env when the whole
# environment should be deleted.

PACKAGE_NAME="optimizer-v3"
ENV_NAME="${OPTIMIZER_V3_ENV:-optimizer_v3}"
STATE_FILE="${OPTIMIZER_V3_STATE_FILE:-${HOME}/.optimizer_v3/install.json}"
REMOVE_ENV=0

usage() {
  cat <<'USAGE'
Usage: ./uninstall.sh [options]

Options:
  --env-name NAME       Conda environment name. Defaults to install-state value,
                        then OPTIMIZER_V3_ENV, then optimizer_v3.
  --state-file PATH     Install-state JSON path. Default: ~/.optimizer_v3/install.json
  --remove-env          Remove the whole conda environment after uninstalling package.
  -h, --help            Show this help.
USAGE
}

json_value() {
  local file="$1"
  local key="$2"
  local pybin=""
  if command -v python3 >/dev/null 2>&1; then
    pybin="python3"
  elif command -v python >/dev/null 2>&1; then
    pybin="python"
  else
    return 1
  fi
  "${pybin}" - "${file}" "${key}" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
try:
    data = json.loads(open(path, encoding="utf-8").read())
except Exception:
    sys.exit(1)
value = data
for part in key.split("."):
    value = value.get(part, "") if isinstance(value, dict) else ""
print(value if value is not None else "")
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="${2:?Missing value for --env-name}"
      shift 2
      ;;
    --state-file)
      STATE_FILE="${2:?Missing value for --state-file}"
      shift 2
      ;;
    --remove-env)
      REMOVE_ENV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 1
fi

if [[ -f "${STATE_FILE}" ]]; then
  STATE_ENV="$(json_value "${STATE_FILE}" env_name || true)"
  STATE_PACKAGE="$(json_value "${STATE_FILE}" package || true)"
  if [[ -n "${STATE_ENV}" && "${ENV_NAME}" == "${OPTIMIZER_V3_ENV:-optimizer_v3}" ]]; then
    ENV_NAME="${STATE_ENV}"
  fi
  if [[ -n "${STATE_PACKAGE}" ]]; then
    PACKAGE_NAME="${STATE_PACKAGE}"
  fi
fi

env_prefix() {
  conda env list | awk -v name="${ENV_NAME}" '$1 == name {print $NF; exit}'
}

if [[ -z "$(env_prefix)" ]]; then
  echo "Conda environment not found: ${ENV_NAME}"
  if [[ -f "${STATE_FILE}" ]]; then
    echo "Removing stale install state: ${STATE_FILE}"
    rm -f "${STATE_FILE}"
  fi
  exit 0
fi

echo "Removing OPTIMIZER v3 editable package from: ${ENV_NAME}"
conda run -n "${ENV_NAME}" python -m pip uninstall -y "${PACKAGE_NAME}" || true

if [[ "${REMOVE_ENV}" == "1" ]]; then
  echo "Removing conda environment: ${ENV_NAME}"
  conda env remove -n "${ENV_NAME}" -y
else
  echo "Environment kept: ${ENV_NAME}"
  echo "Remove it later with: ./uninstall.sh --remove-env"
fi

if [[ -f "${STATE_FILE}" ]]; then
  echo "Removing install state: ${STATE_FILE}"
  rm -f "${STATE_FILE}"
fi
