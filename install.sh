#!/usr/bin/env bash
set -euo pipefail

# Build or update the conda environment used for OPTIMIZER v3, then install this
# repository in editable mode.  Editable mode means Python imports the live source
# tree, so code edits are visible after restarting Python without reinstalling.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="optimizer-v3"
IMPORT_NAME="optimizer"
ENV_NAME="${OPTIMIZER_V3_ENV:-optimizer_v3}"
ENV_FILE="${ROOT_DIR}/environment.yml"
STATE_FILE="${OPTIMIZER_V3_STATE_FILE:-${HOME}/.optimizer_v3/install.json}"
RUN_TESTS=0

usage() {
  cat <<'USAGE'
Usage: ./install.sh [options]

Options:
  --env-name NAME       Conda environment name. Default: optimizer_v3
  --state-file PATH     Install-state JSON path. Default: ~/.optimizer_v3/install.json
  --run-tests           Run the unit test suite after installation.
  -h, --help            Show this help.

Environment overrides:
  OPTIMIZER_V3_ENV
  OPTIMIZER_V3_STATE_FILE
USAGE
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
    --run-tests)
      RUN_TESTS=1
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

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing environment file: ${ENV_FILE}" >&2
  exit 1
fi

env_prefix() {
  conda env list | awk -v name="${ENV_NAME}" '$1 == name {print $NF; exit}'
}

if [[ -n "$(env_prefix)" ]]; then
  echo "Updating existing conda environment: ${ENV_NAME}"
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}"
else
  echo "Creating conda environment: ${ENV_NAME}"
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

echo "Installing OPTIMIZER v3 editable package into: ${ENV_NAME}"
conda run -n "${ENV_NAME}" python -m pip install --no-deps --no-build-isolation -e "${ROOT_DIR}"

echo "Verifying import surface"
conda run -n "${ENV_NAME}" python -c "import optimizer as opt; print('optimizer import ok'); print('public namespaces: optimizers, utils, guesses, schedules'); print('example method: ' + opt.optimizers.adam.__name__)"

if [[ "${RUN_TESTS}" == "1" ]]; then
  echo "Running unit tests inside ${ENV_NAME}"
  conda run -n "${ENV_NAME}" python -m unittest discover -s "${ROOT_DIR}/tests"
fi

STATE_DIR="$(dirname "${STATE_FILE}")"
mkdir -p "${STATE_DIR}"

ENV_PREFIX="$(env_prefix)"
PYTHON_EXE="$(conda run -n "${ENV_NAME}" python -c "import sys; print(sys.executable)")"
CONDA_EXE="$(command -v conda)"
GIT_REMOTE="$(git -C "${ROOT_DIR}" remote get-url origin 2>/dev/null || true)"
GIT_BRANCH="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true)"

echo "Writing install state: ${STATE_FILE}"
STATE_FILE="${STATE_FILE}" \
ROOT_DIR="${ROOT_DIR}" \
ENV_NAME="${ENV_NAME}" \
ENV_PREFIX="${ENV_PREFIX}" \
PYTHON_EXE="${PYTHON_EXE}" \
CONDA_EXE="${CONDA_EXE}" \
ENV_FILE="${ENV_FILE}" \
PACKAGE_NAME="${PACKAGE_NAME}" \
IMPORT_NAME="${IMPORT_NAME}" \
GIT_REMOTE="${GIT_REMOTE}" \
GIT_BRANCH="${GIT_BRANCH}" \
GIT_COMMIT="${GIT_COMMIT}" \
conda run -n "${ENV_NAME}" python -c "import datetime as d, json, os, pathlib; p=pathlib.Path(os.environ['STATE_FILE']); data={'schema_version':1,'package':os.environ['PACKAGE_NAME'],'import_name':os.environ['IMPORT_NAME'],'install_mode':'editable','repo_root':os.environ['ROOT_DIR'],'environment_file':os.environ['ENV_FILE'],'env_name':os.environ['ENV_NAME'],'env_prefix':os.environ['ENV_PREFIX'],'python':os.environ['PYTHON_EXE'],'conda':os.environ['CONDA_EXE'],'installed_at_utc':d.datetime.now(d.UTC).isoformat().replace('+00:00','Z'),'git':{'remote_origin':os.environ['GIT_REMOTE'],'branch':os.environ['GIT_BRANCH'],'commit':os.environ['GIT_COMMIT']}}; p.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')"

echo
echo "Setup complete."
echo "Activate with: conda activate ${ENV_NAME}"
echo "State file: ${STATE_FILE}"
