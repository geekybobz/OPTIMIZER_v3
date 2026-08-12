#!/usr/bin/env bash
set -euo pipefail

# Build a wheel outside this checkout and install it non-editably into the
# release environment.  A tagged release must never be a live view of its source.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_EXE:-/Users/billabobz/miniconda3/bin/conda}"
PACKAGE_NAME="optimizer-v4"
IMPORT_NAME="optimizer"
ENV_NAME="${OPTIMIZER_V4_ENV:-optimizer_v4}"
ENV_FILE="${ROOT_DIR}/environment.yml"
STATE_FILE="${OPTIMIZER_V4_STATE_FILE:-${HOME}/.optimizer_v4/install.json}"
RUN_TESTS=0

usage() {
  cat <<'USAGE'
Usage: ./install.sh [options]

Options:
  --env-name NAME       Conda environment name. Default: optimizer_v4
  --state-file PATH     Install-state JSON path. Default: ~/.optimizer_v4/install.json
  --run-tests           Run the unit test suite after installation.
  -h, --help            Show this help.

Environment overrides:
  OPTIMIZER_V4_ENV
  OPTIMIZER_V4_STATE_FILE
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

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable was not found: ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing environment file: ${ENV_FILE}" >&2
  exit 1
fi

env_prefix() {
  "${CONDA_BIN}" env list | awk -v name="${ENV_NAME}" '$1 == name {print $NF; exit}'
}

if [[ -n "$(env_prefix)" ]]; then
  echo "Updating existing conda environment: ${ENV_NAME}"
  "${CONDA_BIN}" env update -n "${ENV_NAME}" -f "${ENV_FILE}"
else
  echo "Creating conda environment: ${ENV_NAME}"
  "${CONDA_BIN}" env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

WHEEL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/optimizer-v4-wheel.XXXXXX")"
cleanup() { rm -rf "${WHEEL_DIR}"; }
trap cleanup EXIT

echo "Building release wheel outside the checkout"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -B -m pip wheel --no-deps --no-build-isolation -w "${WHEEL_DIR}" "${ROOT_DIR}"
WHEEL_PATH="$(find "${WHEEL_DIR}" -maxdepth 1 -name '*.whl' -print -quit)"
if [[ -z "${WHEEL_PATH}" ]]; then
  echo "Wheel build produced no wheel." >&2
  exit 1
fi
WHEEL_SHA256="$(shasum -a 256 "${WHEEL_PATH}" | awk '{print $1}')"

echo "Installing non-editable release wheel into: ${ENV_NAME}"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -B -m pip install --no-deps --force-reinstall "${WHEEL_PATH}"

echo "Verifying import surface"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -B -c "import optimizer as opt; print('optimizer import ok'); print('public namespaces: optimizers, utils, guesses, schedules'); print('example method: ' + opt.optimizers.adam.__name__)"

if [[ "${RUN_TESTS}" == "1" ]]; then
  echo "Running unit tests inside ${ENV_NAME}"
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -B -m unittest discover -s "${ROOT_DIR}/tests"
fi

STATE_DIR="$(dirname "${STATE_FILE}")"
mkdir -p "${STATE_DIR}"

ENV_PREFIX="$(env_prefix)"
PYTHON_EXE="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -B -c "import sys; print(sys.executable)")"
CONDA_EXE="${CONDA_BIN}"
GIT_REMOTE="$(git -C "${ROOT_DIR}" remote get-url origin 2>/dev/null || true)"
GIT_BRANCH="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true)"
GIT_TAG="$(git -C "${ROOT_DIR}" describe --exact-match --tags HEAD 2>/dev/null || true)"

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
GIT_TAG="${GIT_TAG}" \
WHEEL_SHA256="${WHEEL_SHA256}" \
"${CONDA_BIN}" run -n "${ENV_NAME}" python -B -c "import datetime as d, json, os, pathlib; p=pathlib.Path(os.environ['STATE_FILE']); data={'schema_version':1,'package':os.environ['PACKAGE_NAME'],'import_name':os.environ['IMPORT_NAME'],'install_mode':'wheel','repo_root':os.environ['ROOT_DIR'],'environment_file':os.environ['ENV_FILE'],'env_name':os.environ['ENV_NAME'],'env_prefix':os.environ['ENV_PREFIX'],'python':os.environ['PYTHON_EXE'],'conda':os.environ['CONDA_EXE'],'wheel_sha256':os.environ['WHEEL_SHA256'],'installed_at_utc':d.datetime.now(d.UTC).isoformat().replace('+00:00','Z'),'git':{'remote_origin':os.environ['GIT_REMOTE'],'branch':os.environ['GIT_BRANCH'],'commit':os.environ['GIT_COMMIT'],'tag':os.environ['GIT_TAG']}}; p.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')"

echo
echo "Setup complete."
echo "Activate with: conda activate ${ENV_NAME}"
echo "State file: ${STATE_FILE}"
