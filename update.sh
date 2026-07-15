#!/usr/bin/env bash
set -euo pipefail

# Fetch the latest Git revision for this checkout, fast-forward when possible, then
# rerun the editable installer.  This is intentionally conservative: dirty worktrees
# and diverged branches stop with an explanation instead of overwriting local work.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${OPTIMIZER_V3_REMOTE:-origin}"
BRANCH="${OPTIMIZER_V3_BRANCH:-}"
CHECK_ONLY=0
ALLOW_DIRTY=0
RUN_TESTS=0

usage() {
  cat <<'USAGE'
Usage: ./update.sh [options]

Options:
  --check           Only report whether updates are available.
  --allow-dirty     Allow update with local uncommitted changes.
  --run-tests       Run tests after reinstalling.
  --remote NAME     Git remote to fetch. Default: origin
  --branch NAME     Remote branch to compare/pull. Default: current branch/upstream.
  -h, --help        Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      CHECK_ONLY=1
      shift
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    --run-tests)
      RUN_TESTS=1
      shift
      ;;
    --remote)
      REMOTE="${2:?Missing value for --remote}"
      shift 2
      ;;
    --branch)
      BRANCH="${2:?Missing value for --branch}"
      shift 2
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

if ! git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This folder is not a git checkout: ${ROOT_DIR}" >&2
  exit 1
fi

if ! git -C "${ROOT_DIR}" remote get-url "${REMOTE}" >/dev/null 2>&1; then
  echo "Git remote not found: ${REMOTE}" >&2
  echo "Add a remote first, for example: git remote add origin <repo-url>" >&2
  exit 1
fi

CURRENT_BRANCH="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD)"
if [[ -z "${BRANCH}" ]]; then
  BRANCH="${CURRENT_BRANCH}"
fi
REMOTE_REF="${REMOTE}/${BRANCH}"

echo "Fetching ${REMOTE}"
git -C "${ROOT_DIR}" fetch "${REMOTE}"

if ! git -C "${ROOT_DIR}" rev-parse --verify "${REMOTE_REF}" >/dev/null 2>&1; then
  echo "Remote branch not found: ${REMOTE_REF}" >&2
  exit 1
fi

LOCAL_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
REMOTE_COMMIT="$(git -C "${ROOT_DIR}" rev-parse "${REMOTE_REF}")"
BASE_COMMIT="$(git -C "${ROOT_DIR}" merge-base HEAD "${REMOTE_REF}")"

if [[ "${LOCAL_COMMIT}" == "${REMOTE_COMMIT}" ]]; then
  echo "Already up to date: ${LOCAL_COMMIT}"
  exit 0
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  if [[ "${BASE_COMMIT}" == "${LOCAL_COMMIT}" ]]; then
    echo "Update available: ${LOCAL_COMMIT} -> ${REMOTE_COMMIT}"
    git -C "${ROOT_DIR}" log --oneline "${LOCAL_COMMIT}..${REMOTE_REF}"
    exit 0
  fi
  echo "No clean fast-forward update available. Local and remote have diverged."
  exit 1
fi

if [[ "${BASE_COMMIT}" != "${LOCAL_COMMIT}" ]]; then
  if [[ "${BASE_COMMIT}" == "${REMOTE_COMMIT}" ]]; then
    echo "Local checkout is ahead of ${REMOTE_REF}; nothing to pull."
    exit 0
  fi
  echo "Local and remote branches have diverged. Resolve with git manually." >&2
  exit 1
fi

if [[ "${ALLOW_DIRTY}" != "1" && -n "$(git -C "${ROOT_DIR}" status --porcelain)" ]]; then
  echo "Worktree has uncommitted changes. Commit/stash them or pass --allow-dirty." >&2
  git -C "${ROOT_DIR}" status --short
  exit 1
fi

echo "Fast-forwarding ${CURRENT_BRANCH} from ${LOCAL_COMMIT} to ${REMOTE_COMMIT}"
git -C "${ROOT_DIR}" pull --ff-only "${REMOTE}" "${BRANCH}"

INSTALL_ARGS=()
if [[ "${RUN_TESTS}" == "1" ]]; then
  INSTALL_ARGS+=(--run-tests)
fi

"${ROOT_DIR}/install.sh" "${INSTALL_ARGS[@]}"

