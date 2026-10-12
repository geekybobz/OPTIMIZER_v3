#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_ONLY=0
case "${1:-}" in
  --check) CHECK_ONLY=1;;
  -h|--help) printf '%s\n' 'Usage: ./update.sh [--check]' 'Fetch origin/main and fast-forward source only. Global releases require explicit install.sh promotion.'; exit 0;;
  '') ;;
  *) exit 2;;
esac
git -C "$ROOT_DIR" fetch origin main
if [[ "$CHECK_ONLY" == 1 ]]; then
  git -C "$ROOT_DIR" log --oneline HEAD..origin/main
else
  [[ "$(git -C "$ROOT_DIR" branch --show-current)" == main ]] || { printf '%s\n' 'Switch to main first.' >&2; exit 1; }
  [[ -z "$(git -C "$ROOT_DIR" status --porcelain)" ]] || { printf '%s\n' 'Commit or stash source changes first.' >&2; exit 1; }
  git -C "$ROOT_DIR" merge --ff-only origin/main
fi
