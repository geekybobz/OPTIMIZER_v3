#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  --remove-launchers) ;;
  *) printf '%s\n' 'Usage: ./uninstall.sh --remove-launchers' 'Removes managed global launchers only. Environment, experiments and release artifacts are retained.'; exit 2;;
esac
for name in optimizer optimizer-mcp optimizer-system; do
  target="$HOME/.local/bin/$name"
  if [[ -f "$target" ]] && grep -q '^# Managed OPTIMIZER release launcher;' "$target"; then
    rm -- "$target"
  fi
done
