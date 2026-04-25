#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT_DIR/.run_pids"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No .run_pids found. Nothing to stop."
  exit 0
fi

# shellcheck disable=SC1090
source "$PID_FILE"

for PID in "${BACKEND_PID:-}" "${FRONTEND_PID:-}"; do
  if [[ -n "${PID}" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" || true
    echo "Stopped PID $PID"
  else
    echo "PID ${PID:-unknown} already stopped or unavailable"
  fi
done

rm -f "$PID_FILE"
echo "All done."
