#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "Starting FairSight backend on http://127.0.0.1:8000 ..."
cd "$ROOT_DIR"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload >"$LOG_DIR/backend.out.log" 2>"$LOG_DIR/backend.err.log" &
BACKEND_PID=$!

echo "Starting FairSight frontend on http://127.0.0.1:5173 ..."
cd "$ROOT_DIR/frontend"
npm run dev -- --host 127.0.0.1 --port 5173 >"$LOG_DIR/frontend.out.log" 2>"$LOG_DIR/frontend.err.log" &
FRONTEND_PID=$!

cat >"$ROOT_DIR/.run_pids" <<EOF
BACKEND_PID=$BACKEND_PID
FRONTEND_PID=$FRONTEND_PID
EOF

echo "Started."
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Use ./stop_all.sh to stop both services."
