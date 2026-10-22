#!/usr/bin/env bash
# Easy ML Viewer — macOS / Linux launcher (Phase 40 hardened).
# Boots backend 127.0.0.1:30211 + frontend 127.0.0.1:30222, 100% offline.
# Pre-flight stale-port sweeper + process-GROUP cleanup via trap (no zombies).
set -euo pipefail
cd "$(dirname "$0")/.."

free_port() { lsof -ti tcp:"$1" 2>/dev/null | xargs -r kill -9 2>/dev/null || true; }
echo "[1/3] Sweeping stale ports 30211 and 30222..."
free_port 30211; free_port 30222

echo "[2/3] Starting backend on 127.0.0.1:30211 ..."
(cd backend && python3 -m uvicorn app.main:app --host 127.0.0.1 --port 30211) &
BACKEND_PID=$!
sleep 3
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then echo "[ERROR] backend failed to start"; exit 1; fi

echo "[3/3] Starting frontend on 127.0.0.1:30222 ..."
(cd frontend && npm run dev -- --host 127.0.0.1 --port 30222 --strictPort) &
FRONTEND_PID=$!

cleanup() {
  echo; echo "Stopping EasyML servers (backend $BACKEND_PID, frontend $FRONTEND_PID)..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM
echo "Backend PID $BACKEND_PID | Frontend PID $FRONTEND_PID — http://127.0.0.1:30222 (100% offline)"
(xdg-open http://127.0.0.1:30222 2>/dev/null || open http://127.0.0.1:30222 2>/dev/null || true) &
wait
