#!/usr/bin/env bash
# Easy ML Viewer — macOS / Linux launcher. Cleans ports, boots both servers, traps EXIT.
set -euo pipefail
cd "$(dirname "$0")/.."

free_port() { lsof -ti tcp:"$1" 2>/dev/null | xargs -r kill -9 2>/dev/null || true; }
echo "[1/3] Freeing ports 30211 and 30222..."
free_port 30211; free_port 30222

echo "[2/3] Starting backend :30211 ..."
(cd backend && python3 -m uvicorn app.main:app --host 127.0.0.1 --port 30211) &
BACKEND_PID=$!
sleep 3

echo "[3/3] Starting frontend :30222 ..."
(cd frontend && npm run dev) &
FRONTEND_PID=$!

cleanup() { echo; echo "Stopping servers..."; kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
echo "Backend PID $BACKEND_PID | Frontend PID $FRONTEND_PID — opening http://localhost:30222"
(xdg-open http://localhost:30222 2>/dev/null || open http://localhost:30222 2>/dev/null || true) &
wait
