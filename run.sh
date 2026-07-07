#!/usr/bin/env bash
# Single-command startup (no Docker): starts the FastAPI backend and the
# Streamlit UI together, and stops both on Ctrl+C.
set -e

if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi

echo "Starting FastAPI backend on http://localhost:8000 ..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

cleanup() {
  echo "Stopping backend (pid $API_PID)..."
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2
echo "Starting Streamlit UI on http://localhost:8501 ..."
streamlit run ui/streamlit_app.py --server.enableCORS=false --server.enableXsrfProtection=false
