#!/usr/bin/env bash
# Model Surgery setup for Linux / Jetson: creates .venv, installs the backend, builds the frontend if npm exists.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3.11}
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e "backend[dev]"

if command -v npm >/dev/null 2>&1; then
  (cd frontend && { [ -f package-lock.json ] && npm ci || npm install; } && npm run build)
else
  echo "npm not found: build frontend/dist on another machine and copy it here."
fi

echo
echo "Done. Start with:  .venv/bin/model-surgery serve"
