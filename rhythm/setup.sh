#!/usr/bin/env bash
# Idempotent one-time environment setup for the repository rhythm pipeline.
# Creates rhythm/.venv (Python 3.12, matching transcribe/.venv) and installs
# the pinned rhythm/requirements.txt (librosa 1.0.0 and its closure).
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PIPELINE_DIR/.venv"

VENV_PYTHON=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    VENV_PYTHON="$candidate"
    break
  fi
done
if [[ -z "$VENV_PYTHON" ]]; then
  echo "No python3.12, python3.11 or python3 on PATH; cannot build $VENV." >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating virtualenv at $VENV with $VENV_PYTHON ..."
  "$VENV_PYTHON" -m venv "$VENV"
else
  echo "Virtualenv already exists at $VENV."
fi

# `python -m pip` rather than bin/pip: a console script's shebang records the
# absolute interpreter path, so it breaks if the virtualenv is ever relocated.
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$PIPELINE_DIR/requirements.txt"

echo "Setup complete: $VENV/bin/python"
