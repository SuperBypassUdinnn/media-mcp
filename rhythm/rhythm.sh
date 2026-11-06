#!/usr/bin/env bash
# Run rhythm analysis inside the repository virtualenv (rhythm/.venv).
# Bootstraps that virtualenv via rhythm/setup.sh on first use, so agents
# never touch system pip. Prints one JSON object to stdout (see onsets.py).
#
# MEDIA_RHYTHM_BIN is deliberately ignored here. The server forwards to this
# script through that variable, so honouring it would let the two call each
# other; only the forwarder reads it.
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PIPELINE_DIR/.venv"
PY="$VENV/bin/python"
ONSETS="$PIPELINE_DIR/onsets.py"

if [[ ! -x "$PY" ]]; then
  echo "Rhythm virtualenv not found; running one-time setup ..." >&2
  bash "$PIPELINE_DIR/setup.sh"
fi

exec "$PY" "$ONSETS" "$@"
