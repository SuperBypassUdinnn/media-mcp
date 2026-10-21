#!/usr/bin/env bash
# Frozen copy of the pre-move skill wrapper, the "before" side of
# tests/test_regression_transcribe.py.
#
# The skill wrapper used to resolve `SKILL_DIR` from its own location and run
# `<skill>/.venv/bin/python <skill>/scripts/transcribe.py`. The promotion moved
# the virtualenv, transcribe.py and the setup script into the repository, so
# this snapshot differs from transcribe/transcribe.sh in exactly two ways and
# no more:
#   1. it resolves the moved pipeline directory explicitly instead of from its
#      own location, because its own location is now tests/snapshots;
#   2. it takes the token file from MEDIA_TRANSCRIBE_ENV with the skill path as
#      the default, again because its own location moved.
# Everything else (the .env parser, the 600 tightening, the bootstrap, the
# `exec` line) is the wrapper as it stood before the move, so a diff of the two
# runs isolates the wrapper change rather than the pipeline.
set -euo pipefail

PIPELINE_DIR="$HOME/.clone/Github/media-mcp/transcribe"
VENV="$PIPELINE_DIR/.venv"
PY="$VENV/bin/python"
TRANSCRIBE="$PIPELINE_DIR/transcribe.py"

if [[ ! -x "$PY" ]]; then
  echo "Transcription virtualenv not found; running one-time setup ..." >&2
  bash "$PIPELINE_DIR/setup.sh"
fi

# Credentials live only in the token file (chmod 600), never in chat or
# config. Explicit environment variables take precedence over the file.
ENV_FILE="${MEDIA_TRANSCRIBE_ENV:-$HOME/.config/opencode/skills/transcribe-diarize/.env}"
if [[ -f "$ENV_FILE" ]]; then
  perms="$(stat -c %a "$ENV_FILE")"
  if [[ "$perms" != "600" ]]; then
    echo "WARNING: $ENV_FILE has permissions $perms; tightening to 600." >&2
    chmod 600 "$ENV_FILE"
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="$(echo "$key" | tr -d '[:space:]')"
    value="$(echo "$value" | sed -e 's/^[[:space:]"]*//' -e 's/[[:space:]"]*$//')"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -z "${!key:-}" ]]; then
      export "$key=$value"
    fi
  done < "$ENV_FILE"
fi

exec "$PY" "$TRANSCRIBE" "$@"
