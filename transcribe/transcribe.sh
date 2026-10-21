#!/usr/bin/env bash
# Run transcription inside the repository virtualenv (transcribe/.venv).
# Bootstraps that virtualenv via transcribe/setup.sh on first use, so agents
# never touch system pip.
#
# MEDIA_TRANSCRIBE_BIN is deliberately ignored here. The skill wrapper forwards
# to this script through that variable, so honouring it would let the two call
# each other; only the forwarder reads it.
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PIPELINE_DIR/.venv"
PY="$VENV/bin/python"
TRANSCRIBE="$PIPELINE_DIR/transcribe.py"

if [[ ! -x "$PY" ]]; then
  echo "Transcription virtualenv not found; running one-time setup ..." >&2
  bash "$PIPELINE_DIR/setup.sh"
fi

# Credentials live only in the token file (chmod 600), never in chat or config.
# The default is the skill's file because the secrets stayed there when the
# pipeline moved; MEDIA_TRANSCRIBE_ENV points somewhere else when they differ.
# Explicit environment variables take precedence over the file.
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
