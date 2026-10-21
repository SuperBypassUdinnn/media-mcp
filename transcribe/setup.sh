#!/usr/bin/env bash
# Idempotent one-time environment setup for the repository transcription
# pipeline. Creates transcribe/.venv and installs transcribe/requirements.txt.
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PIPELINE_DIR/.venv"

# Python 3.12 is preferred: current torch/torchaudio wheels compatible with
# pyannote.audio 3.x exist for 3.12, while newer interpreters only offer
# torchaudio releases that removed APIs pyannote 3.x still calls.
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

# deepfilternet declares numpy<2, which breaks scipy; numpy 2.x is verified
# working with DeepFilterNet, so upgrade past the cap after requirements.
"$VENV/bin/python" -m pip install "numpy>=2.0"

# Credential store: created once, never overwritten, always owner-only. It
# lives with the skill because the secrets did not move with the pipeline.
ENV_FILE="${MEDIA_TRANSCRIBE_ENV:-$HOME/.config/opencode/skills/transcribe-diarize/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  mkdir -p "$(dirname "$ENV_FILE")"
  cat > "$ENV_FILE" <<'EOF'
# Hugging Face token for gated pyannote models (diarization).
# Paste your token after the = sign. Keep this file chmod 600.
HF_TOKEN=
EOF
fi
chmod 600 "$ENV_FILE"

echo "Setup complete: $VENV/bin/python"
