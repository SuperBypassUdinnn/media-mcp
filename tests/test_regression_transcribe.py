"""The promoted pipeline reproduces the pre-move wrapper's segments.

Both sides run the same moved pipeline: `transcribe/transcribe.sh` and the
frozen `tests/snapshots/transcribe-pre-move.sh` both execute
`transcribe/.venv/bin/python` on `transcribe/transcribe.py`. Only the wrapper
logic differs, because the snapshot is the skill wrapper captured before the
move with just its venv, script and token-file paths adjusted to the moved
locations; the comparison therefore isolates the wrapper change rather than the
pipeline.

The sample is synthesised with espeak-ng into the tmp directory on every run, so
no generated media is committed and the comparison cannot be vacuous (a tone
would leave both sides with empty segment lists). Diarization is off on both
sides: this regression is about the wrapper, and an off run needs no Hugging
Face token.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WRAPPER = REPO_ROOT / "transcribe" / "transcribe.sh"
SNAPSHOT = REPO_ROOT / "tests" / "snapshots" / "transcribe-pre-move.sh"
SENTENCE = (
    "Hello world. This is a transcription regression test. Numbers: one two three."
)

# `tiny` is the smallest model that produces stable segments; the environment
# override exists so a machine without network can point both sides at a model
# that is already cached.
MODEL = os.environ.get("MEDIA_TRANSCRIBE_TEST_MODEL", "tiny")
TIMEOUT_SECONDS = int(os.environ.get("MEDIA_TRANSCRIBE_TEST_TIMEOUT", "900"))


def _speech_sample(directory: Path) -> Path:
    """Render `SENTENCE` with espeak-ng, or skip when it is not installed."""
    if shutil.which("espeak-ng") is None:
        pytest.skip("espeak-ng is not installed, so no speech sample can be made")
    sample = directory / "speech.wav"
    subprocess.run(
        ["espeak-ng", "-s", "140", "-w", str(sample), SENTENCE],
        check=True,
        capture_output=True,
        timeout=60,
    )
    assert sample.is_file() and sample.stat().st_size > 0
    return sample


def _run_pipeline(wrapper: Path, sample: Path, out_dir: Path) -> list[str]:
    """Run one wrapper over `sample` and return `speaker|text` per segment.

    The wrapper is invoked through `bash`, so a missing executable bit cannot
    decide the comparison.
    """
    done = subprocess.run(
        [
            "bash",
            str(wrapper),
            str(sample),
            "--no-diarize",
            "--model",
            MODEL,
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )
    assert done.returncode == 0, (
        f"{wrapper} exited {done.returncode}\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    )
    transcript = out_dir / sample.stem / f"{sample.stem}.json"
    assert transcript.is_file(), f"the run wrote no {transcript}"
    payload = json.loads(transcript.read_text(encoding="utf-8"))
    return [
        f"{segment.get('speaker')}|{segment.get('text')}"
        for segment in payload["segments"]
    ]


def test_moved_pipeline_matches_the_original_segments(tmp_path: Path) -> None:
    """Both wrappers drive the same moved pipeline over the same speech."""
    sample = _speech_sample(tmp_path)

    moved = _run_pipeline(WRAPPER, sample, tmp_path / "moved")
    frozen = _run_pipeline(SNAPSHOT, sample, tmp_path / "frozen")

    assert moved, "the sample must yield at least one segment to mean anything"
    assert moved == frozen
