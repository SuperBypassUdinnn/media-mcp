#!/usr/bin/env python3
"""Rhythm analysis for one media file: onsets, tempo, and a beat grid.

The librosa-based engine of the media server's ``media_beats`` tool. Reads
the media path from argv[1], decodes mono audio through librosa, and prints
one JSON object to stdout:

    {"engine": "librosa", "degraded": false, "duration": 10.0,
     "tempo": 120.0, "beats": [0.5, 1.0, ...], "onsets": [0.5, ...]}

``tempo`` is null when no reliable periodicity was found, and ``beats`` is
empty in that case; ``onsets`` is still the measured note-boundary list.
Any failure prints nothing on stdout and exits non-zero — the caller
treats a non-zero exit or unparseable stdout as "this engine is not
available" and falls back to the pure-numpy baseline.
"""

from __future__ import annotations

import json
import sys
import warnings


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: onsets.py <media-path>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    # librosa/numba emit benign JIT and resample warnings; they are noise on
    # a stdout that must stay pure JSON, so they are suppressed wholesale.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import subprocess
        import tempfile
        from pathlib import Path

        import librosa
        import numpy as np

        # libsndfile (librosa's default reader) cannot decode compressed
        # containers such as mp4/aac, so every source goes through one ffmpeg
        # pass into a temp wav first; that also makes the engine's accepted
        # inputs identical to the numpy baseline's.
        with tempfile.TemporaryDirectory() as scratch:
            wav = Path(scratch) / "rhythm-in.wav"
            done = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    path,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "22050",
                    "-c:a",
                    "pcm_s16le",
                    str(wav),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if done.returncode != 0:
                print(done.stderr.strip() or "ffmpeg decode failed", file=sys.stderr)
                return 1
            y, sr = librosa.load(str(wav), sr=None, mono=True)
        duration = float(len(y) / sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, units="time"
        )
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, units="time"
        )
        tempo_value = float(np.atleast_1d(tempo)[0])
        if not np.isfinite(tempo_value) or tempo_value <= 0:
            tempo_value = None
    payload = {
        "engine": "librosa",
        "degraded": False,
        "duration": round(duration, 3),
        "tempo": round(tempo_value, 2) if tempo_value is not None else None,
        "beats": [round(float(b), 3) for b in beat_frames],
        "onsets": [round(float(o), 3) for o in onsets],
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
