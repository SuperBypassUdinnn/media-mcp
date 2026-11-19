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

import numpy as np


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
        beat_list = [round(float(b), 3) for b in beat_frames]
        onset_list = [round(float(o), 3) for o in onsets]
        confidence = None
        if beat_list and onset_list:
            hits = sum(
                1
                for beat in beat_list
                if min(abs(beat - onset) for onset in onset_list) <= 0.07
            )
            confidence = round(hits / len(beat_list), 3)
        beat_details = _compute_beat_details(beat_list, y, sr)
    payload = {
        "engine": "librosa",
        "degraded": False,
        "duration": round(duration, 3),
        "tempo": round(tempo_value, 2) if tempo_value is not None else None,
        "beats": beat_list,
        "onsets": onset_list,
        "confidence": confidence,
        "beat_details": beat_details,
    }
    print(json.dumps(payload))
    return 0


def _band_channels(sr: float, n_fft: int = 2048) -> list[slice]:
    """FFT-frequency slices for low (<150 Hz), mid (150-2500 Hz), high (>2500 Hz)."""
    import librosa
    import numpy as np

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    idx_150 = int(np.argmax(freqs >= 150.0)) if (freqs >= 150.0).any() else len(freqs)
    idx_2500 = (
        int(np.argmax(freqs >= 2500.0)) if (freqs >= 2500.0).any() else len(freqs)
    )
    return [slice(0, idx_150), slice(idx_150, idx_2500), slice(idx_2500, None)]


def _compute_beat_details(beat_times: list[float], y: np.ndarray, sr: float) -> list:
    """Per-beat strength, band, downbeat phase, and phrase energy (librosa).

    Returns a list of dicts matching the ``BeatDetail`` field names so the
    caller can serialize them directly into the JSON payload.
    """
    import librosa
    import numpy as np

    if not beat_times:
        return []
    hop_length = 512
    n_fft = 2048
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S = librosa.amplitude_to_db(stft)
    band_envs = librosa.onset.onset_strength_multi(
        S=S,
        sr=sr,
        hop_length=hop_length,
        channels=_band_channels(sr, n_fft),
    )
    band_names = ("low", "mid", "high")
    n_frames = band_envs.shape[1]
    bands = (band_envs[0], band_envs[1], band_envs[2])
    floors = [float(np.median(b)) + 1e-9 for b in bands]
    times = np.array(beat_times, dtype=np.float64)
    frames = np.clip(
        librosa.time_to_frames(times, sr=sr, hop_length=hop_length), 0, n_frames - 1
    )

    def sample(band: np.ndarray, frame: int) -> float:
        return float(band[frame])

    totals = np.array(
        [
            sum(sample(bands[bi], frames[i]) for bi in range(3))
            for i in range(len(beat_times))
        ],
        dtype=np.float64,
    )
    scale = float(np.percentile(totals, 98)) or 1.0
    strengths = np.clip(totals / scale, 0.0, 1.0)

    band_idxs = [
        int(np.argmax([sample(bands[bi], frames[i]) / floors[bi] for bi in range(3)]))
        for i in range(len(beat_times))
    ]

    low_strengths = np.array(
        [sample(bands[0], frames[i]) for i in range(len(beat_times))],
        dtype=np.float64,
    )
    n = len(beat_times)
    best_phase, best_score = 0, -1.0
    for p in range(4):
        mask = np.arange(n) % 4 == p
        score = float(low_strengths[mask].sum()) / (
            float(low_strengths[~mask].sum()) + 1e-9
        )
        if score > best_score:
            best_score, best_phase = score, p

    half = 4
    phrase = np.array(
        [strengths[max(0, i - half) : min(n, i + half + 1)].mean() for i in range(n)],
        dtype=np.float64,
    )
    phrase /= float(phrase.max()) or 1.0

    details = []
    for i, t in enumerate(beat_times):
        beat_in_bar = ((i - best_phase) % 4) + 1
        bar_index = (i - best_phase) // 4
        details.append(
            {
                "t": round(float(t), 3),
                "strength": round(float(strengths[i]), 3),
                "band": band_names[band_idxs[i]],
                "is_downbeat": bool(beat_in_bar == 1),
                "bar_index": int(bar_index),
                "beat_in_bar": int(beat_in_bar),
                "phrase_energy": round(float(phrase[i]), 3),
            }
        )
    return details


if __name__ == "__main__":
    raise SystemExit(main())
