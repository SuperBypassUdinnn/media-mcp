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

    if not beat_times:
        return []
    hop_length = 512
    n_fft = 2048
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    band_envs = librosa.onset.onset_strength_multi(
        S=stft,
        sr=sr,
        hop_length=hop_length,
        channels=_band_channels(sr, n_fft),
        aggregate=np.sum,
        center=False,
    )
    band_names = ("low", "mid", "high")
    n_frames = band_envs.shape[1]
    n = len(beat_times)
    times = np.array(beat_times, dtype=np.float64)
    frames = np.clip(
        librosa.time_to_frames(times, sr=sr, hop_length=hop_length), 0, n_frames - 1
    )
    # Sample local max within ~60 ms around each beat frame to absorb grid jitter.
    radius = max(1, round(0.06 * sr / hop_length))

    band_vals = np.array(
        [
            [
                float(
                    band_envs[
                        bi, max(0, int(f) - radius) : min(n_frames, int(f) + radius + 1)
                    ].max()
                )
                for bi in range(3)
            ]
            for f in frames
        ],
        dtype=np.float64,
    )

    band_medians = np.median(band_envs, axis=1)
    excess = np.maximum(0.0, band_vals - band_medians)
    active = excess if float(excess.max()) > 0.0 else band_vals

    totals = active.sum(axis=1)
    scale = float(np.percentile(totals, 98)) or 1.0
    strengths = np.clip(totals / scale, 0.0, 1.0)

    band_idxs = [
        int(np.argmax(active[i]))
        if float(active[i].max()) > 0.0
        else int(np.argmax(band_vals[i]))
        for i in range(n)
    ]

    # Downbeat phase: favor low-band impact combined with total strength; fall
    # back to total strength alone when the track has negligible sub-bass.
    low_vals = active[:, 0]
    low_scale = float(np.percentile(low_vals, 98))
    if low_scale > 0.0 and low_scale >= 0.05 * scale:
        low_norm = np.clip(low_vals / low_scale, 0.0, 1.0)
        downbeat_energy = 0.7 * low_norm + 0.3 * strengths
    else:
        downbeat_energy = strengths

    best_phase, best_score = 0, -1.0
    indices = np.arange(n)
    for p in range(min(4, n)):
        mask = indices % 4 == p
        if not mask.any():
            continue
        on_mean = float(downbeat_energy[mask].mean())
        off_mean = float(downbeat_energy[~mask].mean()) if (~mask).any() else 0.0
        score = on_mean / (off_mean + 1e-9)
        if score > best_score:
            best_score, best_phase = score, p

    # 8-beat moving window average: [i - 4, i + 4) -> up to 8 beats.
    phrase = np.array(
        [strengths[max(0, i - 4) : min(n, i + 4)].mean() for i in range(n)],
        dtype=np.float64,
    )
    phrase /= float(phrase.max()) or 1.0

    # Keep bar_index non-negative: pickup beats before best_phase live in bar 0
    # and the first full bar starts at bar 1 (or bar 0 when best_phase == 0).
    bar_offset = 1 if best_phase > 0 else 0
    details = []
    for i, t in enumerate(beat_times):
        beat_in_bar = ((i - best_phase) % 4) + 1
        bar_index = (i - best_phase) // 4 + bar_offset
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
