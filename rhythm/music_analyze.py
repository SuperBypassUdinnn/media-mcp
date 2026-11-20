#!/usr/bin/env python3
"""Music-core analysis for one media file: key, HPSS character, arrangement.

The librosa-based engine of the media server's ``media_music`` tool. Reads the
media path from argv[1], decodes mono audio through librosa (ffmpeg pass into a
temp wav first, same decode block as ``onsets.py``), and prints one JSON object
to stdout:

    {"engine": "librosa", "degraded": false, "duration": 6.0,
     "key": "C major", "key_confidence": 0.42, "mode": "major",
     "harmonic_ratio": 0.61, "transient_density": 2.0, "onset_density": 2.0,
     "flux": 0.34, "low_band_ratio": 0.12, "mood_tags": ["cheerful"],
     "arrangement": [{"start": 0.0, "end": 4.0, "level": "low", "label": "intro"}],
     "loopiness": 0.18}

Any failure prints nothing on stdout and exits non-zero — the caller treats a
non-zero exit or unparseable stdout as "this engine is not available" and falls
back to the pure-numpy baseline.
"""

from __future__ import annotations

import json
import math
import sys
import warnings

import numpy as np


def _major_profiles() -> list[tuple[str, str, list[float]]]:
    """Krumhansl-Kessler major / minor key profiles (12 pitch classes each)."""
    major_template = [
        6.35,
        2.23,
        3.48,
        2.33,
        4.38,
        4.09,
        2.59,
        5.19,
        2.39,
        3.66,
        2.29,
        2.86,
    ]
    minor_template = [
        6.33,
        2.68,
        3.52,
        5.38,
        2.60,
        3.53,
        2.54,
        4.75,
        3.98,
        2.69,
        3.34,
        3.17,
    ]
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    profiles: list[tuple[str, str, list[float]]] = []
    for i in range(12):
        rotated_major = major_template[i:] + major_template[:i]
        rotated_minor = minor_template[i:] + minor_template[:i]
        profiles.append((names[i], "major", rotated_major))
        profiles.append((names[i], "minor", rotated_minor))
    return profiles


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((a * b).sum()) / denom


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: music_analyze.py <media-path>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    # librosa/numba emit benign JIT and resample warnings; they are noise on a
    # stdout that must stay pure JSON, so they are suppressed wholesale.
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
            wav = Path(scratch) / "music-in.wav"
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
        if duration <= 0.0 or float(np.abs(y).max()) <= 1e-12:
            # Digital silence: a valid but empty payload. key is None,
            # harmonic_ratio 0.0, no mood tags, nothing NaN.
            payload = {
                "engine": "librosa",
                "degraded": False,
                "duration": round(duration, 3),
                "key": None,
                "key_confidence": None,
                "mode": None,
                "harmonic_ratio": 0.0,
                "transient_density": 0.0,
                "onset_density": 0.0,
                "flux": 0.0,
                "low_band_ratio": 0.0,
                "mood_tags": [],
                "arrangement": [],
                "loopiness": 0.0,
            }
            print(json.dumps(payload))
            return 0

        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        mag = np.abs(stft)
        phase = np.angle(stft)
        # HPSS on the STFT magnitude -> harmonic / percussive magnitudes.
        harmonic_mag, percussive_mag = librosa.decompose.hpss(mag)
        harmonic_ratio = float(
            (harmonic_mag**2).sum()
            / ((harmonic_mag**2).sum() + (percussive_mag**2).sum())
            if (harmonic_mag**2).sum() + (percussive_mag**2).sum() > 0
            else 0.0
        )

        # Reconstruct harmonic audio (original phase) for chroma key detection.
        harmonic_audio = librosa.istft(
            harmonic_mag * np.exp(1j * phase), hop_length=512
        )

        # Key detection: mean chroma of the harmonic part vs KS profiles.
        chroma = librosa.feature.chroma_cqt(y=harmonic_audio, sr=sr)
        mean_chroma = chroma.mean(axis=1)
        profiles = _major_profiles()
        correlations = [
            _correlate(mean_chroma, np.array(prof, dtype=np.float64))
            for _, _, prof in profiles
        ]
        best_idx = int(np.argmax(correlations))
        best_name, best_mode, _ = profiles[best_idx]
        best_corr = float(correlations[best_idx])
        # Normalize peak correlation into [0, 1] via min-max over profiles.
        c_min, c_max = float(min(correlations)), float(max(correlations))
        confidence = (best_corr - c_min) / (c_max - c_min) if c_max > c_min else 0.0
        if float(np.abs(mean_chroma).max()) <= 1e-9:
            key_value = None
            mode_value = None
            confidence_value = None
        else:
            key_value = f"{best_name} {best_mode}"
            mode_value = best_mode
            confidence_value = round(confidence, 3)

        # Transient / onset density.
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        onset_list = [float(o) for o in onsets]
        transient_density = len(onset_list) / duration
        onset_density = transient_density

        # Character profile: spectral flux + low-band ratio.
        flux_frames = librosa.onset.onset_strength(y=y, sr=sr)
        flux = (
            float(np.mean(np.maximum(0.0, np.diff(flux_frames))))
            if len(flux_frames) > 1
            else 0.0
        )
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        low_mask = (freqs >= 40.0) & (freqs <= 160.0)
        total_energy = float((mag**2).sum())
        low_energy = float((mag[low_mask] ** 2).sum()) if low_mask.any() else 0.0
        low_band_ratio = low_energy / total_energy if total_energy > 0 else 0.0

        # Mood tags (rule-based, ordered).
        mood_tags: list[str] = []
        if transient_density >= 4.0 and flux >= 1.0 and 0.3 <= low_band_ratio <= 0.8:
            mood_tags.append("upbeat")
        if mode_value == "major":
            mood_tags.append("cheerful")
        elif mode_value == "minor":
            mood_tags.append("moody")
        if transient_density < 2.0:
            mood_tags.append("ambient")

        # Arrangement: RMS energy over 1-second windows, thresholded at 60% /
        # 130% of the track median.
        hop_length = 512
        n_win = max(1, math.ceil(duration))
        rms = np.array(
            [
                float(np.sqrt(np.mean(y[i * sr : min((i + 1) * sr, len(y))] ** 2)))
                if min((i + 1) * sr, len(y)) > i * sr
                else 0.0
                for i in range(n_win)
            ]
        )
        median_rms = float(np.median(rms))
        low_thresh = 0.60 * median_rms
        high_thresh = 1.30 * median_rms
        levels = []
        for value in rms:
            if value < low_thresh:
                levels.append("low")
            elif value > high_thresh:
                levels.append("high")
            else:
                levels.append("mid")

        arrangement: list[dict] = []
        start = 0.0
        prev_level = levels[0]
        for i in range(1, n_win + 1):
            level = levels[i] if i < n_win else None
            if level is None or level != prev_level:
                end = i
                label = _arrangement_label(prev_level, arrangement, start, end)
                arrangement.append(
                    {
                        "start": round(float(start), 3),
                        "end": round(float(end), 3),
                        "level": prev_level,
                        "label": label,
                    }
                )
                start = float(end)
                prev_level = level

        # Loopiness: mean cosine similarity of each 1-s chroma vector against
        # the 1-bar-lagged vector. Beat-derived bar length; default 2 s at 120 BPM.
        chroma_win = librosa.feature.chroma_cqt(y=y, sr=sr)
        n_chroma_win = chroma_win.shape[1]
        bar_seconds = 2.0
        bar_frames = max(1, int(bar_seconds * sr / hop_length))
        sims: list[float] = []
        for i in range(n_chroma_win):
            j = i - bar_frames
            if j < 0:
                continue
            a = chroma_win[:, i]
            b = chroma_win[:, j]
            denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
            if denom <= 1e-12:
                continue
            sims.append(float(np.dot(a, b) / denom))
        loopiness = float(np.mean(sims)) if sims else 0.0

    payload = {
        "engine": "librosa",
        "degraded": False,
        "duration": round(duration, 3),
        "key": key_value,
        "key_confidence": confidence_value,
        "mode": mode_value,
        "harmonic_ratio": round(harmonic_ratio, 3),
        "transient_density": round(transient_density, 3),
        "onset_density": round(onset_density, 3),
        "flux": round(flux, 3),
        "low_band_ratio": round(low_band_ratio, 3),
        "mood_tags": mood_tags,
        "arrangement": arrangement,
        "loopiness": round(loopiness, 3),
    }
    print(json.dumps(payload))
    return 0


def _arrangement_label(
    level: str, arrangement: list[dict], start: float, end: float
) -> str:
    if level == "high":
        return "body"
    if level == "low":
        # First low section -> intro; low after a high -> breakdown.
        if any(item["level"] == "high" for item in arrangement):
            return "breakdown"
        return "intro"
    return "mid"


if __name__ == "__main__":
    raise SystemExit(main())
