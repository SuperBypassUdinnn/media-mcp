"""Tests for the librosa music-core engine: ``rhythm/music.sh``.

The audio fixtures are synthesized numpy audio written through the stdlib
``wave`` module so they encode exactly once and decode identically everywhere:
- a 6 s C-major triad pad (sines at 261.63 / 329.63 / 392.00 Hz),
- a 10 s 120 BPM click track,
- 3 s of digital silence,
- a click track followed by 4 s of silence.

The librosa engine test runs the repository's ``rhythm/music.sh`` when its
virtualenv exists and skips the librosa-specific assertions when it does not.
"""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RATE = 22050
CLICK_SECONDS = 10.0


def _write_wav(path: Path, samples: list[float], rate: int = RATE) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(
            b"".join(
                int(max(-1.0, min(1.0, s)) * 32767).to_bytes(2, "little", signed=True)
                for s in samples
            )
        )
    return path


def _write_tonal_bed(path: Path) -> Path:
    """6 s C-major triad pad: C4/E4/G4 sines at equal amplitude, slow attack."""
    total = int(RATE * 6.0)
    samples = [0.0] * total
    for freq in (261.63, 329.63, 392.00):
        for index in range(total):
            t = index / RATE
            amp = min(1.0, t / 0.5)
            samples[index] += (1.0 / 3.0) * amp * math.sin(2 * math.pi * freq * t)
    return _write_wav(path, samples)


def _write_click_track(path: Path, bpm: float = 120.0) -> Path:
    """A mono 16-bit WAV: one decaying 1 kHz click per beat at `bpm`."""
    interval = 60.0 / bpm
    total = int(RATE * CLICK_SECONDS)
    samples = [0.0] * total
    click_len = 400
    for index in range(click_len):
        envelope = math.exp(-8.0 * index / click_len)
        samples[index] = 0.8 * envelope * math.sin(2 * math.pi * 1000 * index / RATE)
    beats = [
        int(beat * interval * RATE) for beat in range(int(CLICK_SECONDS / interval))
    ]
    pcm = [0.0] * total
    for start in beats:
        for index in range(click_len):
            if start + index < total:
                pcm[start + index] += samples[index]
    return _write_wav(path, pcm)


def _write_silence(path: Path, seconds: float = 3.0) -> Path:
    return _write_wav(path, [0.0] * int(RATE * seconds))


def _write_click_plus_silence(path: Path, bpm: float = 120.0) -> Path:
    """Click track for 5 s then 4 s of silence (9 s total)."""
    interval = 60.0 / bpm
    click_seconds = 5.0
    total = int(RATE * (click_seconds + 4.0))
    samples = [0.0] * total
    click_len = 400
    for index in range(click_len):
        envelope = math.exp(-8.0 * index / click_len)
        samples[index] = 0.8 * envelope * math.sin(2 * math.pi * 1000 * index / RATE)
    beats = [
        int(beat * interval * RATE) for beat in range(int(click_seconds / interval))
    ]
    pcm = [0.0] * total
    for start in beats:
        for index in range(click_len):
            if start + index < total:
                pcm[start + index] += samples[index]
    return _write_wav(path, pcm)


def _run_music_sh(wrapper: Path, media_path: Path) -> dict:
    import subprocess

    done = subprocess.run(
        [str(wrapper), str(media_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _no_na(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_no_na(v) for v in value.values())
    if isinstance(value, list):
        return all(_no_na(v) for v in value)
    return True


wrapper = REPO_ROOT / "rhythm" / "music.sh"
venv = REPO_ROOT / "rhythm" / ".venv" / "bin" / "python"


@pytest.mark.skipif(not venv.exists(), reason="rhythm/.venv not built")
def test_music_engine_tonal_bed_key_detection(tmp_path):
    path = _write_tonal_bed(tmp_path / "pad.wav")
    payload = _run_music_sh(wrapper, path)
    assert payload["engine"] == "librosa"
    assert payload["degraded"] is False
    assert payload["key"] == "C major" or (
        payload["mode"] == "major" and payload["key"] and payload["key"].startswith("C")
    ), payload
    assert 0.3 <= payload["key_confidence"] <= 1.0, payload
    assert payload["harmonic_ratio"] >= 0.55, payload


@pytest.mark.skipif(not venv.exists(), reason="rhythm/.venv not built")
def test_music_engine_transient_density(tmp_path):
    path = _write_click_track(tmp_path / "clicks.wav")
    payload = _run_music_sh(wrapper, path)
    assert 1.6 <= payload["transient_density"] <= 2.4, payload
    assert 1.6 <= payload["onset_density"] <= 2.4, payload


@pytest.mark.skipif(not venv.exists(), reason="rhythm/.venv not built")
def test_music_engine_silence_payload_valid(tmp_path):
    path = _write_silence(tmp_path / "silence.wav")
    payload = _run_music_sh(wrapper, path)
    assert payload["engine"] == "librosa"
    assert payload["degraded"] is False
    assert payload["key"] is None, payload
    assert payload["harmonic_ratio"] == 0.0, payload
    assert payload["mood_tags"] == [], payload
    assert _no_na(payload), payload


def _write_white_noise(path: Path, seconds: float = 2.0) -> Path:
    """2 s of band-limited white noise (no tonal content)."""
    rng = np.random.default_rng(0)
    total = int(RATE * seconds)
    samples = [float(s) for s in rng.standard_normal(total)]
    return _write_wav(path, samples)


@pytest.mark.skipif(not venv.exists(), reason="rhythm/.venv not built")
def test_music_engine_noise_key_confidence_low(tmp_path):
    path = _write_white_noise(tmp_path / "noise.wav")
    payload = _run_music_sh(wrapper, path)
    assert payload["engine"] == "librosa"
    assert payload["degraded"] is False
    # White noise has no tonal center: either no key, or a low confidence.
    assert payload["key"] is None or payload["key_confidence"] < 0.5, payload


@pytest.mark.skipif(not venv.exists(), reason="rhythm/.venv not built")
def test_music_engine_arrangement_low_high(tmp_path):
    path = _write_click_plus_silence(tmp_path / "clicks_then_silence.wav")
    payload = _run_music_sh(wrapper, path)
    arrangement = payload["arrangement"]
    assert len(arrangement) >= 2, payload
    assert arrangement[0]["start"] == 0.0, payload
    # The appended silence must be detected as a low section after the clicks.
    low_sections = [s for s in arrangement if s["level"] == "low"]
    assert low_sections, payload
    assert low_sections[-1]["end"] == round(payload["duration"], 3), payload
