"""Tests for the rhythm layer: ``detect_beats`` and ``media_beats``.

The audio fixtures are click tracks synthesized with numpy and written
through the stdlib ``wave`` module — 1 kHz decaying clicks at a known BPM —
so tempo and onset assertions have exact ground truth. The librosa-engine
test runs the repository's ``rhythm/rhythm.sh`` when its virtualenv exists
and skips the librosa-specific assertions when it does not (CI builds it;
a bare checkout without ``rhythm/.venv`` still exercises the fallback).
"""

from __future__ import annotations

import asyncio
import json
import math
import subprocess
import wave
from pathlib import Path

from mcp.server.mcpserver.exceptions import ToolError

import media.server
from media import media as media_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
RATE = 22050
CLICK_SECONDS = 10.0


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
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(
            b"".join(
                int(max(-1.0, min(1.0, s)) * 32767).to_bytes(2, "little", signed=True)
                for s in pcm
            )
        )
    return path


def _expected_onsets(bpm: float = 120.0) -> list[float]:
    interval = 60.0 / bpm
    return [round(beat * interval, 3) for beat in range(int(CLICK_SECONDS / interval))]


def _onsets_match(measured: list[float], tolerance: float) -> None:
    """Every measured onset sits within `tolerance` of a true click.

    Clicks at the very start of the file are legitimately missed by onset
    detectors (there is no pre-onset context), so the count may be one
    short; every detected onset must still land on a true click.
    """
    truth = _expected_onsets()
    assert len(measured) in {len(truth) - 1, len(truth)}, (len(measured), len(truth))
    for moment in measured:
        nearest = min(truth, key=lambda t: abs(t - moment))
        assert abs(moment - nearest) < tolerance, (moment, nearest)


def test_detect_beats_librosa_engine(tmp_path):
    wrapper = REPO_ROOT / "rhythm" / "rhythm.sh"
    venv = REPO_ROOT / "rhythm" / ".venv" / "bin" / "python"
    path = _write_click_track(tmp_path / "clicks.wav")
    result = media_mod.detect_beats(path, wrapper=wrapper)
    assert result.engine in {"librosa", "numpy"}
    assert result.degraded == (result.engine == "numpy")
    _onsets_match(list(result.onsets), tolerance=0.06)
    if venv.exists():
        assert result.engine == "librosa"
        assert not result.degraded
        assert result.tempo is not None and 105 <= result.tempo <= 135
        assert result.beats, "librosa should produce a beat grid here"


def test_detect_beats_numpy_fallback_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_RHYTHM_BIN", "/nonexistent/rhythm.sh")
    path = _write_click_track(tmp_path / "clicks.wav")
    result = media_mod.detect_beats(path, wrapper=Path("/nonexistent/rhythm.sh"))
    assert result.engine == "numpy"
    assert result.degraded is True
    _onsets_match(list(result.onsets), tolerance=0.1)
    if result.tempo is not None:
        assert 100 <= result.tempo <= 140


def _write_single_band_track(path: Path, freq: float) -> Path:
    """Mono WAV: one click per beat at 120 BPM, all at `freq` (first click at t=0.5 s)."""
    total = int(RATE * CLICK_SECONDS)
    samples = [0.0] * total
    interval = 60.0 / 120.0
    click_len = 400
    for beat in range(int(CLICK_SECONDS / interval)):
        start = int((beat + 1) * interval * RATE)
        for index in range(click_len):
            if start + index >= total:
                break
            envelope = math.exp(-8.0 * index / click_len)
            samples[start + index] += (
                0.9 * envelope * math.sin(2 * math.pi * freq * index / RATE)
            )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(
            b"".join(
                int(max(-1.0, min(1.0, s)) * 32767).to_bytes(2, "little", signed=True)
                for s in samples
            )
        )
    return path


def _write_accented_click_track(path: Path) -> Path:
    """16-beat click track at 120 BPM with beat 1 of each bar accented (louder)."""
    total = int(RATE * CLICK_SECONDS)
    samples = [0.0] * total
    interval = 60.0 / 120.0
    click_len = 400
    for beat in range(int(CLICK_SECONDS / interval)):
        start = int(beat * interval * RATE)
        amp = 1.0 if beat % 4 == 0 else 0.4
        for index in range(click_len):
            if start + index >= total:
                break
            envelope = math.exp(-8.0 * index / click_len)
            samples[start + index] += (
                amp * envelope * math.sin(2 * math.pi * 1000 * index / RATE)
            )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(
            b"".join(
                int(max(-1.0, min(1.0, s)) * 32767).to_bytes(2, "little", signed=True)
                for s in samples
            )
        )
    return path


def test_beat_details_frequency_bands(tmp_path):
    """80 Hz clicks classify as ``low`` and 5 kHz clicks classify as ``high``."""
    wrapper = REPO_ROOT / "rhythm" / "rhythm.sh"
    for freq, expected in ((80.0, "low"), (5000.0, "high")):
        path = _write_single_band_track(tmp_path / f"{freq}.wav", freq)
        result = media_mod.detect_beats(path, wrapper=wrapper)
        assert result.engine in {"librosa", "numpy"}
        details = list(result.beat_details)
        assert details, "beat_details must be non-empty"
        for detail in details:
            if detail.strength > 0:
                assert detail.band == expected, (freq, detail.band, detail.t)


def test_downbeat_detection_4_4(tmp_path):
    """Accented beat-1 clicks align the 4/4 downbeat phase every four beats."""
    wrapper = REPO_ROOT / "rhythm" / "rhythm.sh"
    path = _write_accented_click_track(tmp_path / "accented.wav")
    result = media_mod.detect_beats(path, wrapper=wrapper)
    assert result.engine in {"librosa", "numpy"}
    details = list(result.beat_details)
    assert details, "beat_details must be non-empty"
    downbeats = [i for i, d in enumerate(details) if d.is_downbeat]
    assert downbeats, "at least one downbeat must be inferred"
    # Downbeats recur every four beats and beat_in_bar cycles 1..4.
    for index in range(1, len(downbeats)):
        assert downbeats[index] - downbeats[index - 1] == 4, downbeats
    for index, detail in enumerate(details):
        assert detail.beat_in_bar in {1, 2, 3, 4}
        expected_bar = (index - downbeats[0]) // 4
        assert detail.bar_index == expected_bar, (index, detail.bar_index, expected_bar)
        expected_beat = ((index - downbeats[0]) % 4) + 1
        assert detail.beat_in_bar == expected_beat, (
            index,
            detail.beat_in_bar,
            expected_beat,
        )
        assert detail.is_downbeat == (expected_beat == 1)


def test_numpy_fallback_emits_beat_details(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_RHYTHM_BIN", "/nonexistent/rhythm.sh")
    path = _write_click_track(tmp_path / "clicks.wav")
    result = media_mod.detect_beats(path, wrapper=Path("/nonexistent/rhythm.sh"))
    assert result.engine == "numpy"
    assert result.degraded is True
    details = list(result.beat_details)
    assert details, "fallback must emit beat_details"
    for detail in details:
        assert detail.band in {"low", "mid", "high"}
        assert 0.0 <= detail.strength <= 1.0
        assert 0.0 <= detail.phrase_energy <= 1.0
        assert detail.beat_in_bar in {1, 2, 3, 4}


def test_detect_beats_requires_audio(tmp_path):
    video_only = tmp_path / "silent.mp4"
    import subprocess

    done = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:size=320x240:rate=25:d=1",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_only),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    try:
        media_mod.detect_beats(video_only, wrapper=None)
    except media_mod.MediaError as error:
        assert "audio stream" in str(error)
    else:
        raise AssertionError("video-only source must raise MediaError")


def test_media_beats_tool(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    monkeypatch.setenv("MEDIA_RHYTHM_BIN", str(REPO_ROOT / "rhythm" / "rhythm.sh"))
    path = _write_click_track(tmp_path / "clicks.wav")
    result = asyncio.run(media.server.media_beats(str(path)))
    assert "# media_beats" in result
    assert f"Source: {path}" in result
    assert "Engine:" in result
    assert "Tempo:" in result
    assert "Artifact:" in result and "beats.json" in result
    beats_json = next(cache.rglob("beats.json"))
    payload = json.loads(beats_json.read_text())
    assert payload["engine"] in {"librosa", "numpy"}
    assert payload["degraded"] == (payload["engine"] == "numpy")
    _onsets_match(list(payload["onsets"]), tolerance=0.1)
    assert payload["source"] == str(path)
    # Spec section 4: beats.json carries a confidence field (beat-onset
    # agreement); the click track's grid should agree with itself strongly.
    assert "confidence" in payload
    if payload["beats"]:
        assert payload["confidence"] is None or payload["confidence"] >= 0.5


def test_media_frames_at_beats(tmp_path, monkeypatch):
    """media_frames(at='beats') rasterizes the cached grid; missing beats
    is an error naming media_beats, never an implicit decode."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    monkeypatch.setenv("MEDIA_RHYTHM_BIN", str(REPO_ROOT / "rhythm" / "rhythm.sh"))
    video = tmp_path / "v.mp4"
    clicks = tmp_path / "clicks.wav"
    _write_click_track(clicks, bpm=120.0)
    done = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:size=320x240:rate=25:d=10",
            "-i",
            str(clicks),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    try:
        asyncio.run(media.server.media_frames(str(video), at="beats"))
    except ToolError as error:
        assert "media_beats" in str(error)
    else:
        raise AssertionError("at='beats' without beats.json must error naming it")
    asyncio.run(media.server.media_beats(str(video)))
    result = asyncio.run(media.server.media_frames(str(video), at="beats"))
    assert "# media_frames" in result
    assert "Timestamps:" in result
    pngs = list(cache.rglob("frames/*.png"))
    assert pngs, result
