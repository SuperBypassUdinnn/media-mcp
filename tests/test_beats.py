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


def _write_alternating_band_track(path: Path) -> Path:
    """Mono WAV at 120 BPM: alternating 80 Hz low kicks (even clicks) and 5 kHz
    high clicks (odd clicks), starting at t=0.0 s."""
    interval = 60.0 / 120.0
    n_clicks = 18
    # End 0.35 s after the final click so a comb grid cannot extrapolate a
    # phantom beat past the last synthesized click.
    duration = (n_clicks - 1) * interval + 0.35
    total = int(RATE * duration)
    samples = [0.0] * total
    for click in range(n_clicks):
        start = int(click * interval * RATE)
        if click % 2 == 0:
            freq, amp, click_len = 80.0, 0.95, 1600
        else:
            freq, amp, click_len = 5000.0, 0.45, 600
        for index in range(click_len):
            if start + index >= total:
                break
            envelope = math.exp(-6.0 * index / click_len)
            samples[start + index] += (
                amp * envelope * math.sin(2 * math.pi * freq * index / RATE)
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


def _write_accented_click_track(path: Path) -> tuple[Path, list[float]]:
    """16-beat click track at 120 BPM with low-accented beat 1 of each bar.

    Returns ``(path, downbeat_times)`` where ``downbeat_times`` lists the true
    timestamps of every accented beat-1 click (strong 80 Hz kick + 1 kHz click)
    while beats 2, 3, and 4 carry softer 1 kHz clicks.
    """
    interval = 60.0 / 120.0
    n_beats = 16
    duration = (n_beats - 1) * interval + 0.35
    total = int(RATE * duration)
    samples = [0.0] * total
    downbeat_times: list[float] = []
    for beat in range(n_beats):
        t = beat * interval
        start = int(t * RATE)
        if beat % 4 == 0:
            downbeat_times.append(t)
            # Strong low-end kick (80 Hz) plus crisp mid attack (1 kHz)
            for freq, amp, length in ((80.0, 0.9, 1500), (1000.0, 0.4, 500)):
                for index in range(length):
                    if start + index >= total:
                        break
                    envelope = math.exp(-6.0 * index / length)
                    samples[start + index] += (
                        amp * envelope * math.sin(2 * math.pi * freq * index / RATE)
                    )
        else:
            length = 500
            for index in range(length):
                if start + index >= total:
                    break
                envelope = math.exp(-6.0 * index / length)
                samples[start + index] += (
                    0.4 * envelope * math.sin(2 * math.pi * 1000.0 * index / RATE)
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
    return path, downbeat_times


def test_beat_details_frequency_bands(tmp_path):
    """Alternating 80 Hz (even) and 5 kHz (odd) clicks in one track map to
    ``low`` and ``high`` with non-zero strength on every beat."""
    wrapper = REPO_ROOT / "rhythm" / "rhythm.sh"
    path = _write_alternating_band_track(tmp_path / "alternating.wav")
    interval = 60.0 / 120.0
    for engine_wrapper in (wrapper, Path("/nonexistent/rhythm.sh")):
        result = media_mod.detect_beats(path, wrapper=engine_wrapper)
        details = list(result.beat_details)
        assert len(details) >= 14, (result.engine, len(details))
        bands = [detail.band for detail in details]
        assert set(bands) == {"low", "high"}, (result.engine, bands)
        for detail in details:
            assert detail.strength > 0.0, (result.engine, detail)
            click_idx = round(detail.t / interval)
            true_t = click_idx * interval
            assert abs(detail.t - true_t) <= 0.08, (result.engine, detail.t, true_t)
            expected = "low" if click_idx % 2 == 0 else "high"
            assert detail.band == expected, (
                result.engine,
                click_idx,
                detail.t,
                detail.band,
                expected,
            )


def test_downbeat_detection_4_4(tmp_path):
    """Accented beat-1 clicks align detected downbeats with true beat-1 times."""
    wrapper = REPO_ROOT / "rhythm" / "rhythm.sh"
    path, true_downbeats = _write_accented_click_track(tmp_path / "accented.wav")
    for engine_wrapper in (wrapper, Path("/nonexistent/rhythm.sh")):
        result = media_mod.detect_beats(path, wrapper=engine_wrapper)
        details = list(result.beat_details)
        assert len(details) >= 12, (result.engine, len(details))
        detected_downbeats = [d for d in details if d.is_downbeat]
        assert len(detected_downbeats) >= 3, (result.engine, detected_downbeats)
        # Every detected downbeat must land on a true accented beat-1 timestamp.
        for db in detected_downbeats:
            nearest = min(true_downbeats, key=lambda t: abs(t - db.t))
            assert abs(db.t - nearest) <= 0.07, (
                result.engine,
                db.t,
                nearest,
                true_downbeats,
            )
        # Non-downbeats must NOT land on a true accented beat-1 timestamp.
        for detail in details:
            assert detail.bar_index >= 0, (result.engine, detail)
            assert detail.beat_in_bar in {1, 2, 3, 4}, (result.engine, detail)
            assert detail.is_downbeat == (detail.beat_in_bar == 1)
            if not detail.is_downbeat:
                nearest = min(true_downbeats, key=lambda t: abs(t - detail.t))
                assert abs(detail.t - nearest) > 0.1, (
                    result.engine,
                    detail.t,
                    nearest,
                )
        # Bar index increments cleanly on consecutive downbeats.
        db_indices = [i for i, d in enumerate(details) if d.is_downbeat]
        for idx in range(1, len(db_indices)):
            assert db_indices[idx] - db_indices[idx - 1] == 4
            assert (
                details[db_indices[idx]].bar_index
                == details[db_indices[idx - 1]].bar_index + 1
            )


def test_numpy_fallback_emits_beat_details(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_RHYTHM_BIN", "/nonexistent/rhythm.sh")
    path = _write_click_track(tmp_path / "clicks.wav")
    result = media_mod.detect_beats(path, wrapper=Path("/nonexistent/rhythm.sh"))
    assert result.engine == "numpy"
    assert result.degraded is True
    details = list(result.beat_details)
    assert details, "fallback must emit beat_details"
    assert len(details) == len(result.beats)
    for detail in details:
        assert detail.band in {"low", "mid", "high"}
        assert 0.0 <= detail.strength <= 1.0
        assert 0.0 <= detail.phrase_energy <= 1.0
        assert detail.bar_index >= 0
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
    # Spec section 5: beats.json carries per-beat band/downbeat details.
    details = payload["beat_details"]
    assert isinstance(details, list)
    expected_keys = {
        "t",
        "strength",
        "band",
        "is_downbeat",
        "bar_index",
        "beat_in_bar",
        "phrase_energy",
    }
    for item in details:
        assert set(item) == expected_keys, (item, expected_keys)


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
