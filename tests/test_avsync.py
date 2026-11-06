"""Tests for the alignment layer: ``align_cuts_to_onsets`` and
``media_avsync``.

The tool-level fixture is an AMV in miniature: four one-second color
segments concatenated into hard cuts at exactly 1.0/2.0/3.0 s, muxed with a
60 BPM click track whose clicks land on the same boundaries — so "synced"
has exact ground truth. The pure tests cover the verdict rules, the A/V
offset correction, and the empty-input degenerate case.
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


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(argv, check=False, capture_output=True, text=True)
    if done.returncode != 0:
        raise AssertionError(f"fixture ffmpeg failed: {done.stderr}")
    return done


def _write_beat_clicks(path: Path, bpm: float = 60.0, seconds: float = 4.0) -> None:
    """Mono 16-bit WAV: one decaying 1 kHz click per beat."""
    interval = 60.0 / bpm
    total = int(RATE * seconds)
    pcm = [0.0] * total
    click_len = 400
    beat_times = [b * interval for b in range(int(seconds / interval) + 1)]
    for beat in beat_times:
        start = int(beat * RATE)
        for index in range(click_len):
            if start + index < total:
                envelope = math.exp(-8.0 * index / click_len)
                pcm[start + index] += (
                    0.8 * envelope * math.sin(2 * math.pi * 1000 * index / RATE)
                )
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


def _write_cut_on_beats_video(path: Path, tmp: Path) -> Path:
    """Four 1 s color segments (cuts at 1/2/3 s) + clicks on the same times."""
    colors = ["red", "green", "blue", "yellow"]
    wav = tmp / "clicks.wav"
    _write_beat_clicks(wav)
    argv = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    for color in colors:
        argv += [
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:size=320x240:rate=25:d=1",
        ]
    argv += ["-i", str(wav)]
    argv += [
        "-filter_complex",
        "".join(f"[{i}:v]" for i in range(len(colors)))
        + f"concat=n={len(colors)}:v=1:a=0[v]",
        "-map",
        "[v]",
        "-map",
        f"{len(colors)}:a",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(path),
    ]
    _run(argv)
    return path


def test_align_synced_cuts():
    cuts = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    beats = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    result = media_mod.align_cuts_to_onsets(cuts, cuts, duration=4.0, beats=beats)
    assert result.verdict == "synced"
    assert result.share_within_tolerance == 1.0
    assert result.median_distance_s == 0.0
    assert result.baseline_share is not None and result.baseline_share < 0.5
    assert result.beat_phase is not None and result.beat_phase < 0.05


def test_align_offset_correction():
    # Onsets live in audio time; the container shows them 0.4 s later.
    result = media_mod.align_cuts_to_onsets(
        [0.9, 1.4, 1.9], [0.5, 1.0, 1.5], duration=3.0, offset=0.4, tolerance=0.1
    )
    assert result.verdict == "synced"
    assert result.share_within_tolerance == 1.0
    assert result.offset_s == 0.4


def test_align_desynced_cuts():
    result = media_mod.align_cuts_to_onsets(
        [0.13, 1.77, 2.31], [0.5, 1.0, 1.5, 2.0, 2.5], duration=3.0
    )
    assert result.verdict == "unsynced"
    assert result.share_within_tolerance == 0.0


def test_align_empty_inputs():
    result = media_mod.align_cuts_to_onsets([], [0.5], duration=3.0)
    assert result.verdict == "inconclusive"
    assert result.share_within_tolerance is None
    assert result.baseline_share is None


def test_media_avsync_tool(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    monkeypatch.setenv("MEDIA_RHYTHM_BIN", str(REPO_ROOT / "rhythm" / "rhythm.sh"))
    path = _write_cut_on_beats_video(tmp_path / "amv.mp4", tmp_path)
    asyncio.run(media.server.media_beats(str(path)))
    result = asyncio.run(media.server.media_avsync(str(path)))
    assert "# media_avsync" in result
    assert "Verdict: synced" in result, result
    assert "Artifact:" in result and "sync.json" in result
    sync_json = next(cache.rglob("sync.json"))
    payload = json.loads(sync_json.read_text())
    assert payload["verdict"] == "synced"
    assert payload["cuts"] >= 2, payload
    assert payload["share_within_tolerance"] >= 0.5


def test_media_avsync_joins_burst_starts(tmp_path, monkeypatch):
    """Spec section 5: motion-burst starts join the scdet cuts.

    A crossfade has no scdet cut; its motion burst start is the only visual
    event, so alignment must see it once media_motion has run.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    monkeypatch.setenv("MEDIA_RHYTHM_BIN", str(REPO_ROOT / "rhythm" / "rhythm.sh"))
    path = _write_cut_on_beats_video(tmp_path / "amv.mp4", tmp_path)
    asyncio.run(media.server.media_beats(str(path)))
    asyncio.run(media.server.media_motion(str(path)))
    result = asyncio.run(media.server.media_avsync(str(path)))
    assert "motion.json bursts" in result, result
    # With motion present the comparison set can only grow, never shrink.
    sync_json = next(cache.rglob("sync.json"))
    payload = json.loads(sync_json.read_text())
    assert payload["cuts"] >= 2, payload


def test_media_avsync_requires_beats(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    video = tmp_path / "v.mp4"
    _run(
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
            "color=c=gray:size=320x240:rate=25:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ]
    )
    try:
        asyncio.run(media.server.media_avsync(str(video)))
    except ToolError as error:
        assert "media_beats" in str(error)
    else:
        raise AssertionError("missing beats.json must raise ToolError naming it")
