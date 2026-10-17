"""Tests for the synthetic fixture generator the whole suite builds on."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests import fixtures

SCENE_LINE = re.compile(
    r"lavfi\.scd\.score: (?P<score>[0-9.]+), lavfi\.scd\.time: (?P<time>[0-9.]+)"
)
VOLUME_LINE = re.compile(r"max_volume: (?P<db>-?[0-9.]+) dB")


def test_make_sample_creates_playable_video(tmp_path: Path) -> None:
    out = fixtures.make_sample(tmp_path / "sample.mp4", duration=6)
    assert out.exists() and out.stat().st_size > 0
    probe = __import__("subprocess").run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0
    assert 5.0 <= float(probe.stdout.strip()) <= 7.0


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=False, capture_output=True, text=True)


def _frame_png(source: Path, timestamp: float, out: Path) -> Path:
    done = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-update",
            "1",
            str(out),
        ]
    )
    assert done.returncode == 0, done.stderr
    return out


def _ocr(png: Path) -> str:
    done = _run(["tesseract", str(png), "stdout"])
    assert done.returncode == 0, done.stderr
    return done.stdout


def _scene_scores(source: Path, threshold: float = 10.0) -> list[tuple[float, float]]:
    done = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(source),
            "-vf",
            f"scdet=threshold={threshold}",
            "-f",
            "null",
            "-",
        ]
    )
    assert done.returncode == 0, done.stderr
    return [
        (float(match.group("time")), float(match.group("score")))
        for match in SCENE_LINE.finditer(done.stderr)
    ]


def _max_volume_db(source: Path, start: float, length: float) -> float:
    done = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-ss",
            str(start),
            "-t",
            str(length),
            "-i",
            str(source),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ]
    )
    assert done.returncode == 0, done.stderr
    match = VOLUME_LINE.search(done.stderr)
    assert match is not None, done.stderr
    return float(match.group("db"))


def _stream_start_times(source: Path) -> dict[str, float]:
    done = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,start_time",
            "-of",
            "default=nw=1",
            str(source),
        ]
    )
    assert done.returncode == 0, done.stderr
    starts: dict[str, float] = {}
    current: str | None = None
    for line in done.stdout.splitlines():
        key, _, value = line.partition("=")
        if key == "codec_type":
            current = value
        elif key == "start_time" and current is not None:
            starts[current] = float(value)
    return starts


def _stream_types(source: Path) -> list[str]:
    done = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=nw=1:nk=1",
            str(source),
        ]
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.split()


def test_make_sample_has_a_real_hard_cut_at_counter_jump_at(tmp_path: Path) -> None:
    sample = fixtures.make_sample(tmp_path / "cuts.mp4", duration=8, counter_jump_at=4)
    scores = _scene_scores(sample)
    assert scores, "a clip with a hard cut must yield at least one scored second"
    assert max(score for _, score in scores) >= 10.0
    assert any(abs(timestamp - 4.0) <= 1.0 for timestamp, _ in scores)


def test_burned_in_text_is_legible_to_ocr(tmp_path: Path) -> None:
    sample = fixtures.make_sample(tmp_path / "text.mp4", duration=4)
    png = _frame_png(sample, 2.0, tmp_path / "text.png")
    assert "LANJUT" in _ocr(png).upper()


def test_counter_value_changes_at_counter_jump_at(tmp_path: Path) -> None:
    sample = fixtures.make_sample(
        tmp_path / "counter.mp4", duration=8, counter_jump_at=4
    )
    before = _ocr(_frame_png(sample, 2.0, tmp_path / "before.png"))
    after = _ocr(_frame_png(sample, 6.0, tmp_path / "after.png"))
    assert fixtures.COUNTER_BEFORE in before
    assert fixtures.COUNTER_AFTER in after


def test_spike_is_much_louder_than_the_base_tone(tmp_path: Path) -> None:
    sample = fixtures.make_sample(tmp_path / "spike.mp4", duration=8, spike_at=4)
    base = _max_volume_db(sample, 1.0, 0.5)
    spike = _max_volume_db(sample, 4.2, 0.5)
    assert spike - base >= 12.0, f"base {base} dB, spike {spike} dB"


def test_audio_offset_delays_the_audio_stream(tmp_path: Path) -> None:
    aligned = _stream_start_times(
        fixtures.make_sample(tmp_path / "aligned.mp4", duration=6)
    )
    delayed = _stream_start_times(
        fixtures.make_sample(tmp_path / "delayed.mp4", duration=6, audio_offset=1.0)
    )
    assert abs(aligned["audio"] - aligned["video"]) < 0.5
    assert delayed["audio"] - delayed["video"] >= 0.5


def test_make_silent_video_is_motionless(tmp_path: Path) -> None:
    silent = fixtures.make_silent_video(tmp_path / "silent.mp4")
    assert "video" in _stream_types(silent)
    first = _frame_png(silent, 0.5, tmp_path / "first.png").read_bytes()
    second = _frame_png(silent, 2.5, tmp_path / "second.png").read_bytes()
    assert first == second


def test_make_audio_only_has_no_video_stream(tmp_path: Path) -> None:
    audio = fixtures.make_audio_only(tmp_path / "only.m4a")
    assert _stream_types(audio) == ["audio"]


def test_png_size_reads_the_ihdr_header(tmp_path: Path) -> None:
    png = _frame_png(
        fixtures.make_silent_video(tmp_path / "source.mp4"), 0.5, tmp_path / "x.png"
    )
    width, height = fixtures.png_size(png.read_bytes())
    assert (width, height) == fixtures.VIDEO_SIZE


def test_png_size_rejects_bytes_without_a_png_header() -> None:
    with pytest.raises(ValueError):
        fixtures.png_size(b"not a png at all")
