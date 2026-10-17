"""Tests for OCR over tesseract and the value-change tracker."""

from __future__ import annotations

import struct
import zlib
from dataclasses import replace
from pathlib import Path

import pytest

from media import media
from tests import fixtures

# Exactly as the brief writes them ------------------------------------------------


def test_ocr_recovers_burned_in_text(tmp_path):
    png = media.extract_frame(
        make_sample(tmp_path / "s.mp4", duration=4, text="LANJUT 42"), 2.0, None, 960
    )
    assert "LANJUT" in media.ocr_png(png).text.upper()


def test_tracking_finds_the_injected_jump():
    seq = [media.OcrResult(i * 0.5, "Nilai 80", 0.9) for i in range(4)]
    seq.append(media.OcrResult(2.0, "Nilai 48", 0.9))
    seq += [media.OcrResult(2.5 + i * 0.5, "Nilai 80", 0.9) for i in range(3)]
    found = media.track_values(seq)
    assert len(found) == 1
    assert found[0].timestamp == 2.0
    assert found[0].value == "48"
    assert found[0].before == "80" and found[0].after == "80"


def test_tracking_on_text_without_numbers_returns_nothing():
    seq = [media.OcrResult(i, "tidak ada angka", 0.8) for i in range(3)]
    assert media.track_values(seq) == []


def test_missing_language_pack_lists_available_languages(tmp_path):
    src = make_sample(tmp_path / "s.mp4", duration=4, text="LANJUT 42")
    png = media.extract_frame(src, 1.0, None, 320)
    with pytest.raises(media.MediaError) as err:
        media.ocr_png(png, lang="zzz")
    assert "eng" in str(err.value)


# Resolutions of the brief --------------------------------------------------------


def test_available_languages_include_eng():
    langs = media.ocr_available_langs()
    assert langs and "eng" in langs


def test_ocr_confidence_lands_in_the_unit_interval(tmp_path):
    png = media.extract_frame(
        make_sample(tmp_path / "s.mp4", duration=4, text="LANJUT 42"), 2.0, None, 960
    )
    result = media.ocr_png(png)
    assert "42" in result.text
    assert 0.0 < result.confidence <= 1.0


def test_featureless_white_frame_yields_empty_text_and_zero_confidence():
    result = media.ocr_png(_solid_png(320, 180, (255, 255, 255)))
    assert result.text == ""
    assert result.confidence == 0.0


def test_step_change_and_sequence_edges_yield_no_candidate():
    def seq(values):
        return [
            media.OcrResult(i * 0.5, f"Nilai {value}", 0.9)
            for i, value in enumerate(values)
        ]

    # A held step change equals its next neighbour, so it is no flicker.
    assert media.track_values(seq(["80", "48", "48", "48"])) == []
    # The ends are only ever neighbours, never candidates: a difference that
    # opens the sequence and one that closes it surface nothing.
    assert media.track_values(seq(["48", "80", "80", "80"])) == []
    assert media.track_values(seq(["80", "80", "80", "48"])) == []
    # With a genuine interior glitch, the ends stay out of the result.
    found = media.track_values(seq(["80", "80", "48", "80", "80"]))
    assert [(candidate.timestamp, candidate.value) for candidate in found] == [
        (1.0, "48")
    ]


def test_decimal_token_is_extracted():
    seq = [media.OcrResult(i * 0.5, "Nilai 12.5", 0.9) for i in range(3)]
    seq[1] = media.OcrResult(0.5, "Nilai 8.25", 0.9)
    found = media.track_values(seq)
    assert len(found) == 1
    assert found[0].value == "8.25"
    assert found[0].before == "12.5" and found[0].after == "12.5"


def test_value_flicker_fixture_is_found_by_real_ocr(tmp_path):
    clip = fixtures.make_value_flicker(tmp_path / "flick.mp4")
    results = []
    for timestamp in (3.5, 4.5, 5.5, 6.5):
        png = media.extract_frame(clip, timestamp, None, 960)
        # ocr_png receives only pixels; the caller stamps the real time.
        result = replace(media.ocr_png(png), timestamp=timestamp)
        assert any(ch.isdigit() for ch in result.text), (
            f"no digits OCR'd at {timestamp}s: {result.text!r}"
        )
        results.append(result)
    found = media.track_values(results)
    assert len(found) == 1
    assert found[0].timestamp == 4.5
    assert found[0].value != "80"
    assert found[0].before == "80" and found[0].after == "80"


# Helpers -------------------------------------------------------------------------


def make_sample(path: Path, **kwargs) -> Path:
    """The brief calls an unqualified `make_sample`; bind the real fixture."""
    return fixtures.make_sample(path, **kwargs)


def _solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Build a solid-color PNG with the standard library alone."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )
