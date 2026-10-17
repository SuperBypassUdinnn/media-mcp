"""Tests for the timeline projection and the pure search over cached artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from media import artifacts


def _write_transcript(entry: Path, segments: list[dict[str, object]]) -> None:
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "transcript.json").write_text(
        json.dumps({"meta": {}, "segments": segments}), encoding="utf-8"
    )


def _write_ocr(entry: Path, stamp: str, timestamp: float, text: str) -> None:
    ocr_dir = entry / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (ocr_dir / f"{stamp}.json").write_text(
        json.dumps({"timestamp": timestamp, "text": text, "confidence": 0.9}),
        encoding="utf-8",
    )


def test_timeline_is_sorted_and_greppable(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    artifacts.write_timeline(
        entry,
        [
            artifacts.TimelineEvent(12.5, "speech", "halo semua"),
            artifacts.TimelineEvent(0.5, "scene", "cut"),
        ],
    )
    lines = (entry / "timeline.md").read_text().splitlines()
    assert lines[0] == "[00:00:00.500] scene | cut"
    assert lines[1] == "[00:00:12.500] speech | halo semua"


def test_format_timestamp_is_uniform():
    assert artifacts.format_timestamp(0.5) == "00:00:00.500"
    assert artifacts.format_timestamp(12.5) == "00:00:12.500"
    assert artifacts.format_timestamp(3661.2) == "01:01:01.200"


def test_timeline_rewrite_is_deterministic(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    events = [
        artifacts.TimelineEvent(2.0, "audio", "loud spike"),
        artifacts.TimelineEvent(1.0, "scene", "cut"),
    ]
    artifacts.write_timeline(entry, events)
    artifacts.write_timeline(entry, list(reversed(events)))
    first = (entry / "timeline.md").read_bytes()
    artifacts.write_timeline(entry, events)
    second = (entry / "timeline.md").read_bytes()
    assert first == second


def test_unknown_layer_is_rejected_not_written(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    artifacts.write_timeline(entry, [artifacts.TimelineEvent(1.0, "scene", "cut")])
    before = (entry / "timeline.md").read_bytes()
    with pytest.raises(ValueError) as err:
        artifacts.TimelineEvent(2.0, "visual", "zoom")
    assert "visual" in str(err.value)
    assert (entry / "timeline.md").read_bytes() == before


def test_search_scores_and_caps_k(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    # Twenty-four matching transcript segments plus two matching OCR records:
    # k=99 must collapse to the hard cap of 20 hits, ordered score-descending.
    _write_transcript(
        entry,
        [
            {
                "start": float(i),
                "end": float(i) + 1,
                "speaker": "SPEAKER_00",
                "text": f"slider intro {i}",
            }
            for i in range(24)
        ],
    )
    _write_ocr(entry, "030", 30.0, "slider panel value 80")
    _write_ocr(entry, "031", 31.0, "slider panel value 48")
    hits = artifacts.search(entry, "slider", k=99)
    assert len(hits) <= 20
    assert hits[0].score >= hits[-1].score
    assert [hit.timestamp for hit in hits] == sorted(hit.timestamp for hit in hits)
    assert artifacts.search(entry, "slider", scope="ocr")[0].layer == "ocr"


def test_transcript_scope_reads_transcript_segments(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    _write_transcript(
        entry,
        [
            {
                "start": 12.5,
                "end": 14.0,
                "speaker": "SPEAKER_01",
                "text": "slider bergerak",
            },
            {"start": 20.0, "end": 21.0, "speaker": "SPEAKER_01", "text": "yang lain"},
        ],
    )
    _write_ocr(entry, "010", 10.0, "slider panel")
    hits = artifacts.search(entry, "slider", scope="transcript")
    assert len(hits) == 1
    assert hits[0].layer == "speech"
    assert hits[0].timestamp == 12.5
    assert hits[0].text == "SPEAKER_01: slider bergerak"


def test_transcript_without_speaker_still_searches(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    _write_transcript(entry, [{"start": 5.0, "end": 6.0, "text": "slider halus"}])
    hits = artifacts.search(entry, "slider", scope="transcript")
    assert len(hits) == 1
    assert hits[0].text == "slider halus"


def test_ocr_scope_reads_ocr_records(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    _write_ocr(entry, "006", 6.25, "slider panel 80")
    _write_ocr(entry, "012", 12.5, "slider panel 48")
    _write_transcript(entry, [{"start": 3.0, "end": 4.0, "text": "slider suara"}])
    hits = artifacts.search(entry, "slider", scope="ocr")
    assert len(hits) == 2
    assert {hit.layer for hit in hits} == {"ocr"}
    assert [hit.timestamp for hit in hits] == [6.25, 12.5]
    assert hits[0].text == "slider panel 80"


def test_all_scope_unions_both_sources_ranked_together(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    _write_transcript(
        entry,
        [
            {
                "start": 20.0,
                "end": 21.0,
                "speaker": "SPEAKER_00",
                "text": "slider muncul sekali",
            },
            {
                "start": 21.0,
                "end": 22.0,
                "speaker": "SPEAKER_00",
                "text": "tanpa topik",
            },
        ],
    )
    _write_ocr(entry, "005", 5.0, "slider panel dua kali")
    hits = artifacts.search(entry, "slider panel")
    assert len(hits) == 2
    # The OCR record carries the full token overlap, the segment only half of
    # it; the score orders them across sources, not the collection order.
    assert hits[0].timestamp == 5.0
    assert hits[0].layer == "ocr" and hits[0].score == 1.0
    assert hits[1].layer == "speech" and hits[1].score == 0.5


def test_search_is_case_insensitive(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    _write_transcript(
        entry,
        [
            {
                "start": 4.0,
                "end": 5.0,
                "speaker": "SPEAKER_00",
                "text": "Slider Panel 42",
            },
        ],
    )
    hits = artifacts.search(entry, "SLIDER panel", scope="transcript")
    assert len(hits) == 1
    assert hits[0].score == 1.0


def test_search_missing_artifacts_return_no_hits(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    assert artifacts.search(entry, "slider") == []
    assert artifacts.search(entry, "slider", scope="transcript") == []
    assert artifacts.search(entry, "slider", scope="ocr") == []


def test_malformed_transcript_does_not_suppress_ocr_hits(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    (entry / "transcript.json").write_text("{not json at all", encoding="utf-8")
    _write_ocr(entry, "010", 10.0, "slider tahan")
    hits = artifacts.search(entry, "slider")
    assert len(hits) == 1
    assert hits[0].layer == "ocr" and hits[0].text == "slider tahan"


def test_malformed_ocr_file_does_not_suppress_transcript_hits(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    _write_transcript(
        entry,
        [
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "slider lepas"},
        ],
    )
    _write_ocr(entry, "010", 10.0, "slider panel")
    (entry / "ocr" / "011.json").write_text("[broken", encoding="utf-8")
    hits = artifacts.search(entry, "slider")
    # The malformed sibling file is skipped and counted; both healthy
    # artifacts still surface their hits.
    assert len(hits) == 2
    assert [hit.layer for hit in hits] == ["speech", "ocr"]
    assert hits[0].text == "SPEAKER_00: slider lepas"
    assert hits[1].text == "slider panel"


def test_search_ranks_full_overlap_above_partial_overlap(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    _write_transcript(
        entry,
        [
            {"start": 10.0, "end": 11.0, "text": "slider panel appears"},
            {"start": 20.0, "end": 21.0, "text": "slider mentioned once"},
        ],
    )
    hits = artifacts.search(entry, "slider panel", scope="transcript")
    assert len(hits) == 2
    assert hits[0].timestamp == 10.0
    assert hits[0].score > hits[-1].score


def test_search_is_deterministic(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    _write_transcript(
        entry,
        [
            {
                "start": 1.0,
                "end": 2.0,
                "speaker": "SPEAKER_00",
                "text": "slider value 80",
            },
            {
                "start": 2.0,
                "end": 3.0,
                "speaker": "SPEAKER_00",
                "text": "slider bergerak",
            },
        ],
    )
    _write_ocr(entry, "004", 4.0, "slider tidak bergerak")
    assert artifacts.search(entry, "slider") == artifacts.search(entry, "slider")


def test_unknown_scope_is_rejected(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    with pytest.raises(ValueError) as err:
        artifacts.search(entry, "slider", scope="frames")
    assert "scope" in str(err.value)


def test_empty_k_returns_no_hits(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    _write_transcript(
        entry,
        [
            {"start": 1.0, "end": 2.0, "text": "slider"},
        ],
    )
    assert artifacts.search(entry, "slider", k=0) == []
