"""Tests for the timeline projection and the pure search over it."""

from __future__ import annotations

import pytest

from media import artifacts


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
    # Twenty-four matches across the ocr layer plus one speech match: k=99
    # must collapse to the hard cap of 20 hits, ordered score-descending.
    events = [
        artifacts.TimelineEvent(float(i), "ocr", f"slider value {i}") for i in range(24)
    ]
    events.insert(3, artifacts.TimelineEvent(3.0, "speech", "slider intro"))
    artifacts.write_timeline(entry, events)
    hits = artifacts.search(entry, "slider", k=99)
    assert len(hits) <= 20
    assert hits[0].score >= hits[-1].score
    assert [hit.timestamp for hit in hits] == sorted(hit.timestamp for hit in hits)
    assert artifacts.search(entry, "slider", scope="ocr")[0].layer == "ocr"


def test_search_is_case_insensitive(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    artifacts.write_timeline(
        entry, [artifacts.TimelineEvent(4.0, "ocr", "Slider Panel 42")]
    )
    assert artifacts.search(entry, "SLIDER panel")[0].timestamp == 4.0


def test_search_ranks_full_overlap_above_partial_overlap(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    artifacts.write_timeline(
        entry,
        [
            artifacts.TimelineEvent(10.0, "ocr", "slider panel appears"),
            artifacts.TimelineEvent(20.0, "speech", "slider mentioned once"),
        ],
    )
    hits = artifacts.search(entry, "slider panel")
    assert len(hits) == 2
    assert hits[0].timestamp == 10.0
    assert hits[0].score > hits[-1].score


def test_search_is_deterministic(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    artifacts.write_timeline(
        entry,
        [
            artifacts.TimelineEvent(1.0, "ocr", "slider value 80"),
            artifacts.TimelineEvent(2.0, "speech", "slider bergerak"),
            artifacts.TimelineEvent(3.0, "ocr", "slider tidak bergerak"),
        ],
    )
    assert artifacts.search(entry, "slider") == artifacts.search(entry, "slider")


def test_search_without_timeline_returns_no_hits(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    assert artifacts.search(entry, "slider") == []


def test_search_skips_malformed_lines_without_corrupting_ranking(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    artifacts.write_timeline(entry, [artifacts.TimelineEvent(1.0, "ocr", "slider top")])
    path = entry / "timeline.md"
    path.write_text(
        "not a timeline line\n[garbage} scene | junk\n" + path.read_text(),
        encoding="utf-8",
    )
    hits = artifacts.search(entry, "slider")
    assert len(hits) == 1
    assert hits[0].timestamp == 1.0
    assert hits[0].score == 1.0
    assert hits[0].layer == "ocr" and hits[0].text == "slider top"


def test_unknown_scope_is_rejected(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    with pytest.raises(ValueError) as err:
        artifacts.search(entry, "slider", scope="frames")
    assert "scope" in str(err.value)


def test_empty_k_returns_no_hits(tmp_path):
    entry = tmp_path / "e"
    entry.mkdir()
    artifacts.write_timeline(entry, [artifacts.TimelineEvent(1.0, "ocr", "slider")])
    assert artifacts.search(entry, "slider", k=0) == []
