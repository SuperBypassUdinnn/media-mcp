"""Tests for scene detection, candidate sampling, and ranking."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from media import media
from tests import fixtures

# Exactly as the brief writes them ------------------------------------------------


def test_scene_scores_are_parsed_from_a_real_clip(tmp_path):
    sample = fixtures.make_sample(tmp_path / "cuts.mp4", duration=8, counter_jump_at=4)
    scores = media.detect_scenes(sample)
    assert scores, "a clip with a hard cut must yield at least one scored second"
    assert max(s.score for s in scores) >= 10.0, (
        "the hard cut must clear the default threshold"
    )
    assert all(s.timestamp >= 0.0 for s in scores)


def test_dedup_drops_near_identical_candidates():
    same = media.diff_hash_of(b"identical")
    kept = media.sample_candidates(
        [], duration=10, interval=1, dedup_distance=6, extract=lambda t: same
    )
    assert len(kept) == 1


def test_ranking_puts_scene_boundaries_first(tmp_path):
    cands = [media.Candidate(1.0, "fill", 0.0), media.Candidate(2.0, "scene", 0.9)]
    ranked, accounting = media.rank_candidates(cands, [], budget=1)
    assert ranked[0].reason == "scene"
    assert accounting["dropped"] == 1


def test_budget_accounting_is_complete(tmp_path):
    cands = [media.Candidate(float(i), "interval", 0.0) for i in range(10)]
    _, accounting = media.rank_candidates(cands, [], budget=4)
    assert sum(accounting[k] for k in ("scene", "audio", "interval", "fill")) == 4
    assert accounting["dropped"] == 6


# Resolutions of the brief --------------------------------------------------------


def test_clip_without_a_cut_above_the_threshold_yields_no_scores(tmp_path):
    sample = fixtures.make_silent_video(tmp_path / "still.mp4")
    assert media.detect_scenes(sample) == []


def test_dedup_keeps_the_scene_when_a_scene_and_interval_collide():
    same = media.diff_hash_of(b"same")
    kept = media.sample_candidates(
        [media.SceneScore(0.0, 31.710)],
        duration=5,
        interval=1,
        dedup_distance=6,
        extract=lambda t: same,
    )
    assert len(kept) == 1
    assert kept[0].reason == "scene"
    assert kept[0].score == 31.710


def test_hamming_stays_within_its_bounds():
    full = (1 << 64) - 1
    assert media.hamming(0, full) == 64
    assert media.hamming(full, 0) == 64
    assert media.hamming(123, 123) == 0
    for a, b in (
        (1, 2),
        (full, 1),
        (media.diff_hash_of(b"x"), media.diff_hash_of(b"y")),
    ):
        assert media.hamming(a, b) <= 64


def test_diff_hash_of_is_deterministic_and_pure():
    assert media.diff_hash_of(b"stable") == media.diff_hash_of(b"stable")
    assert media.diff_hash_of(b"stable") != media.diff_hash_of(b"changed")
    assert 0 <= media.diff_hash_of(b"stable") < (1 << 64)


def test_ranking_ties_break_on_earliest_timestamp():
    cands = [
        media.Candidate(5.0, "interval", 0.0),
        media.Candidate(3.0, "interval", 0.0),
    ]
    ranked, _ = media.rank_candidates(cands, [], budget=1)
    assert ranked[0].timestamp == 3.0


def test_ranking_accepts_audio_event_objects_and_plain_timestamps():
    @dataclass(frozen=True)
    class Event:
        timestamp: float

    ranked, accounting = media.rank_candidates([], [Event(2.0), 1.0], budget=5)
    assert [c.timestamp for c in ranked] == [1.0, 2.0]
    assert all(c.reason == "audio" for c in ranked)
    assert accounting["audio"] == 2
    assert accounting["dropped"] == 0
    assert accounting["scene"] == accounting["interval"] == accounting["fill"] == 0


def test_ranking_never_invents_audio_candidates_for_an_empty_event_list():
    cands = [media.Candidate(0.0, "scene", 9.0)]
    ranked, accounting = media.rank_candidates(cands, [], budget=5)
    assert ranked == [media.Candidate(0.0, "scene", 9.0)]
    assert accounting["audio"] == 0
    assert accounting["dropped"] == 0


def test_fill_only_appears_for_sparse_kept_sets():
    # Two kept timestamps four seconds apart with a gap at least two bare
    # intervals wide leave room and something to fill; the caller's budget
    # must not be exhausted by kept candidates either.
    cands = [media.Candidate(0.0, "scene", 30.0), media.Candidate(8.0, "scene", 20.0)]
    ranked, accounting = media.rank_candidates(cands, [], budget=3)
    assert accounting["fill"] == 1
    fills = [c for c in ranked if c.reason == "fill"]
    assert fills[0].timestamp == 4.0
    assert {c.reason for c in ranked} == {"scene", "fill"}


def test_fill_never_appears_when_there_is_nothing_to_fill(tmp_path):
    # Budget is already consumed, and a dense kept set has no wide gap:
    # neither case may synthesize a fill candidate.
    dense = [media.Candidate(float(i), "interval", 0.0) for i in range(3)]
    _, dense_accounting = media.rank_candidates(dense, [], budget=3)
    assert dense_accounting["fill"] == 0

    empty, empty_accounting = media.rank_candidates([], [], budget=10)
    assert empty == []
    assert empty_accounting == {
        "scene": 0,
        "audio": 0,
        "interval": 0,
        "fill": 0,
        "dropped": 0,
    }


def test_diff_hash_downscales_through_ffmpeg(tmp_path):
    sample = fixtures.make_sample(tmp_path / "clip.mp4", duration=8, counter_jump_at=4)
    early = _extract_frame(sample, tmp_path / "early.png", at=1.0)
    late = _extract_frame(sample, tmp_path / "late.png", at=6.0)
    first = media.diff_hash(early)
    again = media.diff_hash(early)
    assert first == again
    assert 0 <= first < (1 << 64)
    assert media.hamming(first, media.diff_hash(late)) > 0


def test_diff_hash_rejects_a_raster_that_is_not_the_documented_size(monkeypatch):
    # ffmpeg hashes whatever gray raster it emitted; a future build that
    # returns a different geometry must not be hashed silently, so the byte
    # count is pinned to the documented 32x32x1 raster.
    expected = media._GRAY_SIZE * media._GRAY_SIZE

    def short_raster(*args, **kwargs):
        return media.CompletedProcess(
            returncode=0,
            stdout="g" * (expected - 1),
            stderr="",
            timed_out=False,
            killed=False,
        )

    monkeypatch.setattr(media, "run_child", short_raster)
    with pytest.raises(media.MediaError) as err:
        media.diff_hash(b"\x89PNG\r\n\x1a\n")
    message = str(err.value)
    assert str(expected - 1) in message, "the actual byte count must be named"
    assert str(expected) in message, "the expected byte count must be named"


def test_diff_hash_hashes_exactly_the_documented_raster(monkeypatch):
    raster = bytes(range(256)) * (media._GRAY_SIZE * media._GRAY_SIZE // 256)

    def full_raster(*args, **kwargs):
        return media.CompletedProcess(
            returncode=0,
            stdout=raster.decode("latin-1"),
            stderr="",
            timed_out=False,
            killed=False,
        )

    monkeypatch.setattr(media, "run_child", full_raster)
    assert media.diff_hash(b"a frame") == media.diff_hash_of(raster)


def test_run_child_stdin_bytes_reach_the_child_binary_safe():
    payload = bytes(range(256))
    done = media.run_child(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())",
        ],
        stdin_bytes=payload,
        timeout_seconds=30,
    )
    assert done.returncode == 0
    assert done.stdout.encode("latin-1") == payload


def _extract_frame(clip: Path, destination: Path, at: float) -> bytes:
    done = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{at}",
            "-i",
            str(clip),
            "-frames:v",
            "1",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise AssertionError(f"frame extraction failed: {done.stderr.strip()}")
    return destination.read_bytes()
