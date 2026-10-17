"""Tests for content identity and the on-disk cache store."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from media import artifacts
from tests import fixtures

_HEX_KEY = re.compile(r"^[0-9a-f]{16}$")


@pytest.fixture()
def sample(tmp_path: Path) -> Path:
    return fixtures.make_sample(tmp_path / "sample.mp4", duration=4)


def test_sampled_identity_is_stable_and_changes_with_mtime(tmp_path):
    f = fixtures.make_sample(tmp_path / "a.mp4", duration=4)
    first = artifacts.resolve_source(str(f))
    assert first.key == artifacts.resolve_source(str(f)).key
    os.utime(f, (1.0, 1.0))
    assert artifacts.resolve_source(str(f)).key != first.key


def test_strict_identity_ignores_mtime(tmp_path):
    f = fixtures.make_sample(tmp_path / "b.mp4", duration=4)
    strict = artifacts.resolve_source(str(f), identity="strict")
    os.utime(f, (1.0, 1.0))
    assert artifacts.resolve_source(str(f), identity="strict").key == strict.key
    assert artifacts.resolve_source(str(f)).key != strict.key


def test_url_identity_is_normalized_without_network():
    a = artifacts.resolve_source("https://youtu.be/AbC?t=5")
    b = artifacts.resolve_source("https://youtu.be/AbC?t=90")
    assert a.kind == "youtube" and a.key == b.key
    assert a.detail["video_id"] == "AbC"
    assert artifacts.resolve_source("http://example.com/x.mp4").kind == "url"


def test_meta_round_trip_and_touch(tmp_path, sample):
    entry = artifacts.entry_dir(artifacts.resolve_source(str(sample)), tmp_path)
    artifacts.write_meta(entry, {"created": 1.0, "last_used": 1.0})
    assert artifacts.read_meta(entry)["created"] == 1.0
    artifacts.touch(entry, now=99.0)
    assert artifacts.read_meta(entry)["last_used"] == 99.0


def test_identity_keys_are_16_lowercase_hex(tmp_path):
    f = fixtures.make_sample(tmp_path / "hex.mp4", duration=4)
    for ident in (
        artifacts.resolve_source(str(f)),
        artifacts.resolve_source(str(f), identity="strict"),
        artifacts.resolve_source("https://youtu.be/AbC?t=5"),
        artifacts.resolve_source("http://example.com/x.mp4"),
    ):
        assert _HEX_KEY.fullmatch(ident.key)


def test_sampled_identity_kinds(tmp_path):
    f = fixtures.make_sample(tmp_path / "kinds.mp4", duration=4)
    sampled = artifacts.resolve_source(str(f))
    strict = artifacts.resolve_source(str(f), identity="strict")
    assert sampled.kind == "file-sampled"
    assert strict.kind == "file-strict"


def test_youtube_watch_url_matches_short_url():
    short = artifacts.resolve_source("https://youtu.be/zz-9_8?si=track")
    watch = artifacts.resolve_source("https://www.youtube.com/watch?si=track&v=zz-9_8")
    assert short.kind == "youtube"
    assert short.detail["video_id"] == "zz-9_8"
    assert watch.detail["video_id"] == "zz-9_8"
    assert short.key == watch.key


def test_sampled_identity_detail_folds_size_and_chunks(tmp_path):
    f = fixtures.make_sample(tmp_path / "detail.mp4", duration=4)
    ident = artifacts.resolve_source(str(f))
    assert ident.detail["size"] == f.stat().st_size
    assert ident.detail["mtime_ns"] == f.stat().st_mtime_ns
    for chunk in ("head", "middle", "tail"):
        digest = ident.detail[chunk]
        assert isinstance(digest, str) and len(digest) == 64


def test_strict_detail_carries_full_hash(tmp_path):
    f = fixtures.make_sample(tmp_path / "strict.mp4", duration=4)
    ident = artifacts.resolve_source(str(f), identity="strict")
    assert (
        ident.detail["sha256"]
        == artifacts.resolve_source(str(f), identity="strict").detail["sha256"]
    )


def test_entry_dir_lives_under_cache_root_entries(tmp_path):
    ident = artifacts.resolve_source("https://youtu.be/AbC")
    entry = artifacts.entry_dir(ident, tmp_path)
    assert entry == tmp_path / "entries" / ident.key
    assert entry.is_dir()


def test_missing_meta_reads_as_none(tmp_path):
    entry = tmp_path / "entries" / ("0" * 16)
    entry.mkdir(parents=True)
    assert artifacts.read_meta(entry) is None


def test_touch_on_missing_meta_writes_both_timestamps(tmp_path, sample):
    entry = artifacts.entry_dir(artifacts.resolve_source(str(sample)), tmp_path)
    assert artifacts.read_meta(entry) is None
    artifacts.touch(entry, now=55.0)
    meta = artifacts.read_meta(entry)
    assert meta == {"created": 55.0, "last_used": 55.0}


def test_touch_preserves_created(sample, tmp_path):
    entry = artifacts.entry_dir(artifacts.resolve_source(str(sample)), tmp_path)
    artifacts.write_meta(entry, {"created": 3.0, "last_used": 3.0})
    artifacts.touch(entry, now=10.0)
    meta = artifacts.read_meta(entry)
    assert meta["created"] == 3.0
    assert meta["last_used"] == 10.0


def test_stage_path_for_files_and_directories(tmp_path):
    entry = tmp_path / "entries" / ("k" * 16)
    entry.mkdir(parents=True)
    probe = artifacts.stage_path(entry, "probe.json")
    assert probe == entry / "probe.json"
    assert not probe.exists()

    frames = artifacts.stage_path(entry, "frames/")
    assert frames == entry / "frames"
    assert frames.is_dir()

    nested = artifacts.stage_path(entry, "ocr/1234.json")
    assert nested == entry / "ocr" / "1234.json"
    assert nested.parent.is_dir()


def test_meta_json_persists_as_json(tmp_path, sample):
    entry = artifacts.entry_dir(artifacts.resolve_source(str(sample)), tmp_path)
    artifacts.write_meta(entry, {"created": 7.0, "last_used": 7.0, "bytes": 128})
    on_disk = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
    assert on_disk == {"created": 7.0, "last_used": 7.0, "bytes": 128}
