"""The resolver's ordering rules, end to end at the tool boundary.

The Task 10c review found that `_entry_for` identified (and therefore opened
and hashed) a local source before `validate_source` ran. That regressed four
local-path behaviours: `~` paths stopped resolving, an unreadable file leaked a
bare `PermissionError` past `_guarded`, the `MEDIA_ALLOWED_READ_ROOTS` check
ran after the file had been read, and every refusal left a phantom cache entry
behind. These tests pin the order that fixes it: a local path is validated
before it is identified, and a URL is guarded before its entry directory is
created.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from media import artifacts, ingest, media, server
from tests import fixtures


def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **env: str):
    """Settings rooted in `tmp_path`, with the read-root and URL switches clear."""
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MEDIA_ALLOWED_READ_ROOTS", raising=False)
    monkeypatch.delenv("MEDIA_ALLOW_PRIVATE_URLS", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return server.load_settings()


def _entries(settings) -> Path:
    return settings.cache_dir / "entries"


def test_a_tilde_path_is_expanded_and_identified_from_the_validated_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    clip = fixtures.make_sample(tmp_path / "clip.mp4", duration=4)
    settings = _settings(tmp_path, monkeypatch)

    resolved, path, entry, _note = server._entry_for(
        settings, "~/clip.mp4", "sampled", server._Report()
    )

    assert path == clip, "the tilde must be expanded before the file is opened"
    assert resolved.source == str(clip)
    assert resolved.key == artifacts.resolve_source(str(clip), "sampled").key
    assert entry == _entries(settings) / resolved.key


def test_a_source_outside_the_allowed_roots_is_refused_before_any_entry_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = fixtures.make_sample(tmp_path / "outside.mp4", duration=2)
    settings = _settings(tmp_path, monkeypatch, MEDIA_ALLOWED_READ_ROOTS=str(allowed))

    with pytest.raises(media.MediaError) as refused:
        server._entry_for(settings, str(outside), "sampled", server._Report())

    assert "allowed roots" in str(refused.value)
    assert not _entries(settings).exists(), (
        "a refused source must not create a cache entry: validation runs first"
    )


def test_an_unreadable_source_is_a_media_error_not_a_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = fixtures.make_sample(tmp_path / "clip.mp4", duration=2)
    clip.chmod(0o000)
    try:
        if os.access(clip, os.R_OK):
            pytest.skip("running with privileges that ignore the mode bits")
        settings = _settings(tmp_path, monkeypatch)

        with pytest.raises(media.MediaError) as refused:
            server._entry_for(settings, str(clip), "sampled", server._Report())

        assert "not readable" in str(refused.value)
        assert not _entries(settings).exists()
    finally:
        clip.chmod(0o644)


def test_a_refused_loopback_url_creates_no_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)

    with pytest.raises(media.MediaError) as refused:
        server._entry_for(
            settings, "http://127.0.0.1:9/clip.mp4", "sampled", server._Report()
        )

    message = str(refused.value)
    assert "127.0.0.1" in message
    assert "MEDIA_ALLOW_PRIVATE_URLS" in message
    assert not _entries(settings).exists(), (
        "the private-range guard must run before the entry directory exists"
    )


def test_a_fractional_duration_cap_reaches_ingest_untruncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`MEDIA_MAX_SOURCE_SECONDS=0.5` must not become `0` and refuse everything."""
    settings = _settings(tmp_path, monkeypatch, MEDIA_MAX_SOURCE_SECONDS="0.5")
    assert settings.max_source_seconds == 0.5
    seen: dict[str, object] = {}

    def fake_ingest(source, entry, **kwargs):
        seen.update(kwargs)
        landed = Path(entry) / "media" / "clip.mp4"
        landed.parent.mkdir(parents=True, exist_ok=True)
        landed.write_bytes(b"x")
        return ingest.IngestedSource(landed, 1, False, "")

    monkeypatch.setattr(server.ingest, "ingest_url", fake_ingest)

    _resolved, path, entry, _note = server._entry_for(
        settings, "http://93.184.216.34/clip.mp4", "sampled", server._Report()
    )

    assert seen["max_seconds"] == 0.5
    assert path == entry / "media" / "clip.mp4"
