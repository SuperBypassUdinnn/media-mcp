"""Tests for URL ingestion: yt-dlp into the cache entry's `media/` directory.

The design promises `yt-dlp` for YouTube and direct URLs, the downloaded bytes
under the entry's `media/` directory, a hard 4 GiB / 4 hour refusal cap, and
yt-dlp's stderr surfaced verbatim. These tests drive the ingest layer through a
POSIX-sh stand-in (`tests/fake_ytdlp.sh`) for the controlled cases, plus one
real `yt-dlp` run against a loopback `http.server`, and one stdio-level
`media_probe` call with the stand-in on the server subprocess's `PATH`.

Every byte lands under `tmp_path`; no fixture is committed and no test touches
the internet.
"""

from __future__ import annotations

import functools
import http.server
import os
import shutil
import threading
from pathlib import Path

import pytest

from media import ingest, media
from tests import fixtures
from tests.test_stdio_contract import ServerSession

FAKE_YTDLP = Path(__file__).with_name("fake_ytdlp.sh")

# Generous caps for the cases that must not refuse; the refusal cases pass
# their own smaller numbers.
_ROOMY_BYTES = 64 * 1024 * 1024
_ROOMY_SECONDS = 600.0
_TIMEOUT = 60


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files without writing a request log line per test."""

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture
def served(tmp_path: Path):
    """A loopback HTTP server over a fixture clip; yields (clip, url)."""
    root = tmp_path / "served"
    root.mkdir()
    clip = fixtures.make_sample(root / "sample.mp4", duration=4)
    handler = functools.partial(_QuietHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield clip, f"http://127.0.0.1:{server.server_address[1]}/sample.mp4"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.fixture
def shim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A PATH directory whose `yt-dlp` is the committed stand-in."""
    directory = tmp_path / "shim"
    directory.mkdir()
    copy = directory / "yt-dlp"
    shutil.copy(FAKE_YTDLP, copy)
    copy.chmod(0o755)
    monkeypatch.setenv(
        "PATH", f"{directory}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}"
    )
    return directory


def _marks(path: Path) -> int:
    """Count the lines the stand-in appended, or zero when it never ran."""
    if not path.is_file():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line])


def _downloaded(entry: Path) -> list[Path]:
    """Every regular file under the entry's `media/` directory."""
    media_dir = entry / "media"
    if not media_dir.is_dir():
        return []
    return sorted(path for path in media_dir.iterdir() if path.is_file())


def test_is_url_truth_table() -> None:
    assert ingest.is_url("http://example.com/clip.mp4") is True
    assert ingest.is_url("https://example.invalid/a") is True
    assert ingest.is_url("/tmp/clip.mp4") is False
    assert ingest.is_url("clip.mp4") is False
    assert ingest.is_url("file:///tmp/clip.mp4") is False
    assert ingest.is_url("ftp://example.com/clip.mp4") is False
    assert ingest.is_url("not a url at all") is False
    assert ingest.is_url("") is False


def test_a_loopback_target_is_refused_then_accepted_with_allow_private(
    served, shim, tmp_path, monkeypatch
) -> None:
    clip, url = served
    entry = tmp_path / "entry"
    marks = tmp_path / "marks"
    monkeypatch.setenv("FAKE_YTDLP_SOURCE", str(clip))
    monkeypatch.setenv("FAKE_YTDLP_MARKFILE", str(marks))

    with pytest.raises(media.MediaError) as refused:
        ingest.ingest_url(
            url,
            entry,
            max_bytes=_ROOMY_BYTES,
            max_seconds=_ROOMY_SECONDS,
            allow_private=False,
            timeout_seconds=_TIMEOUT,
        )
    message = str(refused.value)
    assert "127.0.0.1" in message
    assert "MEDIA_ALLOW_PRIVATE_URLS" in message
    assert _marks(marks) == 0, "the private-range guard must refuse before any child"
    assert _downloaded(entry) == []

    accepted = ingest.ingest_url(
        url,
        entry,
        max_bytes=_ROOMY_BYTES,
        max_seconds=_ROOMY_SECONDS,
        allow_private=True,
        timeout_seconds=_TIMEOUT,
    )
    assert accepted.path.is_file()
    assert _marks(marks) == 1


def test_the_happy_path_lands_bytes_under_the_entry_media_directory(
    served, shim, tmp_path, monkeypatch
) -> None:
    clip, url = served
    entry = tmp_path / "entry"
    monkeypatch.setenv("FAKE_YTDLP_SOURCE", str(clip))

    result = ingest.ingest_url(
        url,
        entry,
        max_bytes=_ROOMY_BYTES,
        max_seconds=_ROOMY_SECONDS,
        allow_private=True,
        timeout_seconds=_TIMEOUT,
    )

    assert result.from_cache is False
    assert result.path.parent == entry / "media"
    assert result.path.name == "sample.mp4"
    assert result.bytes == clip.stat().st_size
    assert result.path.read_bytes() == clip.read_bytes()
    assert media.probe(result.path).has_video
    assert "fake yt-dlp: wrote" in result.stderr_tail


def test_a_second_call_reuses_the_cached_download_without_running_the_shim(
    served, shim, tmp_path, monkeypatch
) -> None:
    clip, url = served
    entry = tmp_path / "entry"
    marks = tmp_path / "marks"
    monkeypatch.setenv("FAKE_YTDLP_SOURCE", str(clip))
    monkeypatch.setenv("FAKE_YTDLP_MARKFILE", str(marks))

    first = ingest.ingest_url(
        url,
        entry,
        max_bytes=_ROOMY_BYTES,
        max_seconds=_ROOMY_SECONDS,
        allow_private=True,
        timeout_seconds=_TIMEOUT,
    )
    second = ingest.ingest_url(
        url,
        entry,
        max_bytes=_ROOMY_BYTES,
        max_seconds=_ROOMY_SECONDS,
        allow_private=True,
        timeout_seconds=_TIMEOUT,
    )

    assert second.from_cache is True
    assert second.path == first.path
    assert second.bytes == first.bytes
    assert _marks(marks) == 1, "a cache hit must not run yt-dlp again"


def test_the_byte_cap_refuses_with_the_measured_size_and_deletes_the_file(
    served, shim, tmp_path, monkeypatch
) -> None:
    clip, url = served
    entry = tmp_path / "entry"
    monkeypatch.setenv("FAKE_YTDLP_SOURCE", str(clip))
    cap = clip.stat().st_size - 1

    with pytest.raises(media.MediaError) as refused:
        ingest.ingest_url(
            url,
            entry,
            max_bytes=cap,
            max_seconds=_ROOMY_SECONDS,
            allow_private=True,
            timeout_seconds=_TIMEOUT,
        )
    message = str(refused.value)
    assert str(clip.stat().st_size) in message
    assert str(cap) in message
    assert "deleted" in message
    assert _downloaded(entry) == []


def test_the_duration_cap_refuses_with_the_measured_duration_and_deletes_the_file(
    served, shim, tmp_path, monkeypatch
) -> None:
    clip, url = served
    entry = tmp_path / "entry"
    monkeypatch.setenv("FAKE_YTDLP_SOURCE", str(clip))
    measured = media.probe(clip).duration
    cap = 1.0
    assert measured > cap, (
        "the fixture must exceed the cap for this test to mean anything"
    )

    with pytest.raises(media.MediaError) as refused:
        ingest.ingest_url(
            url,
            entry,
            max_bytes=_ROOMY_BYTES,
            max_seconds=cap,
            allow_private=True,
            timeout_seconds=_TIMEOUT,
        )
    message = str(refused.value)
    assert f"{measured:.3f}" in message
    assert f"{cap:g} s" in message
    assert "deleted" in message
    assert _downloaded(entry) == []


def test_a_shim_failure_surfaces_the_exit_code_and_the_stderr_verbatim(
    served, shim, tmp_path, monkeypatch
) -> None:
    _clip, url = served
    entry = tmp_path / "entry"
    monkeypatch.setenv("FAKE_YTDLP_FAIL", "1")
    monkeypatch.setenv(
        "FAKE_YTDLP_STDERR",
        "ERROR: this video is unavailable\nERROR: giving up after 3 attempts",
    )

    with pytest.raises(media.MediaError) as failed:
        ingest.ingest_url(
            url,
            entry,
            max_bytes=_ROOMY_BYTES,
            max_seconds=_ROOMY_SECONDS,
            allow_private=True,
            timeout_seconds=_TIMEOUT,
        )
    message = str(failed.value)
    assert "exit code 7" in message
    assert "ERROR: this video is unavailable" in message
    assert "ERROR: giving up after 3 attempts" in message
    assert _downloaded(entry) == []


def test_a_real_yt_dlp_downloads_a_loopback_url(served, tmp_path) -> None:
    """The real binary, not the stand-in: the production invocation works.

    No shim fixture is requested, so `yt-dlp` resolves to the installed
    binary; the loopback server means the test never reaches the internet.
    """
    clip, url = served
    entry = tmp_path / "entry"
    assert shutil.which("yt-dlp") is not None, "yt-dlp is a documented dependency"

    result = ingest.ingest_url(
        url,
        entry,
        max_bytes=_ROOMY_BYTES,
        max_seconds=_ROOMY_SECONDS,
        allow_private=True,
        timeout_seconds=_TIMEOUT,
    )

    assert result.from_cache is False
    assert result.path.is_file()
    assert result.bytes == clip.stat().st_size
    assert media.probe(result.path).has_video


def test_a_download_that_exceeds_its_timeout_is_a_media_error(
    served, shim, tmp_path, monkeypatch
) -> None:
    _clip, url = served
    entry = tmp_path / "entry"
    monkeypatch.setenv("FAKE_YTDLP_SLEEP", "5")

    with pytest.raises(media.MediaError) as stopped:
        ingest.ingest_url(
            url,
            entry,
            max_bytes=_ROOMY_BYTES,
            max_seconds=_ROOMY_SECONDS,
            allow_private=True,
            timeout_seconds=1,
        )
    message = str(stopped.value)
    assert "timeout" in message.lower()
    assert "yt-dlp" in message
    assert _downloaded(entry) == []


def test_probe_takes_a_url_source_over_stdio_and_reports_the_file_duration(
    served, shim, tmp_path
) -> None:
    """End to end through the MCP server: a URL `source` behaves like the file."""
    clip, url = served
    root = tmp_path / "server"
    cache = root / "cache"
    cache.mkdir(parents=True)
    env = {
        "MEDIA_CACHE_DIR": str(cache),
        "MEDIA_ALLOW_PRIVATE_URLS": "1",
        "MEDIA_CALL_TIMEOUT_SECONDS": "120",
        "PATH": f"{shim}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
        "FAKE_YTDLP_SOURCE": str(clip),
        "FAKE_YTDLP_MARKFILE": str(root / "marks"),
    }
    session = ServerSession(
        env, root / "server.stderr", root / "unused.pid", root / "marks"
    )
    try:
        from_url = session.call("media_probe", {"source": url})
        from_file = session.call("media_probe", {"source": str(clip)})
        again = session.call("media_probe", {"source": url})
        assert not from_url.is_error, from_url.text
        assert not from_file.is_error, from_file.text
        assert not again.is_error, again.text
        assert _duration_line(from_url.text) == _duration_line(from_file.text)
        assert "download media/" in from_url.text
        assert "cache: media/" in again.text
        assert session.invocations() == 1
    finally:
        session.close()


def _duration_line(text: str) -> str:
    """The one response line that states the probed duration."""
    lines = [line for line in text.splitlines() if line.startswith("Duration: ")]
    assert len(lines) == 1, f"expected exactly one Duration line in:\n{text}"
    return lines[0]
