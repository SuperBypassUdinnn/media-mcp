"""Tests for the process registry, the VRAM probe, and the shared transcript lock.

These exercise the three additions Task 11 makes to `media/media.py` plus the
server's single-flight lock: a live child can be stopped by the shutdown path,
the VRAM measurement is honest about being unavailable, and `probe` now runs
under the bounded runner instead of escaping it.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import threading
import time

import pytest

from media import media, server


def _await(predicate, deadline: float = 5.0) -> bool:
    """Poll `predicate` until it holds or `deadline` seconds pass."""
    stop = time.monotonic() + deadline
    while time.monotonic() < stop:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _completed(stdout: str = "", returncode: int = 0) -> media.CompletedProcess[str]:
    return media.CompletedProcess(
        returncode=returncode, stdout=stdout, stderr="", timed_out=False, killed=False
    )


# The active-child registry -------------------------------------------------------


def test_terminate_active_children_stops_a_live_run():
    results: list[media.CompletedProcess[str]] = []

    def run() -> None:
        results.append(media.run_child(["sleep", "30"], timeout_seconds=60))

    worker = threading.Thread(target=run, name="registry-runner")
    worker.start()
    assert _await(lambda: bool(media._ACTIVE_CHILDREN)), (
        "the child was never registered"
    )
    assert media.terminate_active_children() == 1
    worker.join(timeout=10)
    assert not worker.is_alive(), "the runner must finish once its child is stopped"
    assert results and results[0].returncode != 0
    assert not media._ACTIVE_CHILDREN


def test_terminate_active_children_is_quiet_with_nothing_running():
    assert media.terminate_active_children() == 0
    assert media.terminate_active_children(grace=0.0) == 0


# The VRAM measurement ------------------------------------------------------------


def test_free_vram_mib_parses_this_host(monkeypatch):
    if shutil.which("nvidia-smi") is None:
        assert media.free_vram_mib() is None
    else:
        value = media.free_vram_mib()
        assert value is None or (isinstance(value, int) and value >= 0)

    monkeypatch.setattr(media, "run_child", lambda *a, **k: _completed("3729\n"))
    assert media.free_vram_mib() == 3729
    monkeypatch.setattr(media, "run_child", lambda *a, **k: _completed("1024\n2048\n"))
    assert media.free_vram_mib() == 2048


def test_free_vram_mib_is_none_when_unmeasurable(monkeypatch):
    def missing(*args, **kwargs):
        raise media.MediaError("could not start nvidia-smi: [Errno 2] missing")

    monkeypatch.setattr(media, "run_child", missing)
    assert media.free_vram_mib() is None

    monkeypatch.setattr(media, "run_child", lambda *a, **k: _completed("N/A\n"))
    assert media.free_vram_mib() is None

    monkeypatch.setattr(media, "run_child", lambda *a, **k: _completed(""))
    assert media.free_vram_mib() is None

    monkeypatch.setattr(media, "run_child", lambda *a, **k: _completed("x", 9))
    assert media.free_vram_mib() is None


# The tier downgrade branch -------------------------------------------------------


def _auto_settings(tmp_path, monkeypatch) -> server.Settings:
    """Load `Settings` with the VRAM knobs at their documented defaults."""
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("MEDIA_DEVICE", raising=False)
    monkeypatch.delenv("MEDIA_VRAM_FLOOR", raising=False)
    return server.load_settings()


def test_the_tier_drops_to_medium_when_free_vram_is_below_the_floor(
    tmp_path, monkeypatch
):
    """The measured MiB and the floor are both named in the downgrade reason."""
    settings = _auto_settings(tmp_path, monkeypatch)
    measured = settings.vram_floor_mib - 1
    monkeypatch.setattr(media, "free_vram_mib", lambda: measured)
    tier, device, rationale = server._choose_tier(settings, None, None)
    assert (tier, device) == ("medium", "cpu")
    assert f"{measured} MiB" in rationale
    assert f"{settings.vram_floor_mib} MiB floor" in rationale


def test_the_tier_drops_to_medium_when_free_vram_is_unmeasurable(tmp_path, monkeypatch):
    """An unmeasurable driver is treated as below the floor, and says so."""
    settings = _auto_settings(tmp_path, monkeypatch)
    monkeypatch.setattr(media, "free_vram_mib", lambda: None)
    tier, device, rationale = server._choose_tier(settings, None, None)
    assert (tier, device) == ("medium", "cpu")
    assert "unmeasurable" in rationale
    assert f"{settings.vram_floor_mib} MiB" in rationale


def test_the_top_tier_is_kept_at_exactly_the_vram_floor(tmp_path, monkeypatch):
    """The floor is inclusive: free VRAM at the floor does not downgrade."""
    settings = _auto_settings(tmp_path, monkeypatch)
    monkeypatch.setattr(media, "free_vram_mib", lambda: settings.vram_floor_mib)
    tier, device, rationale = server._choose_tier(settings, None, None)
    assert (tier, device) == ("large-v3-turbo", "auto")
    assert f"{settings.vram_floor_mib} MiB floor" in rationale


# probe runs under the bounded runner --------------------------------------------


def test_probe_runs_through_the_bounded_runner(tmp_path, monkeypatch):
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"not really a video")
    calls: list[list[str]] = []

    def stop(argv, **kwargs):
        calls.append(list(argv))
        raise media.MediaError("stub runner refused the call")

    monkeypatch.setattr(media, "run_child", stop)
    with pytest.raises(media.MediaError) as err:
        media.probe(target)
    assert str(target) in str(err.value)
    assert calls and calls[0][0] == "ffprobe"


# The single-flight transcription lock -------------------------------------------


def test_transcribe_lock_fails_fast_with_the_holder_named(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(tmp_path))
    settings = server.load_settings()
    lock_path = tmp_path / "locks" / "transcribe.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(server.MediaError) as err:
            server.acquire_transcribe_lock(
                settings,
                source="clip.mp4",
                queue="fail",
                cancel=None,
                timeout_seconds=5,
            )
        message = str(err.value).lower()
        assert "transcribe.lock" in message
        assert "queue='fail'" in message
    finally:
        fcntl.flock(holder_fd, fcntl.LOCK_UN)
        os.close(holder_fd)


def test_transcribe_lock_waits_and_reports_the_holder(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(tmp_path))
    settings = server.load_settings()
    lock_path = tmp_path / "locks" / "transcribe.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "locks" / "transcribe.lock.holder").write_text(
        '{"pid": 4242, "source": "other.mp4", "started": 1.0}\n', encoding="utf-8"
    )
    holder_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    released = threading.Timer(0.3, lambda: fcntl.flock(holder_fd, fcntl.LOCK_UN))
    released.start()
    started = time.monotonic()
    try:
        note = server.acquire_transcribe_lock(
            settings,
            source="clip.mp4",
            queue="wait",
            cancel=None,
            timeout_seconds=10,
        )
        elapsed = time.monotonic() - started
        assert elapsed >= 0.2
        assert note is not None and "4242" in note
    finally:
        released.cancel()
        fcntl.flock(holder_fd, fcntl.LOCK_UN)
        os.close(holder_fd)
