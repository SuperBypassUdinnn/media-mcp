"""Tests for the bounded, cancellable subprocess runner."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import threading
import time as time_module

import pytest

from media import media

# Exactly as the brief writes them ------------------------------------------------


def test_timeout_kills_the_process_group():
    with pytest.raises(media.MediaError) as err:
        media.run_child(["sleep", "30"], timeout_seconds=1)
    assert "timeout" in str(err.value).lower()
    assert (
        subprocess.run(
            ["pgrep", "-f", "sleep 30"], check=False, capture_output=True
        ).returncode
        != 0
    )


def test_cancel_event_kills_the_child():
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    with pytest.raises(media.MediaError):
        media.run_child(["sleep", "30"], timeout_seconds=60, cancel=cancel)
    assert (
        subprocess.run(
            ["pgrep", "-f", "sleep 30"], check=False, capture_output=True
        ).returncode
        != 0
    )


def test_stderr_is_captured_and_returned():
    done = media.run_child(["ffprobe", "-v", "error", "/nonexistent.mp4"])
    assert done.returncode != 0 and done.stderr.strip()


# Resolutions of the brief --------------------------------------------------------


def test_timeout_stop_reports_sigterm_was_enough():
    # `sleep` dies on SIGTERM, so the stop report must say SIGTERM sufficed
    # rather than mentioning an escalation that never happened.
    with pytest.raises(media.MediaError) as err:
        media.run_child(["sleep", "30"], timeout_seconds=1)
    message = str(err.value).lower()
    assert "sigterm" in message
    assert "sigkill" not in message


def test_cancel_stop_reports_sigterm_was_enough():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(media.MediaError) as err:
        media.run_child(["sleep", "30"], timeout_seconds=60, cancel=cancel)
    message = str(err.value).lower()
    assert "sigterm" in message
    assert "sigkill" not in message


def test_missing_binary_raises_media_error_naming_it():
    with pytest.raises(media.MediaError) as err:
        media.run_child(["run-child-no-such-binary-xyz"], timeout_seconds=5)
    assert "run-child-no-such-binary-xyz" in str(err.value)


def test_successful_run_returns_fields_and_result_is_frozen():
    done = media.run_child([sys.executable, "-c", "print('ok')"], timeout_seconds=10)
    assert done.returncode == 0
    assert done.stdout.strip() == "ok"
    assert done.timed_out is False and done.killed is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        done.returncode = 1  # type: ignore[misc]


def test_nonzero_exit_is_data_not_exception():
    code = "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"
    done = media.run_child([sys.executable, "-c", code], timeout_seconds=10)
    assert done.returncode == 3
    assert "boom" in done.stderr
    assert done.stdout == ""


def test_streams_are_capped_with_a_truncation_marker():
    limit = media._STREAM_CAPTURE_LIMIT
    size = 2 * limit
    code = (
        "import sys\n"
        f"sys.stdout.write('x' * {size}); sys.stdout.flush()\n"
        f"sys.stderr.write('e' * {size}); sys.stderr.flush()\n"
    )
    done = media.run_child([sys.executable, "-c", code], timeout_seconds=20)
    assert done.returncode == 0
    assert len(done.stdout) < size
    assert len(done.stdout) >= limit
    assert media._TRUNCATION_MARKER in done.stdout
    assert media._TRUNCATION_MARKER in done.stderr
    assert done.stderr.count(media._TRUNCATION_MARKER) == 1


def test_timeout_escalates_to_sigkill_when_sigterm_is_ignored():
    marker = "run-child-sigterm-ignorer"
    code = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"print('{marker}', flush=True)\n"
        "time.sleep(60)\n"
    )
    started = time_module.monotonic()
    with pytest.raises(media.MediaError) as err:
        media.run_child([sys.executable, "-c", code], timeout_seconds=1)
    elapsed = time_module.monotonic() - started
    # The deadline was reached (so the child really ran) and SIGKILL needed the
    # grace period on top of it, which the error message must report.
    assert elapsed >= 1
    assert "timeout" in str(err.value).lower()
    assert "sigkill" in str(err.value).lower()
    assert (
        subprocess.run(
            ["pgrep", "-f", marker], check=False, capture_output=True
        ).returncode
        != 0
    )
