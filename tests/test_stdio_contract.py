"""Contract tests for the stdio server: what a real client actually sees.

The server runs as a subprocess (`media/.venv/bin/python media/server.py`) and
the tests speak MCP to it through `stdio_client` and `ClientSession`, the
transport both shipped clients use. Every test writes only under `tmp_path`:
the cache root, the fake pipeline's pid file and the server's stderr all live
there, so nothing lands in the repository.

A failure the server raises as a `ToolError` arrives as
`CallToolResult(is_error=True)` with the message in the text block, so the
`call()` helper exposes exactly `.is_error` and `.text`; a failure the SDK
raises as a protocol error is caught and reported the same way.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError

from media import artifacts
from tests import fixtures

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / "media" / ".venv" / "bin" / "python"
SERVER_SCRIPT = REPO_ROOT / "media" / "server.py"
FAKE_TRANSCRIBE = Path(__file__).with_name("fake_transcribe.sh")
TOOL_NAMES = {
    "media_probe",
    "media_manifest",
    "media_frames",
    "media_ocr",
    "media_transcript",
    "media_search",
    "media_cache",
}


# Driving the server from a synchronous test --------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """What one tool call produced, in the spelling the SDK uses."""

    is_error: bool
    text: str
    content: list = field(default_factory=list)


class ServerSession:
    """A live stdio connection to the media server, driven from sync test code.

    The MCP session lives in its own asyncio loop on a background thread; the
    test thread submits coroutines to it, so the fixture can expose the plain
    `list_tools()` and `call()` methods the brief's tests use.
    """

    def __init__(
        self,
        env: dict[str, str],
        stderr_path: Path,
        pidfile: Path,
        markfile: Path,
    ) -> None:
        self.pidfile = pidfile
        self.markfile = markfile
        self._env = env
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stopped: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._error: BaseException | None = None
        self._pending = None
        self._stderr = stderr_path.open("w", encoding="utf-8")
        self._thread = threading.Thread(
            target=self._serve, name="media-server-session", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=60):
            raise RuntimeError("the media server session never became ready")
        if self._error is not None:
            raise self._error

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except BaseException as err:  # noqa: BLE001 - re-raised in the constructor
            self._error = err
        finally:
            self._ready.set()
            self._loop.close()

    async def _connect(self) -> None:
        params = StdioServerParameters(
            command=str(VENV_PYTHON),
            args=[str(SERVER_SCRIPT)],
            env=self._env,
            cwd=str(REPO_ROOT),
        )
        async with (
            stdio_client(params, errlog=self._stderr) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            self._session = session
            self._stopped = asyncio.Event()
            self._ready.set()
            await self._stopped.wait()

    def submit(self, coro):
        """Schedule `coro` on the session loop and return its future."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def list_tools(self, timeout: float = 60):
        assert self._session is not None
        return self.submit(self._session.list_tools()).result(timeout=timeout)

    async def _call_raw(self, name: str, arguments: dict | None):
        assert self._session is not None
        return await self._session.call_tool(name, arguments)

    def call(self, name: str, arguments: dict | None = None, timeout: float = 120):
        assert self._session is not None
        future = self.submit(self._call_raw(name, arguments))
        try:
            result = future.result(timeout=timeout)
        except (
            MCPError,
            TimeoutError,
            OSError,
            anyio.ClosedResourceError,
            anyio.BrokenResourceError,
        ) as err:  # a protocol-level error the SDK raised
            return ToolCall(True, str(err), [])
        return _as_tool_call(result)

    def call_async(self, name: str, arguments: dict | None = None):
        """Submit a tool call without waiting, and return its future."""
        assert self._session is not None
        return self.submit(self._call_raw(name, arguments))

    def invocations(self) -> int:
        """How many times the fake pipeline was actually entered."""
        if not self.markfile.is_file():
            return 0
        text = self.markfile.read_text(encoding="utf-8")
        return len([line for line in text.splitlines() if line.strip()])

    def start_transcript(self, source: str, deadline: float = 15.0) -> int:
        """Start a transcript call in the background; return its child's pid."""
        assert self._session is not None
        self._pending = self.submit(
            self._call_raw("media_transcript", {"source": source})
        )
        assert _wait_for_file(self.pidfile, deadline), (
            f"the fake transcribe never wrote {self.pidfile}"
        )
        return _read_pid(self.pidfile)

    async def _transcript_then_cancel(self, source: str) -> int:
        assert self._session is not None
        task = asyncio.create_task(
            self._session.call_tool("media_transcript", {"source": source})
        )
        deadline = time.monotonic() + 15.0
        while not self.pidfile.is_file() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if not self.pidfile.is_file():
            raise AssertionError(f"the fake transcribe never wrote {self.pidfile}")
        pid = _read_pid(self.pidfile)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return pid

    def cancel_transcript(self, source: str, timeout: float = 30.0) -> int:
        """Run a transcript call, cancel it, and return the child's pid."""
        assert self._session is not None
        return self.submit(self._transcript_then_cancel(source)).result(timeout=timeout)

    def close(self) -> None:
        """Stop the session; safe to call more than once."""
        if self._stopped is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stopped.set)
        self._thread.join(timeout=30)
        with contextlib.suppress(Exception):
            self._stderr.close()


def _as_tool_call(result) -> ToolCall:
    blocks = list(getattr(result, "content", None) or [])
    texts = [
        text
        for text in (getattr(block, "text", None) for block in blocks)
        if isinstance(text, str)
    ]
    return ToolCall(
        is_error=bool(getattr(result, "is_error", False)),
        text="\n".join(texts),
        content=blocks,
    )


def _read_pid(pidfile: Path) -> int:
    return int(pidfile.read_text(encoding="utf-8").strip())


def _wait_for_file(path: Path, deadline: float) -> bool:
    stop = time.monotonic() + deadline
    while time.monotonic() < stop:
        if path.is_file() and path.read_text(encoding="utf-8").strip():
            return True
        time.sleep(0.05)
    return path.is_file()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_death(pid: int, deadline: float = 10.0) -> bool:
    """Poll until `pid` is gone, up to `deadline` seconds."""
    stop = time.monotonic() + deadline
    while time.monotonic() < stop:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


# Fixtures -------------------------------------------------------------------------


def _install_fake_transcribe(directory: Path) -> Path:
    fake = directory / "fake_transcribe.sh"
    shutil.copy(FAKE_TRANSCRIBE, fake)
    fake.chmod(0o755)
    return fake


def _start_session(
    tmp_path_factory: pytest.TempPathFactory,
    name: str,
    overrides: dict[str, str],
) -> ServerSession:
    root = tmp_path_factory.mktemp(name)
    cache_dir = root / "cache"
    cache_dir.mkdir()
    pidfile = root / "fake-transcribe.pid"
    markfile = root / "fake-transcribe.marks"
    env = {
        "MEDIA_CACHE_DIR": str(cache_dir),
        "MEDIA_TRANSCRIBE_BIN": str(_install_fake_transcribe(root)),
        "MEDIA_TRANSCRIBE_ENV": str(root / "transcribe.env"),
        "FAKE_TRANSCRIBE_PIDFILE": str(pidfile),
        "FAKE_TRANSCRIBE_MARKFILE": str(markfile),
        "MEDIA_CALL_TIMEOUT_SECONDS": "120",
    }
    env.update(overrides)
    return ServerSession(env, root / "server.stderr", pidfile, markfile)


@pytest.fixture(scope="module")
def server_session(tmp_path_factory):
    session = _start_session(tmp_path_factory, "server", {})
    yield session
    session.close()


@pytest.fixture(scope="module")
def slow_session(tmp_path_factory):
    """A session whose fake transcribe sleeps, for cancellation."""
    session = _start_session(tmp_path_factory, "slow", {"FAKE_TRANSCRIBE_SLEEP": "30"})
    yield session
    session.close()


@pytest.fixture(scope="module")
def orphan_session(tmp_path_factory):
    """A session that leaves a sleeping fake running, closed by its own test."""
    session = _start_session(
        tmp_path_factory, "orphan", {"FAKE_TRANSCRIBE_SLEEP": "30"}
    )
    yield session
    session.close()


@pytest.fixture(scope="module")
def timeout_session(tmp_path_factory):
    """A session with a one-second child timeout and a slow fake."""
    session = _start_session(
        tmp_path_factory,
        "timeout",
        {"FAKE_TRANSCRIBE_SLEEP": "30", "MEDIA_CALL_TIMEOUT_SECONDS": "1"},
    )
    yield session
    session.close()


@pytest.fixture(scope="module")
def nofont_session(tmp_path_factory):
    """A session with an unusable font, so tile labels must degrade."""
    session = _start_session(
        tmp_path_factory,
        "nofont",
        {"MEDIA_FONT_FILE": str(Path("/nonexistent/media-mcp/font.ttf"))},
    )
    yield session
    session.close()


@pytest.fixture(scope="module")
def coalesce_session(tmp_path_factory):
    """A session whose fake pipeline sleeps, so two calls truly overlap."""
    session = _start_session(
        tmp_path_factory, "coalesce", {"FAKE_TRANSCRIBE_SLEEP": "3"}
    )
    yield session
    session.close()


@pytest.fixture(scope="module")
def small_cache_session(tmp_path_factory):
    """A session with a tiny cache cap, so a heavy job must evict to fit."""
    session = _start_session(
        tmp_path_factory, "small-cap", {"MEDIA_CACHE_CAP": "500000"}
    )
    yield session
    session.close()


@pytest.fixture
def sample(tmp_path):
    return fixtures.make_sample(tmp_path / "sample.mp4", duration=4, counter_jump_at=2)


# The brief's tests -----------------------------------------------------------------


def test_every_tool_is_advertised_with_a_schema(server_session):
    tools = server_session.list_tools().tools
    names = {t.name for t in tools}
    assert names == TOOL_NAMES
    assert all(t.input_schema for t in tools)


def test_malformed_parameter_returns_an_error_not_a_crash(server_session):
    result = server_session.call(
        "media_frames", {"source": "x.mp4", "at": [1, 2], "window": 5}
    )
    assert result.is_error and "window" in result.text.lower()


def test_cancelled_request_terminates_cleanly(slow_session, sample):
    pid = slow_session.cancel_transcript(str(sample))
    assert _wait_for_death(pid), f"the cancelled child {pid} survived"
    # A following cheap tool must still answer on the same connection.
    after = slow_session.call("media_cache", {"action": "list"})
    assert not after.is_error, after.text


def test_manifest_response_ends_with_a_status_block(server_session, sample):
    text = server_session.call("media_manifest", {"source": str(sample)}).text
    assert "computed:" in text and "cache:" in text
    assert text.strip().splitlines()[-1].startswith("skipped:")


def test_a_cut_free_clip_reports_no_scene_change_reached_the_threshold(
    server_session, tmp_path
):
    """Motionless content cannot trip `scdet`, and the manifest says so."""
    clip = fixtures.make_silent_video(tmp_path / "silent.mp4")
    result = server_session.call("media_manifest", {"source": str(clip)})
    assert not result.is_error, result.text
    assert "No scene change reached the threshold." in result.text
    assert "0 scene boundary candidate" in result.text


# Lifecycle: no child may outlive the session --------------------------------------


def test_orphaned_child_dies_when_the_session_closes(orphan_session, sample):
    pid = orphan_session.start_transcript(str(sample))
    assert _pid_alive(pid), "the fake transcribe should be running"
    orphan_session.close()
    assert _wait_for_death(pid), f"the orphaned child {pid} outlived the session"


# Tool behaviour -------------------------------------------------------------------


def test_probe_reports_measurements_and_stage_costs(server_session, sample):
    result = server_session.call("media_probe", {"source": str(sample)})
    assert not result.is_error, result.text
    assert "1280x720" in result.text.replace(" ", "")
    assert "duration" in result.text.lower()
    assert "start_time" in result.text.lower()
    assert "audio-video offset" in result.text.lower()
    for weight in ("light", "medium", "heavy"):
        assert weight in result.text.lower()
    assert result.text.strip().splitlines()[-1].startswith("skipped:")


def test_probe_names_the_container_layout_and_both_stream_start_times(
    server_session, sample
):
    result = server_session.call("media_probe", {"source": str(sample)})
    assert not result.is_error, result.text
    lowered = result.text.lower()
    assert "container" in lowered
    assert "mp4" in result.text
    assert "audio layout" in lowered and "mono" in lowered
    start_line = next(
        line
        for line in result.text.splitlines()
        if line.lower().startswith("stream start_time")
    )
    assert "video=" in start_line and "audio=" in start_line
    assert "audio-video offset" in start_line


def test_manifest_digest_tables_and_the_loudness_gate(server_session, sample):
    text = server_session.call("media_manifest", {"source": str(sample)}).text
    assert "scene" in text.lower()
    assert "audio" in text.lower()
    # An empty loudness list must name the gate and the margin (finding R30).
    assert "-30" in text and "10" in text
    assert "candidates" in text.lower()


def test_full_profile_runs_and_states_its_caps(server_session, sample):
    result = server_session.call(
        "media_manifest",
        {"source": str(sample), "profile": "full", "frame_budget": 2},
    )
    assert not result.is_error, result.text
    assert "transcript" in result.text.lower()
    assert "ocr" in result.text.lower()
    assert "capped:" in result.text
    assert "capped: (none)" not in result.text


def test_frames_write_pngs_and_point_at_read(server_session, sample):
    result = server_session.call(
        "media_frames", {"source": str(sample), "at": [1.0, 2.0]}
    )
    assert not result.is_error, result.text
    paths = _frame_paths(result.text)
    assert len(paths) == 2
    for path in paths:
        assert path.is_absolute() and path.is_file()
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "00:00:01.000" in result.text and "00:00:02.000" in result.text
    assert "read" in result.text.lower()


def test_frames_sheet_mode_writes_one_composite(server_session, sample):
    result = server_session.call(
        "media_frames", {"source": str(sample), "at": [1.0, 3.0], "tile": "sheet"}
    )
    assert not result.is_error, result.text
    paths = _frame_paths(result.text)
    assert len(paths) == 1 and paths[0].is_file()


def test_frames_with_only_a_source_returns_the_default_contact_sheet(
    server_session, sample
):
    """`at` defaults to 'scenes', whose tile mode is a single contact sheet."""
    result = server_session.call("media_frames", {"source": str(sample)})
    assert not result.is_error, result.text
    assert "sheet" in result.text.lower()
    paths = _frame_paths(result.text)
    assert len(paths) == 1 and paths[0].is_file()


def test_ocr_with_only_a_source_runs_instead_of_demanding_a_timestamp(
    server_session, sample
):
    """`at` defaults to 'scenes' for media_ocr too."""
    result = server_session.call("media_ocr", {"source": str(sample)})
    assert not result.is_error, result.text
    assert "a timestamp is required" not in result.text.lower()
    assert "Recognized text per timestamp" in result.text


def test_frames_on_an_audio_only_source_names_the_missing_video_stream(
    server_session, tmp_path
):
    clip = fixtures.make_audio_only(tmp_path / "audio-only-frames.m4a")
    result = server_session.call("media_frames", {"source": str(clip)})
    assert result.is_error, result.text
    assert "no video stream" in result.text.lower(), result.text


def test_ocr_on_an_audio_only_source_names_the_missing_video_stream(
    server_session, tmp_path
):
    clip = fixtures.make_audio_only(tmp_path / "audio-only-ocr.m4a")
    result = server_session.call("media_ocr", {"source": str(clip)})
    assert result.is_error, result.text
    assert "no video stream" in result.text.lower(), result.text


def test_frames_without_a_usable_font_still_return_images(nofont_session, sample):
    result = nofont_session.call(
        "media_frames", {"source": str(sample), "at": [1.0, 3.0], "labels": True}
    )
    assert not result.is_error, result.text
    paths = _frame_paths(result.text)
    assert len(paths) == 2 and all(path.is_file() for path in paths)
    assert (
        "unlabelled" in result.text.lower() or "no usable font" in result.text.lower()
    )


def test_ocr_returns_text_per_timestamp(server_session, sample):
    result = server_session.call("media_ocr", {"source": str(sample), "at": [1.0, 3.0]})
    assert not result.is_error, result.text
    assert "LANJUT" in result.text.upper()
    assert "00:00:01.000" in result.text and "00:00:03.000" in result.text
    assert "computed:" in result.text


def test_ocr_track_reports_a_change_candidate(server_session, tmp_path):
    clip = fixtures.make_value_flicker(tmp_path / "flicker.mp4")
    result = server_session.call(
        "media_ocr",
        {
            "source": str(clip),
            "at": 4.5,
            "window": 3.0,
            "fps": 2,
            "track": True,
            "crop": {"x": 0, "y": 0, "width": 1280, "height": 240},
        },
    )
    assert not result.is_error, result.text
    assert "candidate" in result.text.lower()
    assert "48" in result.text


def test_transcript_reports_tier_and_artifact_paths(server_session, sample):
    result = server_session.call("media_transcript", {"source": str(sample)})
    assert not result.is_error, result.text
    assert "halo" in result.text
    assert "00:00:00.000" in result.text
    assert ".srt" in result.text and ".vtt" in result.text and ".md" in result.text
    assert "tier" in result.text.lower()


def test_concurrent_transcripts_coalesce_onto_one_pipeline_run(
    coalesce_session, sample
):
    """Two overlapping calls for one source run the pipeline exactly once."""
    first = coalesce_session.call_async("media_transcript", {"source": str(sample)})
    assert _wait_for_file(coalesce_session.pidfile, 15.0), (
        "the first transcript job never reached the pipeline"
    )
    second = coalesce_session.call_async("media_transcript", {"source": str(sample)})
    first_result = _as_tool_call(first.result(timeout=120))
    second_result = _as_tool_call(second.result(timeout=120))
    assert not first_result.is_error, first_result.text
    assert not second_result.is_error, second_result.text
    assert coalesce_session.invocations() == 1, (
        "the second identical call re-ran the pipeline instead of reusing the "
        "cache entry"
    )
    assert "transcript.json" in _cache_line(second_result.text), second_result.text
    assert "waited for transcribe.lock" in second_result.text
    assert "held by" in second_result.text and "pid " in second_result.text


def test_a_timeout_is_an_error_result_naming_the_timeout(timeout_session, sample):
    result = timeout_session.call("media_transcript", {"source": str(sample)})
    assert result.is_error
    assert "timeout" in result.text.lower()


def test_search_names_the_missing_stages_and_then_finds_speech(server_session, sample):
    missing = server_session.call(
        "media_search", {"source": str(sample), "query": "halo"}
    )
    assert missing.is_error
    lowered = missing.text.lower()
    assert "transcript" in lowered and "media_transcript" in lowered

    transcript = server_session.call("media_transcript", {"source": str(sample)})
    assert not transcript.is_error, transcript.text

    found = server_session.call(
        "media_search", {"source": str(sample), "query": "halo"}
    )
    assert not found.is_error, found.text
    assert "00:00:00.000" in found.text
    assert "halo" in found.text.lower()
    assert "speech" in found.text.lower()


def test_search_finds_a_cached_ocr_record_by_its_text(server_session, sample):
    """`scope="ocr"` reads the `<entry>/ocr/*.json` records `media_ocr` wrote."""
    ocr = server_session.call("media_ocr", {"source": str(sample), "at": [1.0]})
    assert not ocr.is_error, ocr.text

    found = server_session.call(
        "media_search",
        {"source": str(sample), "query": "LANJUT", "scope": "ocr"},
    )
    assert not found.is_error, found.text
    assert "LANJUT" in found.text.upper()
    assert "00:00:01.000" in found.text
    assert "ocr" in found.text.lower()


def test_a_strict_transcript_records_its_own_identity_in_meta(server_session, tmp_path):
    """A strict run's meta.json carries the strict identity, not the sampled one."""
    # A clip of its own: the strict identity hashes content only, so two
    # byte-identical fixtures in different tests would share one cache entry.
    clip = fixtures.make_sample(
        tmp_path / "strict-sample.mp4", duration=5, counter_jump_at=3
    )
    result = server_session.call(
        "media_transcript", {"source": str(clip), "identity": "strict"}
    )
    assert not result.is_error, result.text
    strict = artifacts.resolve_source(str(clip), "strict")
    sampled = artifacts.resolve_source(str(clip), "sampled")
    assert strict.key != sampled.key
    meta = _entry_meta(server_session, strict.key)
    assert meta["identity"] == strict.key
    assert meta["kind"] == "file-strict"


def test_a_changed_identity_names_the_previous_entry_and_the_reason(
    server_session, tmp_path
):
    """A new identity states the previous entry was not reused, and why."""
    clip = fixtures.make_sample(tmp_path / "changed.mp4", duration=6, counter_jump_at=2)
    first = server_session.call("media_probe", {"source": str(clip)})
    assert not first.is_error, first.text
    sampled = artifacts.resolve_source(str(clip), "sampled")
    strict = artifacts.resolve_source(str(clip), "strict")

    second = server_session.call(
        "media_probe", {"source": str(clip), "identity": "strict"}
    )
    assert not second.is_error, second.text
    assert sampled.key in second.text, second.text
    assert strict.key in second.text
    lowered = second.text.lower()
    assert "not reused" in lowered
    assert "identity changed" in lowered


def test_a_repeated_identity_change_does_not_renotify_on_cache_hits(
    server_session, tmp_path
):
    """Once the changed entry exists, a further call reports no change."""
    clip = fixtures.make_sample(
        tmp_path / "changed-again.mp4", duration=7, counter_jump_at=3
    )
    server_session.call("media_probe", {"source": str(clip)})
    first_strict = server_session.call(
        "media_probe", {"source": str(clip), "identity": "strict"}
    )
    assert "not reused" in first_strict.text.lower()
    second_strict = server_session.call(
        "media_probe", {"source": str(clip), "identity": "strict"}
    )
    assert "not reused" not in second_strict.text.lower()
    assert "probe.json" in _cache_line(second_strict.text)


def test_a_heavy_job_evicts_to_stay_within_the_cache_cap(small_cache_session, sample):
    """Eviction runs after the job itself, with no media_cache(clean) call."""
    cache_root = Path(small_cache_session._env["MEDIA_CACHE_DIR"])
    cap = 500000
    old = cache_root / "entries" / "0000000000000001"
    old.mkdir(parents=True)
    (old / "meta.json").write_text(
        json.dumps(
            {
                "identity": "0000000000000001",
                "kind": "file-sampled",
                "source": "/old/clip.mp4",
                "created": 1.0,
                "last_used": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (old / "blob.bin").write_bytes(b"\0" * 600000)
    assert artifacts.total_bytes(cache_root) > cap

    result = small_cache_session.call(
        "media_frames", {"source": str(sample), "at": [1.0]}
    )
    assert not result.is_error, result.text
    assert not old.exists(), "the older entry should have been evicted by the job"
    assert artifacts.total_bytes(cache_root) <= cap
    assert "evicted" in _cache_line(result.text), result.text


def test_cache_lists_stats_and_cleans(server_session, sample):
    assert not server_session.call("media_probe", {"source": str(sample)}).is_error

    listing = server_session.call("media_cache", {"action": "list"})
    assert not listing.is_error, listing.text
    assert "entries" in listing.text.lower() and "sample.mp4" in listing.text

    stats = server_session.call("media_cache", {"action": "stats"})
    assert not stats.is_error, stats.text
    assert "total_bytes" in stats.text
    assert "cap" in stats.text.lower()
    assert "dependencies" in stats.text.lower()
    assert "transcribe" in stats.text.lower()

    clean = server_session.call("media_cache", {"action": "clean"})
    assert not clean.is_error, clean.text
    assert "evicted" in clean.text.lower()


def test_an_unknown_action_is_an_error_naming_the_allowed_values(server_session):
    result = server_session.call("media_cache", {"action": "wipe"})
    assert result.is_error and "list" in result.text.lower()


def test_a_missing_source_is_an_actionable_error(server_session, tmp_path):
    result = server_session.call("media_probe", {"source": str(tmp_path / "gone.mp4")})
    assert result.is_error
    assert "gone.mp4" in result.text


def test_a_url_source_says_url_ingestion_is_not_implemented(server_session):
    result = server_session.call(
        "media_probe", {"source": "https://example.invalid/clip.mp4"}
    )
    assert result.is_error
    assert "url" in result.text.lower()


# Helpers --------------------------------------------------------------------------


def _frame_paths(text: str) -> list[Path]:
    """Return the absolute paths the frames tool printed, in order."""
    paths = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            candidate = stripped.split("]", 1)[1].strip()
            if candidate.startswith("/"):
                paths.append(Path(candidate))
        elif stripped.startswith("/") and stripped.endswith(".png"):
            paths.append(Path(stripped))
    return paths


def _cache_line(text: str) -> str:
    """Return the status block's `cache:` line, or `""` when it is absent."""
    for line in text.splitlines():
        if line.startswith("cache:"):
            return line
    return ""


def _entry_meta(session: ServerSession, key: str) -> dict:
    """Read an entry's `meta.json` by cache key."""
    path = Path(session._env["MEDIA_CACHE_DIR"]) / "entries" / key / "meta.json"
    assert path.is_file(), f"no meta.json for entry {key}"
    return json.loads(path.read_text(encoding="utf-8"))


def _entry_for_source(session: ServerSession, source: Path) -> Path:
    """Return the cache entry directory whose meta names `source`."""
    entries = Path(session._env["MEDIA_CACHE_DIR"]) / "entries"
    for child in sorted(entries.iterdir()):
        meta_path = child / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        recorded = meta.get("source")
        if isinstance(recorded, str) and Path(recorded).resolve() == source.resolve():
            return child
    raise AssertionError(f"no cache entry records the source {source}")


def test_the_session_wrote_nothing_into_the_repository(server_session, sample):
    """The repository stays clean: artifacts land under the cache root only."""
    before = _repo_fingerprint()
    assert not server_session.call(
        "media_frames", {"source": str(sample), "at": [1.0]}
    ).is_error
    assert _repo_fingerprint() == before


def test_the_repository_fingerprint_catches_an_in_place_rewrite(tmp_path):
    """A same-size rewrite of one file is invisible to a path-set fingerprint."""
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("original\n", encoding="utf-8")
    before = _repo_fingerprint(tmp_path)
    tracked.write_text("modified\n", encoding="utf-8")
    after = _repo_fingerprint(tmp_path)
    assert set(after) == set(before), "the path set alone cannot tell them apart"
    assert len("original\n") == len("modified\n")
    assert after != before, "the fingerprint must hash content, not just names"


def _repo_fingerprint(root: Path = REPO_ROOT) -> dict[str, str]:
    """Fingerprint every file under `root` by relative path and content hash.

    Content, not just the path set: the requirement this guards ("the server
    writes only inside the cache entry") is violated just as much by an
    in-place rewrite of a tracked file as by a new one, and only a hash sees
    the difference. The venv, git metadata, and the interpreter/tooling caches
    (`__pycache__`, `.pytest_cache`, `.ruff_cache`) stay excluded: they are not
    the repository's deliverables, they are written by the test process and the
    interpreter rather than by the server, and a `__pycache__` header can carry
    the source file's mtime on interpreters that use timestamp invalidation, so
    hashing them would make this test depend on the bytecode-cache mode.
    """
    excluded = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and not excluded.intersection(path.parts)
    }


def test_the_status_block_is_machine_readable_json_free_text(server_session, sample):
    text = server_session.call("media_probe", {"source": str(sample)}).text
    last_four = text.strip().splitlines()[-4:]
    labels = [line.split(":", 1)[0] for line in last_four]
    assert labels == ["computed", "cache", "capped", "skipped"]
    # The probe caches its result, so the second call reports the reuse.
    again = server_session.call("media_probe", {"source": str(sample)}).text
    assert "probe.json" in again


def test_probe_json_is_cached_under_the_entry(server_session, sample):
    server_session.call("media_probe", {"source": str(sample)})
    entries = list((Path(server_session._env["MEDIA_CACHE_DIR"]) / "entries").iterdir())
    assert entries, "the probe should have created a cache entry"
    assert (entries[0] / "probe.json").is_file()


# The timeline projection (design lines 268 and 285) --------------------------------


def _timeline_layers(timeline: Path) -> list[str]:
    """Return the layer of every line of a projected `timeline.md`."""
    return [
        line.split("] ", 1)[1].split(" | ", 1)[0]
        for line in timeline.read_text(encoding="utf-8").splitlines()
        if line.startswith("[")
    ]


def test_a_manifest_projects_scene_and_audio_lines_into_the_timeline(
    server_session, sample
):
    """The manifest rebuilds `timeline.md` from the entry's cached artifacts."""
    result = server_session.call("media_manifest", {"source": str(sample)})
    assert not result.is_error, result.text
    timeline = _entry_for_source(server_session, sample) / "timeline.md"
    assert timeline.is_file(), "a manifest must project the entry's timeline"
    layers = _timeline_layers(timeline)
    assert "scene" in layers and "audio" in layers
    assert layers.count("scene") == 1


def test_a_transcript_projects_speech_lines_into_the_timeline(server_session, sample):
    """`media_transcript` projects its segments onto the `speech` layer."""
    result = server_session.call("media_transcript", {"source": str(sample)})
    assert not result.is_error, result.text
    timeline = _entry_for_source(server_session, sample) / "timeline.md"
    assert timeline.is_file(), "a transcript must project the entry's timeline"
    text = timeline.read_text(encoding="utf-8")
    assert "[00:00:00.000] speech | SPEAKER_00: halo" in text, text


def test_an_ocr_track_run_records_a_candidate_that_a_manifest_keeps(
    server_session, tmp_path
):
    """Track candidates are persisted, and a later rebuild does not erase them."""
    clip = fixtures.make_value_flicker(
        tmp_path / "flicker-timeline.mp4", flick_seconds=0.5
    )
    result = server_session.call(
        "media_ocr",
        {
            "source": str(clip),
            "at": 4.0,
            "window": 1.0,
            "fps": 2,
            "track": True,
        },
    )
    assert not result.is_error, result.text
    entry = _entry_for_source(server_session, clip)
    timeline = entry / "timeline.md"
    assert timeline.is_file(), "an OCR run must project the entry's timeline"
    projected = [
        line
        for line in timeline.read_text(encoding="utf-8").splitlines()
        if line.startswith("[") and "ocr-change | " in line
    ]
    assert len(projected) == 1, projected
    assert projected[0].startswith("[00:00:04.000] ocr-change | ")
    assert "48" in projected[0]
    assert (entry / "ocr_changes.json").is_file()

    again = server_session.call("media_manifest", {"source": str(clip)})
    assert not again.is_error, again.text
    after = timeline.read_text(encoding="utf-8")
    assert projected[0] in after, "a manifest rebuild must not erase track candidates"
    assert "ocr | " in after, "the OCR records stay projected too"
