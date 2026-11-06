"""The MCP tool layer: seven tools over the earlier media modules.

This module is the only place where the environment becomes configuration
(`load_settings`), where the shared parameters are validated, where a
`profile` expands into declared work, and where every response is rendered
with the status block the design requires: what was computed, what came from
cache, what was capped, and what was skipped and why.

Each tool body is a plain synchronous function; the registered tool wraps it
and runs it on a worker thread, because every stage shells out through
`media.run_child`. Running the body off the event loop means a cancelled MCP
request can arrive at an await point, and the handler can then stop the child
process group it started (`media.terminate_active_children`) before the
cancellation propagates. `main` runs the same stop in a `finally`, so stdin
EOF, SIGTERM and SIGINT all leave no ffmpeg or whisper process behind.

No environment variable is read at import time: every tool calls
`load_settings` when it runs. `media.py` remains the only module that reads
`MEDIA_FONT_FILE` directly.
"""

from __future__ import annotations

import base64
import fcntl
import json
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# The server is launched exactly as the clients launch it (`python
# media/server.py`), so the script's own directory is on `sys.path` and the
# repository root — the parent of this package — is not. Adding it here is
# what lets the `media.*` package imports below resolve; the tests and the
# clients both exercise this same entry point.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ContentBlock, ImageContent, TextContent

from media import artifacts, contracts, ingest, media
from media.media import MediaError, _image_stats, quality_tier

# The shared parameter ceilings of the design document. `count` and `fps` and
# `k` are hard caps; `timeout` has a default and a ceiling. The frame budget
# shares `count`'s default and cap (8, hard cap 60): it bounds the ranked
# candidate list a manifest hands to the visual stages, never a
# rasterization (that is `count`'s own job).
COUNT_DEFAULT = 8
COUNT_CAP = 60
# `media_frames(image_inline=true)` attaches each returned PNG to the tool
# result as image content. The frame budget's own cap is 60, and inlining 60
# PNGs is not a defensible default, so the inline route is bounded here; when
# more frames exist than this, the first `INLINE_IMAGE_CAP` are attached and
# the status block reports the cap.
INLINE_IMAGE_CAP = 8
FPS_DEFAULT = 2
FPS_CAP = 30
K_DEFAULT = 8
K_CAP = 20
TIMEOUT_DEFAULT = 900
TIMEOUT_CEILING = 21600
FRAME_BUDGET_DEFAULT = 8
SCENE_THRESHOLD_DEFAULT = 10.0
# `media_transcript`'s declared defaults: two speakers and the pipeline's own
# `auto` language choice, so a plain call diarizes and lets the pipeline detect
# the language. The manifest's transcript stage runs the same defaults.
NUM_SPEAKERS_DEFAULT = 2
LANGUAGE_DEFAULT = "auto"

# The absolute silence gate and the relative loudness margin that
# `media.audio_events` applies by default. The manifest quotes both whenever the
# loud list is empty, so a reader never mistakes "no peak cleared the rule" for
# "nothing was loud" (finding R30).
_SILENCE_GATE_DB = -30.0
_LOUD_MARGIN_DB = media._LOUD_MARGIN_DB
_LOUDNESS_QUALIFICATION = (
    f"The loudness gate is {media._plain_decimal(_SILENCE_GATE_DB)} dB and a loud peak "
    f"must clear the file's own median frame level by "
    f"{media._plain_decimal(_LOUD_MARGIN_DB)} dB, so an empty loud list means no peak "
    "cleared that rule, not no loud moment."
)
MAX_EDGE_DEFAULT = 1600
CACHE_CAP_DEFAULT = 20 * 1024**3
MAX_SOURCE_BYTES_DEFAULT = 4 * 1024**3
MAX_SOURCE_SECONDS_DEFAULT = 14400.0
VRAM_FLOOR_DEFAULT = 2800

# `profile` presets over the six pipeline stages. A profile is the only place
# heavy work is implied; every other tool runs exactly what its parameters ask
# for. Bridging `read`/`visual` caps into `full` is what "both capped" means.
_STAGES = ("probe", "scenes", "audio", "transcript", "ocr", "frames")
PROFILES: dict[str, dict[str, bool]] = {
    "manifest": {
        "probe": True,
        "scenes": True,
        "audio": True,
        "transcript": False,
        "ocr": False,
        "frames": False,
    },
    "read": {
        "probe": True,
        "scenes": True,
        "audio": True,
        "transcript": True,
        "ocr": False,
        "frames": False,
    },
    "visual": {
        "probe": True,
        "scenes": True,
        "audio": True,
        "transcript": False,
        "ocr": True,
        "frames": True,
    },
    "full": dict.fromkeys(_STAGES, True),
}

# The timestamp forms the tools accept. The result is always rendered back
# through `artifacts.format_timestamp`, so the project has exactly one
# timestamp spelling on the wire.
_STAMPED = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")

# The file-name stamp of a frame: the project timestamp with the separators
# replaced (`00:00:01.000` -> `000001-000`), so a `frames/` directory sorts in
# timeline order and needs no index.
_TILE_MODES = ("sheet", "filmstrip", "individual")
_QUEUE_MODES = ("wait", "fail")
_SEARCH_SCOPES = ("all", "transcript", "ocr")
_IDENTITIES = ("sampled", "strict")

# Worker-thread poll cadence, and how long a stop path may wait.
_JOB_POLL_SECONDS = 0.05

# The one lock that serializes heavy work, relative to the cache root.
_LOCK_NAME = "transcribe.lock"
_HOLDER_NAME = "transcribe.lock.holder"

# Open descriptors of held locks, keyed by resolved lock path, so the public
# `acquire_transcribe_lock` can return a note rather than a file object while
# still keeping the lock held for the caller.
_HELD_LOCKS: dict[str, int] = {}


@dataclass(frozen=True)
class Settings:
    """The environment, resolved once per call into plain values."""

    cache_dir: Path
    cache_cap_bytes: int
    device: str
    vram_floor_mib: int
    frame_max_edge: int
    max_source_bytes: int
    max_source_seconds: float
    allowed_read_roots: list[Path] | None
    call_timeout_seconds: int
    allow_private_urls: bool


_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
}
_SIZE = re.compile(r"^(\d+(?:\.\d+)?)\s*([a-z]*)$")


def _parse_size(text: str) -> int:
    """Parse `20 GiB` / `1 GiB` / a plain byte count into an integer."""
    match = _SIZE.fullmatch(text.strip().lower())
    if match is None or match.group(2) not in _SIZE_UNITS:
        raise ValueError(f"not a size: {text!r}")
    return int(float(match.group(1)) * _SIZE_UNITS[match.group(2)])


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value is not None and value.strip() else None


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _env_size(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return _parse_size(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_flag(name: str) -> bool:
    value = _env(name)
    return value is not None and value.strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_roots(name: str) -> list[Path] | None:
    value = _env(name)
    if value is None:
        return None
    roots = [Path(part).expanduser() for part in value.split(":") if part.strip()]
    return roots or None


def load_settings() -> Settings:
    """Read every `MEDIA_*` setting, each call, into a frozen `Settings`.

    A value that cannot be parsed falls back to the documented default rather
    than failing every tool call: one typo in a client configuration must not
    make the whole server unusable. The per-call timeout is clamped to
    `TIMEOUT_CEILING` so an over-eager environment value cannot ask for an
    unbounded child.
    """
    timeout = _env_int("MEDIA_CALL_TIMEOUT_SECONDS", TIMEOUT_DEFAULT)
    timeout = max(1, min(timeout, TIMEOUT_CEILING))
    allowed_roots = _env_roots("MEDIA_ALLOWED_READ_ROOTS")
    return Settings(
        cache_dir=Path(
            _env("MEDIA_CACHE_DIR") or Path.home() / ".cache" / "media-mcp"
        ).expanduser(),
        cache_cap_bytes=_env_size("MEDIA_CACHE_CAP", CACHE_CAP_DEFAULT),
        device=_env("MEDIA_DEVICE") or "auto",
        vram_floor_mib=_env_int("MEDIA_VRAM_FLOOR", VRAM_FLOOR_DEFAULT),
        frame_max_edge=_env_int("MEDIA_FRAME_MAX_EDGE", MAX_EDGE_DEFAULT),
        max_source_bytes=_env_size("MEDIA_MAX_SOURCE_BYTES", MAX_SOURCE_BYTES_DEFAULT),
        max_source_seconds=_env_float(
            "MEDIA_MAX_SOURCE_SECONDS", MAX_SOURCE_SECONDS_DEFAULT
        ),
        allowed_read_roots=allowed_roots,
        call_timeout_seconds=timeout,
        allow_private_urls=_env_flag("MEDIA_ALLOW_PRIVATE_URLS"),
    )


# ---------------------------------------------------------------------------
# Shared parameter semantics
# ---------------------------------------------------------------------------


def parse_timestamp(value: object) -> float:
    """Parse `HH:MM:SS.mmm`, `MM:SS`, or plain seconds into a float.

    Negative and unparseable values raise a `MediaError` naming the offending
    value, because a timestamp the caller cannot retype is unusable. The
    result is always a non-negative number of seconds in content time.
    """
    if isinstance(value, bool):
        raise MediaError(f"a boolean is not a timestamp: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        text = value.strip()
        match = _STAMPED.fullmatch(text)
        if match is not None:
            hours, minutes, rest = match.groups()
            seconds = int(hours or 0) * 3600 + int(minutes) * 60 + float(rest)
        else:
            try:
                seconds = float(text)
            except ValueError:
                raise MediaError(
                    f"cannot parse {value!r} as a timestamp; use HH:MM:SS.mmm, "
                    "MM:SS, or a number of seconds"
                ) from None
    else:
        raise MediaError(f"cannot parse {value!r} as a timestamp")
    if seconds < 0:
        raise MediaError(f"a timestamp may not be negative: {value!r}")
    return seconds


def _stamp(seconds: float) -> str:
    """Render a timestamp with the project's one formatter."""
    return artifacts.format_timestamp(seconds)


def _file_stamp(seconds: float) -> str:
    """Render a file-name-safe stamp for the `frames/` directory."""
    return _stamp(seconds).replace(":", "").replace(".", "-")


def _counted(count: int, noun: str) -> str:
    """Write ``<count> <noun>`` with a real plural, so prose reads as English."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _stream_start(seconds: float | None) -> str:
    """Render a stream's raw `start_time`, or `(absent)` when it has none."""
    return "(absent)" if seconds is None else f"{seconds:.3f} s"


def validate_count(
    value: object,
    default: int = COUNT_DEFAULT,
    cap: int = COUNT_CAP,
    name: str = "count",
) -> int:
    """Validate a whole-number parameter, naming it and the cap when refused.

    `name` lets a caller with its own bounded integer (the OCR `psm` range)
    reuse this check without the refusal naming the wrong parameter.
    """
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MediaError(
            f"{name} must be a whole number between 1 and {cap}, not {value!r}"
        )
    if float(value) != int(value):
        raise MediaError(
            f"{name} must be a whole number between 1 and {cap}, not {value!r}"
        )
    number = int(value)
    if number < 1 or number > cap:
        raise MediaError(f"{name} must be between 1 and {cap}, not {value!r}")
    return number


def validate_fps(value: object, default: int = FPS_DEFAULT, cap: int = FPS_CAP) -> int:
    """Validate a `window` sampling rate against its hard cap."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MediaError(
            f"fps must be a whole number between 1 and {cap}, not {value!r}"
        )
    if float(value) != int(value):
        raise MediaError(
            f"fps must be a whole number between 1 and {cap}, not {value!r}"
        )
    number = int(value)
    if number < 1 or number > cap:
        raise MediaError(f"fps must be between 1 and {cap}, not {value!r}")
    return number


def validate_k(value: object, default: int = K_DEFAULT, cap: int = K_CAP) -> int:
    """Validate a search `k` against its hard cap."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MediaError(f"k must be a whole number between 1 and {cap}, not {value!r}")
    if float(value) != int(value):
        raise MediaError(f"k must be a whole number between 1 and {cap}, not {value!r}")
    number = int(value)
    if number < 1 or number > cap:
        raise MediaError(f"k must be between 1 and {cap}, not {value!r}")
    return number


def validate_window(at: object, window: object) -> float | None:
    """Validate the `window` parameter against the shape of `at`.

    `window` is a span in seconds around a single timestamp. A window with a
    list of timestamps, with the literal `"scenes"`, or with no timestamp at
    all is an ambiguous request and it is refused rather than silently applied
    to the first element. Returns the span, or `None` when no window was
    requested.
    """
    if window is None:
        return None
    if isinstance(window, bool) or not isinstance(window, (int, float)):
        raise MediaError(f"window must be a number of seconds, not {window!r}")
    span = float(window)
    if span <= 0:
        raise MediaError(f"window must be a positive number of seconds, not {window!r}")
    if at is None or isinstance(at, (list, tuple)):
        raise MediaError(
            "window requires a single timestamp, so it cannot be combined with a "
            f"list of timestamps or with no timestamp (at={at!r})"
        )
    if isinstance(at, str) and at.strip().lower() == "scenes":
        raise MediaError(
            "window requires a single timestamp, so it cannot be combined with "
            "at='scenes'"
        )
    return span


def validate_crop(spec: object) -> object:
    """Validate a `crop` request: `None`, `"auto"`, or a complete pixel box.

    Only this layer's integrality and completeness checks live here; fitting
    the box inside the measured frame happens against the real probe, so the
    rejection can quote the frame size the file actually has.
    """
    if spec is None or spec == "auto":
        return spec
    if not isinstance(spec, dict):
        raise MediaError(
            f"crop must be None, 'auto', or a box with x/y/width/height keys, not {spec!r}"
        )
    allowed = {"x", "y", "width", "height"}
    unknown = sorted(key for key in spec if key not in allowed)
    if unknown:
        raise MediaError(
            f"crop has unknown key(s) {', '.join(repr(key) for key in unknown)}; "
            "crop accepts only x, y, width, and height"
        )
    box: dict[str, int] = {}
    for key in ("x", "y", "width", "height"):
        if key not in spec:
            raise MediaError(
                f"crop is missing the required key {key!r} (crop needs all of "
                "x, y, width, and height)"
            )
        value = spec[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MediaError(f"crop key {key!r} must be a number, not {value!r}")
        if float(value) != int(value):
            raise MediaError(f"crop key {key!r} must be a whole number, not {value!r}")
        box[key] = int(value)
    if box["width"] <= 0:
        raise MediaError(f"crop width must be positive, not {box['width']}")
    if box["height"] <= 0:
        raise MediaError(f"crop height must be positive, not {box['height']}")
    if box["x"] < 0:
        raise MediaError(f"crop x must not be negative, not {box['x']}")
    if box["y"] < 0:
        raise MediaError(f"crop y must not be negative, not {box['y']}")
    return box


def validate_choice(
    value: object, default: str, allowed: tuple[str, ...], name: str
) -> str:
    """Validate a small enumerated parameter, naming the allowed values."""
    if value is None:
        return default
    if not isinstance(value, str) or value not in allowed:
        raise MediaError(f"{name} must be one of {', '.join(allowed)}, not {value!r}")
    return value


# ---------------------------------------------------------------------------
# State honesty
# ---------------------------------------------------------------------------


def status_block(
    computed: list[str],
    cached: list[str],
    capped: list[str],
    skipped: dict[str, str],
) -> str:
    """Render the four status lines appended to every tool response.

    The four labels are always present and always in this order, so the block
    is machine-readable free text: a reader (human or agent) can see what was
    computed, what was reused from cache, what was capped, and what was
    skipped with a reason. An empty group says `(none)` rather than vanishing;
    a capped or skipped stage is always stated, never silent.
    """
    return "\n".join(
        (
            _status_line("computed", computed),
            _status_line("cache", cached),
            _status_line("capped", capped),
            _status_line("skipped", skipped),
        )
    )


def _status_line(label: str, group: object) -> str:
    if isinstance(group, dict):
        items = [f"{key}: {reason}" for key, reason in group.items()]
    else:
        items = [str(item) for item in group]  # type: ignore[union-attr]
    if not items:
        return f"{label}: (none)"
    return f"{label}: " + "; ".join(items)


@dataclass
class _Report:
    """The running account of one tool call, rendered by `status_block`."""

    computed: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    capped: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    extras: list[str] = field(default_factory=list)

    def status(self) -> str:
        return status_block(self.computed, self.cached, self.capped, self.skipped)


# ---------------------------------------------------------------------------
# The single-flight transcription lock
# ---------------------------------------------------------------------------


def _lock_paths(settings: Settings) -> tuple[Path, Path]:
    locks = settings.cache_dir / "locks"
    return locks / _LOCK_NAME, locks / _HOLDER_NAME


def _read_holder(holder_path: Path) -> str:
    """Describe the holder recorded in the sidecar file, or say it is unknown."""
    try:
        data = json.loads(holder_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "holder unknown"
    if not isinstance(data, dict):
        return "holder unknown"
    parts = []
    if "pid" in data:
        parts.append(f"pid {data['pid']}")
    if data.get("source"):
        parts.append(f"source {data['source']}")
    if data.get("started"):
        parts.append(f"since {data['started']}")
    return ", ".join(parts) if parts else "holder unknown"


def acquire_transcribe_lock(
    settings: Settings,
    source: str,
    queue: str,
    cancel: threading.Event | None = None,
    timeout_seconds: int | None = None,
) -> str | None:
    """Take the exclusive transcription lock, reporting who held it.

    `queue="fail"` returns immediately with the holder's identity when the
    lock is taken; `queue="wait"` reports the same facts and then waits, and
    its return value is a note naming the holder and the wait, which the
    caller states in its response. The lock is an exclusive `flock` on
    `<cache>/locks/transcribe.lock` with a JSON sidecar naming the holder, so
    a second process can explain the wait instead of guessing. A `cancel`
    event (or the deadline) interrupts a wait and raises `MediaError`.
    """
    lock_path, holder_path = _lock_paths(settings)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    limit = (
        settings.call_timeout_seconds if timeout_seconds is None else timeout_seconds
    )
    deadline = time.monotonic() + max(1, int(limit))
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    waited = False
    held_by = "holder unknown"
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                held_by = _read_holder(holder_path)
                if queue == "fail":
                    raise MediaError(
                        f"transcription is busy: another job holds "
                        f"{lock_path.name} ({held_by}); queue='fail' returns "
                        "immediately instead of waiting"
                    ) from None
                waited = True
                if cancel is not None and cancel.is_set():
                    raise MediaError(
                        f"cancelled while waiting for {lock_path.name} ({held_by})"
                    ) from None
                if time.monotonic() >= deadline:
                    raise MediaError(
                        f"timed out waiting for {lock_path.name} ({held_by})"
                    ) from None
                time.sleep(_JOB_POLL_SECONDS)
        _HELD_LOCKS[str(lock_path)] = descriptor
        _write_holder(
            holder_path,
            {"pid": os.getpid(), "source": source, "started": round(time.time(), 3)},
        )
    except BaseException:
        os.close(descriptor)
        raise
    if not waited:
        return None
    return f"waited for {lock_path.name}; it was held by {held_by}, then handed over"


def release_transcribe_lock(settings: Settings) -> None:
    """Release a lock this process holds; a no-op when it holds none."""
    lock_path, holder_path = _lock_paths(settings)
    descriptor = _HELD_LOCKS.pop(str(lock_path), None)
    if descriptor is None:
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
        holder_path.unlink(missing_ok=True)


def _write_holder(holder_path: Path, payload: dict[str, object]) -> None:
    holder_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Running a tool body off the event loop
# ---------------------------------------------------------------------------


async def _execute(work: Any) -> Any:
    """Run `work(cancel)` on a daemon thread, bounded by cancellation.

    The body is synchronous because every stage shells out through
    `media.run_child`, which blocks. Parking it on a thread leaves the event
    loop free to deliver the peer's cancellation at the `await` here; when
    that happens the cancel event is set and every live child process group is
    stopped before the cancellation propagates, so a cancelled request leaves
    no ffmpeg or whisper process behind and the connection stays usable. The
    body's own value is returned untouched, so a tool may answer with a plain
    text string or with a list of MCP content blocks.
    """
    cancel = threading.Event()
    outcome: dict[str, Any] = {}

    def runner() -> None:
        try:
            outcome["value"] = work(cancel)
        except BaseException as err:  # noqa: BLE001 - re-raised below
            outcome["error"] = err

    thread = threading.Thread(target=runner, daemon=True, name="media-job")
    thread.start()
    try:
        while thread.is_alive():
            await anyio.sleep(_JOB_POLL_SECONDS)
    except BaseException:
        cancel.set()
        media.terminate_active_children()
        raise
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value", "")


async def _guarded(work: Any) -> Any:
    """Turn an anticipated `MediaError` into an actionable `ToolError`.

    The SDK withholds the text of any exception that is not a `ToolError`, so
    every failure a caller can act on is converted here; a cancellation is a
    `BaseException` and passes through untouched.
    """
    try:
        return await _execute(work)
    except MediaError as err:
        raise ToolError(str(err)) from err


# ---------------------------------------------------------------------------
# Cache entries and artifacts
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _record_entry(entry: Path, identity: artifacts.Identity) -> None:
    """Write the entry's `meta.json`, keeping `created` and refreshing `last_used`."""
    meta = artifacts.read_meta(entry)
    meta = dict(meta) if isinstance(meta, dict) else {}
    now = time.time()
    meta["identity"] = identity.key
    meta["kind"] = identity.kind
    meta["source"] = identity.source
    meta.setdefault("created", now)
    meta["last_used"] = now
    artifacts.write_meta(entry, meta)


def _evict_after_job(settings: Settings, entry: Path, report: _Report) -> None:
    """Enforce `MEDIA_CACHE_CAP` now that this job's artifacts are on disk.

    The design states the cap is enforced after each job and that eviction runs
    once the current job finishes, so an in-flight job is never evicted
    underneath itself: the entry this call just wrote is protected and every
    other entry is a candidate, oldest `last_used` first. When entries were
    removed the status block's `cache:` line carries the count, so a caller can
    see that the cache shrank without running `media_cache(action="clean")`.
    """
    evicted = artifacts.evict(
        settings.cache_dir, settings.cache_cap_bytes, protect=entry
    )
    if evicted:
        report.cached.append(f"evicted {len(evicted)}")


def _local_source_path(settings: Settings, source: str) -> Path:
    """Validate a local `source` and return the path every later stage reads.

    The scheme refusal, the `MEDIA_ALLOWED_READ_ROOTS` allowlist and the
    readable-file check all run here, before anything identifies, opens or
    hashes the file.
    """
    parsed = urlparse(source)
    if parsed.scheme and len(parsed.scheme) > 1:
        raise MediaError(
            f"unsupported source scheme {parsed.scheme!r} in {source!r}: only "
            "local paths and http/https URLs are accepted"
        )
    return media.validate_source(source, settings.allowed_read_roots)


def _download_source(
    settings: Settings, source: str, entry: Path, report: _Report
) -> Path:
    """Download an http/https `source` into `entry`'s `media/` directory.

    `media.ingest` applies the private-range guard (again, after the caller's
    pre-flight check) and the byte and duration caps; the download or cache hit
    is folded into `report`, so every tool that takes a source states it in the
    same status block.
    """
    ingested = ingest.ingest_url(
        source,
        entry,
        max_bytes=settings.max_source_bytes,
        max_seconds=settings.max_source_seconds,
        allow_private=settings.allow_private_urls,
        timeout_seconds=settings.call_timeout_seconds,
    )
    _record_ingest(report, ingested)
    return ingested.path


def _identity_for(source: str, identity: str) -> artifacts.Identity:
    """Resolve a source to its content identity, converting failures."""
    try:
        return artifacts.resolve_source(source, identity)
    except (FileNotFoundError, ValueError) as err:
        raise MediaError(f"could not identify source {source}: {err}") from err


def _record_ingest(report: _Report, ingested: ingest.IngestedSource) -> None:
    """State the URL download, or the cache hit that replaced it.

    A first download is work this call computed and names the new
    `media/<file>`; a reused one is a cache hit under the same name. When the
    child left stderr behind, its last line travels with the download, so a
    yt-dlp warning is visible without a second run.
    """
    name = f"media/{ingested.path.name}"
    if ingested.from_cache:
        report.cached.append(name)
        return
    note = f"download {name} ({ingested.bytes} bytes)"
    tail = ingested.stderr_tail.splitlines()
    if tail:
        note += f"; yt-dlp stderr: {tail[-1]}"
    report.computed.append(note)


def _entry_for(
    settings: Settings, source: str, identity: str, report: _Report
) -> tuple[artifacts.Identity, Path, Path, str | None]:
    """Resolve a source to its identity, local path, and cache entry.

    Order matters here, and tests assert it: a local path is validated (scheme
    refusal, allowlist, readability) *before* it is identified, so a refused
    path is never opened and never creates a cache entry; an http/https URL that
    is *not* already cached is guarded against private targets *before* its
    entry directory exists, while a URL that already holds a completed download
    is served without any network access (not even a DNS lookup), which is the
    cache promise a hit makes. A URL's identity is the URL itself. The fourth
    value is a note stating that an older entry for the same source with a
    different key was not reused, and why. Design section 12 requires that
    sentence whenever a source resolves to a new entry, so the detection lives
    here, at the one place every entry resolution passes through, and every tool
    body surfaces the note in its status block. `report` receives the URL
    download (or cache hit).
    """
    identity = validate_choice(identity, "sampled", _IDENTITIES, "identity")
    if ingest.is_url(source):
        resolved = _identity_for(source, identity)
        entry = artifacts.entry_path(resolved, settings.cache_dir)
        # `cached_download` names an unreadable entry as a MediaError, which
        # `/guarded` turns into an actionable tool error.
        if ingest.cached_download(entry) is None:
            ingest.check_target_allowed(source, settings.allow_private_urls)
            entry = artifacts.entry_dir(resolved, settings.cache_dir)
        path = _download_source(settings, source, entry, report)
    else:
        path = _local_source_path(settings, source)
        resolved = _identity_for(str(path), identity)
        entry = artifacts.entry_dir(resolved, settings.cache_dir)
    return resolved, path, entry, _previous_entry_note(settings, entry, resolved)


def _previous_entry_note(
    settings: Settings, entry: Path, resolved: artifacts.Identity
) -> str | None:
    """Name an older entry for the same source that was not reused, or `None`.

    Only a genuinely new entry is examined: an entry that already holds a
    readable `meta.json` is this identity's own cache hit, and rescanning for a
    previous entry on every cache hit would be work with no change to report.
    Otherwise the cache is scanned for an entry whose recorded `source` (the
    resolved path it was created from) matches and whose identity differs.
    """
    try:
        own_meta = artifacts.read_meta(entry)
    except (OSError, ValueError):
        own_meta = None
    if own_meta is not None:
        return None
    entries_root = settings.cache_dir / "entries"
    if not entries_root.is_dir():
        return None
    for child in sorted(entries_root.iterdir()):
        if child == entry or not child.is_dir():
            continue
        try:
            meta = artifacts.read_meta(child)
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict) or meta.get("source") != resolved.source:
            continue
        older_key = meta.get("identity")
        if older_key == resolved.key:
            continue
        older_kind = str(meta.get("kind") or "unknown")
        older_name = str(older_key) if older_key else child.name
        return (
            f"the previous entry {older_name} ({older_kind}) for this source was "
            f"not reused because the identity changed to {resolved.key} "
            f"({resolved.kind}); a changed file sample or a different requested "
            "identity mode are the reasons a key changes"
        )
    return None


def _record_identity_note(report: _Report, note: str | None) -> None:
    """Surface an identity-change note in the response's status block."""
    if note is not None:
        report.skipped["identity"] = note


def _ensure_probe(entry: Path, path: Path, report: _Report) -> media.ProbeResult:
    """Return the probe result, from `probe.json` when it is already cached."""
    probe_path = entry / "probe.json"
    data = _read_json(probe_path)
    if isinstance(data, dict):
        try:
            result = media.ProbeResult(**data)
        except TypeError as err:
            report.skipped["probe.json"] = (
                f"cached probe.json does not match the probe shape ({err}); re-probing"
            )
        else:
            report.cached.append("probe.json")
            return result
    result = media.probe(path)
    _write_json(probe_path, asdict(result))
    report.computed.append("probe")
    return result


def _ensure_audio_events(
    entry: Path, path: Path, report: _Report
) -> list[media.AudioEvent]:
    """Return the audio events, from `audio_events.json` when cached."""
    events_path = entry / "audio_events.json"
    data = _read_json(events_path)
    if isinstance(data, list):
        try:
            events = [
                media.AudioEvent(
                    timestamp=float(item["timestamp"]),
                    kind=str(item["kind"]),
                    detail=str(item["detail"]),
                )
                for item in data
            ]
        except (KeyError, TypeError, ValueError) as err:
            report.skipped["audio_events.json"] = (
                f"cached audio_events.json is unreadable ({err}); re-detecting"
            )
        else:
            report.cached.append("audio_events.json")
            return events
    events = media.detect_audio_events(path)
    _write_json(
        events_path,
        [
            {"timestamp": event.timestamp, "kind": event.kind, "detail": event.detail}
            for event in events
        ],
    )
    report.computed.append("audio")
    return events


def _hash_extractor(
    path: Path, crop: media.Box | None, max_edge: int, report: _Report
) -> Any:
    """Build the perceptual peek `sample_candidates` uses for dedup.

    A candidate whose frame cannot be sampled keeps a stable, unique digest
    instead of collapsing into its neighbours: a failure to look at a frame is
    stated in `skipped`, never silently merged away.
    """

    def extract(timestamp: float) -> int:
        try:
            return media.diff_hash(media.extract_frame(path, timestamp, crop, max_edge))
        except MediaError as err:
            report.skipped.setdefault(
                "frames",
                f"a candidate at {_stamp(timestamp)} could not be sampled ({err}); "
                "kept rather than merged",
            )
            return media.diff_hash_of(f"unsampled:{timestamp}".encode())

    return extract


def _ensure_scene_candidates(
    entry: Path,
    path: Path,
    result: media.ProbeResult,
    threshold: float,
    budget: int,
    report: _Report,
) -> tuple[list[media.SceneScore], list[media.Candidate], dict[str, int]]:
    """Return scenes, ranked candidates, and the budget accounting.

    Everything is cached in `scenes.json`: when that file carries a candidate
    list the whole scene/rank stage is reused, so the perceptual hashing of
    stage 2 is paid once per entry.
    """
    scenes_path = entry / "scenes.json"
    events = _ensure_audio_events(entry, path, report)
    data = _read_json(scenes_path)
    if isinstance(data, dict) and "scenes" in data and "candidates" in data:
        try:
            scenes = [
                media.SceneScore(float(item["timestamp"]), float(item["score"]))
                for item in data["scenes"]
            ]
            candidates = [
                media.Candidate(
                    float(item["timestamp"]), str(item["reason"]), float(item["score"])
                )
                for item in data["candidates"]
            ]
            accounting = {
                str(key): int(value)
                for key, value in dict(data.get("accounting") or {}).items()
            }
        except (KeyError, TypeError, ValueError) as err:
            report.skipped["scenes.json"] = (
                f"cached scenes.json is unreadable ({err}); re-detecting"
            )
        else:
            report.cached.append("scenes.json")
            return scenes, candidates, accounting

    if isinstance(data, dict) and "scenes" in data:
        try:
            scenes = [
                media.SceneScore(float(item["timestamp"]), float(item["score"]))
                for item in data["scenes"]
            ]
        except (KeyError, TypeError, ValueError):
            scenes = []
        else:
            report.cached.append("scenes.json")
    else:
        scenes = media.detect_scenes(path, threshold=threshold)
        report.computed.append("scenes")

    sampled = media.sample_candidates(
        scenes,
        result.duration,
        interval=media.interval,
        dedup_distance=media.dedup_distance,
        extract=_hash_extractor(path, None, MAX_EDGE_DEFAULT, report),
    )
    candidates, accounting = media.rank_candidates(sampled, events, budget=budget)
    _write_json(
        scenes_path,
        {
            "threshold": threshold,
            "duration": result.duration,
            "scenes": [
                {"timestamp": scene.timestamp, "score": scene.score} for scene in scenes
            ],
            "candidates": [
                {
                    "timestamp": candidate.timestamp,
                    "reason": candidate.reason,
                    "score": candidate.score,
                }
                for candidate in candidates
            ],
            "accounting": accounting,
        },
    )
    return scenes, candidates, accounting


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def _choose_tier(
    settings: Settings, model: str | None, device: str | None
) -> tuple[str, str, str]:
    """Choose the model tier and device, stating any downgrade and its reason.

    An explicit `model` or `device` is honoured exactly, precheck or not. When
    both are left to the server, free VRAM is measured once: at or above
    `MEDIA_VRAM_FLOOR` the top tier is kept; below it the tier drops to
    `medium` on CPU, and the measured free MiB is part of the reason so the
    change is never silent. An unmeasurable driver is treated as below the
    floor, because guessing upward would risk the 4 GB card.
    """
    requested_model = model if isinstance(model, str) else ""
    requested_device = device if isinstance(device, str) else ""
    model_explicit = bool(requested_model) and requested_model != "auto"
    device_choice = requested_device or settings.device
    device_explicit = bool(device_choice) and device_choice != "auto"
    tier = requested_model if model_explicit else "large-v3-turbo"
    if model_explicit or device_explicit:
        return (
            tier,
            device_choice,
            (
                f"explicit choice (model={tier!r}, device={device_choice!r}); the "
                "VRAM precheck is bypassed"
            ),
        )
    free = media.free_vram_mib()
    if free is None:
        return (
            "medium",
            "cpu",
            (
                f"free VRAM is unmeasurable and the floor is {settings.vram_floor_mib} MiB; "
                "dropped to medium on CPU"
            ),
        )
    if free < settings.vram_floor_mib:
        return (
            "medium",
            "cpu",
            (
                f"free VRAM {free} MiB is below the {settings.vram_floor_mib} MiB floor; "
                "dropped to medium on CPU"
            ),
        )
    return (
        tier,
        "auto",
        (
            f"free VRAM {free} MiB is at or above the {settings.vram_floor_mib} MiB floor; "
            f"kept {tier}"
        ),
    )


def _ensure_transcript(
    settings: Settings,
    entry: Path,
    path: Path,
    source: str,
    model: str | None,
    num_speakers: int,
    language: str | None,
    device: str | None,
    queue: str,
    report: _Report,
    cancel: threading.Event,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, Path], str]:
    """Return the transcript segments, meta, artifact paths, and tier.

    A cached `transcript.json` is reused as-is. Otherwise the delegated
    pipeline runs once under the single-flight lock, with an `--out` inside
    the entry so nothing lands in a working directory. The cache is consulted
    again *after* the lock is taken: a caller that waited for a holder finds
    the entry that holder just wrote and reports it as a cache hit instead of
    decoding the same source a second time, and only then is the stale
    `transcribe/` work directory wiped. The tier is reported only from
    evidence: the pipeline's confirmed `meta["tier"]`/`meta["model"]` (via
    `contracts.run_transcript`), never from the server's own assumption.
    """
    transcript_path = entry / "transcript.json"
    cached = _cached_transcript(transcript_path)
    if cached is not None:
        report.cached.append("transcript.json")
        return cached

    note = acquire_transcribe_lock(
        settings,
        source=source,
        queue=queue,
        cancel=cancel,
        timeout_seconds=settings.call_timeout_seconds,
    )
    if note is not None:
        report.capped.append(f"transcribe lock: {note}")
    try:
        # A second identical call waits on the flock above; once it is handed
        # the lock the holder's `transcript.json` is already in place, so it is
        # reported as a cache hit rather than a second pipeline run (design
        # section 12, "the second job runs after the first and then reuses the
        # cache entry", and the stale-workdir wipe is skipped with it).
        cached = _cached_transcript(transcript_path)
        if cached is not None:
            report.cached.append("transcript.json")
            return cached

        tier, device_choice, rationale = _choose_tier(settings, model, device)
        report.extras.append(f"tier: {tier} ({rationale})")
        out_dir = entry / "transcribe"
        _empty_dir(out_dir)
        argv = contracts.build_argv(
            str(path),
            out_dir,
            tier,
            int(num_speakers or 0),
            language or "",
            device_choice,
            queue == "fail",
            settings.call_timeout_seconds,
        )
        result = contracts.run_transcript(argv, settings.call_timeout_seconds, cancel)
    finally:
        release_transcribe_lock(settings)

    meta = dict(result.meta)
    meta.setdefault("tier", result.tier)
    meta["requested_tier"] = tier
    meta["device"] = device_choice
    _write_json(transcript_path, {"meta": meta, "segments": result.segments})
    report.computed.append("transcript")
    return result.segments, meta, result.artifacts, result.tier


def _cached_transcript(
    transcript_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, Path], str] | None:
    """Read a usable `transcript.json`, or `None` when there is nothing cached."""
    data = _read_json(transcript_path)
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        return None
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    tier = str(meta.get("tier") or meta.get("model") or "unknown")
    return list(data["segments"]), dict(meta), {}, tier


def _empty_dir(path: Path) -> None:
    """Create `path`, removing any stale contents, so an `--out` is fresh."""
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                _remove_tree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


# ---------------------------------------------------------------------------
# Frames, OCR, and the visual stage
# ---------------------------------------------------------------------------


def _window_timestamps(
    center: float, span: float, fps: int, duration: float, cap: int
) -> list[float]:
    """Sample `[center - span/2, center + span/2]` at `fps`, capped by `count`."""
    start = max(0.0, center - span / 2.0)
    stop = center + span / 2.0
    if duration > 0:
        stop = min(stop, duration)
    step = 1.0 / max(1, fps)
    stamps: list[float] = []
    current = start
    while current <= stop + 1e-9 and len(stamps) < cap:
        stamps.append(round(current, 6))
        current += step
    if not stamps:
        stamps.append(max(0.0, center))
    return stamps


def _requested_timestamps(
    at: object, span: float | None, fps: int, duration: float, cap: int
) -> list[float]:
    """Resolve the `at`/`window`/`fps` shape into an explicit timestamp list."""
    if at is None:
        raise MediaError(
            "a timestamp is required: pass a number, HH:MM:SS.mmm, a list of "
            "timestamps, or 'scenes'"
        )
    if isinstance(at, str) and at.strip().lower() == "scenes":
        raise MediaError("at='scenes' is resolved by the caller, not here")
    if isinstance(at, (list, tuple)):
        stamps = [parse_timestamp(item) for item in at]
        if not stamps:
            raise MediaError("the timestamp list is empty")
        return stamps[:cap]
    center = parse_timestamp(at)
    if span is not None:
        return _window_timestamps(center, span, fps, duration, cap)
    return [center]


def _require_video_stream(
    path: Path, result: media.ProbeResult, consequence: str
) -> None:
    """Refuse frame work on a source whose probe found no video stream.

    The design's edge case is explicit: an audio-only source makes the frame
    and OCR tools return a clear "no video stream" error rather than an empty
    result. The check runs on the probe that all frame paths already need, so
    it holds whichever way `at` selects timestamps.
    """
    if not result.has_video:
        raise MediaError(f"{path} has no video stream, so {consequence}")


def _resolve_crop_box(
    crop_spec: object,
    result: media.ProbeResult,
    path: Path,
    start: float,
    stop: float,
    fps: int,
    report: _Report,
) -> media.Box | None:
    """Resolve a validated crop spec into a frame-pixel box, or None."""
    if crop_spec is None:
        return None
    if crop_spec == "auto":
        box = media.resolve_crop("auto", result, path, start, stop, fps)
        if box is None:
            report.skipped["crop"] = (
                "crop='auto' found no movement to bound, so the full frame is used"
            )
        return box
    return media.resolve_crop(crop_spec, result, path, start, stop, fps)


def _render_frame(
    timestamp: float,
    png: bytes,
    labels: bool,
    max_edge: int,
    scratch_dir: Path | None = None,
    notes: list[str] | None = None,
) -> bytes:
    """Return one labelled, bounded frame image (a one-tile sheet).

    `scratch_dir` stages the tile and ASS intermediates in the caller's cache
    entry rather than the OS temp directory, and `notes` receives the reason a
    label could not be burned instead of failing the whole render.
    """
    return media.render_sheet([(timestamp, png)], labels, max_edge, scratch_dir, notes)


def _report_label_notes(report: _Report, notes: list[str]) -> None:
    """Add per-tile label degradations to the status block, never overwriting it.

    A tile whose label could not be burned is returned unlabelled, and the
    reason has to reach the caller: the design forbids a silent degradation.
    """
    if not notes:
        return
    joined = "; ".join(notes)
    prior = report.skipped.get("labels")
    report.skipped["labels"] = f"{prior}; {joined}" if prior else joined


def _visual_stage(
    entry: Path,
    path: Path,
    result: media.ProbeResult,
    candidates: list[media.Candidate],
    count: int,
    labels: bool,
    max_edge: int,
    psm: int,
    lang: str,
    report: _Report,
) -> list[dict[str, object]]:
    """Rasterize up to `count` selected candidates and OCR each one.

    Each selected timestamp yields a PNG under `frames/` and an OCR record
    under `ocr/`, both reused when they already exist. The record's text and
    confidence are what `media_search(scope="ocr")` reads later.
    """
    label_notes: list[str] = []
    selected = candidates[:count]
    if not selected:
        report.skipped["ocr"] = "no frame candidates were available to OCR"
        return []
    font_ok = media.resolve_font() is not None
    if labels and not font_ok:
        report.skipped["labels"] = (
            "no usable font was found, so tiles are returned unlabelled"
        )
    records: list[dict[str, object]] = []
    frames_dir = artifacts.stage_path(entry, "frames/")
    ocr_dir = artifacts.stage_path(entry, "ocr/")
    for candidate in selected:
        timestamp = candidate.timestamp
        stamp = _file_stamp(timestamp)
        frame_path = frames_dir / f"{stamp}.png"
        ocr_path = ocr_dir / f"{stamp}.json"
        cached_ocr = _read_json(ocr_path)
        if (
            frame_path.is_file()
            and isinstance(cached_ocr, dict)
            and "text" in cached_ocr
        ):
            report.cached.append(f"frames/{stamp}.png")
            records.append(
                {
                    "timestamp": timestamp,
                    "frame": frame_path,
                    "text": str(cached_ocr.get("text", "")),
                }
            )
            continue
        png = media.extract_frame(path, timestamp, None, max_edge)
        frame_path.write_bytes(
            _render_frame(timestamp, png, labels, max_edge, entry, label_notes)
        )
        recognized = media.ocr_png(png, psm=psm, lang=lang)
        _write_json(
            ocr_path,
            {"timestamp": timestamp, "text": recognized.text},
        )
        records.append(
            {
                "timestamp": timestamp,
                "frame": frame_path,
                "text": recognized.text,
            }
        )
    _report_label_notes(report, label_notes)
    report.computed.append("frames")
    report.computed.append("ocr")
    return records


def _candidate_timestamps(
    entry: Path,
    path: Path,
    result: media.ProbeResult,
    threshold: float,
    budget: int,
    report: _Report,
) -> list[float]:
    """Rank frame candidates and return their timestamps (for `at="scenes"`).

    The candidate list stays complete in `scenes.json`; only the timestamps
    handed to rasterization are bounded to what this source can actually
    produce. A scene boundary or, more often, a silence boundary reported at
    the probed duration is a real moment, but `-ss` cannot reach past the last
    frame's presentation time (`duration - 1/fps`), so such a candidate would
    fail extraction with an empty output. It is dropped here — and the drop is
    stated in `skipped`, never silent — rather than merged away in the list.
    """
    _scenes, candidates, _accounting = _ensure_scene_candidates(
        entry, path, result, threshold, budget, report
    )
    last = result.duration - (1.0 / result.fps) if result.fps > 0 else result.duration
    stamps = [
        candidate.timestamp for candidate in candidates if candidate.timestamp <= last
    ]
    omitted = len(candidates) - len(stamps)
    if omitted:
        report.skipped["scenes"] = (
            f"{omitted} candidate(s) at or beyond the last frame "
            f"({_stamp(last)}) of a {_stamp(result.duration)} clip: omitted, "
            "no frame exists to rasterize"
        )
    return stamps


# ---------------------------------------------------------------------------
# The timeline projection
# ---------------------------------------------------------------------------


def _number(value: object) -> float | None:
    """Coerce a JSON value to a float, or `None` when it is not a number."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_dicts(value: object) -> list[dict[str, object]]:
    """Return the object items of a JSON array, ignoring anything else."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _stored_list(path: Path) -> list[dict[str, object]]:
    """Read a JSON array of objects from disk, or an empty list."""
    return _as_dicts(_read_json(path))


def _timeline_events(entry: Path) -> list[artifacts.TimelineEvent]:
    """Project the entry's cached artifacts into timeline events.

    One layer per stage: `scene` boundaries from `scenes.json`, `audio` events
    from `audio_events.json`, `speech` from `transcript.json`, `ocr` strings
    from the `ocr/` records, and the persisted value-tracking candidates from
    `ocr_changes.json`. Reading the artifacts instead of accumulating events
    in memory is what lets a later stage rebuild the whole file without
    dropping what an earlier stage contributed. A malformed record is skipped
    rather than crashing the tool: the projection is a convenience over the
    artifacts, never their source of truth.
    """
    events: list[artifacts.TimelineEvent] = []

    scenes = _read_json(entry / "scenes.json")
    if isinstance(scenes, dict):
        for item in _as_dicts(scenes.get("scenes")):
            timestamp = _number(item.get("timestamp"))
            if timestamp is not None:
                score = _number(item.get("score")) or 0.0
                events.append(
                    artifacts.TimelineEvent(timestamp, "scene", f"score {score:.3f}")
                )

    for item in _stored_list(entry / "audio_events.json"):
        timestamp = _number(item.get("timestamp"))
        if timestamp is not None:
            kind = str(item.get("kind") or "audio")
            detail = str(item.get("detail") or "")
            events.append(
                artifacts.TimelineEvent(timestamp, "audio", f"{kind} {detail}".strip())
            )

    transcript = _read_json(entry / "transcript.json")
    if isinstance(transcript, dict):
        for item in _as_dicts(transcript.get("segments")):
            timestamp = _number(item.get("start"))
            if timestamp is not None:
                speaker = str(item.get("speaker") or "(unknown)")
                events.append(
                    artifacts.TimelineEvent(
                        timestamp,
                        "speech",
                        f"{speaker}: {item.get('text') or ''}",
                    )
                )

    ocr_dir = entry / "ocr"
    if ocr_dir.is_dir():
        for record in sorted(ocr_dir.glob("*.json")):
            item = _read_json(record)
            if isinstance(item, dict):
                timestamp = _number(item.get("timestamp"))
                if timestamp is not None:
                    text = str(item.get("text") or "").strip()
                    events.append(
                        artifacts.TimelineEvent(timestamp, "ocr", text or "(no text)")
                    )

    for item in _stored_list(entry / "ocr_changes.json"):
        timestamp = _number(item.get("timestamp"))
        if timestamp is not None:
            events.append(
                artifacts.TimelineEvent(timestamp, "ocr-change", _change_payload(item))
            )

    return events


def _change_payload(item: dict[str, object]) -> str:
    """Render one persisted value-tracking candidate as a short timeline payload."""
    return (
        f"value {item.get('value')}, neighbours {item.get('before')} "
        f"then {item.get('after')}"
    )


def _refresh_timeline(entry: Path, report: _Report) -> None:
    """Rewrite the entry's `timeline.md` as the projection of its artifacts."""
    artifacts.write_timeline(entry, _timeline_events(entry))
    report.computed.append("timeline")


def _record_ocr_changes(entry: Path, candidates: list[media.ChangeCandidate]) -> None:
    """Persist the value-tracking candidates beside the OCR records.

    Keyed and merged by timestamp rather than replaced, so running OCR over a
    second window appends to the record instead of erasing the first window's
    candidates; the timeline can then be rebuilt at any later call without
    losing them.
    """
    stored: dict[float, dict[str, object]] = {}
    for item in _stored_list(entry / "ocr_changes.json"):
        timestamp = _number(item.get("timestamp"))
        if timestamp is not None:
            stored[timestamp] = item
    for candidate in candidates:
        stored[candidate.timestamp] = {
            "timestamp": candidate.timestamp,
            "value": candidate.value,
            "before": candidate.before,
            "after": candidate.after,
            "confidence": candidate.confidence,
        }
    _write_json(
        entry / "ocr_changes.json",
        [stored[stamp] for stamp in sorted(stored)],
    )


# ---------------------------------------------------------------------------
# Per-tool work bodies
# ---------------------------------------------------------------------------


def _probe_work(
    settings: Settings, source: str, identity: str, cancel: threading.Event
) -> str:
    report = _Report()
    resolved, path, entry, identity_note = _entry_for(
        settings, source, identity, report
    )
    _record_identity_note(report, identity_note)
    result = _ensure_probe(entry, path, report)
    _record_entry(entry, resolved)

    offset = media.av_offset_seconds(result)
    lines = [
        "# media_probe",
        "",
        f"Source: {resolved.source}",
        f"Identity: {resolved.kind} {resolved.key}",
        "",
        f"Resolution: {result.width}x{result.height}",
        f"Duration: {_stamp(result.duration)} ({result.duration:.3f} s)",
        f"Frame rate: {result.fps:.3f} fps",
        f"Container: {result.container or '(unknown)'}",
        f"Video codec: {result.video_codec or '(none)'}",
        f"Audio codec: {result.audio_codec or '(none)'}",
        f"Audio layout: {result.audio_layout or '(none)'}",
        f"Streams: video={_yes_no(result.has_video)}, audio={_yes_no(result.has_audio)}",
        (
            "Stream start_time: "
            f"video={_stream_start(result.video_start_time)}, "
            f"audio={_stream_start(result.audio_start_time)} "
            f"(audio-video offset {offset:+.3f} s)"
        ),
        f"Source size: {result.size_bytes} bytes",
    ]
    if abs(offset) > 0.5:
        lines.append(
            f"WARNING: the audio and video start_time values differ by {offset:+.3f} s, "
            "so a timestamp is correlated to audio with that much error."
        )
    if result.size_bytes > settings.max_source_bytes:
        lines.append(
            f"WARNING: the source size {result.size_bytes} bytes exceeds the "
            f"{settings.max_source_bytes} byte cap."
        )
    if result.duration > settings.max_source_seconds:
        lines.append(
            f"WARNING: the source duration {result.duration:.3f} s exceeds the "
            f"{settings.max_source_seconds:g} s cap."
        )
    lines += [
        "",
        "Estimated processing cost per stage:",
        "- probe: light",
        "- scenes: light",
        "- audio: light",
        "- transcript: heavy",
        "- ocr: medium",
        "- frames: medium",
        "",
        report.status(),
    ]
    return "\n".join(lines)


def _manifest_work(
    settings: Settings,
    source: str,
    profile: str,
    scene_threshold: object,
    frame_budget: object,
    identity: str,
    cancel: threading.Event,
) -> str:
    profile = validate_choice(profile, "manifest", tuple(PROFILES), "profile")
    stages = PROFILES[profile]
    threshold = (
        float(scene_threshold)
        if scene_threshold is not None
        else SCENE_THRESHOLD_DEFAULT
    )
    budget = validate_count(frame_budget, FRAME_BUDGET_DEFAULT, COUNT_CAP)
    report = _Report()
    resolved, path, entry, identity_note = _entry_for(
        settings, source, identity, report
    )
    _record_identity_note(report, identity_note)
    result = _ensure_probe(entry, path, report)
    _record_entry(entry, resolved)
    scenes, candidates, accounting = _ensure_scene_candidates(
        entry, path, result, threshold, budget, report
    )
    reasons = ", ".join(
        f"{reason}={accounting.get(reason, 0)}"
        for reason in ("scene", "audio", "interval", "fill")
    )
    dropped_line = (
        f"{_counted(accounting.get('dropped', 0), 'candidate')} dropped by the budget "
        f"(by reason: {reasons})."
    )

    lines = [
        "# media_manifest",
        "",
        f"Source: {resolved.source}",
        f"Profile: {profile}",
        f"Identity: {resolved.kind} {resolved.key}",
        "",
        (
            f"Duration {_stamp(result.duration)} ({result.duration:.3f} s), "
            f"{result.width}x{result.height}, {result.fps:.3f} fps"
        ),
        "",
        "## Scene events",
        (
            f"{_counted(len(scenes), 'scene boundary candidate')} scored at or above "
            f"the threshold {threshold:g} in scdet's own units."
        ),
        (
            f"{_counted(len(candidates), 'frame candidate')} after fixed-interval "
            f"sampling and perceptual dedup; the frame budget is {budget}."
        ),
        dropped_line,
    ]
    if scenes:
        lines += ["", "| timestamp | score |", "| --- | --- |"]
        lines += [
            f"| {_stamp(scene.timestamp)} | {scene.score:.3f} |" for scene in scenes
        ]
    else:
        lines.append("No scene change reached the threshold.")

    events = _ensure_audio_events(entry, path, report)
    lines += ["", "## Audio events"]
    if events:
        lines += ["", "| timestamp | kind | detail |", "| --- | --- | --- |"]
        lines += [
            f"| {_stamp(event.timestamp)} | {event.kind} | {event.detail} |"
            for event in events
        ]
        if not any(event.kind == "loud" for event in events):
            lines += ["", _LOUDNESS_QUALIFICATION]
    else:
        lines.append("No audio events. " + _LOUDNESS_QUALIFICATION)

    offset = media.av_offset_seconds(result)
    lines += ["", "## Audio-video offset", f"{offset:+.3f} s"]
    if abs(offset) > 0.5:
        lines.append(
            "WARNING: frame-to-audio correlation carries this error; a reported "
            "timestamp is approximate."
        )

    if stages["frames"] and len(candidates) > budget:
        report.capped.append(
            f"frames: kept {budget} of {len(candidates)} candidates "
            f"(frame_budget={budget})"
        )
    if stages["ocr"]:
        report.capped.append(
            f"ocr: at most {COUNT_DEFAULT} timestamps per call (count cap {COUNT_CAP})"
        )
    if stages["transcript"]:
        report.capped.append("transcript: one heavy job at a time (single-flight lock)")

    if stages["transcript"]:
        segments, _meta, artifacts_paths, tier = _ensure_transcript(
            settings,
            entry,
            path,
            str(path),
            "auto",
            NUM_SPEAKERS_DEFAULT,
            "auto",
            None,
            "wait",
            report,
            cancel,
        )
        lines += ["", "## Transcript", f"Tier: {tier}"]
        if segments:
            lines += ["", "| start | speaker | text |", "| --- | --- | --- |"]
            lines += [
                "| {start} | {speaker} | {text} |".format(
                    start=_stamp(float(segment.get("start") or 0.0)),
                    speaker=segment.get("speaker") or "(unknown)",
                    text=str(segment.get("text", "")).replace("|", "\\|"),
                )
                for segment in segments
            ]
        else:
            lines.append("The pipeline returned no speech segments.")
        if artifacts_paths:
            lines.append(
                "Artifacts: "
                + ", ".join(
                    f"{key}={value}" for key, value in sorted(artifacts_paths.items())
                )
            )

    if stages["ocr"] or stages["frames"]:
        records = _visual_stage(
            entry,
            path,
            result,
            candidates,
            COUNT_DEFAULT,
            True,
            settings.frame_max_edge,
            6,
            "eng",
            report,
        )
        lines += ["", "## OCR and frames"]
        if records:
            lines += [
                f"[{_stamp(float(record['timestamp']))}] {record['frame']} | "
                f"{str(record['text']).strip() or '(no text)'}"
                for record in records
            ]
        else:
            lines.append("No frames were selected for OCR.")

    _refresh_timeline(entry, report)
    _evict_after_job(settings, entry, report)
    lines += ["", report.status()]
    return "\n".join(lines)


def _frames_work(
    settings: Settings,
    source: str,
    at: object,
    window: object,
    fps: object,
    count: object,
    tile: object,
    crop: object,
    labels: bool,
    image_inline: bool,
    max_edge: object,
    identity: str,
    cancel: threading.Event,
) -> str | list[ContentBlock]:
    # The pure validators run before the source is resolved, so an ambiguous
    # request reports the ambiguous parameter rather than a source problem.
    span = validate_window(at, window)
    count_value = validate_count(count)
    fps_value = validate_fps(fps)
    crop_spec = validate_crop(crop)
    report = _Report()
    resolved, path, entry, identity_note = _entry_for(
        settings, source, identity, report
    )
    _record_identity_note(report, identity_note)
    result = _ensure_probe(entry, path, report)
    _record_entry(entry, resolved)
    _require_video_stream(path, result, "no frame can be extracted")
    edge = int(max_edge) if max_edge is not None else settings.frame_max_edge

    if isinstance(at, str) and at.strip().lower() == "scenes":
        stamps = _candidate_timestamps(
            entry, path, result, SCENE_THRESHOLD_DEFAULT, count_value, report
        )[:count_value]
        default_tile = "sheet"
    elif isinstance(at, (list, tuple)):
        stamps = _requested_timestamps(
            at, span, fps_value, result.duration, count_value
        )
        if len(at) > count_value:
            report.capped.append(
                f"frames: {count_value} of {len(at)} requested timestamps "
                f"(count cap {COUNT_CAP})"
            )
        default_tile = "individual"
    else:
        stamps = _requested_timestamps(
            at, span, fps_value, result.duration, count_value
        )
        default_tile = "filmstrip" if span is not None else "individual"
    if not stamps:
        raise MediaError("no timestamps were selected to rasterize")

    mode = validate_choice(tile, default_tile, _TILE_MODES, "tile")
    start = stamps[0]
    stop = stamps[-1]
    if len(stamps) > 1:
        stop = min(result.duration, stop + (1.0 / max(1, fps_value)))
    box = _resolve_crop_box(crop_spec, result, path, start, stop, fps_value, report)
    font_ok = media.resolve_font() is not None
    if labels and not font_ok:
        report.skipped["labels"] = (
            "no usable font was found, so frames are returned unlabelled"
        )

    frames_dir = artifacts.stage_path(entry, "frames/")
    written: list[tuple[float, Path]] = []
    label_notes: list[str] = []
    if mode == "individual":
        for timestamp in stamps:
            png = media.extract_frame(path, timestamp, box, edge)
            target = frames_dir / f"{_file_stamp(timestamp)}.png"
            target.write_bytes(
                _render_frame(timestamp, png, labels, edge, entry, label_notes)
            )
            written.append((timestamp, target))
    else:
        pairs = [
            (timestamp, media.extract_frame(path, timestamp, box, edge))
            for timestamp in stamps
        ]
        if mode == "sheet":
            composite = media.render_sheet(pairs, labels, edge, entry, label_notes)
        else:
            composite = media.render_filmstrip(pairs, labels, edge, entry, label_notes)
        target = frames_dir / (
            f"{_file_stamp(stamps[0])}-{_file_stamp(stamps[-1])}.png"
        )
        target.write_bytes(composite)
        written.append((stamps[0], target))
    _report_label_notes(report, label_notes)
    report.computed.append("frames")
    if box is not None:
        report.extras.append(
            f"crop: x={box.x} y={box.y} width={box.width} height={box.height}"
        )

    lines = [
        "# media_frames",
        "",
        f"Source: {resolved.source}",
        f"Tile mode: {mode}",
        f"Timestamps: {len(stamps)}",
        "",
        f"Frames written ({'one file per timestamp' if mode == 'individual' else 'one composite'}):",
    ]
    if mode == "individual":
        for timestamp, target in written:
            lines.append(f"[{_stamp(timestamp)}] {target}")
    else:
        lines.append(f"[{_stamp(stamps[0])}..{_stamp(stamps[-1])}] {written[0][1]}")
    lines += [
        "",
        "Point the read tool at an absolute path above to view the image.",
    ]
    if labels and not font_ok:
        lines.append(
            "No usable font was found for tile labels, so the images are "
            "returned unlabelled."
        )
    inline: list[ContentBlock] = []
    if image_inline:
        present = [target for _timestamp, target in written if target.is_file()]
        notes: list[str] = []
        if len(present) < len(written):
            notes.append(
                f"{_counted(len(written) - len(present), 'PNG')} missing from disk"
            )
        for target in present[:INLINE_IMAGE_CAP]:
            try:
                payload = base64.b64encode(target.read_bytes()).decode("ascii")
            except OSError:
                notes.append(f"{target.name} could not be read")
                continue
            inline.append(
                ImageContent(type="image", data=payload, mime_type="image/png")
            )
        if notes:
            report.skipped["inline"] = (
                f"{'; '.join(notes)}; only the paths are returned"
            )
        if len(present) > INLINE_IMAGE_CAP:
            report.capped.append(
                f"inline: attached {len(inline)} of {len(present)} PNGs "
                f"(inline cap {INLINE_IMAGE_CAP})"
            )
        if inline:
            lines.append(
                f"Inline: {_counted(len(inline), 'PNG')} above are also attached "
                "to this response as image content."
            )
    _evict_after_job(settings, entry, report)
    lines += ["", report.status()]
    text = "\n".join(lines)
    if not inline:
        return text
    return [TextContent(type="text", text=text), *inline]


def _ocr_work(
    settings: Settings,
    source: str,
    at: object,
    window: object,
    fps: object,
    count: object,
    crop: object,
    track: bool,
    psm: object,
    lang: str,
    identity: str,
    cancel: threading.Event,
) -> str:
    span = validate_window(at, window)
    count_value = validate_count(count)
    fps_value = validate_fps(fps)
    crop_spec = validate_crop(crop)
    psm_value = validate_count(psm, 6, 13, name="psm")
    language = str(lang or "eng")
    report = _Report()
    resolved, path, entry, identity_note = _entry_for(
        settings, source, identity, report
    )
    _record_identity_note(report, identity_note)
    result = _ensure_probe(entry, path, report)
    _record_entry(entry, resolved)
    _require_video_stream(path, result, "no frame can be extracted for OCR")

    if isinstance(at, str) and at.strip().lower() == "scenes":
        stamps = _candidate_timestamps(
            entry, path, result, SCENE_THRESHOLD_DEFAULT, count_value, report
        )[:count_value]
    else:
        stamps = _requested_timestamps(
            at, span, fps_value, result.duration, count_value
        )
        if isinstance(at, (list, tuple)) and len(at) > count_value:
            report.capped.append(
                f"ocr: {count_value} of {len(at)} requested timestamps "
                f"(count cap {COUNT_CAP})"
            )
    if not stamps:
        raise MediaError("no timestamps were selected for OCR")

    start, stop = stamps[0], stamps[-1]
    if len(stamps) > 1:
        stop = min(result.duration, stop + (1.0 / max(1, fps_value)))
    box = _resolve_crop_box(crop_spec, result, path, start, stop, fps_value, report)

    ocr_dir = artifacts.stage_path(entry, "ocr/")
    results: list[media.OcrResult] = []
    for timestamp in stamps:
        png = media.extract_frame(path, timestamp, box, settings.frame_max_edge)
        recognized = media.ocr_png(png, psm=psm_value, lang=language)
        record = media.OcrResult(
            timestamp=timestamp,
            text=recognized.text,
            confidence=recognized.confidence,
        )
        results.append(record)
        _write_json(
            ocr_dir / f"{_file_stamp(timestamp)}.json",
            {"timestamp": timestamp, "text": record.text},
        )
    report.computed.append("ocr")

    lines = [
        "# media_ocr",
        "",
        f"Source: {resolved.source}",
        f"Language: {language} (psm {psm_value})",
        "",
        "Recognized text per timestamp:",
    ]
    for record in results:
        lines.append(
            f"[{_stamp(record.timestamp)}] {record.text.strip() or '(no text)'} "
            f"(confidence {record.confidence:.2f})"
        )
    if track:
        candidates = media.track_values(results)
        _record_ocr_changes(entry, candidates)
        lines += ["", "Value tracking candidates (candidate events, not conclusions):"]
        if candidates:
            for candidate in candidates:
                lines.append(
                    f"candidate: value {candidate.value} at "
                    f"{_stamp(candidate.timestamp)}, neighbours {candidate.before} "
                    f"then {candidate.after} (confidence {candidate.confidence:.2f})"
                )
        else:
            lines.append(
                "No numeric value was found that disagrees with both neighbours, so "
                "there is no candidate event in this window."
            )
    _refresh_timeline(entry, report)
    _evict_after_job(settings, entry, report)
    lines += ["", report.status()]
    return "\n".join(lines)


def _transcript_work(
    settings: Settings,
    source: str,
    model: str | None,
    num_speakers: object,
    language: str | None,
    device: str | None,
    queue: str,
    identity: str,
    cancel: threading.Event,
) -> str:
    queue = validate_choice(queue, "wait", _QUEUE_MODES, "queue")
    report = _Report()
    resolved, path, entry, identity_note = _entry_for(
        settings, source, identity, report
    )
    _record_identity_note(report, identity_note)
    _ensure_probe(entry, path, report)
    _record_entry(entry, resolved)
    speakers = int(num_speakers) if num_speakers is not None else 0
    segments, _meta, artifacts_paths, tier = _ensure_transcript(
        settings,
        entry,
        path,
        str(path),
        model,
        speakers,
        language,
        device,
        queue,
        report,
        cancel,
    )
    report.capped.append("transcript: one heavy job at a time (single-flight lock)")

    lines = [
        "# media_transcript",
        "",
        f"Source: {resolved.source}",
        f"Tier: {tier}",
    ]
    lines += [line for line in report.extras if line.startswith("tier:")]
    lines += [f"Segments: {len(segments)}", ""]
    if segments:
        lines += ["| start | speaker | text |", "| --- | --- | --- |"]
        for segment in segments:
            lines.append(
                f"| {_stamp(float(segment.get('start') or 0.0))} | "
                f"{segment.get('speaker') or '(unknown)'} | "
                f"{str(segment.get('text', '')).replace('|', chr(92) + '|')} |"
            )
    else:
        lines.append("The pipeline returned no speech segments.")
    if artifacts_paths:
        lines += ["", "Artifacts:"]
        lines += [f"- {key}: {value}" for key, value in sorted(artifacts_paths.items())]
    else:
        lines += ["", "Artifacts: (reused from cache)"]
    _refresh_timeline(entry, report)
    _evict_after_job(settings, entry, report)
    lines += ["", report.status()]
    return "\n".join(lines)


def _search_work(
    settings: Settings,
    source: str,
    query: str,
    k: object,
    scope: str,
    identity: str,
    cancel: threading.Event,
) -> str:
    k_value = validate_k(k)
    scope = validate_choice(scope, "all", _SEARCH_SCOPES, "scope")
    report = _Report()
    resolved, _path, entry, identity_note = _entry_for(
        settings, source, identity, report
    )
    _record_identity_note(report, identity_note)
    if not str(query).strip():
        raise MediaError("query must be a non-empty search string")

    transcript_present = (entry / "transcript.json").is_file()
    ocr_present = (
        any((entry / "ocr").glob("*.json")) if (entry / "ocr").is_dir() else False
    )
    wanted_transcript = scope in ("all", "transcript")
    wanted_ocr = scope in ("all", "ocr")
    if (
        wanted_transcript
        and not transcript_present
        and (scope == "transcript" or not ocr_present)
    ):
        raise MediaError(
            f"no searchable transcript is cached for {resolved.source}: "
            f"{entry / 'transcript.json'} is missing; run media_transcript "
            "for this source first"
        )
    if scope == "ocr" and not ocr_present:
        raise MediaError(
            f"no searchable OCR is cached for {resolved.source}: "
            f"{entry / 'ocr'} holds no records; run media_ocr for this source first"
        )
    if scope == "all" and not transcript_present and not ocr_present:
        raise MediaError(
            f"nothing is searchable for {resolved.source} yet: both "
            "transcript.json and the ocr/ records are missing; run "
            "media_transcript or media_ocr for this source first"
        )

    hits = artifacts.search(entry, str(query), k=k_value, scope=scope)
    if transcript_present and wanted_transcript:
        report.cached.append("transcript.json")
    if ocr_present and wanted_ocr:
        report.cached.append("ocr/*.json")

    lines = [
        "# media_search",
        "",
        f"Source: {resolved.source}",
        f'Query: "{query}" (scope {scope}, k {k_value})',
        "",
    ]
    if hits:
        lines.append("Ranked moments:")
        for index, hit in enumerate(hits, start=1):
            lines.append(
                f"{index}. [{_stamp(hit.timestamp)}] {hit.layer} | {hit.text} "
                f"(score {hit.score:.2f})"
            )
    else:
        lines.append(
            "No cached moment matched this query. Layer coverage: "
            f"transcript={'yes' if transcript_present else 'no'}, "
            f"ocr={'yes' if ocr_present else 'no'}."
        )
    lines += ["", report.status()]
    return "\n".join(lines)


def _entry_bytes(entry: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(entry):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).lstat().st_size
            except OSError:
                continue
    return total


def _cache_work(settings: Settings, action: str, cancel: threading.Event) -> str:
    action = validate_choice(action, "list", ("list", "stats", "clean"), "action")
    report = _Report()
    entries_root = settings.cache_dir / "entries"
    entries = (
        sorted(child for child in entries_root.iterdir() if child.is_dir())
        if entries_root.is_dir()
        else []
    )
    lines = ["# media_cache", "", f"Cache root: {settings.cache_dir}"]
    if action == "list":
        report.computed.append("cache:list")
        lines += ["", f"Cache entries: {len(entries)}", ""]
        if entries:
            lines += ["| key | source | bytes |", "| --- | --- | --- |"]
            for entry in entries:
                meta = artifacts.read_meta(entry) or {}
                lines.append(
                    f"| {entry.name} | {meta.get('source', '(unknown)')} | "
                    f"{_entry_bytes(entry)} |"
                )
        else:
            lines.append("No entries are cached yet.")
    elif action == "stats":
        report.computed.append("cache:stats")
        total = artifacts.total_bytes(settings.cache_dir)
        status = contracts.self_check()
        lines += [
            "",
            f"total_bytes: {total} of cap {settings.cache_cap_bytes} bytes",
            f"entries: {len(entries)}",
            f"cap_bytes: {settings.cache_cap_bytes}",
            f"device: {settings.device}",
            "",
            "dependencies:",
            f"- ok: {status.ok}",
            f"- path: {status.path}",
            f"- detail: {status.detail}",
        ]
    else:
        report.computed.append("cache:clean")
        evicted = artifacts.evict(settings.cache_dir, settings.cache_cap_bytes)
        lines += [
            "",
            f"evicted {len(evicted)} entr(y/ies)",
            (
                "Evicted keys: " + ", ".join(evicted)
                if evicted
                else "Nothing was over the cap, so nothing was evicted."
            ),
        ]
    lines += ["", report.status()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The seven registered tools
# ---------------------------------------------------------------------------


async def media_probe(source: str, identity: str = "sampled") -> str:
    """Measure a media source, local path or http/https URL, and cache the probe."""
    return await _guarded(
        lambda cancel: _probe_work(load_settings(), source, identity, cancel)
    )


async def media_manifest(
    source: str,
    profile: str = "manifest",
    scene_threshold: float = SCENE_THRESHOLD_DEFAULT,
    frame_budget: int = FRAME_BUDGET_DEFAULT,
    identity: str = "sampled",
) -> str:
    """Assemble a Markdown digest, running only the profile's stages."""
    return await _guarded(
        lambda cancel: _manifest_work(
            load_settings(),
            source,
            profile,
            scene_threshold,
            frame_budget,
            identity,
            cancel,
        )
    )


async def media_frames(
    source: str,
    at: list | str | float | None = "scenes",
    window: float | None = None,
    fps: int | None = None,
    count: int | None = None,
    tile: str | None = None,
    crop: dict | str | None = None,
    labels: bool = True,
    image_inline: bool = False,
    max_edge: int | None = None,
    identity: str = "sampled",
) -> str | list[ContentBlock]:
    """Extract frames at given timestamps and write PNG artifacts."""
    return await _guarded(
        lambda cancel: _frames_work(
            load_settings(),
            source,
            at,
            window,
            fps,
            count,
            tile,
            crop,
            labels,
            image_inline,
            max_edge,
            identity,
            cancel,
        )
    )


async def media_ocr(
    source: str,
    at: list | str | float | None = "scenes",
    window: float | None = None,
    fps: int | None = None,
    count: int | None = None,
    crop: dict | str | None = None,
    track: bool = False,
    psm: int = 6,
    lang: str = "eng",
    identity: str = "sampled",
) -> str:
    """Recognize on-screen text at given timestamps."""
    return await _guarded(
        lambda cancel: _ocr_work(
            load_settings(),
            source,
            at,
            window,
            fps,
            count,
            crop,
            track,
            psm,
            lang,
            identity,
            cancel,
        )
    )


async def media_transcript(
    source: str,
    model: str = "auto",
    num_speakers: int = NUM_SPEAKERS_DEFAULT,
    language: str = LANGUAGE_DEFAULT,
    device: str | None = None,
    queue: str = "wait",
    identity: str = "sampled",
) -> str:
    """Transcribe speech by delegating to the MEDIA_TRANSCRIBE_BIN pipeline."""
    return await _guarded(
        lambda cancel: _transcript_work(
            load_settings(),
            source,
            model,
            num_speakers,
            language,
            device,
            queue,
            identity,
            cancel,
        )
    )


async def media_search(
    source: str,
    query: str,
    k: int | None = None,
    scope: str = "all",
    identity: str = "sampled",
) -> str:
    """Rank cached transcript and OCR moments for a query."""
    return await _guarded(
        lambda cancel: _search_work(
            load_settings(), source, query, k, scope, identity, cancel
        )
    )


async def media_image_quality(
    source: str, identity: str = "sampled", flag_poor: bool = False
) -> str:
    """Assess one image's quality: blur, brightness, contrast, resolution, tier."""
    return await _guarded(
        lambda cancel: _quality_work(
            load_settings(), source, identity, cancel, flag_poor
        )
    )


async def media_image_sort(
    sources: list,
    by: str = "quality",
    order: str = "desc",
    min_quality: str | None = None,
    identity: str = "sampled",
    output: str | None = None,
) -> str:
    """Rank a list of images by a quality metric, with optional filtering.

    The ranking is also written to a JSON file — `output` when given, or a
    timestamped file under ``<cache>/sorts/`` by default — so the result
    survives conversation compaction. The response names the artifact path.
    """
    return await _guarded(
        lambda cancel: _sort_work(
            load_settings(), sources, by, order, min_quality, identity, cancel, output
        )
    )


# ---------------------------------------------------------------------------
# Photo review/sorting work functions. These sit in the server layer because
# they need the entry-resolution, report, and validation helpers that live
# here; the pure image statistics they consume live in media/media.py.
# ---------------------------------------------------------------------------


def _quality_work(
    settings: Settings,
    source: str,
    identity: str,
    cancel: threading.Event,
    flag_poor: bool = False,
) -> str:
    report = _Report()
    resolved, path, entry, identity_note = _entry_for(
        settings, source, identity, report
    )
    _record_identity_note(report, identity_note)
    metrics = _image_stats(path)
    tier = quality_tier(metrics)
    _write_json(
        entry / "quality.json",
        {"metrics": metrics, "tier": tier, "source": resolved.source},
    )
    _record_entry(entry, resolved)
    lines = [
        "# media_image_quality",
        "",
        f"Source: {resolved.source}",
        f"Identity: {resolved.kind} {resolved.key}",
        f"Quality tier: {tier}",
        "",
        f"Resolution: {metrics['width']}x{metrics['height']} ({metrics['resolution']:,} px)",
        f"Format: {metrics['format']}  Size: {metrics['size_bytes']:,} bytes",
        f"Blur score (Laplacian variance): {metrics['blur_score']}",
        f"Brightness (mean gray): {metrics['brightness']}",
        f"Contrast (gray std): {metrics['contrast']}",
        f"Channel means (BGR): {metrics['channels']}",
    ]
    if metrics.get("brisque_score") is not None:
        lines.append(f"BRISQUE score (lower is better): {metrics['brisque_score']}")
    lines += [
        "",
        (
            "Interpretation: blur below 100 is likely out of focus; brightness "
            "below 40 or above 210 is extreme; contrast std below 25 or above "
            "75 is flat or harsh; BRISQUE below 25 is good, above 50 is poor. "
            "Overall tier is the worst of all metrics (worst wins)."
        ),
    ]
    if flag_poor and tier == "poor":
        lines.append("FLAG: POOR")
    lines += [
        "",
        report.status(),
    ]
    return "\n".join(lines)


def _sort_work(
    settings: Settings,
    sources: list,
    by: str,
    order: str,
    min_quality: str | None,
    identity: str,
    cancel: threading.Event,
    output: str | None = None,
) -> str:
    """Rank a list of images by a quality metric, with optional filtering."""
    report = _Report()
    BY_ALLOWED = ("blur", "brightness", "contrast", "size", "resolution", "quality")
    by = validate_choice(by, "quality", BY_ALLOWED, "by")
    order = validate_choice(order, "desc", ("asc", "desc"), "order")
    TIER_ORDER = {"poor": 0, "fair": 1, "good": 2}
    min_tier = None
    if min_quality is not None:
        min_tier = validate_choice(
            min_quality, None, ("poor", "fair", "good"), "min_quality"
        )
        min_tier = TIER_ORDER[min_tier]

    results: list[dict[str, object]] = []
    for source in sources:
        resolved, path, _entry, note = _entry_for(settings, source, identity, report)
        _record_identity_note(report, note)
        metrics = _image_stats(path)
        tier = quality_tier(metrics)
        results.append(
            {
                "source": resolved.source,
                "key": resolved.key,
                "tier": tier,
                "tier_rank": TIER_ORDER[tier],
                "metrics": metrics,
            }
        )

    kept = [r for r in results if min_tier is None or r["tier_rank"] >= min_tier]
    if len(kept) < len(results):
        report.skipped["filter"] = (
            f"removed {len(results) - len(kept)} below min_quality={min_quality}"
        )

    if by == "quality":
        keyfn = lambda r: (r["tier_rank"], r["metrics"]["blur_score"])
    elif by == "blur":
        keyfn = lambda r: r["metrics"]["blur_score"]
    elif by == "brightness":
        keyfn = lambda r: r["metrics"]["brightness"]
    elif by == "contrast":
        keyfn = lambda r: r["metrics"]["contrast"]
    elif by == "size":
        keyfn = lambda r: r["metrics"]["size_bytes"]
    else:  # by == "resolution"
        keyfn = lambda r: r["metrics"]["resolution"]
    kept.sort(key=keyfn, reverse=(order == "desc"))

    # Write the ranking to a durable JSON file so the result survives
    # conversation compaction: `output` when given, otherwise a timestamped
    # file under <cache>/sorts/. The response names the path either way.
    kept_keys = {r["key"] for r in kept}
    payload = {
        "tool": "media_image_sort",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": {"by": by, "order": order, "min_quality": min_quality},
        "kept": [dict(r, kept=True) for r in kept],
        "filtered": [dict(r, kept=False) for r in results if r["key"] not in kept_keys],
        "summary": {"total": len(results), "kept": len(kept)},
    }
    if output:
        artifact = Path(output).expanduser().resolve()
    else:
        sorts_dir = settings.cache_dir / "sorts"
        artifact = sorts_dir / f"image-sort-{time.strftime('%Y%m%d-%H%M%S')}.json"
        stem = artifact.stem
        counter = 1
        while artifact.exists():
            artifact = sorts_dir / f"{stem}-{counter}.json"
            counter += 1
    artifact.parent.mkdir(parents=True, exist_ok=True)
    _write_json(artifact, payload)

    lines = ["# media_image_sort", "", f"Sort: by={by} order={order}"]
    if min_quality:
        lines.append(f"Min quality: {min_quality}")
    lines.append(f"Kept: {len(kept)} of {len(results)}")
    lines.append("")
    for i, r in enumerate(kept, 1):
        m = r["metrics"]
        lines.append(
            f"{i}. {r['source']}  [{r['tier']}] "
            f"blur={m['blur_score']} brightness={m['brightness']} "
            f"contrast={m['contrast']} {m['width']}x{m['height']}"
        )
    lines += [
        "",
        f"Artifact: {artifact}",
        (
            "The full ranking (metrics per image, kept and filtered) is saved "
            "there; re-read it after any context compaction."
        ),
        "",
        report.status(),
    ]
    return "\n".join(lines)


async def media_cache(action: str = "list") -> str:
    """Report the artifact cache: entries, cap, dependencies, eviction."""
    return await _guarded(lambda cancel: _cache_work(load_settings(), action, cancel))


_TOOL_DESCRIPTIONS = {
    "media_probe": (
        "Measure a media source (a local path or an http/https URL, downloaded "
        "into the cache with yt-dlp) without decoding it: resolution, duration, "
        "fps, codecs, stream presence, the audio-video start_time offset, and the "
        "estimated cost of each later stage. Caches probe.json in the source's "
        "entry. Use this first to learn what the source contains."
    ),
    "media_manifest": (
        "Assemble a Markdown digest of a media source (a local path or an "
        "http/https URL) from cached "
        "artifacts: scene events, audio events, the audio-video offset, and a "
        "status block. `profile` presets the work: manifest (probe+scenes+audio, "
        "default), read (+transcript), visual (+OCR on selected frames), full "
        "(both). Use this to survey a source cheaply before asking for frames or "
        "a transcript."
    ),
    "media_frames": (
        "Write PNG stills for the timestamps in `at`, returning the absolute "
        "paths. `at` is a list of timestamps, one timestamp, or the literal "
        "'scenes' (the default, which returns one contact sheet of the ranked "
        "scene candidates). `window` samples a span around one timestamp at "
        "`fps`; `tile` chooses sheet (grid), filmstrip (vertical stack), or "
        "individual (one file per timestamp). Point the `read` tool at the "
        "returned paths to view them. `image_inline=true` also attaches each "
        "returned PNG (at most 8) as image content."
    ),
    "media_ocr": (
        "Recognize on-screen text at the timestamps in `at` and return it per "
        "timestamp; `at` is a list, one timestamp, or 'scenes' (the default). "
        "`crop` narrows the region before OCR, and `track=true` reports numeric "
        "values that disagree with both neighbours as candidate events. Use for "
        "slides, subtitles burned into video, and UI numbers."
    ),
    "media_transcript": (
        "Transcribe speech by delegating to the MEDIA_TRANSCRIBE_BIN pipeline, "
        "returning the segments and the paths to the SRT, VTT, JSON, and Markdown "
        "artifacts. Heavy work; the single-flight lock serializes it, and any "
        "model-tier downgrade is reported with its reason."
    ),
    "media_search": (
        "Rank cached transcript and OCR moments for a query, returning timestamp, "
        "layer, and snippet. `scope` picks transcript, ocr, or all; `k` caps the "
        "hits. Run media_transcript or media_ocr first so there is something to "
        "search."
    ),
    "media_image_quality": (
        "Assess one image's quality: blur score (Laplacian variance), brightness, "
        "contrast, resolution, format, size, BRISQUE NR-IQA score, and a "
        "``poor``/``fair``/``good`` tier (worst of all metrics wins). "
        "`source` is an absolute local path or an http/https media URL, downloaded "
        "into the cache with yt-dlp when needed. `flag_poor=true` appends a "
        "``FLAG: POOR`` line when the tier is ``poor``. Caches quality.json in the "
        "source's entry. Use it to pick which photos in a series are worth keeping."
    ),
    "media_image_sort": (
        "Rank a list of images by a quality metric, with optional filtering. "
        "`sources` is a list of local paths or http/https media URLs. `by` picks "
        "blur, brightness, contrast, size, resolution, or ``quality`` (the tier "
        "plus blur as tie-break, default); `order` is ``desc`` or ``asc``; "
        "`min_quality`` (``poor``/``fair``/``good``) drops everything below it. "
        "Returns the kept images ranked, so you can quickly discard bad photos. "
        "Also writes the full ranking to a JSON file — `output` path when given, "
        "or a timestamped file under <cache>/sorts/ by default — and names that "
        "path in the response so the result survives context compaction."
    ),
    "media_cache": (
        "Report the artifact cache: `list` shows the entry table, `stats` shows "
        "total_bytes against the cap plus the transcription dependency self-check, "
        "and `clean` evicts least-recently-used entries and reports what it "
        "removed."
    ),
}


def build_server() -> MCPServer:
    """Create the MCP server and register the seven tools.

    Registration goes through the SDK's `tool(...)` decorator, which is what
    `mcp.server.MCPServer` (the class `FastMCP` became in mcp 2.x) exposes for
    this; `structured_output=False` keeps every response the Markdown text the
    design specifies rather than a JSON envelope.
    """
    server = MCPServer("media")
    for fn in (
        media_probe,
        media_manifest,
        media_frames,
        media_ocr,
        media_transcript,
        media_search,
        media_image_quality,
        media_image_sort,
        media_cache,
    ):
        server.tool(
            name=fn.__name__,
            description=_TOOL_DESCRIPTIONS[fn.__name__],
            structured_output=False,
        )(fn)
    return server


def main() -> None:
    """Serve the seven tools over stdio, stopping every child on the way out."""
    server = build_server()
    try:
        server.run(transport="stdio")
    finally:
        media.terminate_active_children()


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    main()
