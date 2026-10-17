"""The ffprobe layer: source validation, probing, limits, and the A/V offset.

Every later media stage (scenes, frames, OCR, transcript validation) starts
from what this module reports: `validate_source` is the safety gate a source
string passes before anything touches the filesystem, `probe` turns a media
file into a `ProbeResult`, and `MediaError` is the error type the whole server
layer raises. External tools are invoked with argument lists, never through a
shell, and stderr is captured and reported in the error message, never echoed
to the console. ffprobe output is read only as JSON, never as human-readable
text, so a changed banner can never break the parse.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

# The refusal thresholds of the design document, exposed here as plain module
# constants. Nothing in this module reads the environment; Task 11 wires the
# `MEDIA_MAX_SOURCE_BYTES` and `MEDIA_MAX_SOURCE_SECONDS` variables over these.
MEDIA_MAX_SOURCE_BYTES = 4 * 1024**3
MEDIA_MAX_SOURCE_SECONDS = 14400.0

# The warning threshold: half of each hard cap (2 GiB, 2 hours).
_WARN_FRACTION = 0.5

# String values ffprobe emits for numbers it could not measure.
_NOT_A_NUMBER = frozenset({"N/A", "NA", "MISSING", "?", "-", ""})


class MediaError(RuntimeError):
    """The error type every media tool raises.

    When a subprocess produced a non-zero exit, the message always ends with
    the exit code and its captured stderr, so one error message is enough to
    diagnose a failed run without re-doing it by hand. stderr content alone is
    never treated as failure: this machine's ffmpeg tools print `Fontconfig`
    noise on stderr while succeeding, so only the exit code decides.
    """


@dataclass(frozen=True)
class ProbeResult:
    """What one source's metadata says, parsed tolerantly from ffprobe JSON.

    `duration` and `fps` are seconds and frames per second; missing or `N/A`
    values become `0.0` rather than a crash. `audio_offset` is
    `audio.start_time - video.start_time` in seconds: positive when the audio
    stream starts later than the video, zero when there is only one stream or
    the two start together. Frame and scene tools correct positions with
    `av_offset_seconds`.
    """

    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    has_video: bool
    has_audio: bool
    audio_offset: float
    size_bytes: int


def validate_source(source: str, allowed_roots: list[Path] | None = None) -> Path:
    """Validate `source` as a readable local file and return it as a `Path`.

    Raises `MediaError` with an actionable message naming the path and the
    reason for: a missing path, a directory instead of a file, a symlink whose
    resolution escapes `allowed_roots`, or an unreadable file. `allowed_roots`
    is checked against `realpath` resolution, so a link cannot smuggle a read
    from outside; `None` accepts any readable path, the deliberate
    single-user-workstation default. The source file is never modified.
    """
    path = Path(source).expanduser()
    if not path.exists():
        raise MediaError(f"source not found: {path}")
    if path.is_dir():
        raise MediaError(f"source is a directory, not a media file: {path}")
    resolved = path.resolve()
    if allowed_roots is not None:
        roots = sorted({root.expanduser().resolve() for root in allowed_roots})
        if not any(
            resolved == root or _is_relative_to(resolved, root) for root in roots
        ):
            allowed = ": ".join(str(root) for root in roots) if roots else "(none)"
            raise MediaError(
                f"source resolves outside the allowed roots: {resolved} "
                f"(allowed: {allowed})"
            )
    if not os.access(resolved, os.R_OK):
        raise MediaError(f"source file is not readable (permission denied): {resolved}")
    return path


def probe(path: Path) -> ProbeResult:
    """Run ffprobe on `path` and return its parsed `ProbeResult`.

    ffprobe runs as an argument list (`ffprobe -v error -print_format json
    -show_format -show_streams`) with stderr captured. A missing file, a
    directory, or any non-zero ffprobe exit raises `MediaError` naming the
    path and ending with the exit code and the captured stderr; a wrong
    extension or corrupt file is therefore a `MediaError`, never a traceback
    from JSON parsing.
    """
    path = Path(path)
    if not path.exists():
        raise MediaError(f"probe target not found: {path}")
    if path.is_dir():
        raise MediaError(f"probe target is a directory, not a media file: {path}")
    argv = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        done = subprocess.run(argv, check=False, capture_output=True, text=True)
    except FileNotFoundError as err:
        raise MediaError(f"ffprobe is not available on PATH for {path}") from err
    if done.returncode != 0:
        stderr = done.stderr.strip()
        raise MediaError(
            f"ffprobe could not read {path}: {stderr} (exit code {done.returncode})"
        )
    return _probe_result_from_json(done.stdout)


def av_offset_seconds(result: ProbeResult) -> float:
    """Return the audio-stream start offset against the video, in seconds.

    Non-zero when the audio and video streams' `start_time` differ: positive
    means the audio starts later than the video. Zero when there is only one
    stream. This is exactly what the brief's offset test observes for an
    `-itsoffset`-delayed audio stream, which ffprobe reports as an audio
    `start_time` while the video stays at zero.
    """
    return result.audio_offset


def with_duration(result: ProbeResult, seconds: float) -> ProbeResult:
    """Return a copy of `result` carrying `seconds` as its duration.

    A pure helper so limit tests can describe long sources without a 2-hour
    fixture on disk; the original result is untouched.
    """
    return replace(result, duration=seconds)


def source_limits(result: ProbeResult) -> dict[str, bool]:
    """Classify a result against the module's source limits.

    Returns `{"warn": bool, "refuse": bool}`. `refuse` is true when either the
    duration exceeds `MEDIA_MAX_SOURCE_SECONDS` or the size exceeds
    `MEDIA_MAX_SOURCE_BYTES`; `warn` is true when either exceeds half of its
    cap. A duration beyond the warning threshold is reported, never refused;
    only crossing a hard cap refuses.
    """
    seconds_refuse = result.duration > MEDIA_MAX_SOURCE_SECONDS
    seconds_warn = result.duration > MEDIA_MAX_SOURCE_SECONDS * _WARN_FRACTION
    bytes_refuse = result.size_bytes > MEDIA_MAX_SOURCE_BYTES
    bytes_warn = result.size_bytes > MEDIA_MAX_SOURCE_BYTES * _WARN_FRACTION
    return {
        "warn": seconds_warn or bytes_warn,
        "refuse": seconds_refuse or bytes_refuse,
    }


def _is_relative_to(child: Path, root: Path) -> bool:
    """Containment check that also works on the oldest supported pathlib."""
    try:
        child.relative_to(root)
    except ValueError:
        return False
    return True


def _probe_result_from_json(stdout: str) -> ProbeResult:
    """Parse ffprobe's JSON output into a `ProbeResult`, tolerantly.

    Timestamp strings arrive in the caller's project timestamp contract
    elsewhere; here every numeric field is parsed with `_as_float`/`_as_int`
    so a missing, `N/A`, or unmeasurable value decays to zero instead of
    raising. The duration prefers the format-level measure and falls back to
    the longest stream duration; widths and frame rates come from the first
    video stream, and the A/V offset is the audio stream's `start_time` minus
    the video stream's.
    """
    try:
        data = json.loads(stdout)
    except ValueError as err:
        raise MediaError(f"ffprobe returned unparseable JSON: {err}") from err
    streams = data.get("streams", []) if isinstance(data, dict) else []
    fmt = data.get("format", {}) if isinstance(data, dict) else {}

    video = next(
        (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"),
        None,
    )
    audio = next(
        (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"),
        None,
    )
    has_video = video is not None
    has_audio = audio is not None

    duration = _as_float(fmt.get("duration")) if isinstance(fmt, dict) else 0.0
    if duration == 0.0:
        stream_durations = [
            _as_float(s.get("duration")) for s in streams if isinstance(s, dict)
        ]
        duration = max(stream_durations, default=0.0)

    size_bytes = _as_int(fmt.get("size")) if isinstance(fmt, dict) else 0

    width = _as_int(video.get("width")) if has_video else 0
    height = _as_int(video.get("height")) if has_video else 0
    if has_video:
        fps = _ratio(video.get("r_frame_rate")) or _ratio(video.get("avg_frame_rate"))
    else:
        fps = 0.0

    video_offset = _as_float(video.get("start_time")) if has_video else 0.0
    audio_offset_value = _as_float(audio.get("start_time")) if has_audio else 0.0
    if has_video and has_audio:
        audio_offset = audio_offset_value - video_offset
    else:
        audio_offset = 0.0

    return ProbeResult(
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        video_codec=_codec(video),
        audio_codec=_codec(audio),
        has_video=has_video,
        has_audio=has_audio,
        audio_offset=audio_offset,
        size_bytes=size_bytes,
    )


def _codec(stream: dict[str, object] | None) -> str:
    """Return a stream's codec name, or "" when the stream is absent.

    The codec name string comes straight from ffprobe's `codec_name`, which
    ffprobe validates before its output is parsed as JSON, never by parsing
    human-readable ffprobe text.
    """
    if not stream:
        return ""
    codec = stream.get("codec_name")
    return codec if isinstance(codec, str) else ""


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce an ffprobe value to float; missing, `N/A`, or bad is `default`."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text in _NOT_A_NUMBER:
            return default
        try:
            return float(text)
        except ValueError:
            return default
    return default


def _as_int(value: object, default: int = 0) -> int:
    """Coerce an ffprobe value to int through `_as_float`."""
    return int(_as_float(value, float(default)))


def _ratio(value: object) -> float:
    """Coerce an ffprobe frame-rate string like `30000/1001` to float.

    `0/0`, which ffprobe emits for streams with no frame rate, and any
    unparseable form decay to `0.0` rather than crashing on a division by
    zero.
    """
    if not isinstance(value, str) or "/" not in value:
        return _as_float(value)
    numerator, _, denominator = value.partition("/")
    divisor = _as_float(denominator)
    if divisor == 0.0:
        return 0.0
    return _as_float(numerator) / divisor
