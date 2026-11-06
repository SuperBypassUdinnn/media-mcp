"""The ffprobe layer: source validation, probing, limits, and the A/V offset.

Every later media stage (scenes, frames, OCR, transcript validation) starts
from what this module reports: `validate_source` is the safety gate a source
string passes before anything touches the filesystem, `probe` turns a media
file into a `ProbeResult`, and `MediaError` is the error type the whole server
layer raises. External tools are invoked with argument lists, never through a
shell, and stderr is captured and reported in the error message, never echoed
to the console. ffprobe output is read only as JSON, never as human-readable
text, so a changed banner can never break the parse. The same module decides
which frames are worth rasterizing at all: `detect_scenes` scores hard cuts,
`sample_candidates` merges them with interval ticks and collapses near
duplicates by perceptual hash, and `rank_candidates` spends the caller's
frame budget with a complete reason accounting. `detect_audio_events` reads
the audio timeline itself — silence segments from ffmpeg's `silencedetect`
and loudness peaks from its `astats` per-frame report — so the ranker and the
manifest consume measured events instead of invented ones. The frame half
then rasterizes what was chosen: `resolve_crop` turns an explicit box, `"auto"`,
or nothing into crop geometry, `extract_frame` pulls one still through
`crop` and the bounded scale, `render_sheet` and `render_filmstrip`
composite labelled tiles into one PNG, and `resolve_font` finds a font
without ever failing a call because one is missing. Only this module reads
the environment, and for labels that is exactly one variable,
`MEDIA_FONT_FILE`.


Heavy commands (scene scans, frame extraction, OCR, transcription) run through
`run_child`: the child leads its own process group so a timeout or a cancel
event can take down the whole process tree, and streams are capped so a
runaway command cannot exhaust memory.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import re
import signal
import statistics
import struct
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Generic, TypeVar

from media.artifacts import format_timestamp

if TYPE_CHECKING:
    import numpy as np  # annotations only; runtime imports stay lazy

try:
    import cv2
except ImportError:  # cv2 is optional; media_image_quality names it when missing
    cv2 = None

try:
    import brisque as _brisque_pkg
except ImportError:  # brisque is optional; media_image_quality names it when missing
    _brisque_pkg = None

_brisque_model = None  # lazy singleton; BRISQUE() loads the NR-IQA model once

_PayloadT_co = TypeVar("_PayloadT_co", covariant=True)

# The refusal thresholds of the design document, exposed here as plain module
# constants. Nothing in this module reads the environment; Task 11 wires the
# `MEDIA_MAX_SOURCE_BYTES` and `MEDIA_MAX_SOURCE_SECONDS` variables over these.
MEDIA_MAX_SOURCE_BYTES = 4 * 1024**3
MEDIA_MAX_SOURCE_SECONDS = 14400.0

# Image quality thresholds for photo review/sorting.
# These are the defaults agreed in the grill-me session; they can be overridden
# by editing these constants or by passing different values per call in future.
_BLUR_POOR = 100.0  # Laplacian variance below this is likely out of focus
_BLUR_GOOD = 500.0  # at or above this is acceptably sharp
_BRIGHT_DARK = 40.0  # mean gray below this is underexposed
_BRIGHT_BRIGHT = 210.0  # mean gray above this is overexposed
_BRIGHT_GOOD_MIN = 60.0  # good needs brightness at least this high
_BRIGHT_GOOD_MAX = 200.0  # good needs brightness at most this high
_CONTRAST_POOR_LOW = 25.0  # gray std below this is flat/low contrast
_CONTRAST_POOR_HIGH = 75.0  # gray std above this is harsh/high contrast
_CONTRAST_GOOD_MIN = 30.0  # good needs contrast at least this high
_CONTRAST_GOOD_MAX = 70.0  # good needs contrast at most this high
_BRISQUE_GOOD = 25.0  # BRISQUE score below this is good
_BRISQUE_POOR = 50.0  # BRISQUE score above this is poor

# The warning threshold: half of each hard cap (2 GiB, 2 hours).
_WARN_FRACTION = 0.5

# ffprobe's own deadline. Probe runs through the bounded runner like every
# other child, so a hung ffprobe is stopped and its group taken down rather
# than holding a tool call forever (finding R29: probe used to escape both).
_PROBE_TIMEOUT_SECONDS = 300

# nvidia-smi answers instantly when a driver is present and does not exist
# otherwise; this bounds the rare wedged-driver case.
_VRAM_QUERY_TIMEOUT_SECONDS = 30

# String values ffprobe emits for numbers it could not measure.
_NOT_A_NUMBER = frozenset({"N/A", "NA", "MISSING", "?", "-", ""})

# Run_child's per-stream capture cap and the marker it appends when a stream
# exceeds it. The cap is generous for the tool outputs in this project (ffmpeg
# scene reports, OCR text, transcript text) while still bounding a runaway
# child's memory: at most 8 MiB of stdout plus 8 MiB of stderr are retained,
# and everything past the cap is drained and discarded, never buffered.
_STREAM_CAPTURE_LIMIT = 8 * 1024**2
_TRUNCATION_MARKER = "...[output truncated]..."

# How long a SIGTERM'd group gets to exit before SIGKILL lands.
_KILL_GRACE_SECONDS = 2.0
# Poll cadence while waiting on the deadline or the cancel event.
_POLL_SECONDS = 0.05
# How large a chunk each reader thread pulls from a stream pipe.
_READ_CHUNK_BYTES = 64 * 1024

# The frame-rasterization sampler's own constants, unprefixed module names by
# the project's "one module, named constants" resolution: no environment
# variable is read anywhere in this file. Fixed-interval ticks land every
# `interval` seconds across a clip, and two candidates whose extracted
# perceptual hashes sit within `dedup_distance` Hamming bits count as the
# same picture and collapse to one survivor.
interval = 3
dedup_distance = 6
# Hash width diff_hash_of guarantees, and its comparison mask.
_HASH_BITS = 64
_HASH_MASK = (1 << _HASH_BITS) - 1

# The diff-hash raster shape: 32x32 one byte per gray pixel.
_GRAY_SIZE = 32

# ffmpeg's scdet report shape on stderr, matched tolerantly: lines that do
# not match are ignored, everything else (banners, progress, Fontconfig
# noise) is skipped. Scores are opaque detector units, never rescaled here.
_SCD_LINE = re.compile(
    r"lavfi\.scd\.score:\s*(?P<score>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    r"\s*,\s*lavfi\.scd\.time:\s*(?P<time>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)

# ffmpeg's silencedetect report shape on stderr: a start line and, once the
# segment closes, an end line that repeats the boundary and carries the
# measured duration. Matched as tolerantly as the scdet lines, so a changed
# banner prefix never matters.
_SILENCE_START = re.compile(r"silence_start:\s*(?P<time>-?[0-9]+(?:\.[0-9]+)?)")
_SILENCE_END = re.compile(
    r"silence_end:\s*(?P<time>-?[0-9]+(?:\.[0-9]+)?)"
    r"(?:\s*\|\s*silence_duration:\s*(?P<duration>-?[0-9]+(?:\.[0-9]+)?))?"
)

# astats' per-frame report once `ametadata=mode=print:key=...` narrows it to
# one line per frame: the frame header carries the media timestamp and the
# kept line carries the level. `-inf` is what digital silence measures.
_ASTATS_FRAME = re.compile(
    r"^frame:\d+\s+pts:\S+\s+pts_time:(?P<time>-?[0-9]+(?:\.[0-9]+)?)\s*$"
)
_ASTATS_PEAK = re.compile(
    r"^lavfi\.astats\.Overall\.Peak_level="
    r"(?P<level>-?[0-9]+(?:\.[0-9]+)?|-inf|nan)\s*$"
)
_LEVEL_NOT_A_NUMBER = frozenset({"-inf", "inf", "nan"})

# The loud-peak rule: a peak must clear the file's own median frame level by
# this margin. Measured on this host with the spike fixture: the base tone's
# per-frame sample peaks sit at -38.01..-37.32 dB (a 0.69 dB band) while the
# deliberate one-second spike peaks at -4.06 dB, 33.9 dB above the -37.99 dB
# median. Ten decibels clears the measured jitter by more than 9 dB and stays
# well below the spike, so ordinary level drift cannot invent a peak.
_LOUD_MARGIN_DB = 10.0

# astats' frame size for the loudness pass: `aresample` normalizes every
# source to 48 kHz first, so 9600 samples are 0.2 seconds at any input rate.
# Left at the incoming 1024-sample frames, astats prints about 47 frames per
# media second, which fills run_child's 8 MiB capture cap after roughly half
# an hour; 0.2-second frames keep a two-hour source near 3 MB while still
# resolving the design's one-second-scale peaks.
_ASTATS_SAMPLE_RATE = 48000
_ASTATS_FRAME_SAMPLES = 9600

# Candidate reasons in ranking-priority order: the tuple index is the sort
# weight, so `scene` beats `audio` beats `interval` beats `fill`.
_REASON_PRIORITY = {"scene": 0, "audio": 1, "interval": 2, "fill": 3}
_RANKING_REASONS = ("scene", "audio", "interval", "fill")

# Frames: the JPEG fallback of the design (encode PNG, and only when the PNG
# exceeds this size re-encode to mjpeg and report both limits), the auto-crop
# sampler geometry measured on this host (a grid of side-by-side 64-wide
# tiles, 2..32 of them, a difference step of 16 gray levels, and a
# 90-percent "the frame changes everywhere or the grid padded black" bail-out),
# and the burned-label typography measured OCR-legible at 320 pixels wide with
# a 40-pixel font, which every tile scales from.
JPEG_FALLBACK_BYTES = 2 * 1024 * 1024
_AUTOCROP_TILE_WIDTH = 64
_AUTOCROP_MIN_TILES = 2
_AUTOCROP_MAX_TILES = 32
_AUTOCROP_DIFF_STEP = 16
_AUTOCROP_BUSY_TILES = 0.9
_LABEL_FONT_REFERENCE_WIDTH = 320
_LABEL_FONT_REFERENCE_SIZE = 40
_LABEL_FONT_SIZE_MIN = 12

# OCR: the columns of tesseract's TSV report (level page_num block_num
# par_num line_num word_num left top width height conf text), the numeric
# token every tracked value is extracted with, and tesseract's own stderr
# shapes for an uninstalled language pack, measured on this host
# (tesseract 5.5.3): `Error opening data file
# /usr/share/tessdata/zzz.traineddata` and `Failed loading language 'zzz'`.
_TSV_COLUMNS = 12
_VALUE_TOKEN = re.compile(r"\d+(?:\.\d+)?")
_MISSING_LANGUAGE_MARKERS = ("Failed loading language", "Error opening data file")


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

    `container` is ffprobe's `format.format_name` verbatim (for example
    `mov,mp4,m4a,3gp,3g2,mj2`), `video_start_time`/`audio_start_time` carry
    each stream's own raw `start_time` and are `None` only when that stream is
    absent, and `audio_layout` is the audio stream's `channel_layout` or the
    fallback `<n> channels`, and `""` when there is no audio stream at all.
    The raw start times are kept beside the derived offset so a reader never
    has to guess one from the other.
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
    container: str = ""
    video_start_time: float | None = None
    audio_start_time: float | None = None
    audio_layout: str = ""


@dataclass(frozen=True)
class SceneScore:
    """What ffmpeg's `scdet` filter reported for one media second.

    `timestamp` is the candidate second on the media timeline and `score` is
    the detector's raw evidence in its own units — not normalized to 0..1:
    continuous testsrc motion measured about 0.5 on this machine while a
    genuine hard cut measured 34.845. Nothing here rescales, clamps, or
    thresholds; the caller's threshold lives in the same units.
    """

    timestamp: float
    score: float


@dataclass(frozen=True)
class AudioEvent:
    """One moment of the audio timeline: a silence boundary or a loud peak.

    `timestamp` is the media-time second the event marks, and `kind` is one
    of `"silence_start"`, `"silence_end"`, or `"loud"`. `detail` carries the
    measurement in readable form without repeating the kind: the start's
    position and the end's measured span are both `HH:MM:SS.mmm` values from
    `artifacts.format_timestamp`, so a table can render them side by side; a
    loud event's detail is its level against the file's own baseline. The
    objects satisfy the shape `rank_candidates` accepts, because
    `_audio_event_timestamp` reads the `timestamp` attribute directly.
    """

    timestamp: float
    kind: str
    detail: str


@dataclass(frozen=True)
class Candidate:
    """One timestamp worth rasterizing into a frame image.

    `reason` explains why the timestamp was picked: `"scene"` for an scdet
    boundary, `"audio"` for an audio event, `"interval"` for a fixed grid
    tick, `"fill"` for a synthesized gap filler. `score` carries the pick's
    evidence — the scene score in scdet's raw units, 0.0 elsewhere — and
    ranking reads it after the reason priority, never before, so the result
    stays reproducible top to bottom.
    """

    timestamp: float
    reason: str
    score: float


@dataclass(frozen=True)
class CompletedProcess(Generic[_PayloadT_co]):
    """What one child process did, immune to post-hoc mutation.

    Fields are the brief's exactly: `returncode` in the usual Popen sense,
    `stdout`/`stderr` as captured text (each capped at
    `_STREAM_CAPTURE_LIMIT` with the `_TRUNCATION_MARKER` appended once when
    a stream exceeded the cap), and the two stop flags. A deadline stop or a
    cancellation raises `MediaError` instead of returning — its message
    reports which signal finished the group — so `timed_out` is False in
    every returned result; `killed` is true only when SIGKILL escalation was
    needed, i.e. a descendant still holding the pipes after a normal exit.
    """

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    killed: bool


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
    -show_format -show_streams`) through the bounded runner, so a hung probe
    is stopped with its process group and never outlives the call; stderr is
    captured. A missing file, a directory, or any non-zero ffprobe exit raises
    `MediaError` naming the path and ending with the exit code and the captured
    stderr; a wrong extension or corrupt file is therefore a `MediaError`,
    never a traceback from JSON parsing.
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
        done = run_child(argv, timeout_seconds=_PROBE_TIMEOUT_SECONDS)
    except MediaError as err:
        raise MediaError(f"ffprobe could not read {path}: {err}") from err
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


def free_vram_mib() -> int | None:
    """Return the largest free VRAM in MiB the driver reports, or None.

    `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits` runs
    through the bounded runner, so a wedged driver cannot hold a tool call up.
    None means the measurement is unavailable and the caller must fall back
    rather than guess: the binary is absent, it exited non-zero, the run was
    stopped, or not one line parsed as an integer. With several GPUs the
    largest free figure is the one a tier choice can trust.
    """
    argv = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    try:
        done = run_child(argv, timeout_seconds=_VRAM_QUERY_TIMEOUT_SECONDS)
    except MediaError:
        return None
    if done.returncode != 0:
        return None
    values: list[int] = []
    for line in done.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            values.append(int(text))
        except ValueError:
            continue
    return max(values) if values else None


def detect_scenes(path: Path, threshold: float = 10.0) -> list[SceneScore]:
    """Score the hard cuts of `path` with ffmpeg's `scdet` filter.

    Runs `ffmpeg -i <path> -vf scdet=threshold=<t> -f null -` through
    `run_child` and parses the `lavfi.scd.score: <score>, lavfi.scd.time:
    <second>` lines ffmpeg writes on stderr, tolerantly: lines with another
    shape (banners, progress, Fontconfig noise) are ignored, since stderr
    content alone is never failure. `threshold` is formatted as a plain
    decimal in the filter string, never scientific notation, and the scores
    come back untouched in the detector's own units, at most one per media
    second in ascending order. A non-zero ffmpeg exit raises `MediaError`
    naming the path; a clip with nothing above the threshold yields no
    score lines at all, and that empty result returns `[]` — legitimate,
    not an error.
    """
    path = Path(path)
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(path),
        "-vf",
        f"scdet=threshold={_plain_decimal(threshold)}",
        "-f",
        "null",
        "-",
    ]
    done = run_child(argv)
    if done.returncode != 0:
        raise MediaError(
            f"scene detection failed for {path}: ffmpeg exited "
            f"{done.returncode}; captured stderr: {done.stderr.strip() or '(none)'}"
        )
    return [
        SceneScore(
            timestamp=float(match.group("time")), score=float(match.group("score"))
        )
        for match in _SCD_LINE.finditer(done.stderr)
    ]


def detect_audio_events(
    path: Path, *, noise_db: float = -30.0, min_silence: float = 0.5
) -> list[AudioEvent]:
    """Report the silence segments and loudness peaks of `path`'s audio.

    One ffmpeg pass runs both filters over the same decode:
    `silencedetect=noise=<noise_db>dB:d=<min_silence>` writes its boundaries
    to stderr, and `aresample=48000,asetnsamples=9600:p=1` (a fixed
    0.2-second analysis frame, see `_ASTATS_FRAME_SAMPLES`) feeds
    `astats=metadata=1:reset=1`, whose per-frame peak levels
    `ametadata=mode=print:key=...` writes to stdout. Both thresholds are
    passed into silencedetect's own arguments, so a `noise_db` above the
    file's every level really removes the audio from the result instead of
    being checked cosmetically afterwards.

    Each silence segment yields a `silence_start` at its beginning and a
    `silence_end` at its end. The pair is clamped as a unit to the probed
    duration before it becomes events, because decoded audio can run past
    the container (an AAC audio-only source measured a silent tail ending at
    `4.010667` for a 4.0-second file); filtering the two boundaries
    separately would drop the end and leave a silence that never closes. A
    segment that is empty after clamping, or lies entirely past the
    duration, contributes both events or neither. A segment still open at
    end of stream closes at the probed duration, and an end line without a
    start is skipped because it cannot be placed.

    Loudness yields one `loud` event per run of consecutive analysis frames
    whose peak clears the file's own median frame level by
    `_LOUD_MARGIN_DB` and the caller's noise gate. That rule is deliberately
    conservative: because the gate is absolute, a quiet but dynamic
    recording can report no peak at all, and because the median carries the
    baseline, loud frames filling more than half the analysis frames make
    the median itself loud, so nothing clears the margin. An empty loudness
    list therefore means "no peak under this rule", not "no loud moment".

    Events are ordered by timestamp, and timestamps are non-negative and
    never beyond the probed duration. Silence details are `HH:MM:SS.mmm`
    values (the start's position, the end's span), and a loud detail carries
    its level against the baseline.

    A source with no audio stream returns `[]` before any filter runs, and
    a source with nothing to report also returns `[]` rather than raising.
    A non-zero ffmpeg exit raises `MediaError` naming the path, the exit
    code, and the captured stderr.
    """
    path = Path(path)
    result = probe(path)
    if not result.has_audio:
        return []
    filter_graph = (
        f"silencedetect=noise={_plain_decimal(noise_db)}"
        f"dB:d={_plain_decimal(min_silence)},"
        f"aresample={_ASTATS_SAMPLE_RATE},"
        f"asetnsamples=n={_ASTATS_FRAME_SAMPLES}:p=1,"
        "astats=metadata=1:reset=1,"
        "ametadata=mode=print:key=lavfi.astats.Overall.Peak_level:file=-"
    )
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(path),
        "-af",
        filter_graph,
        "-vn",
        "-f",
        "null",
        "-",
    ]
    done = run_child(argv)
    if done.returncode != 0:
        raise MediaError(
            f"audio event detection failed for {path}: ffmpeg exited "
            f"{done.returncode}; captured stderr: {done.stderr.strip() or '(none)'}"
        )
    events = _silence_events(done.stderr, result.duration)
    events += [
        event
        for event in _loud_events(_astats_frames(done.stdout), noise_db)
        if 0.0 <= event.timestamp <= result.duration
    ]
    events.sort(key=lambda event: event.timestamp)
    return events


def _silence_events(stderr: str, duration: float) -> list[AudioEvent]:
    """Pair silencedetect's boundaries into two events per clamped segment.

    A start line is held until its matching end line arrives; the pair then
    goes through `_silence_pair`, which clamps both boundaries to the probed
    `duration` together. Clamping as a unit is what keeps a decoded tail past
    the container from losing only its end: an audio-only AAC source whose
    silent content decodes to `4.010667` seconds of a 4.0-second file still
    closes at `4.0` instead of leaving a silence that never ends. A start
    still open at end of stream is closed at `duration`, and an end without a
    start has no beginning to place and is skipped. Lines of any other shape
    (banners, progress, astats noise) are ignored.
    """
    events: list[AudioEvent] = []
    open_start: float | None = None
    for line in stderr.splitlines():
        started = _SILENCE_START.search(line)
        if started is not None:
            open_start = float(started.group("time"))
            continue
        ended = _SILENCE_END.search(line)
        if ended is None or open_start is None:
            continue
        events.extend(
            _silence_pair(
                open_start,
                float(ended.group("time")),
                ended.group("duration"),
                duration,
            )
        )
        open_start = None
    if open_start is not None:
        events.extend(_silence_pair(open_start, duration, None, duration))
    return events


def _silence_pair(
    start: float, stop: float, reported: str | None, duration: float
) -> list[AudioEvent]:
    """Clamp one silence segment to the probed duration and build its events.

    Both boundaries clamp into `[0.0, duration]` before anything is emitted,
    and a segment whose clamped stop is at or before its clamped start yields
    an empty list, so the start and end are always kept or dropped as a unit.
    The end detail prefers the filter's own `silence_duration` when clamping
    did not alter the boundaries, and the clamped span otherwise, which keeps
    a tail cut short by the container honest. Both details are
    `HH:MM:SS.mmm` positions/spans, never the kind repeated.
    """
    clamped_start = min(max(start, 0.0), duration)
    clamped_stop = min(max(stop, 0.0), duration)
    if clamped_stop <= clamped_start:
        return []
    if reported is not None and clamped_start == start and clamped_stop == stop:
        span = float(reported)
    else:
        span = clamped_stop - clamped_start
    return [
        AudioEvent(
            timestamp=clamped_start,
            kind="silence_start",
            detail=format_timestamp(clamped_start),
        ),
        AudioEvent(
            timestamp=clamped_stop,
            kind="silence_end",
            detail=format_timestamp(span),
        ),
    ]


def _astats_frames(stdout: str) -> list[tuple[float, float]]:
    """Pair astats' frame timestamps with their Overall peak levels.

    `ametadata=mode=print` writes a `frame:... pts_time:<seconds>` header
    followed by the selected metadata line; parsing walks the lines in order
    and pairs each header with the first `lavfi.astats.Overall.Peak_level`
    after it. `-inf` and `nan` (digital silence, an empty frame) are dropped
    so they can neither drag the median down nor fake a peak. Any other line
    is ignored, so a banner or a stray statistic never breaks the parse.
    """
    frames: list[tuple[float, float]] = []
    current: float | None = None
    for line in stdout.splitlines():
        header = _ASTATS_FRAME.match(line)
        if header is not None:
            current = float(header.group("time"))
            continue
        level = _ASTATS_PEAK.match(line)
        if level is None or current is None:
            continue
        value = level.group("level")
        if value not in _LEVEL_NOT_A_NUMBER:
            frames.append((current, float(value)))
        current = None
    return frames


def _loud_events(
    frames: list[tuple[float, float]], noise_db: float
) -> list[AudioEvent]:
    """Promote runs of analysis frames that stand out from the baseline.

    The baseline is the median of the finite frame levels: a minority of
    loud frames cannot drag a median up the way a mean could, so the rule
    compares against the file's own quiet level. A frame qualifies only when
    its peak clears `baseline + _LOUD_MARGIN_DB` and sits above the caller's
    `noise_db` gate, so raising the gate above a source's every level leaves
    no peak to report. Consecutive qualifying frames are one peak, reported
    at the first frame's timestamp with the run's highest level and its
    margin over the baseline in `detail`.

    Both conditions are limits on purpose: a quiet recording whose clear
    relative peak stays below the absolute gate reports nothing, and a clip
    whose loud frames fill more than half the analysis frames makes the
    median itself loud, so no frame clears the margin. An empty list is a
    result of this rule, not evidence that nothing was loud.
    """
    if not frames:
        return []
    baseline = statistics.median(level for _, level in frames)
    threshold = baseline + _LOUD_MARGIN_DB
    runs: list[list[int]] = []
    for index, (_, level) in enumerate(frames):
        if level < threshold or level <= noise_db:
            continue
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    events: list[AudioEvent] = []
    for run in runs:
        peak = max(frames[index][1] for index in run)
        events.append(
            AudioEvent(
                timestamp=frames[run[0]][0],
                kind="loud",
                detail=(
                    f"peak {peak:.1f} dB, {peak - baseline:+.1f} dB over the "
                    f"{baseline:.1f} dB baseline"
                ),
            )
        )
    return events


def diff_hash_of(raw: bytes) -> int:
    """Digest raw bytes into a stable 64-bit integer.

    Pure and testable without a video: the same bytes always digest to the
    same value, distinct bytes almost never collide, and the digest is
    stable across processes — blake2b, not a salted builtin hash — so
    `hamming` comparisons stay honest between calls and runs. Standard
    library only; nothing more is needed by design.
    """
    return int.from_bytes(
        hashlib.blake2b(raw, digest_size=_HASH_BITS // 8).digest(), "big"
    )


def hamming(a: int, b: int) -> int:
    """Count the differing bits of two 64-bit digests, bounded by 0..64.

    The XOR is masked to the hash width first, so inputs wider than
    `diff_hash_of`'s product are truncated like the digest space they claim
    to compare: bigger arguments can never report more than 64 differing
    bits.
    """
    return ((a ^ b) & _HASH_MASK).bit_count()


def diff_hash(png_bytes: bytes) -> int:
    """Hash a PNG frame after a 32x32 grayscale downscale through ffmpeg.

    The image travels into the child's stdin (`-f image2pipe`) and a
    one-byte-per-pixel gray raster comes out of stdout (`-f rawvideo`), so
    the trip never touches a temporary file and never adds an image
    library: the raster's 1024 raw bytes feed `diff_hash_of` directly, and
    bytes survive the text-typed child result through `run_child`'s
    latin-1 bijection. A non-zero ffmpeg exit or an empty raster raises
    `MediaError` — a malformed PNG is the caller's broken frame, named as
    such. The geometry is pinned: a raster that is not exactly 32x32x1
    bytes also raises, naming the byte count it saw, so a future ffmpeg
    that emitted a different size cannot be hashed silently while the
    comparisons keep claiming to be valid.
    """
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-f",
        "image2pipe",
        "-c:v",
        "png",
        "-i",
        "-",
        "-frames:v",
        "1",
        "-vf",
        f"scale={_GRAY_SIZE}:{_GRAY_SIZE}",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]
    done = run_child(argv, stdin_bytes=png_bytes)
    raster = done.stdout.encode("latin-1")
    if done.returncode != 0 or not raster:
        raise MediaError(
            "diff_hash could not downscale a frame to "
            f"{_GRAY_SIZE}x{_GRAY_SIZE} gray: ffmpeg exited "
            f"{done.returncode}; captured stderr: {done.stderr.strip() or '(none)'}"
        )
    expected = _GRAY_SIZE * _GRAY_SIZE
    if len(raster) != expected:
        raise MediaError(
            f"diff_hash expected a {expected}-byte {_GRAY_SIZE}x{_GRAY_SIZE} gray "
            f"raster from ffmpeg, got {len(raster)} bytes; a changed raster "
            "geometry would make hamming comparisons mean something else"
        )
    return diff_hash_of(raster)


# ---------------------------------------------------------------------------
# Motion analysis: the decimated frame-diff layer behind media_motion.
#
# One ffmpeg pass decodes the source at a reduced analysis rate into a raw
# grayscale raster file (not stdout: `run_child` bounds captured output at
# 8 MiB, and a raster for an hour of video is gigabytes, so the child writes
# the raster to a scratch file and keeps its stdout empty). numpy then walks
# the raster frame by frame computing the two signals the design names —
# mean absolute frame difference and 32-bin grayscale histogram correlation —
# and the event extraction turns those curves into bursts, holds, gradual
# transitions, and keyframes. Defaults (analysis_fps=10, width=320) are the
# grill-me verdicts in docs/design-motion-avsync.md section 9.
# ---------------------------------------------------------------------------

_MOTION_HIST_BINS = 64
_MOTION_MAX_FRAMES = 36000  # one hour at the default 10 fps; caps the raster


@dataclass(frozen=True)
class MotionCurvePoint:
    """One analysis frame's motion signals.

    `t` is the frame's media-time second (`index / analysis_fps`), `diff` is
    the mean absolute grayscale difference against the previous analysis
    frame in 0..255 levels (0.0 for the first frame, which has no
    predecessor), and `hist_dist` is the one-dimensional earth mover's
    distance between the two frames' 64-bin grayscale histograms — the mean
    absolute shift of the cumulative distribution, 0..1 (0.0 for the first
    frame). Earth mover's distance is used because flat-color content puts
    all histogram mass in one bin: a gradual fade then shifts the whole
    mass a few bins per frame, which a support-overlap measure reports as
    total displacement 1.0 while EMD reports the honest small shift, and
    motion graphics are full of exactly such flat-color scenes.
    """

    t: float
    diff: float
    hist_dist: float


@dataclass(frozen=True)
class MotionEvent:
    """One derived span of the motion timeline.

    `kind` is `"burst"` (frame difference sustained above the local
    baseline), `"hold"` (difference at or below the static floor),
    `"transition"` (histogram earth mover's distance sustained above the
    shift floor — the gradual blend/wipe signal `scdet` cannot see), or
    `"keyframe"` (a local difference minimum inside a burst: the
    animation's rest point).
    `start`/`end` are media-time seconds; a `keyframe` is a point event with
    `start == end`. `detail` carries the measurement behind the event in
    readable form without repeating the kind.
    """

    kind: str
    start: float
    end: float
    detail: str


def _motion_frame_geometry(
    raster_bytes: int, width: int, source_width: int, source_height: int
) -> tuple[int, int]:
    """Derive (scaled_height, frame_count) from the raster's exact size.

    ffmpeg's `scale=<width>:-2` rounds the scaled height to an even value;
    rather than trusting one rounding rule, the few even candidates around
    the arithmetic expectation are tried against the file's byte count, and
    the one that divides it into whole frames wins. No candidate fitting is
    a `MediaError` naming the byte count, so a changed ffmpeg scaling rule
    fails loudly instead of misreading the raster as shifted frames.
    """
    base = round(source_height * width / source_width)
    frame_bytes = 0
    scaled_height = 0
    for candidate in sorted({_even(base - 2), _even(base), _even(base + 2)}):
        if candidate > 0 and raster_bytes % (width * candidate) == 0:
            scaled_height = candidate
            frame_bytes = width * candidate
            break
    if not frame_bytes:
        raise MediaError(
            f"motion raster of {raster_bytes} bytes does not divide into "
            f"{width}-wide gray frames around expected height {base}; a "
            "changed ffmpeg scale rule would make frame indexing lie"
        )
    return scaled_height, raster_bytes // frame_bytes


def _even(value: int) -> int:
    """Round `value` up to the next even integer (ffmpeg's `-2` behavior)."""
    return value if value % 2 == 0 else value + 1


def motion_curve(
    path: Path,
    *,
    analysis_fps: float = 10.0,
    width: int = 320,
    scratch_dir: Path | None = None,
    cancel: threading.Event | None = None,
) -> list[MotionCurvePoint]:
    """Decode `path` decimated and return its per-frame motion signals.

    One ffmpeg pass runs `fps=<analysis_fps>,scale=<width>:-2,format=gray`
    over the whole source and writes the raw one-byte-per-pixel raster to
    `<scratch_dir>/motion-raster.gray` (or a fresh temp directory when
    `scratch_dir` is `None`); the file is deleted before returning, success
    or failure. Frames are then read one at a time — memory stays at two
    frames no matter how long the source is — and each frame yields a
    `MotionCurvePoint` against its predecessor. The raster is capped at
    `_MOTION_MAX_FRAMES` frames; longer sources analyze their first hour and
    the caller reports the cap. A source with no video stream, non-positive
    parameters, a failed ffmpeg run, or a raster that does not divide into
    whole frames raises `MediaError`.
    """
    import numpy as np

    if analysis_fps <= 0 or width <= 0:
        raise MediaError(
            "motion analysis needs analysis_fps > 0 and width > 0, got "
            f"{analysis_fps} and {width}"
        )
    path = Path(path)
    result = probe(path)
    if not result.has_video:
        raise MediaError(f"motion analysis needs a video stream; {path} has none")
    scratch = Path(scratch_dir) if scratch_dir is not None else Path(tempfile.mkdtemp())
    scratch.mkdir(parents=True, exist_ok=True)
    raster_path = scratch / "motion-raster.gray"
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-vf",
        (
            f"fps={_plain_decimal(analysis_fps)},"
            f"scale={_plain_decimal(width)}:-2,format=gray"
        ),
        "-frames:v",
        str(_MOTION_MAX_FRAMES),
        "-f",
        "rawvideo",
        str(raster_path),
    ]
    try:
        done = run_child(argv, cancel=cancel)
        if done.returncode != 0 or not raster_path.is_file():
            raise MediaError(
                f"motion raster decode failed for {path}: ffmpeg exited "
                f"{done.returncode}; captured stderr: {done.stderr.strip() or '(none)'}"
            )
        scaled_height, frame_count = _motion_frame_geometry(
            raster_path.stat().st_size, width, result.width, result.height
        )
        frame_bytes = width * scaled_height
        points: list[MotionCurvePoint] = []
        prev: np.ndarray | None = None
        with raster_path.open("rb") as raster:
            for index in range(frame_count):
                buffer = raster.read(frame_bytes)
                if len(buffer) < frame_bytes:
                    break  # truncated tail: analyze what decoded cleanly
                current = np.frombuffer(buffer, np.uint8).reshape(scaled_height, width)
                if prev is None:
                    diff, dist = 0.0, 0.0
                else:
                    diff = float(
                        np.abs(current.astype(np.int16) - prev.astype(np.int16)).mean()
                    )
                    dist = _motion_hist_dist(current, prev)
                points.append(
                    MotionCurvePoint(
                        t=index / analysis_fps, diff=round(diff, 4), hist_dist=dist
                    )
                )
                prev = current
        return points
    finally:
        raster_path.unlink(missing_ok=True)


def _motion_hist_dist(current: np.ndarray, previous: np.ndarray) -> float:
    """Earth mover's distance between two frames' 64-bin gray histograms.

    The one-dimensional Wasserstein-1 distance between the normalized
    histograms equals the mean absolute shift of their cumulative
    distributions, divided by the bin count to land in 0..1. It measures how
    far pixel mass traveled between frames, so a flat-color fade (all mass
    in one bin, sliding a few bins per frame) reports a small honest value
    where a support-overlap measure would report total displacement.
    """
    import numpy as np

    bins = _MOTION_HIST_BINS
    a = np.bincount(current.ravel() // (256 // bins), minlength=bins)[:bins]
    b = np.bincount(previous.ravel() // (256 // bins), minlength=bins)[:bins]
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a /= a.sum()
    b /= b.sum()
    shift = float(np.abs(a.cumsum() - b.cumsum()).sum())
    return round(min(1.0, shift / bins), 4)


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window median with edge padding, same length as `values`.

    `window` is forced odd and at least 1; edges are padded by repeating the
    boundary values, so the baseline near the start and end of the clip is
    the local baseline rather than a shrinking window's artifact.
    """
    window = max(1, int(window) | 1)
    import numpy as np

    if values.size == 0 or window == 1:
        return values.astype(np.float64).copy()
    pad = window // 2
    padded = np.pad(values.astype(np.float64), pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1)


def _motion_runs(
    kind: str,
    mask: np.ndarray,
    timestamps: np.ndarray,
    step: float,
    min_duration: float,
    bridge_frames: int,
    detail_of: object,
) -> list[MotionEvent]:
    """Group contiguous `mask` runs into events of one `kind`.

    Gaps of up to `bridge_frames` False values inside a run do not split it
    — frame-rate conversion duplicates and codec noise punch single-frame
    holes into otherwise steady signals, and a real event survives them.
    A run survives when its span (`length * step`) reaches `min_duration`;
    `detail_of(first_index, last_index)` renders the run's measurement for
    the event's `detail`. Runs are emitted in timeline order.
    """
    events: list[MotionEvent] = []
    run_start: int | None = None
    last_active: int | None = None
    for index in range(mask.size + 1):
        active = index < mask.size and bool(mask[index])
        if active:
            if run_start is None:
                run_start = index
            last_active = index
        elif run_start is not None:
            # `active` is False on this branch, so this is a gap check: a gap
            # longer than `bridge_frames` closes the run, a shorter one lets
            # it continue across the hole. End of the mask always closes it.
            at_end = index >= mask.size
            if at_end or last_active is None or index - last_active > bridge_frames:
                span = (last_active + 1 - run_start) * step
                if span >= min_duration:
                    events.append(
                        MotionEvent(
                            kind=kind,
                            start=round(float(timestamps[run_start]), 3),
                            end=round(float(timestamps[last_active]) + step, 3),
                            detail=detail_of(run_start, last_active),
                        )
                    )
                run_start = None
                last_active = None
    return events


def motion_events(
    points: list[MotionCurvePoint],
    *,
    analysis_fps: float,
    burst_k: float = 3.0,
    burst_min_s: float = 0.3,
    hold_max_diff: float = 1.0,
    hold_min_s: float = 0.5,
    transition_min_dist: float = 0.01,
    transition_min_s: float = 0.4,
    bridge_frames: int = 2,
) -> list[MotionEvent]:
    """Turn a motion curve into bursts, holds, transitions, and keyframes.

    A **burst** is a run where the frame difference clears `burst_k` times
    the 5-second rolling median, with an absolute floor of 1.0 level so a
    nearly static clip's tiny baseline cannot flag its own encoding noise;
    runs shorter than `burst_min_s` are dropped. A **hold** is a run at or
    below `hold_max_diff` lasting at least `hold_min_s`. A **transition**
    is a run where the histogram earth mover's distance stays at or above
    `transition_min_dist` for at least `transition_min_s` — sustained
    histogram travel separates a gradual blend from a one-frame hard cut,
    which `scdet` already reports. Runs of every kind bridge gaps of up to
    `bridge_frames` False values, because frame-rate conversion duplicates
    and codec noise punch single-frame holes into steady signals. A
    **keyframe** is a local difference minimum strictly inside a burst
    (both neighbors higher): the animation's rest point worth rasterizing.
    Events return sorted by `(start, kind)`; an empty curve returns `[]`.
    """
    import numpy as np

    if not points:
        return []
    diffs = np.array([p.diff for p in points], dtype=np.float64)
    dists = np.array([p.hist_dist for p in points], dtype=np.float64)
    times = np.array([p.t for p in points], dtype=np.float64)
    step = 1.0 / analysis_fps
    events: list[MotionEvent] = []

    baseline = _rolling_median(diffs, int(analysis_fps * 5))
    burst_mask = diffs > np.maximum(burst_k * baseline, 1.0)
    events += _motion_runs(
        "burst",
        burst_mask,
        times,
        step,
        burst_min_s,
        bridge_frames,
        lambda first, last: (
            f"peak diff {diffs[first : last + 1].max():.2f} levels, "
            f"baseline {baseline[first : last + 1].mean():.2f}"
        ),
    )
    events += _motion_runs(
        "hold",
        diffs <= hold_max_diff,
        times,
        step,
        hold_min_s,
        bridge_frames,
        lambda first, last: f"max diff {diffs[first : last + 1].max():.2f} levels",
    )
    events += _motion_runs(
        "transition",
        dists >= transition_min_dist,
        times,
        step,
        transition_min_s,
        bridge_frames,
        lambda first, last: (
            f"max hist shift {dists[first : last + 1].max():.3f} over "
            f"{(last - first + 1) * step:.1f}s"
        ),
    )

    burst_spans = [
        (event.start, event.end) for event in events if event.kind == "burst"
    ]
    for start, end in burst_spans:
        inside = np.flatnonzero((times > start) & (times < end))
        for index in inside[1:-1] if inside.size > 2 else []:
            if diffs[index] < diffs[index - 1] and diffs[index] < diffs[index + 1]:
                events.append(
                    MotionEvent(
                        kind="keyframe",
                        start=round(float(times[index]), 3),
                        end=round(float(times[index]), 3),
                        detail=f"diff {diffs[index]:.2f} levels inside burst",
                    )
                )

    events.sort(key=lambda event: (event.start, event.kind))
    return events


# ---------------------------------------------------------------------------
# Rhythm analysis: onsets, tempo, and the beat grid behind media_beats.
#
# The strong engine is the repository's rhythm/ pipeline (librosa in its own
# virtualenv, invoked through rhythm/rhythm.sh, the transcribe/ pattern). The
# pure-numpy baseline here is the fallback the grill-me verdict D2 requires:
# mono PCM through one ffmpeg pass, spectral-flux onset envelope, adaptive
# peak picking, autocorrelation tempo in the 60-180 BPM band, and a
# phase-searched beat comb. It reports engine="numpy" with degraded=True so
# callers can state plainly that the result did not come from librosa.
# ---------------------------------------------------------------------------

_BEAT_SAMPLE_RATE = 22050
_BEAT_FRAME = 1024
_BEAT_HOP = 512
_BEAT_MIN_GAP_S = 0.1
_BEAT_ONSET_K = 2.0
_BEAT_BPM_RANGE = (60.0, 180.0)


@dataclass(frozen=True)
class BeatResult:
    """One source's rhythm analysis, engine-honest.

    `engine` names what produced the numbers (`"librosa"` through the
    rhythm/ virtualenv, `"numpy"` through the baseline here), and `degraded`
    is True exactly when the baseline ran — the grill-me D2 flag that lets a
    response say "beat tracking is weaker than librosa-grade". `tempo` is
    the estimated BPM or None when no reliable periodicity exists, `beats`
    is the beat grid in seconds (empty without a tempo), and `onsets` is the
    measured note-boundary list. Timestamps are sorted, finite, and clamped
    to the decoded duration.
    """

    engine: str
    degraded: bool
    duration: float
    tempo: float | None
    beats: tuple[float, ...]
    onsets: tuple[float, ...]


def _decode_mono_pcm(
    path: Path,
    *,
    scratch_dir: Path | None = None,
    cancel: threading.Event | None = None,
) -> np.ndarray:
    """Decode `path`'s audio to mono float samples in [-1, 1).

    One ffmpeg pass (`-vn -ac 1 -ar 22050 -f s16le`) writes the little-endian
    samples to `<scratch_dir>/rhythm-pcm.s16le` (or a fresh temp directory
    when `scratch_dir` is None) — a file, not stdout, because `run_child`
    only decodes binary stdout in its stdin-feeding mode and a text pipe
    would choke on raw PCM the way it chokes on raw video. The file is
    deleted before returning. A source with no audio stream, a failed ffmpeg
    run, or an empty decode raises `MediaError`.
    """
    import numpy as np

    path = Path(path)
    result = probe(path)
    if not result.has_audio:
        raise MediaError(f"rhythm analysis needs an audio stream; {path} has none")
    scratch = Path(scratch_dir) if scratch_dir is not None else Path(tempfile.mkdtemp())
    scratch.mkdir(parents=True, exist_ok=True)
    pcm_path = scratch / "rhythm-pcm.s16le"
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(_BEAT_SAMPLE_RATE),
        "-f",
        "s16le",
        str(pcm_path),
    ]
    try:
        done = run_child(argv, cancel=cancel)
        if done.returncode != 0 or not pcm_path.is_file():
            raise MediaError(
                f"rhythm audio decode failed for {path}: ffmpeg exited "
                f"{done.returncode}; captured stderr: {done.stderr.strip() or '(none)'}"
            )
        raw = pcm_path.read_bytes()
        if not raw:
            raise MediaError(f"rhythm audio decode of {path} produced no samples")
        return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    finally:
        pcm_path.unlink(missing_ok=True)


def _beat_envelope(pcm: np.ndarray) -> tuple[np.ndarray, float]:
    """Spectral-flux onset envelope of `pcm` and the seconds per envelope bin.

    Frames of `_BEAT_FRAME` samples hop `_BEAT_HOP` samples; each frame's
    positive magnitude change against its predecessor sums into the flux.
    The envelope carries one value per hop: `seconds_per_bin = hop / rate`.
    """
    import numpy as np

    frame = _BEAT_FRAME
    hop = _BEAT_HOP
    if pcm.size < frame:
        pcm = np.pad(pcm, (0, frame - pcm.size))
    pad = frame - hop
    padded = np.pad(pcm, (pad, 0))  # center the first window at sample zero
    windows = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    spectrum = np.abs(np.fft.rfft(windows * np.hanning(frame), axis=1))
    flux = np.clip(np.diff(spectrum, axis=0), 0, None).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    return flux, hop / _BEAT_SAMPLE_RATE


def _onset_pick(envelope: np.ndarray, seconds_per_bin: float) -> list[float]:
    """Adaptive peak picking on an onset envelope, returning seconds.

    A bin is an onset when it is a local maximum and clears the envelope's
    median plus `_BEAT_ONSET_K` scaled median absolute deviations — the
    file's own baseline, so a quiet track keeps its onsets without a
    user-set threshold. Peaks closer than `_BEAT_MIN_GAP_S` collapse into
    the stronger one.
    """
    import numpy as np

    if envelope.size < 3:
        return []
    values = envelope.astype(np.float64)
    center = float(np.median(values))
    spread = float(np.median(np.abs(values - center))) or 1e-9
    threshold = center + _BEAT_ONSET_K * spread
    candidates = [
        index
        for index in range(1, values.size - 1)
        if values[index] > threshold
        and values[index] >= values[index - 1]
        and values[index] > values[index + 1]
    ]
    onsets: list[float] = []
    for index in candidates:
        moment = round(index * seconds_per_bin, 3)
        if onsets and moment - onsets[-1] < _BEAT_MIN_GAP_S:
            continue
        onsets.append(moment)
    return onsets


def _tempo_autocorr(envelope: np.ndarray, seconds_per_bin: float) -> float | None:
    """Estimate BPM from the onset envelope's autocorrelation, or None.

    The envelope is mean-subtracted and half-wave-rectified, its linear
    autocorrelation computed through an FFT, and the strongest lag inside
    the 60-180 BPM band wins. A flat envelope (silence, unbroken noise)
    autocorrelates to nothing distinctive; when the peak is not at least
    three times the lag band's mean, no periodicity is claimed and the
    caller gets None rather than a confident wrong number.
    """
    import numpy as np

    values = envelope.astype(np.float64)
    values = np.clip(values - values.mean(), 0, None)
    if not values.any():
        return None
    size = 1 << (2 * values.size - 1).bit_length()
    spectrum = np.fft.rfft(values, size)
    autocorr = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[: values.size]
    lag_low = max(1, round(_BEAT_BPM_RANGE[1] ** -1 / seconds_per_bin))
    lag_high = min(values.size - 1, round(_BEAT_BPM_RANGE[0] ** -1 / seconds_per_bin))
    if lag_high <= lag_low:
        return None
    band = autocorr[lag_low : lag_high + 1]
    best = int(np.argmax(band)) + lag_low
    if band.max() <= 3.0 * band.mean():
        return None
    return round(60.0 / (best * seconds_per_bin), 2)


def _beat_grid(onsets: list[float], tempo: float, duration: float) -> list[float]:
    """Lay a comb grid at `tempo`, phase-locked to the onset density.

    Every phase in [0, period) is scored by how many onsets sit within
    50 ms of a grid tooth; the best phase lays beats for the whole
    duration. This is deliberately simpler than a dynamic-programming
    tracker — the degraded baseline's beat grid is a pacing reference, not
    a transcription.
    """
    period = 60.0 / tempo
    tolerance = 0.05
    best_phase, best_score = 0.0, -1.0
    steps = max(1, int(period / 0.01))
    for step in range(steps):
        phase = step * period / steps
        score = 0
        tooth = phase
        while tooth < duration:
            score += any(abs(o - tooth) <= tolerance for o in onsets)
            tooth += period
        if score > best_score:
            best_phase, best_score = phase, score
    if best_score <= 0:
        return []
    beats: list[float] = []
    moment = best_phase
    while moment < duration:
        beats.append(round(moment, 3))
        moment += period
    return beats


def detect_beats(
    path: Path,
    *,
    wrapper: Path | None = None,
    scratch_dir: Path | None = None,
    cancel: threading.Event | None = None,
) -> BeatResult:
    """Analyze `path`'s rhythm, librosa first and the numpy baseline second.

    When `wrapper` names an existing `rhythm/rhythm.sh`, one child runs it
    and its JSON stdout becomes a librosa-grade `BeatResult`. Any failure
    of that route — missing venv, non-zero exit, unparseable output — falls
    back to the pure-numpy baseline with `degraded=True`, the grill-me D2
    contract. `scratch_dir` receives the fallback's transient PCM file (the
    source's cache entry in production). No wrapper, or a source without an
    audio stream (an error either way), never silently pretends to be
    librosa.
    """
    path = Path(path)
    if wrapper is not None and Path(wrapper).is_file():
        done = run_child(["bash", str(wrapper), str(path)], cancel=cancel)
        if done.returncode == 0 and done.stdout.strip():
            try:
                payload = json.loads(done.stdout)
                tempo = payload.get("tempo")
                return BeatResult(
                    engine=str(payload.get("engine", "librosa")),
                    degraded=bool(payload.get("degraded", False)),
                    duration=float(payload.get("duration", 0.0)),
                    tempo=float(tempo) if tempo is not None else None,
                    beats=tuple(float(b) for b in payload.get("beats", [])),
                    onsets=tuple(float(o) for o in payload.get("onsets", [])),
                )
            except (ValueError, TypeError):
                pass  # unparseable stdout: fall through to the baseline
    pcm = _decode_mono_pcm(path, scratch_dir=scratch_dir, cancel=cancel)
    duration = float(pcm.size / _BEAT_SAMPLE_RATE)
    envelope, seconds_per_bin = _beat_envelope(pcm)
    onsets = _onset_pick(envelope, seconds_per_bin)
    tempo = _tempo_autocorr(envelope, seconds_per_bin)
    beats = _beat_grid(onsets, tempo, duration) if tempo is not None else []
    return BeatResult(
        engine="numpy",
        degraded=True,
        duration=round(duration, 3),
        tempo=tempo,
        beats=tuple(beats),
        onsets=tuple(onsets),
    )


# ---------------------------------------------------------------------------
# Audio-visual alignment: the zero-decode correlation behind media_avsync.
#
# Pure post-processing over timestamps the earlier layers already measured:
# scdet cuts and rhythm onsets (both corrected by the probe's container-level
# A/V offset), compared against a shuffled-onsets baseline so a "synced"
# verdict is measured, not vibes. No decode happens here.
# ---------------------------------------------------------------------------

_SYNC_BASELINE_SHUFFLES = 40


@dataclass(frozen=True)
class SyncResult:
    """How a source's visual cuts relate to its audio onsets.

    `cuts` and `onsets` are the input counts; `median_distance_s` and
    `p90_distance_s` summarize each cut's distance to the nearest onset
    after the A/V offset correction, and `share_within_tolerance` is the
    fraction of cuts landing within `tolerance_s` of one. `baseline_share`
    is the same statistic against randomly shuffled onsets — the chance
    level a real edit must beat. `verdict` is `"synced"` when the share
    clears the baseline by a margin and is at least even, `"unsynced"` when
    it does not beat the baseline at all, else `"inconclusive"`.
    `beat_phase` is the cuts' circular mean position inside the beat period
    (0..1; near 0.0 or 1.0 means cuts land on beats) or None without a
    usable grid. `offset_s` records the correction applied.
    """

    cuts: int
    onsets: int
    tolerance_s: float
    median_distance_s: float | None
    p90_distance_s: float | None
    share_within_tolerance: float | None
    baseline_share: float | None
    verdict: str
    beat_phase: float | None
    offset_s: float


def _nearest_distances(cuts: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """Each cut's distance to the nearest onset (empty-safe)."""
    import numpy as np

    if cuts.size == 0 or onsets.size == 0:
        return np.array([])
    return np.abs(cuts[:, None] - onsets[None, :]).min(axis=1)


def _share_within(distances: np.ndarray, tolerance: float) -> float | None:
    """Fraction of distances within `tolerance`, or None with no distances."""
    if distances.size == 0:
        return None
    return float((distances <= tolerance).mean())


def _circular_mean_phase(fractions: np.ndarray) -> float:
    """Circular mean of fractions in [0, 1), itself folded back into [0, 1)."""
    import numpy as np

    if fractions.size == 0:
        return 0.0
    angles = 2.0 * math.pi * fractions
    mean_angle = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    phase = (mean_angle / (2.0 * math.pi)) % 1.0
    return round(phase, 4)


def align_cuts_to_onsets(
    cuts: list[float],
    onsets: list[float],
    *,
    duration: float,
    offset: float = 0.0,
    tolerance: float = 0.1,
    beats: list[float] | None = None,
) -> SyncResult:
    """Correlate visual cuts with audio onsets, zero decodes.

    Every onset is moved to container time first (`onset + offset`, the
    probe's `audio.start_time - video.start_time`), so a container-level
    offset never masquerades as bad editing. The measured statistic is each
    cut's distance to the nearest onset; the baseline reruns the same
    statistic over `_SYNC_BASELINE_SHUFFLES` fixed-seed shuffles of the
    onsets inside `duration`, which is the share a non-synced edit scores
    by chance. The beat phase uses `beats` when the grid has at least four
    teeth: each cut's position inside the beat period, averaged circularly.
    A source with no cuts or no onsets yields an `"inconclusive"` verdict
    with None statistics rather than an error — "nothing to compare" is a
    finding, not a failure.
    """
    import numpy as np

    cut_arr = np.array([c for c in cuts if 0.0 <= c <= duration], dtype=np.float64)
    onset_arr = np.array(
        [o + offset for o in onsets if 0.0 <= o + offset <= duration],
        dtype=np.float64,
    )
    distances = _nearest_distances(cut_arr, onset_arr)
    share = _share_within(distances, tolerance)
    median = float(np.median(distances)) if distances.size else None
    p90 = float(np.percentile(distances, 90)) if distances.size else None

    baseline_share = None
    if distances.size and onset_arr.size:
        rng = np.random.default_rng(0)
        shares = []
        for _ in range(_SYNC_BASELINE_SHUFFLES):
            sample = rng.permutation(onset_arr.size)
            fake = rng.uniform(0.0, duration, size=cut_arr.size)
            one = _share_within(_nearest_distances(fake, onset_arr[sample]), tolerance)
            if one is not None:
                shares.append(one)
        baseline_share = round(float(np.mean(shares)), 4) if shares else None

    if share is None or baseline_share is None:
        verdict = "inconclusive"
    elif share >= max(0.5, baseline_share + 0.15):
        verdict = "synced"
    elif share <= baseline_share + 0.05:
        verdict = "unsynced"
    else:
        verdict = "inconclusive"

    beat_phase = None
    grid = [b for b in (beats or []) if 0.0 <= b <= duration]
    if len(grid) >= 4:
        periods = np.diff(np.array(grid, dtype=np.float64))
        period = float(np.median(periods))
        if period > 0:
            fractions = ((cut_arr - grid[0]) / period) % 1.0
            beat_phase = _circular_mean_phase(fractions)

    return SyncResult(
        cuts=int(cut_arr.size),
        onsets=int(onset_arr.size),
        tolerance_s=tolerance,
        median_distance_s=round(median, 4) if median is not None else None,
        p90_distance_s=round(p90, 4) if p90 is not None else None,
        share_within_tolerance=round(share, 4) if share is not None else None,
        baseline_share=baseline_share,
        verdict=verdict,
        beat_phase=beat_phase,
        offset_s=round(offset, 3),
    )


def sample_candidates(
    scenes: list[SceneScore],
    duration: float,
    interval: float = 3,
    dedup_distance: int = 6,
    extract: Callable[[float], int] = lambda t: 0,
) -> list[Candidate]:
    """Merge scene boundaries and interval ticks into deduped candidates.

    Scene scores arrive first as `"scene"` candidates carrying their scdet
    evidence, then fixed-interval ticks are laid across `duration` every
    `interval` seconds starting at zero as `"interval"` candidates at score
    0.0. Everything coalesces in timestamp order, and consecutive runs
    within `dedup_distance` Hamming bits of the run's anchor — per
    `extract(timestamp)`, the caller's perceptual peek at each frame —
    collapse into one survivor: the highest-priority reason, then the
    highest score, then the earliest timestamp, deterministically. With no
    scenes, interval 1 across 10 seconds, and one constant hash, exactly
    one tick survives: the earliest. Nothing rasterizes here; `extract` is
    a sampling stub — the video-side caller plugs `diff_hash` over real
    frames in later tasks.
    """
    candidates: list[Candidate] = [
        Candidate(
            timestamp=float(scene.timestamp), reason="scene", score=float(scene.score)
        )
        for scene in scenes
    ]
    step = float(interval)
    if step > 0.0:
        tick = 0.0
        while tick < duration:
            candidates.append(Candidate(timestamp=tick, reason="interval", score=0.0))
            tick += step
    candidates.sort(
        key=lambda candidate: (candidate.timestamp, _REASON_PRIORITY[candidate.reason])
    )
    hashes = [extract(candidate.timestamp) for candidate in candidates]

    kept: list[Candidate] = []
    run_start = 0
    while run_start < len(candidates):
        anchor_hash = hashes[run_start]
        run_end = run_start + 1
        while run_end < len(candidates) and (
            hamming(hashes[run_end], anchor_hash) <= dedup_distance
        ):
            run_end += 1
        kept.append(min(candidates[run_start:run_end], key=_candidate_key))
        run_start = run_end
    return kept


def rank_candidates(
    candidates: list[Candidate],
    audio_events: list[object],
    budget: int = 120,
) -> tuple[list[Candidate], dict[str, int]]:
    """Spend the frame budget on the best candidates and account for the rest.

    `audio_events` may be plain timestamps or objects carrying a timestamp
    (attribute or mapping key) — both are accepted, an unparsable event is
    skipped without crashing, and an empty list invents nothing. The merged
    list sorts by reason priority (scene > audio > interval > fill), then
    score descending, then timestamp ascending, and the first `budget`
    candidates are kept, so the result is reproducible. While the kept set
    leaves room under `budget`, the widest time gaps between the kept
    timestamps, each at least twice the sampler's `interval` wide, gain one
    `"fill"` candidate at the gap midpoint: synthesized fill never
    displaces a real candidate and never appears when there is nothing to
    fill (no real candidates, no wide gap, or no room). The accounting
    counts every kept reason plus `"dropped"`, so the four reason counts
    always sum to the number kept.
    """
    seed = list(candidates)
    for event in audio_events:
        timestamp = _audio_event_timestamp(event)
        if timestamp is not None:
            seed.append(Candidate(timestamp=timestamp, reason="audio", score=0.0))
    seed.sort(key=_candidate_key)
    kept = seed[: max(0, budget)]
    ranked = sorted(kept + _gap_fills(kept, budget), key=_candidate_key)
    accounting: dict[str, int] = dict.fromkeys(_RANKING_REASONS, 0)
    accounting["dropped"] = len(seed) - len(kept)
    for candidate in ranked:
        if candidate.reason in accounting:
            accounting[candidate.reason] += 1
    return ranked, accounting


# Live children, keyed by pid, so one shutdown can stop every one of them ------
#
# `run_child` registers each live process here and removes it again when it
# exits, so a server that is going away can stop the process group of a call
# still in flight: a gracefully exited MCP server leaves its children alive on
# POSIX unless it takes them down itself, and a reparented ffmpeg or whisper
# would otherwise outlive the request that started it. The lock guards only the
# dictionary operations and is never held across a wait, so registering and
# stopping can never deadlock against each other.
_ACTIVE_CHILDREN: dict[int, subprocess.Popen] = {}
_ACTIVE_LOCK = threading.Lock()


def _register_active_child(process: subprocess.Popen) -> None:
    """Record a live child so `terminate_active_children` can find it."""
    with _ACTIVE_LOCK:
        _ACTIVE_CHILDREN[process.pid] = process


def _unregister_active_child(process: subprocess.Popen) -> None:
    """Forget a child that has exited; a repeated call is harmless."""
    with _ACTIVE_LOCK:
        _ACTIVE_CHILDREN.pop(process.pid, None)


def terminate_active_children(grace: float = 2.0) -> int:
    """Stop every live child's process group; return how many it stopped.

    SIGTERM goes to each live child's process group (the group every
    `run_child` child leads, so a grandchild that inherited it dies too), then
    each survivor gets `grace` seconds before SIGKILL; the direct child is
    reaped so no zombie is left behind. This never raises: a process that is
    already gone, a group that cannot be signalled, and an empty registry are
    all normal, and the return value is the number of children that were still
    live when the call began. `main` runs it in a `finally`, so stdin EOF,
    SIGTERM and SIGINT all end with it.
    """
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE_CHILDREN.values())
    stopped = 0
    try:
        live = [process for process in processes if process.poll() is None]
        stopped = len(live)
        for process in live:
            _signal_group(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + max(0.0, grace)
        for process in live:
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(_POLL_SECONDS)
        for process in live:
            if process.poll() is None:
                _signal_group(process.pid, signal.SIGKILL)
        for process in live:
            try:
                process.wait(timeout=_KILL_GRACE_SECONDS)
            except (subprocess.TimeoutExpired, OSError):
                continue
    except OSError:
        pass
    return stopped


def run_child(
    argv: list[str],
    timeout_seconds: int = 900,
    cancel: threading.Event | None = None,
    stdin_bytes: bytes | None = None,
) -> CompletedProcess[str]:
    """Run `argv` in its own process group and return its bounded output.

    The child starts with `start_new_session=True`, so it leads a fresh
    process group: killing the group takes the whole process tree down, not
    just the direct child, and an orphaned ffmpeg can never outlive the call
    that started it. When the deadline passes or `cancel` is set (whichever
    comes first), SIGTERM is sent to the group, the group gets a two-second
    grace period to exit, and SIGKILL lands on anything that survives. The
    error message names `argv`, the elapsed time, and which of the two
    signals finished the group (`SIGTERM was enough` or `SIGKILL was needed
    after the SIGTERM grace period`).

    stdout and stderr are captured separately as text, each capped at
    `_STREAM_CAPTURE_LIMIT` (8 MiB) with the `_TRUNCATION_MARKER` appended
    once past the cap; overflow is drained and discarded, not buffered, so a
    chatty child cannot exhaust memory. When `stdin_bytes` is given, the
    child reads that payload from a piped stdin closed after the last byte,
    and its stdout is decoded one byte per `latin-1` code point — a strict
    bijection, so a binary stdout stream (the diff-hash gray raster) is
    recoverable exactly with `stdout.encode("latin-1")` — while stderr stays
    UTF-8 with replacements for readable diagnostics; without `stdin_bytes`
    the capture behaves exactly as before. A non-zero exit code is data and
    is returned in `returncode`; this machine's ffmpeg
    tools print `Fontconfig` noise on stderr while succeeding, so only the
    exit code decides success. `MediaError` is raised for a stop (timeout or
    cancellation) and for a process that could not start at all — a missing
    error, naming the binary and the OS error. A live child is registered so
    `terminate_active_children` can stop its group if the server goes away
    while the call is in flight.
    """
    started = time.monotonic()
    try:
        if stdin_bytes is None:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        else:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
    except (OSError, ValueError) as err:
        raise MediaError(f"could not start {argv[0]}: {err}; argv: {argv}") from err

    _register_active_child(process)
    try:
        return _collect_child(
            process, argv, timeout_seconds, cancel, stdin_bytes, started
        )
    finally:
        _unregister_active_child(process)


def _collect_child(
    process: subprocess.Popen[str],
    argv: list[str],
    timeout_seconds: int,
    cancel: threading.Event | None,
    stdin_bytes: bytes | None,
    started: float,
) -> CompletedProcess[str]:
    """Drain, bound, and reap one already-started child.

    The body of `run_child` after the process exists: the registry wrapper
    lives there, and everything that waits on the child lives here, so the
    process can never be forgotten while it is being drained or stopped.
    """
    if stdin_bytes is None:
        stdout_stream, stderr_stream = process.stdout, process.stderr
    else:
        stdout_stream = _TextPipe(process.stdout, codec="latin-1")
        stderr_stream = _TextPipe(process.stderr, codec="utf-8")
    writer = (
        _feed_stdin(process.stdin, stdin_bytes) if stdin_bytes is not None else None
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    readers = [
        thread
        for stream, sink in (
            (stdout_stream, stdout_chunks),
            (stderr_stream, stderr_chunks),
        )
        if (thread := _drain(stream, sink)) is not None
    ]

    deadline = started + timeout_seconds
    trigger = ""
    while process.poll() is None:
        if cancel is not None and cancel.is_set():
            trigger = "cancelled"
            break
        if time.monotonic() >= deadline:
            trigger = "timeout"
            break
        time.sleep(_POLL_SECONDS)

    killed = False
    if trigger:
        killed = _stop_group(process)

    # The stdin feed runs alongside the child's own consumption: once the
    # child exited or the group was stopped, a pipe that is still open was
    # abandoned, the writer's next write breaks, and it closes itself out.
    if writer is not None:
        writer.join(timeout=_KILL_GRACE_SECONDS)

    for reader in readers:
        reader.join(timeout=_KILL_GRACE_SECONDS)
    # A grandchild that inherited a pipe can hold its reader open past the
    # process's exit; force the group down so the call cannot hang and no
    # stray process outlives it. The escalation is recorded in `killed`.
    if any(reader.is_alive() for reader in readers):
        killed = True
        _signal_group(process.pid, signal.SIGKILL)
        for reader in readers:
            reader.join(timeout=_KILL_GRACE_SECONDS)

    stdout, stderr = "".join(stdout_chunks), "".join(stderr_chunks)
    if trigger == "timeout":
        elapsed = time.monotonic() - started
        raise MediaError(
            f"timeout: {argv[0]} was stopped after {elapsed:.1f}s "
            f"(limit {timeout_seconds}s; {_stop_report(killed)}); argv: {argv}; "
            f"captured stderr: {stderr.strip() or '(none)'}"
        )
    if trigger == "cancelled":
        elapsed = time.monotonic() - started
        raise MediaError(
            f"cancel: {argv[0]} was stopped after {elapsed:.1f}s, before its "
            f"timeout ({_stop_report(killed)}); argv: {argv}; captured stderr: "
            f"{stderr.strip() or '(none)'}"
        )
    return CompletedProcess(
        returncode=process.returncode or 0,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        killed=killed,
    )


def _stop_report(killed: bool) -> str:
    """Name the signal that finished the group, for the stop error message.

    The observable half of the runner's `killed` flag: since a stop raises,
    the message reports whether SIGTERM alone sufficed or SIGKILL escalation
    was needed after the grace period.
    """
    if killed:
        return "SIGKILL was needed after the SIGTERM grace period"
    return "SIGTERM was enough"


def _stop_group(process: subprocess.Popen[str]) -> bool:
    """SIGTERM `process`'s group, grace-wait, SIGKILL whatever survives.

    Returns whether SIGKILL escalation was needed; that boolean feeds the
    runner's `killed` field. The group signal binds any grandchild that
    inherited the session, so no orphan survives `run_child`.
    """
    _signal_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=_KILL_GRACE_SECONDS)
        return False
    except subprocess.TimeoutExpired:
        pass
    _signal_group(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    return True


def _signal_group(pid: int, sig: signal.Signals) -> None:
    """Signal `pid`'s process group, ignoring an already-gone group."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def _drain(stream: object, sink: list[str]) -> threading.Thread | None:
    """Spawn a reader thread appending `stream` chunks into `sink`, capped.

    Returns the started daemon thread, or None when the stream is not a text
    pipe (not expected with the pipes `run_child` opens but harmless by
    construction). Once the limit is reached, chunks are drained and discarded
    so the child never blocks on a full pipe, and the truncation marker is
    appended exactly once per stream.
    """
    if stream is None:
        return None

    def _append_capped() -> None:
        grabbed = 0
        truncated = False
        for chunk in iter(lambda: stream.read(_READ_CHUNK_BYTES), ""):
            if truncated:
                continue
            if grabbed + len(chunk) > _STREAM_CAPTURE_LIMIT:
                room = _STREAM_CAPTURE_LIMIT - grabbed
                if room:
                    sink.append(chunk[:room])
                grabbed = _STREAM_CAPTURE_LIMIT
                sink.append(_TRUNCATION_MARKER)
                truncated = True
            else:
                sink.append(chunk)
                grabbed += len(chunk)
        stream.close()

    reader = threading.Thread(target=_append_capped, daemon=True)
    reader.start()
    return reader


class _TextPipe:
    """Read a binary pipe as text so the bounded drain treats both alike.

    `run_child`'s drain loop works in text, so a binary pipe (the stdin-fed
    path) is wrapped: every chunk decodes through an incremental decoder.
    With `latin-1` the decode is a strict per-byte bijection — every byte
    becomes one code point and `text.encode("latin-1")` returns the exact
    bytes — which is how a binary stdout stream survives the text-typed
    `CompletedProcess` untouched. With `utf-8`, invalid sequences decay into
    the replacement character instead of raising at a chunk boundary, which
    keeps diagnostics readable. The incremental decoder carries a partial
    multi-byte sequence across a chunk boundary instead of failing there.
    """

    def __init__(self, raw: BinaryIO, codec: str = "utf-8") -> None:
        self._raw = raw
        self._decoder = codecs.getincrementaldecoder(codec)()

    def read(self, size: int) -> str:
        """Return the next decoded text chunk, or "" at end of stream."""
        return (
            self._decoder.decode(raw_chunk)
            if (raw_chunk := self._raw.read(size))
            else ""
        )

    def close(self) -> None:
        """Close the underlying pipe; closing twice is harmless."""
        self._raw.close()


def _feed_stdin(pipe: BinaryIO | None, payload: bytes) -> threading.Thread | None:
    """Start a daemon thread writing `payload` to the child's stdin, closing it.

    A payload larger than the pipe buffer must be written off the caller's
    thread, or the child writing its stdout and the parent filling its stdin
    can deadlock each holding a full pipe. The thread is the caller's
    joinable handle, returned for the runner's shutdown bookkeeping. A child
    that died before consuming stdin makes the next write fail; a broken or
    already-closed pipe drains the error and closes.
    """
    if pipe is None:
        return None

    def _write_and_close() -> None:
        try:
            pipe.write(payload)
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                pipe.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

    writer = threading.Thread(target=_write_and_close, daemon=True)
    writer.start()
    return writer


def _candidate_key(candidate: Candidate) -> tuple[int, float, float]:
    """Sort key ranking candidates: priority, score descending, time ascending.

    The one comparison rule every ranked list here goes through, so results
    are reproducible: `scene` beats `audio` beats `interval` beats `fill`,
    then the stronger score wins, then the earlier timestamp — which makes
    the winner of an exact tie deterministic (earliest).
    """
    return (_REASON_PRIORITY[candidate.reason], -candidate.score, candidate.timestamp)


def _gap_fills(kept: list[Candidate], budget: int) -> list[Candidate]:
    """Synthesize `"fill"` candidates for the sparse kept sets only.

    A fill exists when the kept set is below `budget` and a gap between two
    consecutive kept timestamps is at least twice the sampler's `interval`
    wide — otherwise the timeline is already dense enough that a midpoint
    frame would land right next to a kept one. One fill rises per gap, at
    the gap's midpoint; the widest gaps fill first, each midpoint rounded
    to six decimals for reproducibility. An empty kept set gets nothing to
    fill, and fills never displace a real candidate since they only exist
    within the room `budget` refuses to `kept`.
    """
    room = budget - len(kept)
    if room <= 0 or not kept:
        return []
    minimum_gap = 2 * float(interval)
    times = sorted(candidate.timestamp for candidate in kept)
    gaps = ((times[index + 1] - times[index], index) for index in range(len(times) - 1))
    widest_first = sorted((gap for gap in gaps if gap[0] >= minimum_gap), reverse=True)
    return [
        Candidate(
            timestamp=round(times[index] + span / 2.0, 6),
            reason="fill",
            score=0.0,
        )
        for span, index in widest_first[:room]
    ]


def _audio_event_timestamp(event: object) -> float | None:
    """Read one audio event's timestamp, or None when it carries no readable one.

    A plain number is a timestamp itself; otherwise an object's `timestamp`
    attribute or a mapping's `"timestamp"` key is read. Anything else
    returns None and contributes nothing later — an unparsable event never
    crashes the ranker and never invents a candidate for the caller.
    """
    if isinstance(event, bool):
        return None
    if isinstance(event, (int, float)):
        return float(event)
    if isinstance(event, Mapping):
        value = event.get("timestamp")
    else:
        value = getattr(event, "timestamp", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _plain_decimal(value: float) -> str:
    """Format a number as plain decimal text for a filter argument.

    Always digits, never scientific notation, so `scdet=threshold=<this>`
    reads one number everywhere a locale could disagree with an exponent:
    10.0 renders as `10`, 0.05 as `0.05`, and a huge value stays a digit
    string instead of growing an `e`.
    """
    text = f"{value:.10f}".rstrip("0")
    return text.removesuffix(".")


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
    the video stream's. The container name, each stream's raw `start_time`,
    and the audio channel layout are kept too, because the probe response
    promises them beside the offset.
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
    container = str(fmt.get("format_name") or "") if isinstance(fmt, dict) else ""

    width = _as_int(video.get("width")) if has_video else 0
    height = _as_int(video.get("height")) if has_video else 0
    if has_video:
        fps = _ratio(video.get("r_frame_rate")) or _ratio(video.get("avg_frame_rate"))
    else:
        fps = 0.0

    video_start = _as_float(video.get("start_time")) if has_video else None
    audio_start = _as_float(audio.get("start_time")) if has_audio else None
    video_offset = video_start if video_start is not None else 0.0
    audio_offset_value = audio_start if audio_start is not None else 0.0
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
        container=container,
        video_start_time=video_start,
        audio_start_time=audio_start,
        audio_layout=_audio_layout(audio),
    )


def _audio_layout(stream: dict[str, object] | None) -> str:
    """Return a stream's channel layout, its channel count, or `""`.

    ffprobe's `channel_layout` is the honest answer and is used verbatim; when
    a decoder omits it, the numeric `channels` is reported as
    `<n> channels`; when there is no audio stream (or neither field is
    usable) the answer is `""` rather than an invented layout name.
    """
    if stream is None:
        return ""
    layout = stream.get("channel_layout")
    if isinstance(layout, str) and layout.strip():
        return layout.strip()
    channels = _as_int(stream.get("channels"))
    if channels > 0:
        return f"{channels} channels"
    return ""


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


# Frames: extraction, crop, tiling, and labels ---------------------------------


@dataclass(frozen=True)
class Box:
    """An integer rectangle in frame pixels: top-left corner and extent.

    `resolve_crop` produces these in frame coordinates (never negative,
    never past the measured frame edge) and the auto-crop mask maths
    produces tile-space boxes of the same shape for the caller to scale.
    """

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class FrameRef:
    """One extracted frame with where it stands on the timeline.

    `timestamp` is the media-time second the frame represents, `path` is
    where the frame image persists (the caller's decision, never written
    by extraction itself, which works in bytes only), and the dimensions
    are the frame's own measured extents after crop and scale.
    """

    timestamp: float
    path: Path
    width: int
    height: int


def sniff_image_format(data: bytes) -> str:
    """Return `"png"` or `"jpeg"` for the image format of `data`.

    Magic bytes decide: the 8-byte PNG signature or the `ff d8 ff` JPEG
    start marker. The JPEG fallback path (see `JPEG_FALLBACK_BYTES`) must
    be observably reported, so an unidentifiable payload is a `MediaError`
    here rather than a silent guess that a consumer would trust.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    raise MediaError(
        f"unrecognized image format: a payload of {len(data)} bytes carries "
        "neither a PNG signature nor JPEG magic"
    )


def resolve_crop(
    spec: str | dict | None,
    probe: ProbeResult,
    path: Path,
    start: float,
    end: float,
    fps: float,
    pad: int = 8,
) -> Box | None:
    """Resolve a crop request into a `Box` in frame pixels, or None for none.

    `None` means the whole frame stays. An explicit mapping must carry every
    one of the `x`, `y`, `width`, `height` keys — a missing key is a
    `MediaError` naming it — and the box must fit inside the frame probe
    measured, with that measured size in the rejection message. The string
    `"auto"` runs one ffmpeg pass over the clamped window (clamping past-the-
    end windows is what keeps the tile grid exactly filled, because `tile`
    pads an under-filled grid with black and the difference mask would then
    see motion that is only padding): it downscales the window to
    `_AUTOCROP_TILE_WIDTH`-wide gray tiles, marks the cells whose values move
    more than `_AUTOCROP_DIFF_STEP` across the tiles, and scales the marked
    bounding box back to frame pixels padded by `pad` and clamped. A
    motionless window, and one where 90 percent or more of all cells move
    (frame-filling motion or black padding), resolves to None so the design's
    full-frame fallback applies. Anything other string is a `MediaError`.
    """
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        return _explicit_box(spec, probe)
    if spec == "auto":
        return _auto_box(probe, Path(path), start, end, fps, pad)
    raise MediaError(
        f"crop spec must be None, 'auto', or a mapping with x/y/width/height "
        f"keys, not {spec!r}"
    )


def extract_frame(
    path: Path, timestamp: float, crop: Box | None, max_edge: int
) -> bytes:
    """Extract one still of `path` at `timestamp` seconds as PNG bytes.

    Probing comes first, so a source without video fails with `no video
    stream` in the message instead of letting an ffmpeg error be the only
    signal. When `crop` is given, a `crop` filter runs before the bounded
    scale (`-vf crop=...,scale='if(gt(iw,ih),min(iw,MAX),-2)':'if(gt(iw,ih),
    -2,min(ih,MAX))'`); the still comes back over an image pipe as PNG. An
    empty or non-zero-exit run is a `MediaError` naming the path, the
    timestamp, and ffmpeg's captured stderr; and when the encoded PNG
    exceeds `JPEG_FALLBACK_BYTES` the frame is re-encoded to mjpeg — the
    fallback both limits are reported against, and the caller can tell it
    happened through `sniff_image_format`.
    """
    path = Path(path)
    result = probe(path)
    if not result.has_video:
        raise MediaError(f"{path} has no video stream, so no frame can be extracted")
    if max_edge < 1:
        raise MediaError(
            f"extract_frame needs a max_edge of at least 1, not {max_edge}"
        )
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-ss",
        _plain_decimal(float(timestamp)),
        "-i",
        str(path),
        "-vf",
        ",".join(_frame_filters(crop, max_edge)),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    done = run_child(argv, stdin_bytes=b"")
    frame = done.stdout.encode("latin-1")
    if done.returncode != 0 or not frame:
        raise MediaError(
            f"frame extraction from {path} at {timestamp}s failed: ffmpeg "
            f"exited {done.returncode}; captured stderr: "
            f"{done.stderr.strip() or '(none)'}"
        )
    if len(frame) > JPEG_FALLBACK_BYTES:
        return _reencode_as_mjpeg(frame, f"the frame from {path} at {timestamp}s")
    return frame


def render_sheet(
    pngs: list[tuple[float, bytes]],
    labels: bool,
    max_edge: int,
    scratch_dir: Path | None = None,
    notes: list[str] | None = None,
) -> bytes:
    """Composite frames into one labelled contact sheet, PNG bytes.

    Frames arrive as `(timestamp, image_bytes)` already rasterized; the
    sheet scales each to a `max_edge // cols`-wide tile (`cols` grows with
    the square root of the frame count), stacks them with one `xstack`
    filter, and burns `HH:MM:SS.mmm` per tile in the top-left corner when
    `labels` is true and a font is resolvable. A missing font yields
    unlabelled tiles and no error — a label is decoration, never a
    precondition. The sheet's width stays under `max_edge` by construction.
    One tile's label failing is equally non-fatal: that tile is returned
    unlabelled and the reason is appended to `notes`, so a caller can state
    the degradation instead of hiding it.

    `scratch_dir` is the root for the intermediate tile and ASS files: the
    caller passes a directory on the output filesystem (the cache entry it
    already owns) so intermediates never land in the OS temp directory, and
    `None` keeps the OS temp directory as the default.
    """
    return _render_composite(pngs, labels, max_edge, "sheet", scratch_dir, notes)


def render_filmstrip(
    pngs: list[tuple[float, bytes]],
    labels: bool,
    max_edge: int,
    scratch_dir: Path | None = None,
    notes: list[str] | None = None,
) -> bytes:
    """Composite frames into one vertical filmstrip, PNG bytes.

    One `vstack` filter stacks the frames, so every input is scaled to the
    smallest width among them capped at `max_edge` (even, because vstack
    demands equal widths); each frame keeps whatever crop it already
    carried when it was extracted, and its aspect survives the scale.
    Labels follow the same rule as the sheet, and `scratch_dir` selects the
    staging root exactly as `render_sheet` documents it.
    """
    return _render_composite(pngs, labels, max_edge, "filmstrip", scratch_dir, notes)


def resolve_font() -> Path | None:
    """Find a font file for burned-in labels, or None when nothing qualifies.

    `MEDIA_FONT_FILE` wins when it is set — the set-but-missing case returns
    None on purpose, honouring the owner's miss rather than second-guessing
    it. Otherwise `fc-match monospace` decides, and its file returns only if
    it exists. A missing fontconfig binary, a failed match, or a reported
    path that does not exist are all None with no exception: the design
    forbids losing a frame set to a font. This is the one environment read
    in this module, and it reads only `MEDIA_FONT_FILE`.
    """
    env_value = os.environ.get("MEDIA_FONT_FILE")
    if env_value:
        path = Path(env_value)
        return path if path.exists() else None
    try:
        done = run_child(
            ["fc-match", "monospace", "--format", "%{file}\t%{family}\n"],
            timeout_seconds=30,
        )
    except MediaError:
        return None
    if done.returncode != 0:
        return None
    first = next((line for line in done.stdout.splitlines() if line.strip()), "")
    reported = first.split("\t")[0].strip()
    if not reported:
        return None
    path = Path(reported)
    return path if path.is_file() else None


def _frame_filters(crop: Box | None, max_edge: int) -> list[str]:
    """The per-frame still filters: crop when a box is given, then the scale.

    The scale expression keeps the long edge at or under `max_edge` and the
    short edge at an even width or height (`-2`), so consumers that expect
    even chroma dimensions stay happy without a second pass.
    """
    filters = []
    if crop is not None:
        filters.append(f"crop=w={crop.width}:h={crop.height}:x={crop.x}:y={crop.y}")
    filters.append(
        f"scale='if(gt(iw,ih),min(iw,{int(max_edge)}),-2)':"
        f"'if(gt(iw,ih),-2,min(ih,{int(max_edge)}))'"
    )
    return filters


def _explicit_box(spec: Mapping, probe: ProbeResult) -> Box:
    """Validate an explicit crop mapping against the probe's measured frame.

    Every key is required and named when missing; values must be plain
    numbers (integer-valued is enough, so `128.0` is accepted); and the
    resulting rectangle must lie inside the frame, with the measured frame
    size quoted in the rejection so the caller can correct the spec against
    the file that actually exists.
    """
    values: dict[str, int] = {}
    for key in ("x", "y", "width", "height"):
        if key not in spec:
            raise MediaError(
                f"crop dict is missing the required key '{key}' (crop needs "
                f"all of x, y, width, height)"
            )
        value = spec[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MediaError(f"crop key '{key}' must be a number, not {value!r}")
        values[key] = int(value)
    box = Box(values["x"], values["y"], values["width"], values["height"])
    if (
        box.x < 0
        or box.y < 0
        or box.width <= 0
        or box.height <= 0
        or box.x + box.width > probe.width
        or box.y + box.height > probe.height
    ):
        raise MediaError(
            f"crop box (x={box.x}, y={box.y}, width={box.width}, "
            f"height={box.height}) does not fit inside the measured frame "
            f"{probe.width}x{probe.height} of the source"
        )
    return box


def _auto_box(
    probe: ProbeResult,
    path: Path,
    start: float,
    end: float,
    fps: float,
    pad: int,
) -> Box | None:
    """Measure one tiled pass over the window and scale its motion box back.

    The raw read must be exactly `tiles * tw * th` gray bytes — a shorter or
    longer payload is a truncated (or corrupted) pipeline read and a
    `MediaError`, naming the path and both byte counts. Grid geometry comes
    from the probe's own measured width and height so the scale back is
    faithful.
    """
    if not probe.has_video or probe.width <= 0 or probe.height <= 0:
        raise MediaError(f"auto crop has no measured frame to work from for {path}")
    start = max(0.0, float(start))
    end = min(float(end), probe.duration)
    span = end - start
    if span <= 0.0:
        return None
    tiles = max(_AUTOCROP_MIN_TILES, min(round(span * float(fps)), _AUTOCROP_MAX_TILES))
    tw = _AUTOCROP_TILE_WIDTH
    th = max(2, (tw * probe.height // probe.width) & ~1)
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-ss",
        _plain_decimal(start),
        "-t",
        _plain_decimal(span),
        "-i",
        str(path),
        "-vf",
        (
            f"fps={_plain_decimal(float(fps))},trim=end_frame={tiles},"
            f"scale={tw}:{th},tile={tiles}x1"
        ),
        "-frames:v",
        "1",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]
    done = run_child(argv, stdin_bytes=b"")
    raster = done.stdout.encode("latin-1")
    expected = tiles * tw * th
    if done.returncode != 0 or len(raster) != expected:
        raise MediaError(
            f"auto crop over window [{start}, {end}] of {path} measured "
            f"{len(raster)} raw gray bytes where {expected} were required "
            f"for a {tiles}-tile {tw}x{th} grid (exit code {done.returncode})"
        )
    tile_box = _motion_box(raster, tw, th, tiles)
    if tile_box is None:
        return None
    return _scale_tile_box_back(tile_box, probe.width, probe.height, tw, th, pad)


def _motion_box(
    raster: bytes, tile_width: int, tile_height: int, tiles: int
) -> Box | None:
    """Locate the moving cells of a tiled gray raster and shrink them to a box.

    Pure maths on the bytes, unit-testable without ffmpeg: the grid is
    `tiles` `tile_width`x`tile_height` tiles side by side exactly as ffmpeg's
    `tile=<n>x1` emits them, so cell `(dy, dx)` of tile `t` is
    `raster[dy * stride + t * tile_width + dx]`. A cell is marked when its
    value span across the tiles exceeds `_AUTOCROP_DIFF_STEP`; the marked
    cells' bounding box comes back in tile coordinates. No marked cell means
    nothing moved (None), and 90 percent or more of all cells marked means
    the result is either frame-filling motion or a black-padded grid — both
    poison for a crop, both resolving to None.
    """
    stride = tile_width * tiles
    marked: list[tuple[int, int]] = []
    for dy in range(tile_height):
        row_base = dy * stride
        for dx in range(tile_width):
            values = [raster[row_base + t * tile_width + dx] for t in range(tiles)]
            if max(values) - min(values) > _AUTOCROP_DIFF_STEP:
                marked.append((dy, dx))
    if not marked:
        return None
    if len(marked) >= _AUTOCROP_BUSY_TILES * (tile_height * tile_width):
        # All or nearly all cells moving is the black-padding poison detector
        # just as much as an honest frame-filling motion: no crop result.
        return None
    xs = [dx for _, dx in marked]
    ys = [dy for dy, _ in marked]
    return Box(
        x=min(xs),
        y=min(ys),
        width=max(xs) - min(xs) + 1,
        height=max(ys) - min(ys) + 1,
    )


def _scale_tile_box_back(
    tile_box: Box, width: int, height: int, tw: int, th: int, pad: int
) -> Box:
    """Scale a tile-space box back to frame pixels, pad it, and clamp it.

    `sx` and `sy` come from the probe's measured frame over the measured
    grid, the box grows by `pad` on every side, and the clamps keep the
    result a legal rectangle: non-negative corner, positive extent, and
    inside the frame.
    """
    sx = width / tw
    sy = height / th
    x0 = int(tile_box.x * sx)
    y0 = int(tile_box.y * sy)
    x1 = math.ceil((tile_box.x + tile_box.width) * sx)
    y1 = math.ceil((tile_box.y + tile_box.height) * sy)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)
    return Box(x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _staging_dir(prefix: str, scratch_dir: Path | None) -> tempfile.TemporaryDirectory:
    """A temporary directory, under `scratch_dir` when the caller chose one.

    `None` keeps the OS temp directory as today's default; a caller-chosen
    root stages beside its output so nothing large crosses filesystems.
    """
    if scratch_dir is None:
        return tempfile.TemporaryDirectory(prefix=prefix)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=scratch_dir)


def _render_composite(
    pngs: list[tuple[float, bytes]],
    labels: bool,
    max_edge: int,
    mode: str,
    scratch_dir: Path | None = None,
    notes: list[str] | None = None,
) -> bytes:
    """Stack frames into one image through a single ffmpeg filter graph.

    The per-tile work happens first (label burn through one `ass` filter per
    distinct text, one ASS document per tile written to a scratch directory
    the OS cleans up — or `scratch_dir`, the caller's chosen root), the
    labelled tiles land as files so one compositing ffmpeg can read them as
    inputs, and the stacking itself is one graph — `xstack` for the sheet,
    `vstack` for the filmstrip. A missing font turns labels off silently and
    a per-tile burn failure degrades that tile to unlabelled (its reason
    appended to `notes`); the compositing itself still raises on failure.
    """
    if not pngs:
        raise MediaError(f"{mode} render received no frames to composite")
    if max_edge < 1:
        raise MediaError(
            f"{mode} render needs a max_edge of at least 1, not {max_edge}"
        )
    frames = [frame for _, frame in pngs]
    dimensions = [_image_dimensions(frame) for frame in frames]
    with _staging_dir("media-tiles-", scratch_dir) as scratch:
        root = Path(scratch)
        tiles = [root / f"{index:04d}.png" for index in range(len(frames))]
        for tile_path, frame in zip(tiles, frames):
            tile_path.write_bytes(frame)
        if labels:
            font = resolve_font()
            if font is not None:
                family = _font_family(font) or "monospace"
                for index, (timestamp, frame) in enumerate(pngs):
                    try:
                        labelled = _label_tile(
                            frame,
                            timestamp,
                            dimensions[index],
                            family,
                            font,
                            scratch_dir,
                        )
                    except MediaError as err:
                        if notes is not None:
                            notes.append(
                                f"the label for the tile at "
                                f"{_timestamp_label(timestamp)} could not be "
                                f"burned ({err}); that tile is unlabelled"
                            )
                        continue
                    tiles[index].write_bytes(labelled)
        graph = _composite_graph(mode, max_edge, dimensions)
        argv = ["ffmpeg", "-hide_banner", "-nostdin"]
        for tile_path in tiles:
            argv += ["-i", str(tile_path)]
        argv += [
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
        done = run_child(argv, stdin_bytes=b"")
        composite = done.stdout.encode("latin-1")
        if done.returncode != 0 or not composite:
            raise MediaError(
                f"{mode} compositing of {len(pngs)} frames failed: ffmpeg "
                f"exited {done.returncode}; captured stderr: "
                f"{done.stderr.strip() or '(none)'}"
            )
    if len(composite) > JPEG_FALLBACK_BYTES:
        return _reencode_as_mjpeg(
            composite, f"the {mode} composite of {len(pngs)} frames"
        )
    return composite


def _composite_graph(mode: str, max_edge: int, dimensions: list) -> str:
    """Build the one ffmpeg filter graph that stacks the scaled tiles.

    Sheet mode scales each input to `max_edge // cols` wide (aspect kept,
    heights recomputed per tile from its own measured box) and lays out a
    regular grid left-to-right, top-to-bottom, one row offset per grid row
    so overlapping tiles are impossible. Filmstrip mode scales every input
    to one shared even width — the smallest measured width capped by
    `max_edge` — because vstack demands equal widths. A single frame needs
    no stack at all.
    """
    count = len(dimensions)
    chains: list[str] = []
    if mode == "sheet":
        cols = max(1, math.ceil(math.sqrt(count)))
        tile_w = max(1, max_edge // cols)
        if count == 1:
            chains.append(f"[0:v]scale={tile_w}:-2[out]")
            return ";".join(chains)
        heights = [_even_up(tile_w * section[1] / section[0]) for section in dimensions]
        row_y = 0
        layout_parts: list[str] = []
        for index in range(count):
            if index % cols == 0:
                row_y = sum(
                    max(
                        heights[row * cols + col]
                        for col in range(cols)
                        if row * cols + col < count
                    )
                    for row in range(index // cols)
                )
            layout_parts.append(f"{(index % cols) * tile_w}_{row_y}")
        chains += [f"[{index}:v]scale={tile_w}:-2[s{index}]" for index in range(count)]
        chains.append(
            "".join(f"[s{index}]" for index in range(count))
            + f"xstack=inputs={count}:layout={_stack_layout(layout_parts)}[out]"
        )
        return ";".join(chains)
    # Filmstrip: one shared even width, capped by max_edge.
    narrowest = min(width for width, _ in dimensions)
    strip_w = max(2, min(narrowest, max_edge) & ~1)
    if count == 1:
        chains.append(f"[0:v]scale={strip_w}:-2[out]")
        return ";".join(chains)
    chains += [f"[{index}:v]scale={strip_w}:-2[s{index}]" for index in range(count)]
    chains.append(
        "".join(f"[s{index}]" for index in range(count)) + f"vstack=inputs={count}[out]"
    )
    return ";".join(chains)


def _stack_layout(parts: list[str]) -> str:
    """Join `x_y` position strings the way xstack's layout wants."""
    return "|".join(parts)


def _even_up(value: float) -> int:
    """Round up to an even integer, for heights recomputed per tile."""
    return math.ceil(value / 2.0) * 2


def _label_tile(
    frame: bytes,
    timestamp: float,
    dimensions: tuple[int, int],
    family: str,
    font: Path,
    scratch_dir: Path | None = None,
) -> bytes:
    """Burn `HH:MM:SS.mmm` into one frame's top-left corner through `ass`.

    One ASS document per tile because the text differs per tile; `fontsdir`
    points at the font's own directory so a resolved family always loads.
    The glyph size scales with the tile (see
    `_LABEL_FONT_REFERENCE_*`), so a big sheet and a small filmstrip carry
    equally legible labels in that same measured proportion. Fontconfig
    noise on stderr is not failure: only the exit code and an empty payload
    decide. `scratch_dir` selects where the ASS document is staged, exactly
    as `render_sheet` documents it.
    """
    width, height = dimensions
    size = max(
        _LABEL_FONT_SIZE_MIN,
        round(width * _LABEL_FONT_REFERENCE_SIZE / _LABEL_FONT_REFERENCE_WIDTH),
    )
    with _staging_dir("media-ass-", scratch_dir) as scratch:
        document = Path(scratch) / "label.ass"
        document.write_text(
            _label_ass_document(
                width,
                height,
                _timestamp_label(timestamp),
                family,
                size,
            ),
            encoding="utf-8",
        )
        argv = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg" if sniff_image_format(frame) == "jpeg" else "png",
            "-i",
            "-",
            "-frames:v",
            "1",
            "-vf",
            (f"ass={_filter_quote(document)}:fontsdir={_filter_quote(font.parent)}"),
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
        done = run_child(argv, stdin_bytes=frame)
        labelled = done.stdout.encode("latin-1")
        if done.returncode != 0 or not labelled:
            raise MediaError(
                f"label burn for the tile at {timestamp}s failed: ffmpeg "
                f"exited {done.returncode}; captured stderr: "
                f"{done.stderr.strip() or '(none)'}"
            )
        return labelled


def _label_ass_document(
    width: int, height: int, text: str, family: str, size: int
) -> str:
    """Generate the ASS document carrying one top-left timestamp event.

    `PlayResX`/`PlayResY` match the tile so coordinates stay honest, white
    text over a black outline stays readable on any background, and
    alignment 7 anchors the text at the top-left margin the design wants.
    """
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "ScaledBorderAndShadow: yes\n"
        "WrapStyle: 2\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour"
        ", OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX"
        ", ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment"
        ", MarginL, MarginR, MarginV, Encoding\n"
        "Style: Label,"
        f"{family},{size}"
        ",&H00FFFFFF,&H00000000,&H00000000,&H00000000"
        ",0,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV"
        ", Effect, Text\n"
        f"Dialogue: 0,0:00:00.00,9:59:59.99,Label,,0,0,0,,{text}\n"
    )


def _timestamp_label(seconds: float) -> str:
    """Format a media timestamp as `HH:MM:SS.mmm`, the design's label shape."""
    total_ms = max(0, round(float(seconds) * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _font_family(font: Path) -> str | None:
    """Read a font file's family with fc-scan; None when that lookup fails.

    The family goes into the ASS style line; libass needs the real name, and
    a failed lookup means the caller falls back to the literal `monospace`
    family with `fontsdir` still set.
    """
    try:
        done = run_child(
            ["fc-scan", "--format", "%{family}\n", str(font)],
            timeout_seconds=30,
        )
    except MediaError:
        return None
    if done.returncode != 0:
        return None
    first = next((line for line in done.stdout.splitlines() if line.strip()), "")
    family = first.strip()
    return family or None


def _reencode_as_mjpeg(payload: bytes, what: str) -> bytes:
    """Re-encode an oversized PNG to mjpeg and return those bytes.

    ffmpeg's mjpeg encoder exposes qscale, not a quality percentage, so
    `-q:v 3` is the closest documented stand-in for the design's quality 85;
    this is the fallback the design demands be reported (via
    `sniff_image_format`), never a silent format switch.
    """
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-f",
        "image2pipe",
        "-c:v",
        "png",
        "-i",
        "-",
        "-frames:v",
        "1",
        "-q:v",
        "3",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]
    done = run_child(argv, stdin_bytes=payload)
    jpeg = done.stdout.encode("latin-1")
    if done.returncode != 0 or not jpeg:
        raise MediaError(
            f"jpeg fallback re-encode of {what} failed: ffmpeg exited "
            f"{done.returncode}; captured stderr: "
            f"{done.stderr.strip() or '(none)'}"
        )
    return jpeg


def _image_dimensions(data: bytes) -> tuple[int, int]:
    """Measure an already-encoded image stream, standard library only.

    PNG goes through its IHDR; JPEG through its start-of-frame marker. The
    renderers need each tile's box to lay out the stack and to scale the
    label font, and a producer may have delivered the mjpeg fallback form,
    so both formats are read here. Anything else is a `MediaError`, the same
    rule `sniff_image_format` applies.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        if data[12:16] != b"IHDR":
            raise MediaError(f"PNG stream of {len(data)} bytes carries no IHDR chunk")
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    if data[:3] == b"\xff\xd8\xff":
        return _jpeg_dimensions(data)
    raise MediaError(
        f"cannot measure image dimensions: a payload of {len(data)} bytes is "
        "neither PNG nor JPEG"
    )


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Walk JPEG marker segments for the first start-of-frame marker.

    Only the markers a real encoder emits need walking: standalone markers
    carry no length, everything else does, and any format-signed width and
    height after the marker are big-endian. No start-of-frame marker at all
    is a malformed stream and a `MediaError`.
    """
    index = 2
    limit = len(data)
    while index + 1 < limit:
        if data[index] != 0xFF:
            raise MediaError(
                f"malformed JPEG stream of {len(data)} bytes: expected a "
                f"marker at byte {index}"
            )
        while index < limit and data[index] == 0xFF:
            index += 1
        if index >= limit:
            break
        marker = data[index]
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            # Standalone markers carry no length segment.
            index += 1
            continue
        if index + 3 > limit:
            break
        seg_len = (data[index + 1] << 8) | data[index + 2]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if index + 8 > limit:
                break
            height, width = struct.unpack(">HH", data[index + 4 : index + 8])
            return width, height
        index += 1 + seg_len
    raise MediaError(
        f"malformed JPEG stream of {len(data)} bytes: no start-of-frame marker"
    )


def _filter_quote(value: Path) -> str:
    """Quote a filesystem path for use inside an ffmpeg filter argument."""
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


# OCR and value tracking ---------------------------------------------------------


@dataclass(frozen=True)
class OcrResult:
    """One frame's recognized text with the mean confidence of its words.

    `timestamp` is the media-time second the frame represents. `ocr_png`
    receives only pixels, so it stamps 0.0 and the caller records the real
    timestamp where it stores the result. `confidence` is the mean of the
    kept word confidences scaled from tesseract's percent into 0..1, and
    0.0 when the frame carried no readable word at all — an unreadable
    frame is a result, never an error.
    """

    timestamp: float
    text: str
    confidence: float


@dataclass(frozen=True)
class ChangeCandidate:
    """One frame whose tracked number glitches against both its neighbours.

    `value` is the candidate's own first numeric token, `before` and `after`
    are the neighbouring valued frames' tokens, and `timestamp` and
    `confidence` are the candidate frame's own. A one-frame glitch reports
    with the stable value on both sides; a held step change is not a
    candidate because the new value equals its next neighbour.
    """

    timestamp: float
    value: str
    before: str
    after: str
    confidence: float


def ocr_available_langs() -> list[str]:
    """Return the languages `tesseract --list-langs` reports, in its order.

    The first line is tesseract's header (the tessdata directory and the
    count); every following non-blank line is a language code, kept exactly
    as printed. A non-zero exit raises `MediaError` carrying the exit code
    and the captured stderr.
    """
    done = run_child(["tesseract", "--list-langs"], timeout_seconds=60)
    if done.returncode != 0:
        raise MediaError(
            "tesseract could not list its languages: exited "
            f"{done.returncode}; captured stderr: {done.stderr.strip() or '(none)'}"
        )
    lines = [line.strip() for line in done.stdout.splitlines()]
    return [code for code in lines[1:] if code]


def ocr_png(png_bytes: bytes, psm: int = 6, lang: str = "eng") -> OcrResult:
    """Recognize the text of one PNG frame through tesseract over its pipes.

    The image travels in on stdin and the TSV report comes out on stdout
    (`tesseract - - -l <lang> --psm <psm> tsv` through `run_child`), so the
    frame never touches a temporary file and no shell is involved. `psm` is
    tesseract's page segmentation mode and `lang` its language code. The
    `timestamp` this returns is 0.0 on purpose: `png_bytes` carries no
    timeline, and the caller stamps the real timestamp where it stores the
    result.

    A missing language pack is detected from tesseract's stderr and raised
    as a `MediaError` carrying its first stderr line plus the languages
    `ocr_available_langs` reports, so the caller can name an installed one
    without a second round-trip; any other non-zero exit raises
    `MediaError` with the captured stderr. A frame with nothing readable in
    it comes back as empty text with 0.0 confidence, never an error.
    """
    argv = ["tesseract", "-", "-", "-l", lang, "--psm", str(int(psm)), "tsv"]
    done = run_child(argv, stdin_bytes=png_bytes)
    if done.returncode != 0:
        raise _ocr_failure(done, lang)
    # With `stdin_bytes` given, run_child decodes stdout one byte per
    # latin-1 code point — a strict bijection — so encoding back recovers
    # tesseract's exact UTF-8 bytes for a faithful decode.
    tsv = done.stdout.encode("latin-1").decode("utf-8", "replace")
    text, confidence = _parse_tesseract_tsv(tsv)
    return OcrResult(timestamp=0.0, text=text, confidence=confidence)


def track_values(results: list[OcrResult]) -> list[ChangeCandidate]:
    """Return the frames whose tracked number glitches against both sides.

    Only frames whose text carries a numeric token are valued: the value is
    the first integer-or-decimal token (`\\d+(?:\\.\\d+)?`) in the frame's
    text, and frames without one drop out of the sequence entirely. A
    valued frame is a candidate when its value differs from both its
    immediate previous and its immediate next valued neighbour, so a
    one-frame glitch reports with the stable value on both sides, a held
    step change reports nothing (the new value equals its next neighbour),
    and the first and last valued frames can never be candidates. The
    candidate's `value`, `timestamp`, and `confidence` are its own;
    `before` and `after` are the neighbours' values. Input order is
    preserved, and empty or single-entry sequences return `[]`.
    """
    valued: list[tuple[OcrResult, str]] = []
    for result in results:
        match = _VALUE_TOKEN.search(result.text)
        if match is not None:
            valued.append((result, match.group(0)))
    return [
        ChangeCandidate(
            timestamp=valued[index][0].timestamp,
            value=valued[index][1],
            before=valued[index - 1][1],
            after=valued[index + 1][1],
            confidence=valued[index][0].confidence,
        )
        for index in range(1, len(valued) - 1)
        if valued[index][1] != valued[index - 1][1]
        and valued[index][1] != valued[index + 1][1]
    ]


def _ocr_failure(done: CompletedProcess[str], lang: str) -> MediaError:
    """Build the `MediaError` for a failed tesseract run, by cause.

    An uninstalled language pack leaves the two measured stderr shapes
    behind; that case reports tesseract's first stderr line plus the
    languages that are installed, so the caller can correct `lang` against
    measured facts. Any other failure carries the exit code and the
    captured stderr verbatim.
    """
    stderr = done.stderr.strip()
    if any(marker in stderr for marker in _MISSING_LANGUAGE_MARKERS):
        first_line = next((line for line in stderr.splitlines() if line.strip()), "")
        try:
            available = ", ".join(ocr_available_langs()) or "(none)"
        except MediaError as listing_error:
            # The listing is a courtesy: it must never replace the failure the
            # caller has to act on, so its own failure is appended as context.
            available = f"(the language listing also failed: {listing_error})"
        return MediaError(
            f"tesseract has no language pack {lang!r}: {first_line}; "
            f"available languages: {available}"
        )
    return MediaError(
        f"tesseract OCR failed (exit code {done.returncode}); "
        f"captured stderr: {stderr or '(none)'}"
    )


def _parse_tesseract_tsv(tsv: str) -> tuple[str, float]:
    """Rebuild the page text and mean word confidence from tesseract's TSV.

    Structure rows (page, block, paragraph, line levels) carry conf -1 and
    empty text and are dropped; the kept word rows are grouped by their
    `(block_num, par_num, line_num)` coordinates in first-appearance order,
    the words of a line join with single spaces, and the lines join with
    newlines — tesseract's own reading order. `confidence` is the mean kept
    word confidence divided by 100, so it lands in 0..1; a page with no
    kept word yields `("", 0.0)`.
    """
    words_by_line: dict[tuple[str, str, str], list[str]] = {}
    confidences: list[float] = []
    for row in tsv.splitlines()[1:]:
        fields = row.split("\t", _TSV_COLUMNS - 1)
        if len(fields) < _TSV_COLUMNS:
            continue
        text = fields[11].strip()
        if not text:
            continue
        try:
            confidence = float(fields[10])
        except ValueError:
            continue
        if confidence < 0.0:
            continue
        words_by_line.setdefault((fields[2], fields[3], fields[4]), []).append(text)
        confidences.append(confidence)
    text = "\n".join(" ".join(words) for words in words_by_line.values())
    mean = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
    return text, mean


def _image_stats(path: Path) -> dict[str, object]:
    """Compute quality metrics for one image file.

    Uses OpenCV's Laplacian variance as a model-free proxy for focus (blur)
    quality, plain grayscale statistics for brightness and contrast, and the
    BRISQUE no-reference IQA score (lower = better). Returns a plain dict
    consumable by both the server text renderer and unit tests without the MCP
    transport.
    """
    if cv2 is None:
        raise MediaError(
            "media_image_quality needs opencv-python-headless (cv2); install "
            "it into the server virtualenv, e.g. `uv pip install --python "
            "media/.venv/bin/python opencv-python-headless`"
        )
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise MediaError(f"could not decode image {path}; not an image cv2 reads")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    height, width = gray.shape
    channels = img.shape[2] if img.ndim == 3 else 1
    if channels == 3:
        means = [float(img[:, :, c].mean()) for c in range(3)]
    else:
        means = [float(gray.mean())]
    brightness = float(gray.mean())
    contrast = float(gray.std())
    size_bytes = path.stat().st_size if path.is_file() else 0
    fmt = path.suffix.lstrip(".").upper() or "(unknown)"
    brisque_score = _brisque_score(img)
    return {
        "width": int(width),
        "height": int(height),
        "resolution": int(width * height),
        "format": fmt,
        "size_bytes": int(size_bytes),
        "blur_score": round(blur, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "channels": [round(m, 2) for m in means],
        "brisque_score": brisque_score,
    }


def _brisque_score(img) -> float | None:
    """Compute the BRISQUE no-reference IQA score for an image array.

    Returns ``None`` when the `brisque` package is not installed or scoring
    fails, so the caller can degrade gracefully. The BRISQUE model is loaded
    once and reused across calls.
    """
    global _brisque_model
    if _brisque_pkg is None:
        return None
    if _brisque_model is None:
        _brisque_model = _brisque_pkg.BRISQUE()
    score = None
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        score = _brisque_model.score(rgb)
        return float(score)
    except (TypeError, ValueError):
        # `score` may return a numpy array or scalar; convert defensively.
        try:
            import numpy as np

            arr = np.asarray(score)
            return float(arr.item()) if arr.ndim == 0 else None
        except (TypeError, ValueError):
            return None
    except cv2.error:
        return None


def _blur_tier(blur: float) -> str:
    if blur < _BLUR_POOR:
        return "poor"
    if blur >= _BLUR_GOOD:
        return "good"
    return "fair"


def _brightness_tier(brightness: float) -> str:
    if brightness < _BRIGHT_DARK or brightness > _BRIGHT_BRIGHT:
        return "poor"
    if _BRIGHT_GOOD_MIN <= brightness <= _BRIGHT_GOOD_MAX:
        return "good"
    return "fair"


def _contrast_tier(contrast: float) -> str:
    if contrast < _CONTRAST_POOR_LOW or contrast > _CONTRAST_POOR_HIGH:
        return "poor"
    if _CONTRAST_GOOD_MIN <= contrast <= _CONTRAST_GOOD_MAX:
        return "good"
    return "fair"


def _brisque_tier(score: float | None) -> str:
    if score is None:
        return "fair"  # cannot assess, treat as neutral
    if score > _BRISQUE_POOR:
        return "poor"
    if score < _BRISQUE_GOOD:
        return "good"
    return "fair"


_TIER_RANK = {"poor": 0, "fair": 1, "good": 2}


def quality_tier(metrics: dict[str, object]) -> str:
    """Classify one image's quality into ``poor``, ``fair``, or ``good``.

    Each metric (blur, brightness, contrast, BRISQUE) is classified
    independently and the overall tier is the worst of them ("worst wins").
    ``poor`` when any metric is clearly bad; ``good`` when every metric is in
    a usable range; ``fair`` otherwise. A metric that cannot be measured
    (e.g. BRISQUE when the package is missing) is excluded from the worst-wins
    comparison rather than penalizing the image.
    """
    tiers = [
        _blur_tier(float(metrics["blur_score"])),
        _brightness_tier(float(metrics["brightness"])),
        _contrast_tier(float(metrics["contrast"])),
    ]
    if metrics.get("brisque_score") is not None:
        tiers.append(_brisque_tier(metrics["brisque_score"]))
    return min(tiers, key=lambda t: _TIER_RANK[t])
