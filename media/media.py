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
frame budget with a complete reason accounting.

Heavy commands (scene scans, frame extraction, OCR, transcription) run through
`run_child`: the child leads its own process group so a timeout or a cancel
event can take down the whole process tree, and streams are capped so a
runaway command cannot exhaust memory.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, Generic, TypeVar

_PayloadT_co = TypeVar("_PayloadT_co", covariant=True)

# The refusal thresholds of the design document, exposed here as plain module
# constants. Nothing in this module reads the environment; Task 11 wires the
# `MEDIA_MAX_SOURCE_BYTES` and `MEDIA_MAX_SOURCE_SECONDS` variables over these.
MEDIA_MAX_SOURCE_BYTES = 4 * 1024**3
MEDIA_MAX_SOURCE_SECONDS = 14400.0

# The warning threshold: half of each hard cap (2 GiB, 2 hours).
_WARN_FRACTION = 0.5

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

# Candidate reasons in ranking-priority order: the tuple index is the sort
# weight, so `scene` beats `audio` beats `interval` beats `fill`.
_REASON_PRIORITY = {"scene": 0, "audio": 1, "interval": 2, "fill": 3}
_RANKING_REASONS = ("scene", "audio", "interval", "fill")


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
    such.
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
    return diff_hash_of(raster)


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
    binary, naming the binary and the OS error.
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
