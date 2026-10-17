"""The delegated transcription contract: one stable CLI, parsed honestly.

The server does not implement transcription. It shells out to
`$MEDIA_TRANSCRIBE_BIN` (default `<repo>/transcribe/transcribe.sh`), a wrapper
whose flags, layout, and JSON shape are fixed by the design document. This
module is the seam: `resolve_transcribe_bin` finds the entry point,
`self_check` reports whether it can serve at all, `build_argv` assembles an
argument list the pipeline's own parser accepts, and `run_transcript` runs it
through `run_child` and reads the four artifacts back. A dependency that is
missing or incomplete is reported as a failed `DependencyStatus` or a
`ContractError` naming the variable, the resolved path, and the expected
`<out>/<stem>/` layout; there is no silent fallback and no guessed result.

Queueing and the per-call timeout are the server's concerns, not pipeline
flags: the pipeline has no `--queue` or `--timeout` option and argparse would
exit 2 on one. `build_argv` therefore accepts both values only to document
that they belong to the caller, and `run_transcript` enforces the timeout by
passing it to `run_child`.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from media.media import MediaError, run_child

# The artifact keys the contract promises and the suffix each one carries. A
# success is only a success when every key is present in one directory.
_ARTIFACT_EXTENSIONS = {
    "srt": ".srt",
    "vtt": ".vtt",
    "json": ".json",
    "md": ".md",
}

# The one environment variable this module reads: the entry point override.
_TRANSCRIBE_BIN_VARIABLE = "MEDIA_TRANSCRIBE_BIN"

# The default entry point, relative to the repository root (this file lives
# in `<repo>/media/`, so the root is this module's parent's parent).
_DEFAULT_TRANSCRIBE_BIN = Path("transcribe") / "transcribe.sh"

# `--help` is the startup probe; a wrapper that hangs is as broken as one
# that fails, and the server must still start.
_HELP_TIMEOUT_SECONDS = 30

# How many stderr lines an error message quotes. The tail is where a CLI
# writes its actual reason, after any banner.
_STDERR_TAIL_LINES = 12


class ContractError(MediaError):
    """A transcription dependency or artifact violated the contract.

    Subclasses `MediaError` so the server maps the whole family to one kind
    of actionable response. Messages carry the measured facts: the exit code
    and stderr tail of a failed child, or the directory searched and the
    artifacts actually found.
    """


@dataclass(frozen=True)
class DependencyStatus:
    """What the startup self-check learned about the transcription entry point.

    `ok` is true only when the resolved path exists, is a regular executable
    file, and answers `--help` with exit code 0. `detail` is written for a
    human reading `media_cache(action="stats")`: it names
    `MEDIA_TRANSCRIBE_BIN`, the resolved path, and the layout the variable
    must point at, whether the check passed or failed.
    """

    ok: bool
    path: Path
    detail: str


@dataclass(frozen=True)
class TranscriptResult:
    """One successful transcription: its segments, metadata, and artifacts.

    `segments` are the pipeline's own segment mappings; `artifacts` is keyed
    `srt`, `vtt`, `json`, `md`; `tier` is the model the pipeline confirmed
    (`meta["tier"]`, then `meta["model"]`, then the requested `--model`), or
    the literal `"unknown"` when no source of evidence named one.
    """

    segments: list[dict[str, object]]
    meta: dict[str, object]
    artifacts: dict[str, Path]
    tier: str


def resolve_transcribe_bin() -> Path:
    """Return the entry point the contract will invoke.

    `MEDIA_TRANSCRIBE_BIN` wins whenever it is set, even if the path does not
    exist, because the caller must report the owner's miss rather than
    second-guess it. Otherwise the default is `<repo>/transcribe/transcribe.sh`
    derived from this module's own location. This function only resolves a
    path; it never touches the filesystem.
    """
    override = os.environ.get(_TRANSCRIBE_BIN_VARIABLE)
    if override:
        return Path(override)
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / _DEFAULT_TRANSCRIBE_BIN


def self_check() -> DependencyStatus:
    """Probe the entry point without ever raising, for startup reporting.

    Reports `ok=False` for a missing path, a path that is not a regular file,
    a file without the executable bit, and a `--help` run that exits non-zero
    or cannot start or times out. Every failure is a `DependencyStatus`, not
    an exception, so a broken dependency can never stop the server from
    starting; the detail names `MEDIA_TRANSCRIBE_BIN`, the resolved path, the
    expected `<out>/<stem>/` layout, and what to point the variable at.
    """
    path = resolve_transcribe_bin()
    if not path.exists():
        return _failed(path, "the file does not exist")
    if not path.is_file():
        return _failed(path, "the path is not a regular file")
    if not os.access(path, os.X_OK):
        return _failed(path, "the file is not executable (chmod +x it)")
    try:
        done = run_child([str(path), "--help"], timeout_seconds=_HELP_TIMEOUT_SECONDS)
    except MediaError as err:
        return _failed(path, f"running --help failed: {err}")
    if done.returncode != 0:
        return _failed(
            path,
            f"--help exited {done.returncode} with stderr: {_stderr_tail(done.stderr)}",
        )
    return DependencyStatus(
        ok=True,
        path=path,
        detail=(
            f"MEDIA_TRANSCRIBE_BIN resolved to {path}: exists, is executable, "
            f"and answered --help with exit code 0; {_expected_hint()}"
        ),
    )


def build_argv(
    source: str,
    out_dir: Path,
    model: str,
    num_speakers: int,
    language: str,
    device: str,
    queue_flag: bool,
    timeout_seconds: int,
) -> list[str]:
    """Build the pipeline invocation the contract requires.

    The list is `[<resolved bin>, <source>]`, then only flags the pipeline's
    own parser accepts, then `--out <out_dir>` last. `model` is omitted when
    it is empty or the literal `"auto"`: the server resolves a tier before
    calling, and forwarding `auto` would ask faster-whisper for a model that
    does not exist. A `num_speakers` greater than zero becomes
    `--num-speakers`; zero or less becomes `--no-diarize`, because there is
    no valid speaker count to separate. `language` and `device` are
    forwarded whenever non-empty, since `auto`, `cuda`, and `cpu` are the
    pipeline's own choices.

    `queue_flag` and `timeout_seconds` deliberately emit no flags: the
    pipeline has no such options, and argparse would exit 2 on them. Queue
    policy (`queue="wait"` or `queue="fail"`) belongs to the server's
    single-flight lock, and the timeout is enforced by `run_child` when
    `run_transcript` runs the result.
    """
    argv = [str(resolve_transcribe_bin()), str(source)]
    if isinstance(model, str) and model.strip() and model != "auto":
        argv += ["--model", model]
    if int(num_speakers) > 0:
        argv += ["--num-speakers", str(int(num_speakers))]
    else:
        argv.append("--no-diarize")
    if isinstance(language, str) and language.strip():
        argv += ["--language", language]
    if isinstance(device, str) and device.strip():
        argv += ["--device", device]
    argv += ["--out", str(out_dir)]
    return argv


def run_transcript(
    argv: list[str],
    timeout_seconds: int,
    cancel: threading.Event | None,
) -> TranscriptResult:
    """Run one transcription and return its parsed, artifact-backed result.

    The child runs through `run_child`, so a timeout or cancellation
    propagates as the `MediaError` that runner raises and the whole process
    group is torn down. A non-zero exit becomes a `ContractError` carrying
    the exit code and the stderr tail. On success the four artifacts are
    located inside `<out>/<stem>/` first, then in `<out>` and its immediate
    subdirectories; an incomplete set is a `ContractError` naming the
    expected directory, the files found, and the expected layout, because a
    zero exit with missing files is not a result the server may report.
    """
    if len(argv) < 2:
        raise ContractError(f"transcription argv has no input source: {argv}")
    done = run_child(argv, timeout_seconds=timeout_seconds, cancel=cancel)
    if done.returncode != 0:
        raise ContractError(
            f"transcription failed: {argv[0]} exited {done.returncode}; "
            f"captured stderr tail: {_stderr_tail(done.stderr)}"
        )
    artifacts = _locate_artifacts(argv)
    segments, meta = parse_transcript_json(artifacts["json"])
    return TranscriptResult(
        segments=segments,
        meta=meta,
        artifacts=artifacts,
        tier=_resolve_tier(meta, argv),
    )


def parse_transcript_json(
    path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Read one transcript JSON into `(segments, meta)`.

    The document must be an object with a `segments` list, and each segment
    must be a mapping with `start`, `end`, and `text`. `speaker` is not
    required: it is legitimately absent when diarization is off. `meta` is
    the object's `meta` mapping, or `{}` when the key is absent. An
    unreadable file, invalid JSON, or a wrong shape raises `ContractError`
    naming the path, never a raw `JSONDecodeError` or `AttributeError`.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as err:
        raise ContractError(f"transcript JSON is unreadable at {path}: {err}") from err
    try:
        data = json.loads(raw)
    except ValueError as err:
        raise ContractError(
            f"transcript JSON at {path} is not valid JSON: {err}"
        ) from err
    if not isinstance(data, dict):
        raise ContractError(
            f"transcript JSON at {path} must be an object with a 'segments' "
            f"list, not {type(data).__name__}"
        )
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list):
        raise ContractError(
            f"transcript JSON at {path} has no 'segments' list "
            f"(found {type(raw_segments).__name__})"
        )
    segments: list[dict[str, object]] = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, Mapping):
            raise ContractError(
                f"transcript segment {index} in {path} must be an object, "
                f"not {type(segment).__name__}"
            )
        missing = [key for key in ("start", "end", "text") if key not in segment]
        if missing:
            raise ContractError(
                f"transcript segment {index} in {path} is missing {', '.join(missing)}"
            )
        segments.append(dict(segment))
    raw_meta = data.get("meta")
    if raw_meta is None:
        meta: dict[str, object] = {}
    elif isinstance(raw_meta, Mapping):
        meta = dict(raw_meta)
    else:
        raise ContractError(
            f"transcript JSON at {path} has a 'meta' that is not an object "
            f"but {type(raw_meta).__name__}"
        )
    return segments, meta


def _failed(path: Path, summary: str) -> DependencyStatus:
    """A failed status whose detail carries every fact the owner needs."""
    return DependencyStatus(
        ok=False,
        path=path,
        detail=(
            f"{_TRANSCRIBE_BIN_VARIABLE} resolved to {path}: {summary}; "
            f"{_expected_hint()}"
        ),
    )


def _expected_hint() -> str:
    """The layout and remedy sentence every status detail ends with."""
    return (
        "the contract expects <out>/<stem>/ holding .srt, .vtt, .json and "
        f".md; set {_TRANSCRIBE_BIN_VARIABLE} to the path of an executable "
        "transcribe.sh wrapper that implements it"
    )


def _stderr_tail(stderr: str) -> str:
    """Quote the last stderr lines, where a CLI puts its actual reason."""
    lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
    return " | ".join(lines[-_STDERR_TAIL_LINES:]) or "(none)"


def _locate_artifacts(argv: list[str]) -> dict[str, Path]:
    """Find the four artifacts, preferring the `<out>/<stem>/` layout.

    The primary directory is derived exactly as the pipeline derives it:
    the `--out` value plus the source's stem (URL query and fragment
    stripped). When that directory is incomplete, `<out>` and its immediate
    subdirectories are scanned, because a URL source lands under the
    pipeline's own fixed directory name rather than the URL's file stem.
    Nothing complete anywhere is a `ContractError` naming the expected
    directory, what was found, and what is missing.
    """
    out_value = _flag_value(argv, "--out")
    if out_value is None:
        raise ContractError(f"transcription argv carries no --out directory: {argv}")
    out_dir = Path(out_value).expanduser()
    stem = _source_stem(argv[1])
    expected_dir = out_dir / stem
    found = _artifacts_in(expected_dir, stem)
    if len(found) == len(_ARTIFACT_EXTENSIONS):
        return found

    directories = [out_dir]
    if out_dir.is_dir():
        directories += sorted(path for path in out_dir.iterdir() if path.is_dir())
    best = found
    for directory in directories:
        candidate = _artifacts_in(directory, stem)
        if len(candidate) == len(_ARTIFACT_EXTENSIONS):
            return candidate
        if len(candidate) > len(best):
            best = candidate

    missing = sorted(set(_ARTIFACT_EXTENSIONS) - set(best))
    present = (
        "; ".join(f"{key}={path}" for key, path in sorted(best.items())) or "(none)"
    )
    raise ContractError(
        f"transcription reported success but its artifacts are incomplete: "
        f"expected {expected_dir}/ to hold {stem}"
        f"{{.srt,.vtt,.json,.md}} (the <out>/<stem>/ layout); missing: "
        f"{', '.join(missing)}; found: {present}; scanned: "
        f"{', '.join(str(directory) for directory in directories)}"
    )


def _artifacts_in(directory: Path, stem: str) -> dict[str, Path]:
    """Map present artifact keys to files in one directory, possibly partial."""
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for key, extension in _ARTIFACT_EXTENSIONS.items():
        preferred = directory / f"{stem}{extension}"
        if preferred.is_file():
            found[key] = preferred
            continue
        matches = sorted(
            path for path in directory.glob(f"*{extension}") if path.is_file()
        )
        if matches:
            found[key] = matches[0]
    return found


def _source_stem(source: str) -> str:
    """The source's file stem, with a URL's query and fragment stripped."""
    trimmed = source.split("?", 1)[0].split("#", 1)[0]
    return Path(trimmed).stem


def _flag_value(argv: list[str], flag: str) -> str | None:
    """Return the value after the last occurrence of `flag`, or None."""
    value = None
    for index, token in enumerate(argv[:-1]):
        if token == flag:
            value = argv[index + 1]
    return value


def _resolve_tier(meta: dict[str, object], argv: list[str]) -> str:
    """Name the tier only from evidence, never from an assumption.

    A confirmed `meta["tier"]` wins, then the pipeline's `meta["model"]`, then
    the `--model` the caller requested. With none of those, the answer is
    `"unknown"` — the pipeline never confirmed a tier and the server must not
    imply that it did.
    """
    for key in ("tier", "model"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value
    requested = _flag_value(argv, "--model")
    if requested:
        return requested
    return "unknown"
