"""URL ingestion: `yt-dlp` downloads into the cache entry's `media/` directory.

The design promises `yt-dlp` for YouTube and direct URL sources (line 24), the
downloaded bytes under the cache entry's `media/` directory (line 272), a hard
refusal cap of 4 GiB and 4 hours (lines 442 and 511-513), and yt-dlp's stderr
surfaced verbatim on failure (lines 528 and 583). This module is that producer:
the one place where a URL becomes a local file. It writes only inside the entry
it is handed and never modifies a source.

A completed file already under `<entry>/media/` is reused before any network
access, so a cache hit runs no download at all; it is still measured and probed,
because a lowered cap must not silently accept an over-cap file. A refusal on a
cache hit names the cached file and leaves it in place: the attempt wrote
nothing, and deleting a valid cache entry would only force the same download
again. A download reaches the private-range guard first, then `yt-dlp` with an
argument list; a refusal or failure on the download path deletes what the
attempt wrote, and a run that exits 0 while leaving several files is refused
rather than cached.

Two limits are deliberate and documented rather than enforced. The guard is
pre-flight: it resolves the host once, and `yt-dlp` resolves again and follows
redirects, so a DNS-rebinding answer or a redirect to a private address is not
blocked a second time; `MEDIA_ALLOW_PRIVATE_URLS` is the documented override and
this is a single-user workstation tool. `--max-filesize` is a hint to `yt-dlp`
and does not always abort a direct download early; the measured post-download
check is the enforcement, so up to the cap may land before it is deleted.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from media import media
from media.media import MediaError

# The binary the design names, resolved through PATH so a `PYTHONPATH`-style
# override is an environment concern, not a module constant.
YTDLP = "yt-dlp"

# The entry subdirectory the design reserves for downloaded bytes.
MEDIA_DIR_NAME = "media"

# One output template under the entry's `media/`, with a name the real binary
# resolves for both a YouTube video id and a direct URL's file stem.
_OUTPUT_TEMPLATE = "%(id)s.%(ext)s"

# yt-dlp writes the in-progress download beside the final name; a file with
# this suffix is never a completed download.
_PART_SUFFIX = ".part"

# How many of the child's final stderr lines an error or a success carries.
_TAIL_LINES = 12


def is_url(source: str) -> bool:
    """True only for an `http` or `https` source; false for every other shape.

    Paths, `file://`, and unrecognisable strings are local-path candidates.
    The check is by scheme alone, so a malformed `http` URL still reaches the
    ingest path and is refused there with a message naming what is missing,
    rather than being mistaken for a file name.
    """
    if not isinstance(source, str):
        return False
    return urlparse(source).scheme.lower() in ("http", "https")


@dataclass(frozen=True)
class IngestedSource:
    """One settled URL source: its local file and what ingesting it cost.

    `path` is a file inside `<entry>/media/`, `bytes` its measured size, and
    `from_cache` states whether the download was reused rather than fetched.
    `stderr_tail` carries the child's final stderr lines after a successful
    download (empty on a cache hit), so the tool layer can report a warning
    without a second run.
    """

    path: Path
    bytes: int
    from_cache: bool
    stderr_tail: str


def ingest_url(
    source: str,
    entry: Path,
    *,
    max_bytes: int,
    max_seconds: float,
    allow_private: bool,
    timeout_seconds: int,
) -> IngestedSource:
    """Download `source` into `entry`'s `media/` directory, or reuse it.

    Raises `MediaError` naming the measured value and the cap when the result
    is over either limit, naming the host and `MEDIA_ALLOW_PRIVATE_URLS` when a
    private target is refused, and carrying yt-dlp's exit code and final stderr
    lines verbatim when the child fails. Every refusal and failure on the
    download path deletes what this call produced; a cache hit runs no download,
    and an over-cap cache hit is refused by name and left in place.
    """
    media_dir = Path(entry) / MEDIA_DIR_NAME
    cached = _completed_file(media_dir)
    if cached is not None:
        size = cached.stat().st_size
        if size > max_bytes:
            _refuse_size(source, cached, size, max_bytes, cached=True)
        seconds = _probe_seconds(source, cached, cached=True)
        if seconds > max_seconds:
            _refuse_seconds(source, cached, seconds, max_seconds, cached=True)
        return IngestedSource(cached, size, True, "")

    check_target_allowed(source, allow_private)
    media_dir.mkdir(parents=True, exist_ok=True)
    for stale in media_dir.glob(f"*{_PART_SUFFIX}"):
        stale.unlink(missing_ok=True)

    argv = [
        YTDLP,
        "--no-playlist",
        "--no-progress",
        "--no-cache-dir",
        # A hint to yt-dlp, not a guarantee: it does not always abort a direct
        # download early, so the measured check below is the enforcement.
        "--max-filesize",
        str(int(max_bytes)),
        "--output",
        str(media_dir / _OUTPUT_TEMPLATE),
        source,
    ]
    try:
        done = media.run_child(argv, timeout_seconds=timeout_seconds)
    except BaseException:
        # A timeout or a cancellation is already a MediaError naming the child;
        # the half-written file must still go.
        _delete_downloads(media_dir)
        raise
    tail = _stderr_tail(done.stderr)
    if done.returncode != 0:
        _delete_downloads(media_dir)
        raise MediaError(
            f"yt-dlp could not download {source}: {tail or '(no stderr)'} "
            f"(exit code {done.returncode})"
        )

    files = _completed_files(media_dir)
    if len(files) > 1:
        names = ", ".join(path.name for path in files)
        _delete_downloads(media_dir)
        raise MediaError(
            f"yt-dlp exited 0 for {source} but left {len(files)} files under "
            f"{media_dir} ({names}); refusing to cache an ambiguous download, "
            "which usually means a video/audio pair whose merge needed ffmpeg"
        )
    produced = files[0] if files else None
    if produced is None:
        _delete_downloads(media_dir)
        detail = f"; stderr: {tail}" if tail else ""
        raise MediaError(
            f"yt-dlp exited 0 for {source} but wrote no file under {media_dir}{detail}"
        )

    size = produced.stat().st_size
    if size > max_bytes:
        _refuse_size(source, produced, size, max_bytes)
    seconds = _probe_seconds(source, produced)
    if seconds > max_seconds:
        _refuse_seconds(source, produced, seconds, max_seconds)
    return IngestedSource(produced, size, False, tail)


def _completed_files(media_dir: Path) -> list[Path]:
    """Every completed download in `media_dir`, ordered by name.

    Only non-empty regular files without the yt-dlp `.part` suffix qualify.
    """
    if not media_dir.is_dir():
        return []
    return sorted(
        path
        for path in media_dir.iterdir()
        if path.is_file()
        and not path.name.endswith(_PART_SUFFIX)
        and path.stat().st_size > 0
    )


def _completed_file(media_dir: Path) -> Path | None:
    """The completed download to reuse, or None when there is none.

    A directory is expected to hold one file (one template, no playlists). If an
    older run left several, the largest and then the last name wins, so reuse
    stays deterministic; a *new* download that leaves several files is refused
    by `ingest_url` instead, so only pre-existing state can reach this.
    """
    files = _completed_files(media_dir)
    if not files:
        return None
    return max(files, key=lambda path: (path.stat().st_size, path.name))


def check_target_allowed(source: str, allow_private: bool) -> None:
    """Refuse a URL resolving to loopback, link-local, or RFC 1918 space.

    The host is resolved once with `getaddrinfo`; any private address among the
    answers refuses the call, so a name resolving to both a public and a private
    address cannot be used to reach the private one. `allow_private` is the
    caller's already-resolved `MEDIA_ALLOW_PRIVATE_URLS` switch.

    This guard is public so a caller can run it *before* it creates any state
    for the source (the server calls it before the cache entry directory
    exists). It is pre-flight: `yt-dlp` resolves the host again and follows
    redirects, so it does not pin the address a later connection uses.
    """
    if allow_private:
        return
    host = urlparse(source).hostname
    if not host:
        raise MediaError(
            f"{source} names no host to resolve; pass a complete http or https URL"
        )
    try:
        answers = socket.getaddrinfo(host, None)
    except socket.gaierror as err:
        raise MediaError(
            f"cannot resolve the host {host!r} of {source}: {err}"
        ) from err
    blocked = sorted(
        {str(answer[4][0]) for answer in answers if _is_private(str(answer[4][0]))}
    )
    if blocked:
        raise MediaError(
            f"{source} resolves to {blocked[0]}, a loopback, link-local or private "
            "address; such targets are refused unless MEDIA_ALLOW_PRIVATE_URLS=1 "
            "is set"
        )


def _is_private(address: str) -> bool:
    """Whether a textual IP address is loopback, link-local, or RFC 1918."""
    try:
        parsed = ipaddress.ip_address(address.split("%")[0])
    except ValueError:
        return False
    return parsed.is_loopback or parsed.is_link_local or parsed.is_private


def _probe_seconds(source: str, path: Path, *, cached: bool = False) -> float:
    """The file's probed duration, deleting it when it is unreadable.

    On a cache hit (`cached=True`) the file is left in place: the attempt wrote
    nothing, and an unusable cached file is still the cache's to evict.
    """
    try:
        return media.probe(path).duration
    except MediaError as err:
        what = "cached" if cached else "downloaded"
        if not cached:
            _delete_downloads(path.parent)
            raise MediaError(
                f"{source}: the {what} file {path.name} cannot be probed ({err}); "
                f"the file was deleted from {path.parent}"
            ) from err
        raise MediaError(
            f"{source}: the {what} file {path} cannot be probed ({err}); "
            "the cached file was left in place"
        ) from err


def _refuse_size(
    source: str, path: Path, size: int, cap: int, *, cached: bool = False
) -> None:
    """Refuse an over-size file, naming both numbers and where it came from."""
    what = "cached" if cached else "downloaded"
    tail = (
        "the cached file was left in place"
        if cached
        else f"the file was deleted from {path.parent}"
    )
    if not cached:
        _delete_downloads(path.parent)
    raise MediaError(
        f"{source}: the {what} file {path.name} is {size} bytes, over the {cap} "
        f"byte cap; {tail}"
    )


def _refuse_seconds(
    source: str, path: Path, seconds: float, cap: float, *, cached: bool = False
) -> None:
    """Refuse an over-long file, naming both numbers and where it came from."""
    what = "cached" if cached else "downloaded"
    tail = (
        "the cached file was left in place"
        if cached
        else f"the file was deleted from {path.parent}"
    )
    if not cached:
        _delete_downloads(path.parent)
    raise MediaError(
        f"{source}: the {what} file {path.name} runs {seconds:.3f} s, over the "
        f"{cap:g} s duration cap; {tail}"
    )


def _delete_downloads(media_dir: Path) -> None:
    """Remove every file a download attempt wrote under `media_dir`.

    Only regular files are unlinked, so a directory (which this module never
    creates there) is left for its owner; a missing directory is normal.
    """
    if not media_dir.is_dir():
        return
    for path in media_dir.iterdir():
        if path.is_file():
            path.unlink(missing_ok=True)


def _stderr_tail(stderr: str, limit: int = _TAIL_LINES) -> str:
    """The final `limit` lines of a child's stderr, verbatim."""
    text = stderr.rstrip()
    if not text:
        return ""
    return "\n".join(text.splitlines()[-limit:])
