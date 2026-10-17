"""Content identity and the on-disk artifact cache.

Every source the server accepts resolves here to an `Identity`: a 16-hex
content key plus the kind string that explains how the key was derived. The
key addresses a cache entry directory under `<cache_root>/entries/<key>`, and
each entry owns a `meta.json` plus staged artifacts (`probe.json`,
`scenes.json`, `transcript.json`, `timeline.md`, `audio_events.json`,
`frames/`, `ocr/`). This module does no network access; URL identities carry
only what the URL itself provides, and hashing reads files in chunks so no
media file is ever loaded whole into memory.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# The identity sample width: head, middle, and tail 1 MiB slices for sampled
# identities, and the read granularity for strict full-file hashing.
CHUNK = 1024 * 1024

KINDS = frozenset({"file-sampled", "file-strict", "youtube", "url"})

_META_NAME = "meta.json"


@dataclass(frozen=True)
class Identity:
    """Content identity of a source.

    `key` is 16 lowercase hex characters, the address of the cache entry.
    `kind` explains how the key was derived; `source` is the original string
    the caller passed; `detail` holds the hashed fields so a later mismatch
    can be explained rather than guessed.
    """

    key: str
    kind: str
    source: str
    detail: dict[str, object]


def resolve_source(source: str, identity: str = "sampled") -> Identity:
    """Resolve a source string to a content `Identity` without network access.

    `identity="sampled"` (default) hashes size, `mtime_ns`, and the SHA-256 of
    the head, middle, and tail 1 MiB of a local file; `identity="strict"`
    hashes the whole file in 1 MiB chunks and ignores `mtime`. HTTP(S) sources
    resolve by URL: YouTube URLs normalize to their video id with query
    parameters dropped, other URLs keep the URL itself.
    """
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return _resolve_url(source, parsed)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"source is neither a URL nor a file: {source}")
    if identity == "strict":
        return _resolve_file_strict(source, path)
    return _resolve_file_sampled(source, path)


def entry_dir(identity: Identity, cache_root: Path) -> Path:
    """Return the cache entry directory for `identity`, creating it if missing."""
    entry = Path(cache_root) / "entries" / identity.key
    entry.mkdir(parents=True, exist_ok=True)
    return entry


def read_meta(entry: Path) -> dict[str, object] | None:
    """Read `meta.json` from an entry, or None when it does not exist."""
    meta_path = entry / _META_NAME
    if not meta_path.is_file():
        return None
    data: Any = json.loads(meta_path.read_text(encoding="utf-8"))
    return data


def write_meta(entry: Path, meta: dict[str, object]) -> None:
    """Write `meta.json` into an entry."""
    entry.mkdir(parents=True, exist_ok=True)
    (entry / _META_NAME).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def touch(entry: Path, now: float | None = None) -> None:
    """Update `last_used` in `meta.json`, creating full meta when absent."""
    stamp = time.time() if now is None else now
    meta = read_meta(entry)
    if meta is None:
        meta = {"created": stamp, "last_used": stamp}
    else:
        meta = dict(meta)
        meta["created"] = meta.get("created", stamp)
        meta["last_used"] = stamp
    write_meta(entry, meta)


def stage_path(entry: Path, name: str) -> Path:
    """Return a staged artifact path inside `entry`, creating parents as needed.

    A `name` ending in `/` denotes a directory (for `frames/` and `ocr/`) and
    is created; every other name is a file whose parent is created.
    """
    entry.mkdir(parents=True, exist_ok=True)
    if name.endswith("/"):
        target = entry / name.rstrip("/")
        target.mkdir(parents=True, exist_ok=True)
        return target
    target = entry / name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _detail_string(detail: dict[str, object]) -> str:
    """Serialize identity detail deterministically for key derivation."""
    return json.dumps(detail, sort_keys=True, separators=(",", ":"))


def _key_for(detail: dict[str, object]) -> str:
    """Derive the 16-hex cache key from the identity detail string."""
    digest = hashlib.sha256(_detail_string(detail).encode("utf-8"))
    return digest.hexdigest()[:16]


def _resolve_file_sampled(source: str, path: Path) -> Identity:
    detail: dict[str, object] = {"kind": "file-sampled"}
    with path.open("rb") as stream:
        size = path.stat().st_size
        detail["size"] = size
        detail["mtime_ns"] = path.stat().st_mtime_ns
        for name in ("head", "middle", "tail"):
            offset = _chunk_offset(name, size)
            detail[name] = _hash_range(stream, offset, CHUNK)
    return Identity(_key_for(detail), "file-sampled", source, detail)


def _resolve_file_strict(source: str, path: Path) -> Identity:
    detail: dict[str, object] = {"kind": "file-strict"}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        size = path.stat().st_size
        detail["size"] = size
        while True:
            chunk = stream.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    detail["sha256"] = digest.hexdigest()
    return Identity(_key_for(detail), "file-strict", source, detail)


def _chunk_offset(name: str, size: int) -> int:
    """Pick the byte offset of the head, middle, or tail 1 MiB sample."""
    if name == "head":
        return 0
    if name == "middle":
        return max(size // 2 - CHUNK // 2, 0)
    return max(size - CHUNK, 0)


def _hash_range(stream: Any, offset: int, length: int) -> str:
    """Hash `length` bytes at `offset` from an open binary stream in chunks."""
    digest = hashlib.sha256()
    stream.seek(offset)
    remaining = length
    while remaining > 0:
        chunk = stream.read(min(CHUNK, remaining))
        if not chunk:
            break
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _resolve_url(source: str, parsed: Any) -> Identity:
    youtube_id = _youtube_video_id(parsed)
    if youtube_id is not None:
        detail: dict[str, object] = {
            "kind": "youtube",
            "video_id": youtube_id,
            "url": _normalized_watch_url(youtube_id),
        }
        return Identity(_key_for(detail), "youtube", source, detail)
    normalized = urlunparse(parsed._replace(fragment=""))
    return Identity(
        _key_for({"kind": "url", "url": normalized}),
        "url",
        source,
        {"kind": "url", "url": normalized},
    )


def _youtube_hosts() -> tuple[str, ...]:
    return ("youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be")


def _youtube_video_id(parsed: Any) -> str | None:
    """Extract a YouTube video id from a parsed URL, or None.

    Accepts `youtu.be/<id>`, `youtube.com/watch?v=<id>`, and `/shorts/<id>`,
    `/embed/<id>`, and `/live/<id>` path forms. Query parameters other than
    the video id are ignored.
    """
    host = parsed.netloc.lower()
    host = host.removeprefix("www.")
    if host not in _youtube_hosts():
        return None
    segments = [segment for segment in parsed.path.split("/") if segment]
    if host == "youtu.be":
        return segments[0] if segments else None
    if parsed.path == "/watch":
        query = parse_qs(parsed.query)
        ids = query.get("v", [])
        return ids[0] if ids else None
    if len(segments) >= 2 and segments[0] in ("shorts", "embed", "live"):
        return segments[1]
    return None


def _normalized_watch_url(video_id: str) -> str:
    params = urlencode({"v": video_id})
    return f"https://www.youtube.com/watch?{params}"
