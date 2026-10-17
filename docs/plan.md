# media-mcp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable local MCP server that turns a video or audio source
into model-visible text and image artifacts, and promote the existing
transcription pipeline so both it and two MCP clients share one copy.

**Architecture:** Three Python layers in one repo. `artifacts.py` owns content
identity, the cache store, LRU eviction, and pure search. `media.py` owns every
ffmpeg, ffprobe, and tesseract call plus frame selection and tiling, and
provides the bounded subprocess runner. `contracts.py` owns the delegated
transcription CLI contract. `server.py` is a thin MCP tool layer that validates
parameters, maps `profile` to work, and reports state honestly. The
transcription pipeline is moved from the `transcribe-diarize` skill into
`transcribe/`, and the skill keeps its `SKILL.md` while its wrapper delegates.

**Tech Stack:** Python 3.14, `mcp` Python SDK (>= 1.29) over stdio, pytest,
ruff. External programs: `ffmpeg`, `ffprobe`, `tesseract`, `yt-dlp` from
`/usr/bin`. Heavy inference (`faster-whisper`, `pyannote`) stays in the
existing transcription virtualenv and is never imported by the server.

**Spec:** `docs/design.md`, the approved design document in this repository.
Sections referenced below as "spec section N".

## Global Constraints

- Repository root: `~/.clone/Github/media-mcp`. Working directory for every
  task is that root unless a step says otherwise.
- Design and plan live in the repository: `docs/design.md` is the approved
  design, `docs/plan.md` is this plan. Both are already present before Task 1
  and are committed with it.
- Local inference only. No cloud transcription, vision, or embedding calls,
  and no new network dependency other than `yt-dlp` for URL sources.
- All server configuration is read from `MEDIA_*` environment variables. The
  server never reads an opencode or Antigravity config file.
- The server writes only inside the cache entry directory. Source files are
  never modified, moved, or deleted.
- Cache root default `~/.cache/media-mcp`, cap default `20 GiB`.
- Frames: `max_edge` default `1600` px, PNG, JPEG q85 fallback only above
  `2 MiB`.
- Frame sampling constants, internal and not tool parameters: interval `3` s,
  `dedup_distance` `6`, scene threshold default `10.0` in `scdet` score units
  (`5.0` for screencast-like content).
- Per-call ceilings: `count` 60, `fps` 30, `k` 20, `timeout_seconds` 900
  default and 21600 ceiling.
- Every tool response ends with a status block naming what was computed, what
  came from cache, what was capped, and what was skipped and why.
- All repository text is English. No emoji anywhere in code, docs, or commits.
- `ruff check .` and `ruff format --check .` clean, `python -m py_compile` clean
  on every module, before a task is called done.
- Commit steps below are part of each task, but do not commit until the user
  has instructed execution of this plan.

## Review Focus

These are the input classes the spec implies but whose tests are easy to omit.
Each one is pinned by a test in the named task.

1. A `source` that is a directory, a device file, or missing entirely. A user
   would expect a clear refusal naming the path, not a traceback or an empty
   result. Owned by Task 5.
2. A zero-byte or truncated media file. A user would expect ffprobe's message
   surfaced verbatim with the exit code. Owned by Task 5.
3. A source with no video stream (audio-only file). A user would expect frame
   and OCR tools to say "no video stream" rather than return an empty success.
   Owned by Task 8.
4. Paths and titles containing spaces and Indonesian characters. A user would
   expect them to work unchanged through every subprocess call. Owned by
   Task 5 and Task 8.
5. A source longer than the 2 hour warning threshold. A user would expect a
   warning and continued processing, and a refusal only past the hard cap.
   Owned by Task 5.

---

### Task 1: Repository scaffolding and the synthetic fixture generator

**Files:**
- Present before this task: `docs/design.md`, `docs/plan.md`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `media/__init__.py`
- Create: `tests/fixtures.py`
- Create: `tests/test_fixtures.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fixtures.make_sample(path: Path, duration: int = 60,
  text: str = "LANJUT 42", spike_at: int = 30, counter_jump_at: int = 20,
  audio_offset: float = 0.0) -> Path`, which writes an mp4 with a test
  pattern, burned-in `text`, a burned-in counter that is one value before
  `counter_jump_at` and another after, a sine tone with a loud spike at
  `spike_at`, and an optional audio start offset. Also
  `fixtures.make_silent_video(path) -> Path` and
  `fixtures.make_audio_only(path) -> Path`.

- [ ] **Step 1: Create the repository and the venv**

Run:
```bash
cd ~/.clone/Github
mkdir -p media-mcp/tests media-mcp/media
cd media-mcp && git init
uv venv media/.venv --python 3.14
uv pip install --python media/.venv/bin/python "mcp>=1.29" pytest ruff
```
Expected: `media/.venv/bin/python -c "import mcp; print(mcp.__version__)"`
prints a version `>= 1.29`.

- [ ] **Step 2: Write `.gitignore`, `pyproject.toml`, `README.md`**

`.gitignore` ignores `.venv/`, `__pycache__/`, `.ruff_cache/`, `*.pyc`, and
`.env` (the delegated pipeline keeps its token file outside the repository, and
the ignore line makes a stray copy harmless).
`pyproject.toml` sets `[tool.ruff] line-length = 88` and
`[tool.pytest.ini_options] testpaths = ["tests"]`. `README.md` states the
one-line purpose and nothing else yet; Task 15 writes `docs/client-config.md`
and turns the README into a quickstart that links to it.

- [ ] **Step 3: Write the failing test**

```python
# tests/test_fixtures.py
from pathlib import Path
from tests import fixtures


def test_make_sample_creates_playable_video(tmp_path: Path) -> None:
    out = fixtures.make_sample(tmp_path / "sample.mp4", duration=6)
    assert out.exists() and out.stat().st_size > 0
    probe = __import__("subprocess").run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True,
    )
    assert probe.returncode == 0
    assert 5.0 <= float(probe.stdout.strip()) <= 7.0
```

- [ ] **Step 4: Run it and watch it fail**

Run: `media/.venv/bin/python -m pytest tests/test_fixtures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.fixtures'`.

- [ ] **Step 5: Implement `fixtures.py`**

Use `ffmpeg` with `-f lavfi -i testsrc` for the video pattern and
`-f lavfi -i sine` for audio. Burn text with the `ass` filter and a generated
ASS file: `drawtext` is absent from the ffmpeg build on this machine while
`ass`/`subtitles` (libass) are present, so one mechanism covers the fixture
text, the counter, and the tile labels of Task 8. Overlay the counter as one ASS
event per second, one value before `counter_jump_at` and another after, and put
the hard cut at `counter_jump_at` by concatenating two segments of different
backgrounds so `scdet` has a real cut to find. For `text`, prefer a light solid
background segment: OCR misread the burned-in text when it sat on `testsrc`
("LANJUT 80" came back as "IFANYUT 80"), so the fixture that Task 9 OCRs must
not be the busy pattern. Use `volume` automation or a second short audio input
mixed in with `adelay` for the spike, and `-itsoffset` on the audio input for
`audio_offset`. Use argument lists, never a shell string. Also provide
`png_size(png_bytes: bytes) -> tuple[int, int]`, which the Task 8 tests call.
Add `tests/__init__.py` so `from tests import fixtures` resolves.

- [ ] **Step 6: Run the test again**

Run: `media/.venv/bin/python -m pytest tests/test_fixtures.py -v`
Expected: PASS.

- [ ] **Step 7: Lint and commit**

```bash
media/.venv/bin/python -m ruff check . && media/.venv/bin/python -m ruff format --check .
git add -A && git commit -m "chore: scaffold repo and synthetic fixture generator"
```

---

### Task 2: Content identity and the cache store

**Files:**
- Create: `media/artifacts.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass(frozen=True) class Identity` with `key: str` (16 hex),
    `kind: str` in `{"file-sampled", "file-strict", "youtube", "url"}`,
    `source: str`, `detail: dict[str, object]`.
  - `resolve_source(source: str, identity: str = "sampled") -> Identity`
  - `entry_dir(identity: Identity, cache_root: Path) -> Path` returning
    `<cache_root>/entries/<key>`, created if missing.
  - `read_meta(entry: Path) -> dict[str, object] | None`
  - `write_meta(entry: Path, meta: dict[str, object]) -> None`
  - `touch(entry: Path, now: float | None = None) -> None` updating
    `last_used` in `meta.json`.
  - `stage_path(entry: Path, name: str) -> Path` for
    `probe.json`, `scenes.json`, `transcript.json`, `timeline.md`,
    `audio_events.json`, and a free subdirectory for `frames/` and `ocr/`.

- [ ] **Step 1: Write the failing tests**

```python
def test_sampled_identity_is_stable_and_changes_with_mtime(tmp_path):
    f = make_sample(tmp_path / "a.mp4", duration=4)
    first = artifacts.resolve_source(str(f))
    assert first.key == artifacts.resolve_source(str(f)).key
    os.utime(f, (1.0, 1.0))
    assert artifacts.resolve_source(str(f)).key != first.key


def test_strict_identity_ignores_mtime(tmp_path):
    f = make_sample(tmp_path / "b.mp4", duration=4)
    strict = artifacts.resolve_source(str(f), identity="strict")
    os.utime(f, (1.0, 1.0))
    assert artifacts.resolve_source(str(f), identity="strict").key == strict.key
    assert artifacts.resolve_source(str(f)).key != strict.key


def test_url_identity_is_normalized_without_network():
    a = artifacts.resolve_source("https://youtu.be/AbC?t=5")
    b = artifacts.resolve_source("https://youtu.be/AbC?t=90")
    assert a.kind == "youtube" and a.key == b.key
    assert a.detail["video_id"] == "AbC"
    assert artifacts.resolve_source("http://example.com/x.mp4").kind == "url"


def test_meta_round_trip_and_touch(tmp_path):
    entry = artifacts.entry_dir(artifacts.resolve_source(str(sample)), tmp_path)
    artifacts.write_meta(entry, {"created": 1.0, "last_used": 1.0})
    assert artifacts.read_meta(entry)["created"] == 1.0
    artifacts.touch(entry, now=99.0)
    assert artifacts.read_meta(entry)["last_used"] == 99.0
```

- [ ] **Step 2: Run them and watch them fail**

Run: `media/.venv/bin/python -m pytest tests/test_artifacts.py -v`
Expected: FAIL with `AttributeError: module 'media.artifacts' has no attribute 'resolve_source'`.

- [ ] **Step 3: Implement `artifacts.py`**

Sampled identity hashes `size`, `mtime_ns`, and SHA-256 of the first, middle,
and last 1 MiB. Strict identity hashes the whole file in 1 MiB chunks. YouTube
identity normalizes to the video id and ignores query parameters other than
none. The cache key is `sha256(detail_string).hexdigest()[:16]`.

- [ ] **Step 4: Run the tests again**

Run: `media/.venv/bin/python -m pytest tests/test_artifacts.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
media/.venv/bin/python -m ruff check . && media/.venv/bin/python -m ruff format --check .
git add -A && git commit -m "feat(artifacts): content identity and cache store"
```

---

### Task 3: LRU eviction and size accounting

**Files:**
- Modify: `media/artifacts.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: `entry_dir`, `read_meta`, `touch` from Task 2.
- Produces:
  - `total_bytes(cache_root: Path) -> int`
  - `evict(cache_root: Path, cap_bytes: int, protect: Path | None = None) -> list[str]`
    returning the evicted keys, oldest `last_used` first, never touching
    `protect`.

- [ ] **Step 1: Write the failing tests**

```python
def test_eviction_drops_oldest_last_used_first(tmp_path):
    keys = {}
    for name, used in (("old", 1.0), ("mid", 2.0), ("new", 3.0)):
        entry = artifacts.entry_dir(artifacts.resolve_source(str(entry_src)), tmp_path)
        (entry / "blob.bin").write_bytes(b"x" * 1024)
        artifacts.write_meta(entry, {"created": used, "last_used": used})
        keys[name] = entry
    evicted = artifacts.evict(tmp_path, cap_bytes=2500)
    assert keys["old"].name in evicted or not keys["old"].exists()
    assert keys["new"].exists()


def test_eviction_protects_the_current_entry(tmp_path):
    ...
    evicted = artifacts.evict(tmp_path, cap_bytes=0, protect=keys["new"])
    assert keys["new"].exists() and keys["new"].name not in evicted


def test_eviction_stops_when_under_cap(tmp_path):
    ...
    assert artifacts.evict(tmp_path, cap_bytes=10_000) == []
```

- [ ] **Step 2: Run them and watch them fail**

Run: `media/.venv/bin/python -m pytest tests/test_artifacts.py -v -k eviction`
Expected: FAIL, `AttributeError: ... has no attribute 'evict'`.

- [ ] **Step 3: Implement `total_bytes` and `evict`**

Sum file sizes by walking `entries/*`. Delete whole entry directories.
Return the evicted keys so the caller can report them.

- [ ] **Step 4: Run the tests again**

Run: `media/.venv/bin/python -m pytest tests/test_artifacts.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
media/.venv/bin/python -m ruff check . && media/.venv/bin/python -m ruff format --check .
git add -A && git commit -m "feat(artifacts): LRU eviction and size accounting"
```

---

### Task 4: Timeline projection and pure search

**Files:**
- Modify: `media/artifacts.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `stage_path` from Task 2.
- Produces:
  - `@dataclass(frozen=True) class TimelineEvent` with `timestamp: float`,
    `layer: str` in `{"scene", "audio", "speech", "ocr", "ocr-change"}`,
    `payload: str`.
  - `write_timeline(entry: Path, events: list[TimelineEvent]) -> Path`
    writing `timeline.md`, one line per event, format
    `[HH:MM:SS.mmm] layer | payload`, sorted by timestamp.
  - `@dataclass(frozen=True) class Hit` with `timestamp: float`, `layer: str`,
    `text: str`, `score: float`.
  - `search(entry: Path, query: str, k: int = 8, scope: str = "all") -> list[Hit]`
    over `transcript.json` segments and `ocr/*.json`, case-insensitive token
    overlap, `k` capped at 20, `scope` in `{"all", "transcript", "ocr"}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_timeline_is_sorted_and_greppable(tmp_path):
    entry = tmp_path / "e"; entry.mkdir()
    artifacts.write_timeline(entry, [
        artifacts.TimelineEvent(12.5, "speech", "halo semua"),
        artifacts.TimelineEvent(0.5, "scene", "cut"),
    ])
    lines = (entry / "timeline.md").read_text().splitlines()
    assert lines[0] == "[00:00:00.500] scene | cut"
    assert lines[1] == "[00:00:12.500] speech | halo semua"


def test_search_scores_and_caps_k(tmp_path):
    ...
    hits = artifacts.search(entry, "slider", k=99)
    assert len(hits) <= 20
    assert hits[0].score >= hits[-1].score
    assert artifacts.search(entry, "slider", scope="ocr")[0].layer == "ocr"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `media/.venv/bin/python -m pytest tests/test_search.py -v`
Expected: FAIL, missing attributes.

- [ ] **Step 3: Implement `write_timeline` and `search`**

Timestamps format through one shared `format_timestamp(seconds) -> str`
helper in `artifacts.py`; every module uses it so timestamps are uniform
everywhere they appear.

- [ ] **Step 4: Run the tests again**

Run: `media/.venv/bin/python -m pytest tests/test_search.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
media/.venv/bin/python -m ruff check . && media/.venv/bin/python -m ruff format --check .
git add -A && git commit -m "feat(artifacts): timeline projection and pure search"
```

---

### Task 5: Probe, source validation, and audio-video offset

**Files:**
- Create: `media/media.py`
- Test: `tests/test_probe.py`

**Interfaces:**
- Consumes: `format_timestamp` and `Identity` from Tasks 2 and 4.
- Produces:
  - `@dataclass(frozen=True) class ProbeResult` with `duration: float`,
    `width: int`, `height: int`, `fps: float`, `video_codec: str`,
    `audio_codec: str`, `has_video: bool`, `has_audio: bool`,
    `audio_offset: float`, `size_bytes: int`.
  - `validate_source(source: str, allowed_roots: list[Path] | None) -> Path`
    raising `MediaError` with an actionable message for a missing path, a
    directory, a symlink escaping the allowlist, or an unreadable file.
  - `probe(path: Path) -> ProbeResult`
  - `av_offset_seconds(result: ProbeResult) -> float`, non-zero when audio and
    video stream `start_time` differ.
  - `class MediaError(RuntimeError)` whose message always ends with the
    subprocess exit code and its stderr when a subprocess produced them.

- [ ] **Step 1: Write the failing tests**

```python
def test_probe_reports_duration_and_streams(tmp_path):
    result = media.probe(make_sample(tmp_path / "s.mp4", duration=6))
    assert result.has_video and result.has_audio
    assert 5.0 <= result.duration <= 7.0


def test_offset_is_detected(tmp_path):
    result = media.probe(make_sample(tmp_path / "off.mp4", duration=6, audio_offset=1.0))
    assert abs(media.av_offset_seconds(result)) > 0.5


def test_directory_source_is_refused_with_the_path_named(tmp_path):
    with pytest.raises(media.MediaError) as err:
        media.validate_source(str(tmp_path), None)
    assert str(tmp_path) in str(err.value)


def test_zero_byte_file_surfaces_ffprobe_stderr(tmp_path):
    bad = tmp_path / "empty.mp4"; bad.write_bytes(b"")
    with pytest.raises(media.MediaError) as err:
        media.probe(bad)
    assert "ffprobe" in str(err.value).lower()


def test_unicode_and_space_paths_work(tmp_path):
    f = make_sample(tmp_path / "Rekaman Uji Coba 1.mp4", duration=4)
    assert media.probe(f).has_video


def test_duration_beyond_warning_threshold_is_reported_not_refused(tmp_path):
    result = media.probe(make_sample(tmp_path / "long.mp4", duration=4))
    long = media.with_duration(result, 7300.0)   # pure helper, no 2h fixture
    assert media.source_limits(long)["warn"] is True
    assert media.source_limits(long)["refuse"] is False
    assert media.source_limits(media.with_duration(result, 20000.0))["refuse"] is True
```

- [ ] **Step 2: Run them and watch them fail**

Run: `media/.venv/bin/python -m pytest tests/test_probe.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'media.media'`.

- [ ] **Step 3: Implement `media.py` probe half**

`probe` runs `ffprobe -v error -print_format json -show_format -show_streams`.
Add `with_duration(result, seconds) -> ProbeResult` as a pure copy helper and
`source_limits(result) -> dict[str, bool]` reading
`MEDIA_MAX_SOURCE_SECONDS` (default 14400) and warning at half of it.

- [ ] **Step 4: Run the tests again**

Run: `media/.venv/bin/python -m pytest tests/test_probe.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
media/.venv/bin/python -m ruff check . && media/.venv/bin/python -m ruff format --check .
git add -A && git commit -m "feat(media): probe, source validation, av offset"
```

---

### Task 6: Bounded, cancellable subprocess runner

**Files:**
- Modify: `media/media.py`
- Test: `tests/test_exec.py`

**Interfaces:**
- Consumes: nothing from earlier tasks beyond `MediaError`.
- Produces:
  - `run_child(argv: list[str], timeout_seconds: int = 900,
    cancel: threading.Event | None = None) -> CompletedProcess[str]`
    starting the child in its own process group, capturing stdout and stderr
    as text, killing the whole group on timeout or when `cancel` is set, and
    raising `MediaError` on timeout or cancellation with the argv and elapsed
    time in the message.

- [ ] **Step 1: Write the failing tests**

```python
def test_timeout_kills_the_process_group():
    with pytest.raises(media.MediaError) as err:
        media.run_child(["sleep", "30"], timeout_seconds=1)
    assert "timeout" in str(err.value).lower()
    assert subprocess.run(["pgrep", "-f", "sleep 30"], capture_output=True).returncode != 0


def test_cancel_event_kills_the_child():
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    with pytest.raises(media.MediaError):
        media.run_child(["sleep", "30"], timeout_seconds=60, cancel=cancel)


def test_stderr_is_captured_and_returned():
    done = media.run_child(["ffprobe", "-v", "error", "/nonexistent.mp4"])
    assert done.returncode != 0 and done.stderr.strip()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `media/.venv/bin/python -m pytest tests/test_exec.py -v`
Expected: FAIL, missing `run_child`.

- [ ] **Step 3: Implement `run_child`**

Use `subprocess.Popen` with `start_new_session=True` so the child leads its own
process group, a poll loop honouring both the deadline and `cancel`, and
`os.killpg` with `SIGTERM` then `SIGKILL` after a short grace period.

- [ ] **Step 4: Run the tests again**

Run: `media/.venv/bin/python -m pytest tests/test_exec.py -v`
Expected: PASS, and no `sleep` process remains.

- [ ] **Step 5: Lint and commit**

```bash
media/.venv/bin/python -m ruff check . && media/.venv/bin/python -m ruff format --check .
git add -A && git commit -m "feat(media): bounded cancellable subprocess runner"
```

---

### Task 7: Scene detection, candidate sampling, and ranking

**Files:**
- Modify: `media/media.py`
- Test: `tests/test_scenes.py`

**Interfaces:**
- Consumes: `run_child`, `probe` from Tasks 5 and 6.
- Produces:
  - `@dataclass(frozen=True) class SceneScore` with `timestamp: float`,
    `score: float`.
  - `@dataclass(frozen=True) class Candidate` with `timestamp: float`,
    `reason: str` in `{"scene", "audio", "interval", "fill"}`, `score: float`.
  - `detect_scenes(path: Path, threshold: float = 10.0) -> list[SceneScore]`
    via the `scdet` filter, parsing its log output. Both numbers in each line
    are in score units, and `scdet` emits at most one line per second of media.
  - `diff_hash_of(raw: bytes) -> int` and
    `hamming(a: int, b: int) -> int`, plus
    `diff_hash(png_bytes: bytes) -> int` which downscales to 32x32 grayscale
    first. All three are pure and testable without a video.
  - `sample_candidates(scenes, duration, interval=3, dedup_distance=6,
    extract: Callable[[float], int] = lambda t: 0) -> list[Candidate]`, where
    `extract` returns the perceptual hash of the frame at a timestamp.
  - `rank_candidates(candidates, audio_events, budget=120) -> tuple[list[Candidate], dict[str, int]]`
    returning the kept candidates and accounting
    `{"scene": n, "audio": n, "interval": n, "fill": n, "dropped": n}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_scene_scores_are_parsed_from_a_real_clip(tmp_path):
    sample = make_sample(tmp_path / "cuts.mp4", duration=8, counter_jump_at=4)
    scores = media.detect_scenes(sample)
    assert scores, "a clip with a hard cut must yield at least one scored second"
    assert max(s.score for s in scores) >= 10.0, "the hard cut must clear the default threshold"
    assert all(s.timestamp >= 0.0 for s in scores)


def test_dedup_drops_near_identical_candidates():
    same = media.diff_hash_of(b"identical")
    kept = media.sample_candidates(
        [], duration=10, interval=1, dedup_distance=6, extract=lambda t: same)
    assert len(kept) == 1


def test_ranking_puts_scene_boundaries_first(tmp_path):
    cands = [media.Candidate(1.0, "fill", 0.0), media.Candidate(2.0, "scene", 0.9)]
    ranked, accounting = media.rank_candidates(cands, [], budget=1)
    assert ranked[0].reason == "scene"
    assert accounting["dropped"] == 1


def test_budget_accounting_is_complete(tmp_path):
    cands = [media.Candidate(float(i), "interval", 0.0) for i in range(10)]
    _, accounting = media.rank_candidates(cands, [], budget=4)
    assert sum(accounting[k] for k in ("scene", "audio", "interval", "fill")) == 4
    assert accounting["dropped"] == 6
```

- [ ] **Step 2: Run them and watch them fail**

Run: `media/.venv/bin/python -m pytest tests/test_scenes.py -v`
Expected: FAIL, missing `detect_scenes`.

- [ ] **Step 3: Implement the scene and ranking half of `media.py`**

`detect_scenes` runs
`ffmpeg -i <path> -vf scdet=threshold=<t> -f null -` and parses
`lavfi.scd.score: <score>, lavfi.scd.time: <second>` from stderr, treating the
score as opaque and the time as the candidate second. `diff_hash` downscales to
32x32 grayscale by shelling to ffmpeg and hashes the raw bytes it returns, so no
image library is added. The constants `interval = 3` and `dedup_distance = 6`
live as module-level names so the spec's "one module" rule holds.

- [ ] **Step 4: Run the tests again**

Run: `media/.venv/bin/python -m pytest tests/test_scenes.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
media/.venv/bin/python -m ruff check . && media/.venv/bin/python -m ruff format --check .
git add -A && git commit -m "feat(media): scene detection, candidate sampling, ranking"
```

---

### Task 8: Frame extraction, crop, tiling, and labels

**Files:**
- Modify: `media/media.py`
- Test: `tests/test_frames.py`

**Interfaces:**
- Consumes: `run_child` from Task 6, `ProbeResult` from Task 5.
- Produces:
  - `@dataclass(frozen=True) class Box` with `x: int, y: int, width: int, height: int`.
  - `@dataclass(frozen=True) class FrameRef` with `timestamp: float`, `path: Path`, `width: int`, `height: int`.
  - `resolve_crop(spec: str | dict | None, probe: ProbeResult, path: Path,
    start: float, end: float, fps: float, pad: int = 8) -> Box | None`
    handling an explicit box, `"auto"`, and `None`.
  - `extract_frame(path: Path, timestamp: float, crop: Box | None,
    max_edge: int) -> bytes` returning PNG bytes.
  - `render_sheet(pngs: list[tuple[float, bytes]], labels: bool,
    max_edge: int) -> bytes`
  - `render_filmstrip(pngs: list[tuple[float, bytes]], labels: bool,
    max_edge: int) -> bytes`
  - `resolve_font() -> Path | None` honouring `MEDIA_FONT_FILE` then
    `fc-match monospace`.

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_frame_respects_max_edge(tmp_path):
    png = media.extract_frame(make_sample(tmp_path / "s.mp4", duration=4), 1.0, None, 320)
    width, height = png_size(png)
    assert max(width, height) <= 320


def test_sheet_is_a_single_image_under_max_edge(tmp_path):
    frames = [(float(i), media.extract_frame(src, float(i), None, 320)) for i in range(6)]
    sheet = media.render_sheet(frames, labels=True, max_edge=800)
    assert png_size(sheet)[0] <= 800


def test_filmstrip_of_a_window_is_one_image(tmp_path):
    frames = [(i / 10, media.extract_frame(src, i / 10, None, 240)) for i in range(8)]
    strip = media.render_filmstrip(frames, labels=True, max_edge=600)
    assert png_size(strip)[1] > png_size(strip)[0]


def test_auto_crop_is_tighter_than_the_full_frame(tmp_path):
    box = media.resolve_crop("auto", probe_of(src), src, 4.0, 5.0, fps=10)
    assert box is not None and box.width < probe_of(src).width


def test_auto_crop_on_a_static_window_returns_none(tmp_path):
    assert media.resolve_crop("auto", probe_of(silent), silent, 0.0, 1.0, fps=10) is None


def test_out_of_bounds_crop_is_rejected_with_measured_size(tmp_path):
    with pytest.raises(media.MediaError) as err:
        media.resolve_crop({"x": 0, "y": 0, "width": 99999, "height": 10}, probe_of(src), src, 0, 1, 1)
    assert str(probe_of(src).width) in str(err.value)


def test_missing_font_yields_unlabelled_tiles(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_FONT_FILE", "/nonexistent/font.ttf")
    assert media.resolve_font() is None
    assert media.render_sheet(frames, labels=True, max_edge=400) is not None


def test_audio_only_source_has_no_video_stream(tmp_path):
    with pytest.raises(media.MediaError) as err:
        media.extract_frame(make_audio_only(tmp_path / "a.m4a"), 1.0, None, 320)
    assert "no video stream" in str(err.value).lower()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `media/.venv/bin/python -m pytest tests/test_frames.py -v`
Expected: FAIL, missing `extract_frame`.

- [ ] **Step 3: Implement the frame half of `media.py`**

Extraction is one `ffmpeg` call per requested timestamp with
`-vf scale='if(gt(iw,ih),min(iw,MAX),-2)':'if(gt(iw,ih),-2,min(ih,MAX))'`
and, when a box is given, a `crop` filter before the scale. Auto crop runs one
ffmpeg pass over the window producing a tiled downscale, then unions the
differing regions over a threshold. Tiling composites PIL-free by generating a
single `ffmpeg` filter graph with `xstack` (sheet) or `vstack` (filmstrip), and
labels are burned per input with the `ass` filter from a generated ASS file,
positioned in the top-left corner of each tile. Encode to JPEG q85 only when the
PNG exceeds `2 MiB`, and report both limits when they trigger.

- [ ] **Step 4: Run the tests again**

Run: `media/.venv/bin/python -m pytest tests/test_frames.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
media/.venv/bin/python -m ruff check . && media/.venv/bin/python -m ruff format --check .
git add -A && git commit -m "feat(media): frame extraction, crop, tiling, labels"
```

---

### Task 9: OCR and value-change tracking

**Files:**
- Modify: `media/media.py`
- Test: `tests/test_ocr.py`

**Interfaces:**
- Consumes: `extract_frame`, `resolve_crop`, `run_child`.
- Produces:
  - `@dataclass(frozen=True) class OcrResult` with `timestamp: float`,
    `text: str`, `confidence: float`.
  - `@dataclass(frozen=True) class ChangeCandidate` with `timestamp: float`,
    `value: str`, `before: str`, `after: str`, `confidence: float`.
  - `ocr_png(png_bytes: bytes, psm: int = 6, lang: str = "eng") -> OcrResult`
    and `ocr_available_langs() -> list[str]` for the missing-pack error.
  - `track_values(results: list[OcrResult]) -> list[ChangeCandidate]`
    returning frames whose value differs from both neighbours.

- [ ] **Step 1: Write the failing tests**

```python
def test_ocr_recovers_burned_in_text(tmp_path):
    png = media.extract_frame(make_sample(tmp_path / "s.mp4", duration=4, text="LANJUT 42"), 2.0, None, 960)
    assert "LANJUT" in media.ocr_png(png).text.upper()


def test_tracking_finds_the_injected_jump():
    seq = [media.OcrResult(i * 0.5, f"Nilai 80", 0.9) for i in range(4)]
    seq.append(media.OcrResult(2.0, "Nilai 48", 0.9))
    seq += [media.OcrResult(2.5 + i * 0.5, "Nilai 80", 0.9) for i in range(3)]
    found = media.track_values(seq)
    assert len(found) == 1
    assert found[0].timestamp == 2.0
    assert found[0].before == "80" and found[0].after == "80"


def test_tracking_on_text_without_numbers_returns_nothing():
    seq = [media.OcrResult(i, "tidak ada angka", 0.8) for i in range(3)]
    assert media.track_values(seq) == []


def test_missing_language_pack_lists_available_languages(tmp_path):
    png = media.extract_frame(src, 1.0, None, 320)
    with pytest.raises(media.MediaError) as err:
        media.ocr_png(png, lang="zzz")
    assert "eng" in str(err.value)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `media/.venv/bin/python -m pytest tests/test_ocr.py -v`
Expected: FAIL, missing `ocr_png`.

- [ ] **Step 3: Implement the OCR half of `media.py`**

Tesseract runs with `-l <lang> --psm <psm> tsv`; the TSV gives per-word
confidence for the confidence band. Number extraction takes the first
integer-or-decimal token per frame, preferring the largest confidence. A
missing language is detected from tesseract's stderr and re-raised through
`MediaError` with `tesseract --list-langs` output appended.

- [ ] **Step 4: Run the tests again**

Run: `media/.venv/bin/python -m pytest tests/test_ocr.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
media/.venv/bin/python -m ruff check . && media/.venv/bin/python -m ruff format --check .
git add -A && git commit -m "feat(media): OCR and value-change tracking"
```

---

### Task 10: Delegated transcription contract

**Files:**
- Create: `media/contracts.py`
- Test: `tests/test_contracts.py`
- Test helper: `tests/fake_transcribe.sh`

**Interfaces:**
- Consumes: `run_child` from Task 6, `format_timestamp` from Task 4.
- Produces:
  - `@dataclass(frozen=True) class DependencyStatus` with `ok: bool`,
    `path: Path`, `detail: str`.
  - `@dataclass(frozen=True) class TranscriptResult` with
    `segments: list[dict[str, object]]`, `meta: dict[str, object]`,
    `artifacts: dict[str, Path]`, `tier: str`.
  - `resolve_transcribe_bin() -> Path` reading `MEDIA_TRANSCRIBE_BIN` with the
    default `<repo>/transcribe/transcribe.sh`.
  - `self_check() -> DependencyStatus` verifying existence, executability, and
    a zero exit for `--help`.
  - `build_argv(source, out_dir: Path, model, num_speakers, language, device,
    queue_flag: bool, timeout_seconds) -> list[str]`
  - `run_transcript(argv, timeout_seconds, cancel) -> TranscriptResult`
  - `parse_transcript_json(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]`

- [ ] **Step 1: Write the failing tests**

```python
def test_missing_binary_is_reported_with_the_variable_and_path(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDIA_TRANSCRIBE_BIN", str(tmp_path / "nope.sh"))
    status = contracts.self_check()
    assert status.ok is False
    assert "MEDIA_TRANSCRIBE_BIN" in status.detail and "nope.sh" in status.detail


def test_argv_uses_an_out_dir_inside_the_cache_entry(tmp_path):
    argv = contracts.build_argv("in.mp4", tmp_path / "entry", "large-v3-turbo", 2,
                                "auto", "auto", False, 900)
    assert argv[-2] == "--out" and str(tmp_path / "entry") == argv[-1]


def test_fake_transcribe_produces_the_four_artifacts(tmp_path, monkeypatch):
    fake = tmp_path / "fake.sh"; shutil.copy("tests/fake_transcribe.sh", fake)
    fake.chmod(0o755)
    monkeypatch.setenv("MEDIA_TRANSCRIBE_BIN", str(fake))
    result = contracts.run_transcript(
        contracts.build_argv("src.mp4", tmp_path, "tiny", 1, "en", "cpu", False, 60), 60, None)
    assert set(result.artifacts) == {"srt", "vtt", "json", "md"}
    assert result.segments[0]["speaker"] == "SPEAKER_00"


def test_parse_rejects_a_json_without_segments(tmp_path):
    bad = tmp_path / "x.json"; bad.write_text('{"meta": {}}')
    with pytest.raises(contracts.ContractError):
        contracts.parse_transcript_json(bad)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `media/.venv/bin/python -m pytest tests/test_contracts.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'media.contracts'`.

- [ ] **Step 3: Implement `contracts.py` and the fake**

`fake_transcribe.sh` writes the four files into `<out>/<stem>/` with one
segment `{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "halo"}`.
`ContractError` is a subclass of `MediaError` so the server maps one family of
errors to actionable text.

- [ ] **Step 4: Run the tests again**

Run: `media/.venv/bin/python -m pytest tests/test_contracts.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
media/.venv/bin/python -m ruff check . && media/.venv/bin/python -m ruff format --check .
git add -A && git commit -m "feat(contracts): delegated transcription contract"
```

---

### Task 11: MCP tool layer, profiles, and state honesty

**Files:**
- Create: `media/server.py`
- Test: `tests/test_stdio_contract.py`
- Test: `tests/test_profiles.py`

**Interfaces:**
- Consumes: every earlier module.
- Produces:
  - Seven tools registered with the spec's exact names and parameters:
    `media_probe`, `media_manifest`, `media_frames`, `media_ocr`,
    `media_transcript`, `media_search`, `media_cache`.
  - `PROFILES: dict[str, dict[str, bool]]` mapping `manifest`, `read`,
    `visual`, `full` to `{"probe": ..., "scenes": ..., "audio": ...,
    "transcript": ..., "ocr": ..., "frames": ...}`.
  - `status_block(computed: list[str], cached: list[str], capped: list[str],
    skipped: dict[str, str]) -> str` appended to every response.
  - `parse_timestamp(value: str | float) -> float` and the shared validators
    rejecting `window` with a list, out-of-range `count`, `fps`, `k`, and
    `crop`, and ambiguous combinations.
  - `main() -> None` running `FastMCP("media").run()` over stdio.

- [ ] **Step 1: Write the failing tests**

```python
def test_every_tool_is_advertised_with_a_schema(server_session):
    names = {t.name for t in server_session.list_tools().tools}
    assert names == {"media_probe", "media_manifest", "media_frames", "media_ocr",
                     "media_transcript", "media_search", "media_cache"}
    assert all(t.inputSchema for t in server_session.list_tools().tools)


def test_malformed_parameter_returns_an_error_not_a_crash(server_session):
    result = server_session.call("media_frames", {"source": "x.mp4", "at": [1, 2], "window": 5})
    assert result.isError and "window" in result.text.lower()


def test_cancelled_request_terminates_cleanly(server_session):
    # start a call against a slow fake transcribe, cancel it, then call a
    # cheap tool and require that it still answers
    ...


def test_manifest_response_ends_with_a_status_block(server_session, sample):
    text = server_session.call("media_manifest", {"source": str(sample)}).text
    assert "computed:" in text and "cache:" in text


def test_profiles_map_to_declared_work():
    assert server.PROFILES["manifest"]["transcript"] is False
    assert server.PROFILES["read"]["transcript"] is True
    assert server.PROFILES["visual"]["ocr"] is True
```

- [ ] **Step 2: Run them and watch them fail**

Run: `media/.venv/bin/python -m pytest tests/test_stdio_contract.py tests/test_profiles.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'media.server'`.

- [ ] **Step 3: Implement `server.py` and the stdio test fixtures**

The contract fixture launches `media/.venv/bin/python media/server.py` as a
subprocess and speaks MCP over stdio with `mcp.client.stdio`, which is what
proves the server works as a client sees it. `media_manifest` assembles its
Markdown digest from cached artifacts, running only the stages the profile
declares. `media_cache` reports entry tables, `total_bytes` against the cap,
dependency status from `contracts.self_check()`, and eviction results.

- [ ] **Step 4: Run the tests again**

Run: `media/.venv/bin/python -m pytest -v`
Expected: the whole suite PASSES.

- [ ] **Step 5: Lint and commit**

```bash
media/.venv/bin/python -m ruff check . && media/.venv/bin/python -m ruff format --check .
git add -A && git commit -m "feat(server): seven-tool MCP surface with profiles and status blocks"
```

---

### Task 12: Promote the transcription pipeline and rewire the skill

**Files:**
- Create: `transcribe/transcribe.py`, `transcribe/transcribe.sh`,
  `transcribe/requirements.txt` (moved from
  `~/.config/opencode/skills/transcribe-diarize/scripts/`)
- Modify: `~/.config/opencode/skills/transcribe-diarize/scripts/transcribe.sh`
  (replaced by a forwarding wrapper)
- Test: `tests/test_regression_transcribe.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Produces the `MEDIA_TRANSCRIBE_BIN`
  target that Task 10 faked and Task 11 called.
- Produces: `transcribe/transcribe.sh` with the unchanged CLI contract from
  spec section 8.

- [ ] **Step 1: Move the pipeline files and the virtualenv**

Run:
```bash
cd ~/.clone/Github/media-mcp
mv ~/.config/opencode/skills/transcribe-diarize/scripts/transcribe.py transcribe/
mv ~/.config/opencode/skills/transcribe-diarize/scripts/transcribe.sh transcribe/
mv ~/.config/opencode/skills/transcribe-diarize/requirements.txt transcribe/
mv ~/.config/opencode/skills/transcribe-diarize/.venv transcribe/
```
Expected: `transcribe/transcribe.sh --help` exits 0 using the skill's `.env`
for `HF_TOKEN`; if the wrapper resolved `.env` relative to its old location,
fix the path resolution in this task and record the change. The secrets stay
where they are: `.env` remains in the skill directory (design section 15 step 4),
the moved wrapper resolves it through `MEDIA_TRANSCRIBE_ENV` with that skill
path as the default, and the repository ignores `.env` so no token can ever be
committed.

- [ ] **Step 2: Write the failing regression test**

```python
def test_moved_pipeline_matches_the_original_segments(tmp_path):
    """Both runs use the same short sample and the same model tier."""
    new = run_pipeline("transcribe/transcribe.sh", SAMPLE, tmp_path / "new")
    old = run_pipeline(OLD_WRAPPER_SNAPSHOT, SAMPLE, tmp_path / "old")
    assert ["%s|%s" % (s["speaker"], s["text"]) for s in new] == \
           ["%s|%s" % (s["speaker"], s["text"]) for s in old]
```

Keep an unmodified copy of the pre-move wrapper at
`tests/snapshots/transcribe-pre-move.sh` before editing anything, so the test
can run both sides.

- [ ] **Step 3: Run it and watch it fail**

Run: `media/.venv/bin/python -m pytest tests/test_regression_transcribe.py -v`
Expected: FAIL, because the old snapshot still points at the old venv path.

- [ ] **Step 4: Rewrite the skill wrapper as a forwarder**

```bash
#!/usr/bin/env bash
set -euo pipefail
BIN="${MEDIA_TRANSCRIBE_BIN:-$HOME/.clone/Github/media-mcp/transcribe/transcribe.sh}"
exec "$BIN" "$@"
```
Update the snapshot side to call the moved venv, then run both.

- [ ] **Step 5: Run the regression test again**

Run: `media/.venv/bin/python -m pytest tests/test_regression_transcribe.py -v`
Expected: PASS with identical segment lists.

- [ ] **Step 6: Verify the skill still works standalone through the wrapper**

Run:
```bash
bash ~/.config/opencode/skills/transcribe-diarize/scripts/transcribe.sh \
  tests/data/short.wav --no-diarize --model tiny --denoise --out /tmp/opencode/skill-check
```
Expected: exit 0, four artifacts under `/tmp/opencode/skill-check/`, which
proves a server-unused flag still works.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "refactor(transcribe): promote pipeline and forward from the skill"
```

---

### Task 13: Wire opencode and run its smoke test

**Files:**
- Modify: `~/.config/opencode/opencode.jsonc`
- Create: `docs/opencode-smoke.md` recording the observed result

**Interfaces:**
- Consumes: the running server from Task 11.
- Produces: a connected `media` server in opencode.

- [ ] **Step 1: Add the server entry**

Insert the `mcp.servers.media` block from spec section 10, preserving the
existing servers and comments. The block uses `{env:HOME}`; if this opencode
version does not substitute it, replace it with the literal home path and
record the substitution failure in `docs/opencode-smoke.md`.

- [ ] **Step 2: Verify the connection**

Run: `opencode mcp list`
Expected: `media` listed as connected, not awaiting authorization.

- [ ] **Step 3: Call a tool from a session**

Run: `opencode run "Call media_probe on <absolute path to the synthetic sample> and report duration and stream count."`
Expected: the response contains a duration matching the fixture and both
streams; paste the observed output into `docs/opencode-smoke.md`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore(clients): wire opencode and record smoke evidence"
```

---

### Task 14: Wire Antigravity and run its smoke test

**Files:**
- Modify: `~/.gemini/antigravity/mcp_config.json`
- Create: `docs/antigravity-smoke.md`

**Interfaces:**
- Consumes: the same server.
- Produces: a connected `media` server in Antigravity, sharing the cache.

- [ ] **Step 1: Add the server entry**

Insert the `mcpServers.media` block from spec section 10 into the existing
`mcpServers` object, leaving `supabase`, `github-mcp-server`, `dart-mcp-server`,
`gopls-mcp-server`, and `alloydb-postgresql` untouched.

- [ ] **Step 2: Confirm the server starts under the Antigravity config shape**

Run:
```bash
MEDIA_CACHE_DIR=~/.cache/media-mcp \
  ~/.clone/Github/media-mcp/media/.venv/bin/python -c "import json,os,sys; b=json.load(open(os.path.expanduser('~/.gemini/antigravity/mcp_config.json')))['mcpServers']['media']; print(json.dumps(b, indent=2)); sys.exit(0 if 'HOME' in b['command'] else 1)"
```
Expected: the block prints with the `${HOME}` marker intact, then a manual
`media_probe` call from Antigravity's agent panel succeeds. The tool call is
the real proof that Antigravity substituted the marker; if it did not, replace
`${HOME}` with the literal home path in that file and record the outcome in
`docs/antigravity-smoke.md`.

- [ ] **Step 3: Record the result**

Antigravity may require manual confirmation in its own UI. If so, perform it,
then write the observed tool call and its output into
`docs/antigravity-smoke.md`. Do not record this as verified without the
output.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore(clients): wire antigravity and record smoke evidence"
```

---

### Task 15: Discovery artifacts for both hosts

**Files:**
- Create: `docs/client-config.md` (authoritative client snippets, spec section 10)
- Modify: `README.md` (quickstart that links to it)
- Create: `~/.config/opencode/skills/media-context/SKILL.md`
- Modify: `~/.config/opencode/AGENTS.md` (one pointer line)
- Modify: `~/.gemini/GEMINI.md` (one appended section)

**Interfaces:**
- Consumes: the tool surface from Task 11 and both smoke results.
- Produces: the three-layer discovery described in spec section 10.

- [ ] **Step 1: Write the would-fail check first**

Run: `grep -c "media_probe" ~/.config/opencode/skills/media-context/SKILL.md`
Expected: FAIL, the file does not exist yet.

- [ ] **Step 2: Write `SKILL.md`**

Short and procedural: when to reach for the server, the four profiles with
their cost, the frames-then-`read` pattern, one worked example per video
category the user selected (meeting, screencast, bug recording, device
experiment), and the rule that caps and skips are reported rather than
assumed.

- [ ] **Step 3: Add the one-line pointer to `AGENTS.md` and the GEMINI.md section**

Preserve every existing line in both files. The GEMINI.md section is short:
what the server does, the default profile, and the frames-then-`read` pattern.

- [ ] **Step 4: Verify all three are reachable**

Run:
```bash
grep -c media_probe ~/.config/opencode/skills/media-context/SKILL.md \
  ~/.config/opencode/AGENTS.md ~/.gemini/GEMINI.md
```
Expected: a non-zero count in each file.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "docs(discovery): media-context skill and host pointers"
```

---

### Task 16: Inline image check and the recorded decision

**Files:**
- Create: `tests/probe_image_inline.py` (throwaway probe, kept for the record)
- Create: `docs/image-inline-finding.md`

**Interfaces:**
- Consumes: `image_inline` handling in `server.py` from Task 11.
- Produces: a recorded decision on whether `image_inline` is enabled.

- [ ] **Step 1: Write the probe**

A minimal stdio server exposing one tool that returns both a text path to a
PNG and an MCP `ImageContent` block for the same PNG, plus a
`media_cache`-like status line. Launch it with opencode once and with
Antigravity once, using the model each client is actually configured with.

- [ ] **Step 2: Record what the model received**

For each host and model, write whether the model described the image content
itself or only the path, and whether `image_inline` should default to on. The
spec's expectation is that the model sees the image when the model reports
media-in-tool-results support, and that the chat UI does not render it, so
the path remains the human-visible route.

- [ ] **Step 3: Apply the decision**

If and only if the check is positive for a host, change that host's
configuration or the tool default, and record the exact evidence in
`docs/image-inline-finding.md`. Otherwise leave `image_inline` off and record
the negative result.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "docs: record inline image finding per client and model"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: section 4 layout to
Tasks 1-11, section 5 tool surface and 5.0-5.2 to Task 11, section 6 artifacts
to Tasks 2-4, section 7 frame selection and 7.1-7.2 to Tasks 7-9, section 8
delegation to Tasks 10 and 12, section 9 resources to Tasks 6 and 11, section
10 client integration and 10.1 environment to Tasks 13-15, section 11 security
to Tasks 5, 6, 8, and 12, section 12 edge cases to the Review Focus tests and
the per-task failure tests, section 13 risks to Tasks 9, 11, and 16, section 14
verification to each task's test steps, section 15 migration to Task 12.

**Step scan.** Every step names one action with a checkable result. No step
carries a function body the signature and tests already determine, except the
auto-crop and tiling algorithms, which are included because the tests alone do
not determine them.

**Type consistency.** `MediaError` is defined in Task 5 and raised by Tasks 6,
8, 9; `ContractError` subclasses it in Task 10. `format_timestamp` is defined
in Task 4 and used by Tasks 4 and 10. `ProbeResult` is defined in Task 5 and
consumed by Task 8. `run_child` is defined in Task 6 and consumed by Tasks 7,
8, 9, and 10. `Identity` is defined in Task 2 and consumed by Tasks 3, 5, and
11.

**Review Focus.** Each of the five lines has its test in the owning task:
directory and missing source in Task 5, zero-byte file in Task 5, audio-only
source in Task 8, unicode and space paths in Tasks 5 and 8, and the duration
threshold in Task 5 through the pure `with_duration` helper so no two-hour
fixture is needed.

**Proportion.** The plan is close to the spec's length; code blocks carry test
assertions and interfaces, not implementations.
