# Design: media-mcp, a Portable Local Media Understanding Server

Date: 2026-10-17
Status: draft for review
Scope: new git repo `~/.clone/Github/media-mcp` (MCP server + promoted
transcription pipeline), thin rework of the existing `transcribe-diarize`
skill wrapper, client wiring for opencode (Antigravity was dropped, section 10)

## 1. Context

OpenCode V2 cannot put video into a model request. Per the V2 Attachments
documentation, the only model-visible inputs are UTF-8 text, directory
listings, and PNG/JPEG/GIF/WebP images; audio, video, PDF, and other binary
prompt attachments are excluded from the request. The built-in `read` tool
accepts the same image formats plus PDF, up to 20 MiB. The documentation
directs the caller to convert an unsupported binary to text or a supported
image first.

This machine already owns every primitive needed for that conversion:

- `ffmpeg` / `ffprobe` at `/usr/bin` for probing, scene detection, frame
  extraction, and audio analysis.
- `tesseract` at `/usr/bin` for on-screen text.
- `yt-dlp` at `/usr/bin` for YouTube and direct URL ingestion.
- The `transcribe-diarize` skill, already installed and verified, with a
  prepared virtualenv holding `faster-whisper`, `pyannote 3.1`, and torch.
  It accepts local files and YouTube URLs and emits SRT, VTT, JSON, and
  Markdown under `./transcripts/<stem>/`.
- A working GPU path: `faster-whisper large-v3-turbo` in int8 on a GTX 1650
  with 4 GB VRAM, with an automatic fallback to `medium` and then CPU.

What is missing is the visual half of video understanding and a stable tool
interface that an agent can call directly. The approved spec
`2026-10-16-media-tools-design.md` covers audio-only transcription as a
skill; it remains valid and is not superseded. This spec adds the visual
capability and reorganizes the transcription pipeline so that a second MCP
client can use it.

The second client was Antigravity, installed at `/opt/Antigravity`, reading MCP
servers from `~/.gemini/antigravity/mcp_config.json` in the classic
`{"mcpServers": {"<name>": {"command", "args", "env"}}}` shape with `${VAR}`
substitution. Task 14 wired that entry and proved the command over stdio, but
the in-UI check failed on 2026-10-21: the client reported its customization
budget exceeded and did not list the `media` server. The user dropped Antigravity
support the same day and the entry was removed from that file on 2026-10-22, so
opencode is the only supported client and its snippet is the only live one.

The portability requirement the second client created still holds, and the
design keeps it: the server depends on no opencode-specific directory or
configuration, which is why the transcription pipeline lives in the repository
rather than inside the skill directory. A different stdio client can be wired
from `docs/client-config.md` at any time; it would share the same server, the
same cache, and the same pipeline.

## 2. Goals and non-goals

Goals:

- Let an agent obtain text and image artifacts that describe a video, using
  a stable tool surface, so that "read this video as context" works.
- Keep the default cost of touching a video small, and let the caller raise
  it deliberately through named profiles.
- Serve any stdio MCP client from one server implementation and one artifact
  cache; opencode is the client in use, as section 10 records.
- Reuse the verified transcription pipeline instead of writing a second
  whisper implementation.
- Never silently degrade: every fallback, cap, and refusal is reported.

Non-goals:

- Music analysis (tempo, key, chord, stem separation).
- Translation of transcripts; transcription output stays in the source
  language.
- Real-name speaker identification; only generic `SPEAKER_XX` labels.
- Cloud transcription, vision, or embedding APIs. All inference is local.
- Two heavy GPU jobs at once.
- Subtitle download as a server feature; that remains the `youtube-subs`
  skill's job. Preferring downloaded YouTube subtitles over local
  transcription is therefore also out of scope for V1, even though for a
  lecture video it would be seconds instead of minutes. It is recorded as a
  deliberate deferral, not an oversight, because it couples two components
  that are otherwise independent.
- Speaker-attributed visual analysis, and any claim about what a person felt
  or intended.

## 3. Decisions and rationale

These were settled during design discussion and are recorded so future work
does not relitigate them silently.

| Decision | Choice | Rationale |
| --- | --- | --- |
| Container | MCP server over stdio, not an opencode plugin | Heavy ffmpeg/whisper work must not run inside the opencode process; a CUDA OOM or child-process crash must not take down the session. MCP keeps the heavy stages out of the agent process, and the local convention is already Python stdio servers. |
| Tool naming | `media_*`, server named `media` | The expensive, valuable core (diarized transcript) is identical for audio and video. Splitting them would duplicate cache, config, process, and naming for one capability. |
| Audio scope | Unified into the same server, no extra tool | Audio adds two event types to the manifest (silence segments, loudness peaks) rather than a new tool surface. These events are directly useful for the physical-experiment video category, for example locating a loud moment. |
| Default context cost | Cheap manifest, raised through a `profile` parameter | The caller decides how many tokens to spend. Default is metadata plus event index. |
| Heavy compute | On demand, then cached | Transcription is the only slow step; keeping it out of the default path preserves responsiveness. |
| Frame delivery to the model | Paths by default, inline behind `image_inline` | The guaranteed path is: the tool writes PNG files and returns paths, and the agent calls `read`, which is documented to pass PNG/JPEG/GIF/WebP to the model with `media.image` limits applied. Inline delivery is no longer merely hypothetical: opencode issue 21227 documents that `session/tools.ts` converts MCP `image` content into a `data:` attachment and that it is "correctly forwarded to the model via the `supportsMediaInToolResults` pipeline", with only the chat UI rendering missing. It stays off by default because it depends on the model supporting media in tool results, and client and model behaviour varies elsewhere. |
| Frame presentation | Contact sheet by default, filmstrip for narrow windows, single files on request | A sheet turns N frames into one ordered image, which is far cheaper than N images and preserves reading order. A filmstrip samples densely inside a narrow window and crops to the region of interest, which is how a 100 ms UI flicker becomes visible. |
| Auto crop | `crop="auto"` derives the box from the region that changes across the window | Asking a caller to supply pixel coordinates before they know where the problem is does not work: the only worked example found in the wild required `y: 1730, height: 360` supplied by hand. Deriving the box from frame differences is cheap on a narrow window, cuts token cost, and makes OCR on the region reliable. An explicit `crop` always wins over `"auto"`. |
| Audio-video sync report | `media_probe` and `media_manifest` report per-stream `start_time` offsets and warn beyond 0.5 s | Every audio event we produce is correlated with a frame timestamp, so an unmeasured offset makes that correlation quietly wrong. Reading it costs one ffprobe field. 
| OCR value tracking | Opt-in through `track` on `media_ocr` | Reading a number on each frame and reporting where it disagrees with its neighbours converts "here are frames" into "here is what changed". It is deterministic, cheap, and CPU-only, but it can misread, so it is opt-in and its output is labelled as a candidate event. |
| Hardening | Optional read-root allowlist, symlink resolution, cancellable subprocesses, per-call timeout | Borrowed from a mature local video MCP server so the tool cannot be talked into reading outside its sandbox, and so an abandoned request does not leave ffmpeg running. |
| Frame selection | Four-stage hybrid | Scene boundaries miss gradual change (typing, a slowly rising panel value); fixed sampling misses long static passages and wastes budget. |
| Retrieval | `media_search` is pure search over cached artifacts | It reuses work already paid for and answers "when did this happen" without guessing timestamps. No embeddings, no model downloads. |
| GPU policy | One heavy job at a time, with a precheck | 4 GB VRAM cannot hold two jobs. The lock and the free-VRAM check make the failure mode predictable and visible. |
| Artifact location | `~/.cache/media-mcp/`, overridable by `MEDIA_CACHE_DIR` | The cache is client-neutral and may be shared, so naming it after one client would mislead. |
| Transcription pipeline | Promoted to one neutral copy in the repo; the opencode skill keeps its `SKILL.md` and becomes a thin wrapper | The stated portability trigger fired: a second consumer now exists. One copy avoids a duplicate 5 GB virtualenv and two whisper implementations. Keeping `SKILL.md` preserves the procedural knowledge (Hugging Face token setup, meaning of `--no-vad`, evidence rule) that a tool description cannot carry. |
| Discovery | Self-sufficient tool descriptions, an opencode companion skill, one AGENTS.md pointer | Tool descriptions are the portable layer; the AGENTS.md pointer exists because a host's agents read an instruction file the tool list cannot reach. |

## 4. Repository layout

```
~/.clone/Github/media-mcp/
├── transcribe/                 canonical transcription pipeline
│   ├── transcribe.py
│   ├── transcribe.sh           stable CLI, the single entry point
│   ├── requirements.txt
│   └── .venv/                  local only, not committed
├── media/                      MCP server
│   ├── server.py               tool schemas, validation, profile mapping
│   ├── media.py                ffprobe, scene detection, frames, OCR, audio events
│   ├── artifacts.py            identity, cache store, LRU eviction, search
│   ├── contracts.py            delegated CLI contract and its self-check
│   └── .venv/                  local only, not committed
├── tests/
│   ├── test_artifacts.py
│   ├── test_media.py
│   ├── test_server.py
│   ├── test_stdio_contract.py   drives the server over a real MCP stdio pipe
│   └── fixtures.py              synthetic clip generator used by the tests
├── docs/
│   ├── design.md               this document
│   ├── plan.md                 task-by-task implementation plan
│   ├── client-config.md        opencode snippet and the retired Antigravity one
│   ├── opencode-smoke.md       observed opencode connection result
│   ├── antigravity-smoke.md    why Antigravity support was dropped
│   └── image-inline-finding.md recorded inline-image decision
├── README.md                   purpose, install, quickstart
└── .gitignore                  ignores .venv/, __pycache__/, .ruff_cache/
```

Three layers with one-directional dependencies:

- `artifacts.py` knows about identity, the cache, and the search index. It
  imports nothing from `media.py` or `server.py`.
- `media.py` knows how to run ffmpeg, ffprobe, and tesseract, and how to
  select frames. It imports `artifacts.py` for paths only.
- `server.py` owns tool schemas, parameter validation, and profile semantics.
  It imports both.
- `contracts.py` holds the delegated CLI contract and its startup self-check.
  `server.py` uses it for the dependency status, and `media.py` uses it to
  build the transcription invocation, so the contract is described once.

The payoff is that everything expensive and interesting is testable without
the MCP protocol: `media.py` and `artifacts.py` can be exercised directly by
pytest, and `server.py` stays thin enough to read in one sitting.

## 5. Tool surface

Seven tools, all prefixed `media_`. Descriptions are short because MCP tool
descriptions consume model context.

| Tool | Weight | Parameters | Returns |
| --- | --- | --- | --- |
| `media_probe` | light | `source`, `identity` | Duration, resolution, fps, codecs, audio layout, container, per-stream `start_time` offsets, estimated processing cost per stage |
| `media_manifest` | light | `source`, `profile`, `scene_threshold`, `frame_budget`, `identity` | Markdown digest: summary line, scene event table, audio event table, audio-video offset warning, cache stage status |
| `media_frames` | medium | `source`, `at` (list or `"scenes"`), `window`, `fps`, `count`, `tile`, `crop`, `labels`, `image_inline` (default off), `max_edge`, `identity` | Written PNG paths with timestamps; by default a single contact sheet, plus a note that `read` shows it. With `image_inline` the first 8 PNGs are attached as image content as well |
| `media_ocr` | medium | `source`, `at`, `window`, `fps`, `count`, `crop`, `track`, `psm`, `lang`, `identity` | Text found on screen per timestamp, plus a change report when `track` is set |
| `media_transcript` | heavy | `source`, `model`, `num_speakers`, `language`, `device`, `queue`, `identity` | Transcript text, plus paths to SRT, VTT, JSON, Markdown |
| `media_search` | light | `source`, `query`, `k`, `scope`, `identity` | Ranked moments: timestamp, source layer (transcript or OCR), snippet |
| `media_cache` | light | `action` (`list`, `stats`, `clean`) | Entry table, environment and dependency status, eviction report |

`media_cache` also carries the dependency self-check output, so no eighth
tool is needed to report environment health.

`source` accepts three forms, distinguished by shape rather than by a mode
parameter:

- An absolute path to a local media file.
- An `http` or `https` YouTube URL.
- An `http` or `https` direct media URL.

### 5.0 Shared parameter semantics

These appear on more than one tool, so they are defined once.

- **`source`** is required on every tool except `media_cache`. All three
  forms above are accepted wherever `source` appears.
- **`identity`** is accepted by every tool that takes `source`, so a caller
  can force exact hashing without changing tools. Default `"sampled"`, as
  defined in section 6.1.
- **Timestamps** are accepted as `HH:MM:SS.mmm`, `MM:SS`, or a plain number
  of seconds, and are always returned as `HH:MM:SS.mmm`.
- **`queue`** accepts `"wait"` (default) or `"fail"`, as defined in
  section 9.
- **`window`** is a window in seconds around a single given timestamp. Only
  meaningful together with a single timestamp; with a list it is an error
  rather than a silent guess.
- **`count`** caps how many frames are rasterized or how many frames are
  sampled in a window. Default `8`, hard cap `60`.
- **`fps`** is the sampling rate inside a window and applies only when
  `window` is greater than zero. Default `2`, hard cap `30`. It exists for
  sub-second events; it is not a way to sample a whole video.
- **`crop`** is `{x, y, width, height}` in pixels, normalized to the source
  frame, and applied before tiling or OCR. Cropping an affected control can
  cut token cost by an order of magnitude and is what makes a numeric read
  reliable. Out-of-bounds values are rejected with the measured frame size.
  The literal `"auto"` derives the box from the union of frame-to-frame
  differences across the requested window, padded by `pad` (default 8 px);
  an explicit box always wins over `"auto"`, and a window with no movement
  falls back to the full frame with that fact reported.

Parameter details that are otherwise easy to misread:

- `media_frames` `at` accepts a list of timestamps or the literal `"scenes"`,
  which means the top-ranked candidates from section 7. `tile` accepts
  `"sheet"` (grid, the default when `at` is `"scenes"`), `"filmstrip"`
  (vertical stack, the default when `window` is greater than zero), or
  `"individual"` (one file per frame, the default for an explicit timestamp
  list). `labels` defaults to `true` and burns `HH:MM:SS.mmm` into each tile;
  when no font is usable the tool says so and returns unlabelled tiles rather
  than failing. `image_inline` defaults to `false`, which keeps the response
  path-first and text-only; set to `true` it also attaches the returned PNGs as
  MCP image content, at most `8` per call, and reports that cap in the status
  block when more frames were written. A PNG that is missing from disk degrades
  to its path in the text, with the reason named in the status block.
- `media_ocr` applies to the same timestamps `media_frames` accepts, with
  `psm` defaulting to `6` (uniform block of text, which suits UI and slide
  content) and `lang` defaulting to `eng`. `track` defaults to `false` and is
  detailed in section 7.2.
- `media_search` `scope` accepts `"all"` (default), `"transcript"`, or
  `"ocr"`, and `k` defaults to `8` with a hard cap of `20`.

### 5.1 Profiles

`profile` applies to `media_manifest` and acts as a preset. Every other tool
takes explicit parameters; a tool never runs heavy work that the caller did
not ask for.

| Profile | Work performed | Approximate context cost |
| --- | --- | --- |
| `manifest` (default) | probe, scene detection, audio events | 1 to 3 KB |
| `read` | `manifest` plus transcription | 13 to 15k tokens for one hour of speech |
| `visual` | `manifest` plus OCR on selected frames and the paths of those frames | OCR text plus at most `count` (default 8) frame paths; images are never inlined under this profile |
| `full` | `read` plus `visual`, both capped | the sum, reported explicitly |

### 5.2 State honesty

Every tool response ends with a status block naming what was computed, what
was reused from cache, what was capped, and what was skipped with a reason.
A capped or skipped stage is always stated; silent truncation is a defect.

## 6. Artifact model

### 6.1 Identity and cache key

A cache entry is addressed by a 16-hex-character key derived from a content
identity:

- Local file, `identity="sampled"` (default): size, `mtime_ns`, and SHA-256
  of three 1 MiB chunks (head, middle, tail). Sampling keeps a multi-gigabyte
  file cheap to identify.
- Local file, `identity="strict"`: full SHA-256.
- YouTube URL: normalized URL plus the extractor's video id and duration.
- Direct URL: URL plus `ETag` and `Content-Length` when the server provides
  them.

The identity string is recorded in the entry so a mismatch can be explained
rather than guessed. A changed identity produces a new entry; the old entry
remains until eviction.

### 6.2 Layout

```
~/.cache/media-mcp/
├── entries/<key>/
│   ├── meta.json          identity, source, provenance, sizes, created, last_used, bytes per stage
│   ├── probe.json         reduced ffprobe output, including per-stream start_time
│   ├── scenes.json        scene boundaries, audio events, budget accounting
│   ├── transcript.json    segments with start, end, speaker, text, plus model tier
│   ├── timeline.md        flat text timeline for grep and for reading in pages
│   ├── ocr/<stamp>.json   OCR result per requested timestamp
│   ├── frames/<stamp>.png materialized frames
│   └── media/             downloaded bytes for URL sources
└── locks/transcribe.lock  single-flight lock for heavy transcription
```

No database. `meta.json` per entry plus a directory scan is enough for LRU
eviction and avoids a corruption-prone state file. `last_used` is rewritten on
every cache hit, which is what orders eviction; `created` is kept so an entry's
age remains inspectable after repeated use.

For a URL source, an entry exists because a download succeeded in it, or because
a download is running in it right now: a download that fails removes the entry it
created once nothing is left inside it, and never touches an entry that was
already there. A failed download therefore cannot leave the `(unknown) | 0` line
a directory scan would otherwise report, and a pre-existing entry is never
deleted by a later failure. A local-file call that fails can still leave an entry
directory with no artifacts in it, which the next call for the same source reuses
rather than replacing.

An entry directory that cannot be read, or a completed file that cannot be
stat-ed, is named in a `MediaError` rather than treated as a cache miss — only a
file that has genuinely vanished reads as a miss.

### 6.3 Timeline as text

`timeline.md` is the human- and grep-friendly projection: one line per event
in chronological order, carrying the timestamp, the layer (`scene`, `audio`,
`speech`, `ocr`, `ocr-change`), and a short payload. It exists so that an agent
can use `grep` and paged `read` on a large video without any new tool, which is
the cheapest possible retrieval path.

## 7. Frame selection

Four stages, in order. Stages 1 and 2 produce a candidate list in
`scenes.json`; nothing is rasterized until `media_frames` or `media_ocr` asks
for a specific timestamp.

1. **Scene boundaries.** One decode pass using the `scdet` filter, which
   reports one scene-change score per second in its own units. Measured on this
   machine: continuous motion in `testsrc` scores about `0.5`, while a hard cut
   scores about `35`. The threshold therefore varies by profile in those units
   (default `10.0`, the filter's own default; `5.0` for screencast-like content
   where screens change subtly). The scores are kept because stage 3 ranks with
   them. `scdet` is present in the ffmpeg build on this machine, so no scene
   detection library is added.
2. **Fixed-interval candidates.** Sampling every `interval` seconds
   (default `3`) added to the candidate set, then reduced by perceptual hash:
   downscale to 32x32 grayscale, compute a difference hash, and drop a
   candidate whose Hamming distance to a kept neighbour is within
   `dedup_distance` (default `6`). This is the stage that catches gradual
   change, such as a code editor filling with text.
3. **Ranking.** Scene boundaries first, then candidates aligned with audio
   events (loudness peak, silence boundary), then uniform fill by timestamp.
4. **Budget.** `frame_budget` (default 120) caps the candidate list, and
   `media_manifest` states how many candidates were dropped and by which
   rule.

Of these five knobs, only `scene_threshold` and `frame_budget` are
caller-visible in V1, through `media_manifest`. `interval` (3 s),
`dedup_distance` (6), and the JPEG fallback threshold are internal constants so
that the parameter surface stays small; they are defined in one module and are
changed by editing that constant, not by adding tool parameters.

Frame extraction downscales to `max_edge` (default 1600 px) preserving
aspect ratio, encodes PNG for text fidelity, and re-encodes to JPEG quality
85 only when a PNG exceeds `2 MiB`. Both limits are reported in the response
when they trigger.

### 7.1 Rendering and tiling

Three tile modes, chosen by the caller and defaulted from the request shape as
described in section 5.0.

- `sheet` arranges the frames in a grid whose column count keeps the sheet
  under `max_edge`, writes one PNG, and burns the timestamp of each tile into
  its top-left corner. This is the default whole-video visual, and it exists
  because one image costs far less context than eight.
- `filmstrip` stacks frames sampled at `fps` inside `window` into a single
  vertical strip, after applying `crop`. It is the mode for motion and for
  sub-second glitches, where neighbouring frames disagree visibly.- `individual` writes one PNG per frame, for when the caller wants to point
  `read` at a specific moment.

Labels are drawn by generating an ASS file and burning it in with the `ass`
filter. `drawtext` is absent from the ffmpeg build on this machine while `ass`
and `subtitles` (libass) are present, and one backend then serves both tile
labels and the burned-in text of the test fixtures. A monospace font is
resolved from the system through `fc-match` and named in the ASS style; when no
font is usable the tool reports that and still returns the image without labels.

When `crop="auto"` is requested, the box is computed once per window from the
frame-to-frame difference union, then reused by both `media_frames` and
`media_ocr` for that window, so tiling and OCR see the same region and the
cost of detection is paid once.

### 7.2 Value tracking over OCR

`media_ocr` with `track=true` samples the window at `fps`, runs OCR on each
frame, and extracts numeric or short textual values, preferring the `crop`
region when one is given. It then reports candidates where a frame's value
disagrees with both neighbours, in the form
`value 48 at 00:06.330, neighbours 80 then 80`.

The report is explicitly labelled as a candidate event, not a conclusion,
because OCR misreads produce false positives. Each candidate carries the raw
OCR string and its per-word confidence so the caller can judge it. When no
numeric value is found at all, the tool says so instead of returning an empty
list. Candidates are appended to the `ocr-change` layer of `timeline.md` and of
`scenes.json` so later `media_search` calls can find them.

## 8. Transcription delegation

The server does not implement transcription. It calls one stable entry point.

Contract, which both the server and the opencode skill rely on:

- Entry point: `$MEDIA_TRANSCRIBE_BIN`, default `<repo>/transcribe/transcribe.sh`.
- Invocation: `transcribe.sh <input> [--model M] [--num-speakers N]
  [--no-diarize] [--no-vad] [--denoise] [--language L] [--out DIR]`.
- Input: a local file path or an `http`/`https` URL.
- Success: exit code 0, and `<out>/<stem>/` containing `.srt`, `.vtt`,
  `.json`, and `.md`.
- JSON shape: `{"meta": {...}, "segments": [{"start", "end", "speaker", "text"}]}`.
- The server always passes an `--out` inside the cache entry, so the server
  never writes into a working directory or a source directory.

Startup self-check, run once per server start: the binary exists, is
executable, and exits 0 for `--help`. The result is reported by
`media_cache(action="stats")` under a `dependencies` block. If the check
fails, every transcription tool returns an actionable error naming
`MEDIA_TRANSCRIBE_BIN`, the resolved path, and the expected layout. There is
no silent fallback to a worse path.

The `transcribe-diarize` skill keeps its `SKILL.md` as the procedural record,
but its paths and environment names had to be corrected once the pipeline moved:
a document that still names the old `.venv` and `requirements.txt` locations
would send a reader to a directory that no longer holds them. `scripts/transcribe.sh`
becomes a thin wrapper that resolves the new location through the same variable
and forwards its arguments, and `scripts/setup.sh` becomes the same kind of
wrapper for the moved setup. The skill therefore keeps working standalone,
including flags the server never uses.

Regression check is mandatory and mechanical: run both the old and new
pipeline on the same short sample with the same model, then compare the
`segments` arrays of the JSON output. Identical output means no regression.
Rollback is reverting one wrapper file.

## 9. Resource policy

- **Single flight.** Heavy transcription takes an exclusive `flock` on
  `locks/transcribe.lock`. Two jobs never run at once.
- **Queue behaviour.** `queue="wait"` (default) reports the holder and
  elapsed time, then waits. `queue="fail"` returns immediately with the same
  information.
- **VRAM precheck.** Before `large-v3-turbo`, read free VRAM from
  `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`. Below
  `MEDIA_VRAM_FLOOR` (default 2800 MiB) drop to `medium`, then to CPU.
- **Explicit override.** `device` accepts `auto`, `cuda`, or `cpu`;
  `MEDIA_DEVICE` sets the default. A forced value is honoured even if the
  precheck disagrees.
- **No silent downgrade.** Any tier change is reported in the tool response
  with the reason, including the measured free VRAM.
- **Bounded and cancellable work.** Every child process runs under a
  per-call timeout (`timeout_seconds`, default 900, ceiling 21600) and is
  terminated with its process group when the MCP request is cancelled or the
  timeout expires. A cancelled call leaves no ffmpeg or whisper process
  behind, and the response says the work was cancelled rather than reporting
  a partial result as complete.

## 10. Client integration

The server is client-agnostic: stdio transport, absolute paths in client
configuration, all configuration through `MEDIA_*` environment variables, and
no reads from any client's config directory.

### 10.1 Environment variables

Every setting has a default that works on this machine, so a client
configuration can stay minimal.

| Variable | Default | Effect |
| --- | --- | --- |
| `MEDIA_CACHE_DIR` | `~/.cache/media-mcp` | Cache root |
| `MEDIA_CACHE_CAP` | `20 GiB` | LRU size cap enforced after each job |
| `MEDIA_TRANSCRIBE_BIN` | `<repo>/transcribe/transcribe.sh` | Delegated pipeline entry point |
| `MEDIA_TRANSCRIBE_ENV` | `~/.config/opencode/skills/transcribe-diarize/.env` | Token file the delegated pipeline reads; it stays outside the repository |
| `MEDIA_DEVICE` | `auto` | Default compute device: `auto`, `cuda`, `cpu` |
| `MEDIA_VRAM_FLOOR` | `2800 MiB` | Free VRAM below which the model tier drops |
| `MEDIA_FRAME_MAX_EDGE` | `1600` | Longest frame edge in pixels |
| `MEDIA_MAX_SOURCE_BYTES` | `4 GiB` | Hard refusal threshold for downloads |
| `MEDIA_MAX_SOURCE_SECONDS` | `14400` (4 h) | Hard refusal threshold for duration |
| `MEDIA_ALLOWED_READ_ROOTS` | unset (any readable path) | Colon-separated allowlist; when set, sources outside it are refused after symlink resolution |
| `MEDIA_CALL_TIMEOUT_SECONDS` | `900` | Default per-call ceiling for child processes |
| `MEDIA_FONT_FILE` | unset (resolved by `fc-match monospace`) | Explicit TTF used for tile labels; a missing file yields unlabelled tiles and a stated reason |
| `MEDIA_ALLOW_PRIVATE_URLS` | unset | Set to `1` to permit loopback and RFC 1918 targets |

opencode, in `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "mcp": {
    "servers": {
      "media": {
        "type": "local",
        "command": ["{env:HOME}/.clone/Github/media-mcp/media/.venv/bin/python",
                    "{env:HOME}/.clone/Github/media-mcp/media/server.py"],
        "enabled": true
      }
    }
  }
}
```

The retired Antigravity entry, kept here because it is the shape a `${VAR}`
client needs:

```json
{
  "mcpServers": {
    "media": {
      "command": "${HOME}/.clone/Github/media-mcp/media/.venv/bin/python",
      "args": ["${HOME}/.clone/Github/media-mcp/media/server.py"],
      "env": {
        "MEDIA_CACHE_DIR": "${HOME}/.cache/media-mcp"
      }
    }
  }
}
```

Antigravity support was dropped on 2026-10-21 after its UI check failed (section
1), and the entry was removed from `~/.gemini/antigravity/mcp_config.json` on
2026-10-22, so nothing on this machine loads the server through that client. The
block above is reference material, not a supported path.

Each client substitutes environment variables in its own configuration file, so
a snippet stays machine-independent: opencode uses `{env:VAR}`, Antigravity used
`${VAR}`. If a future client version stops substituting, replace `{env:HOME}` or
`${HOME}` with the literal output of `echo $HOME`; JSON itself never expands `~`.

Discovery, two layers:

1. Tool descriptions are self-sufficient and state when to use the tool.
2. A new opencode skill `media-context` documents the tiered workflow, the
   meaning of each profile, and worked examples. A single pointer line is
   added to `~/.config/opencode/AGENTS.md`.

The `~/.gemini/GEMINI.md` section that Task 15 added for Antigravity was removed
again when support was dropped, so that file is byte-identical to its state
before Task 15.

`docs/client-config.md` holds the authoritative copy of the live snippet, so a
reader of the repository does not have to extract it from this document.
`README.md` carries only a short quickstart that links to it.

An optional opencode command `/video <source>` prepares a manifest and
suggests the next step. It is a convenience, not a requirement.

## 11. Security and safety

- **URL schemes.** Only `http` and `https` are accepted. `file://` and other
  schemes are rejected in the `source` parameter.
- **Private-range guard.** URLs resolving to loopback, link-local, or RFC 1918
  addresses are refused by default, which blocks metadata-endpoint probing.
  `MEDIA_ALLOW_PRIVATE_URLS=1` overrides it for local streaming cases.
- **Size and duration limits.** Download is capped at 4 GiB and 4 hours by
  default, with a warning at 2 GiB or 2 hours. Beyond the hard cap the tool
  refuses and states the measured value.
- **No shell interpolation.** External programs are invoked with argument
  lists, never through a shell. Timestamps, counts, and thresholds are parsed
  and range-checked before use.
- **Sources are read-only.** The server writes only inside the cache entry.
  It never modifies, moves, or deletes a source file.
- **Filesystem confinement.** When `MEDIA_ALLOWED_READ_ROOTS` is set, every
  local source is resolved with `realpath` before the check, so a symlink
  cannot be used to escape the allowlist. Without the variable, any readable
  path is accepted, which is the deliberate default for a single-user
  workstation and is stated in the README rather than left implicit.
- **Secrets.** The server never reads, echoes, or passes `HF_TOKEN` in
  arguments; the delegated pipeline loads its own `.env`. Tokens must not
  appear in process arguments, logs, or tool responses.
- **Failure surfaces.** Subprocess stderr is surfaced verbatim in the error
  so a yt-dlp or ffmpeg failure can be diagnosed without a second run.

## 12. Edge cases

- Silent audio: silence detection reports one cover-all segment; transcription
  proceeds and returns empty or minimal segments without failing.
- No scene change at all: the fixed-interval stage still yields candidates,
  and the manifest states that no scene boundary was detected.
- Audio-only source: probe reports no video stream; frame and OCR tools return
  a clear "no video stream" error rather than an empty result.
- Corrupt or unreadable media: ffprobe error is returned verbatim with the
  exit code.
- Identity changed between calls: a new cache entry is created and the
  response states that the previous entry was not reused and why.
- Cache cap reached mid-run: eviction runs after the current job finishes, so
  an in-flight job is never evicted underneath itself.
- Transcription binary missing: tools fail with the exact override variable
  and expected path.
- `tesseract` language pack missing: the error lists available languages from
  `tesseract --list-langs`.
- Duplicate concurrent requests for the same source: inside one server
  process, an identical in-flight request is coalesced onto the running job.
  Across clients or sessions, the file lock in section 9 serializes heavy work,
  so the second job runs after the first and then reuses the cache entry instead
  of decoding again, which the response reports as a cache hit.
- Cancelled request: the child process group is terminated, no cache entry is
  marked complete, and the next call for the same source recomputes rather
  than trusting a partial artifact.
- `crop` larger than the frame, or with a zero dimension: rejected with the
  measured frame size in the error.
- `track=true` on a region with no numbers: the tool reports that no numeric
  value was found instead of returning an empty candidate list.
- `window` given together with a list of timestamps: rejected as an
  ambiguous request rather than silently applied to the first element.
- No usable font for labels: tiles are returned unlabelled and the response
  says why.
- Audio and video start offsets differ by more than 0.5 s: `media_probe` and
  `media_manifest` report the measured offsets and warn that frame-to-audio
  correlation carries that error, so an event reported at a timestamp is
  understood to be approximate.
- `crop="auto"` on a window with no movement: the full frame is used and the
  response says the box could not be derived, rather than returning a
  degenerate one-pixel crop.

## 13. Risks and open verification items

| Risk | Mitigation | How it is settled |
| --- | --- | --- |
| Inline image delivery depends on the model, not only the client | Path-first default; `image_inline` off unless the model in use supports media in tool results | opencode issue 21227 confirms MCP `image` content is converted to an attachment and forwarded to the model, with only UI rendering missing. Settled by measurement: a vision model described a randomised probe image correctly, so the flag ships off by default. Recorded per host and model in `docs/image-inline-finding.md`. |
| OCR value tracking produces false positives on noisy frames | Opt-in only, every candidate carries the raw string and per-word confidence, and output is labelled a candidate event | Synthetic slider fixture with a known injected jump; the report must name the correct timestamp, and a fixture with no numbers must return "none found". |
| A contact sheet becomes illegible when tiles are small | `max_edge` bounds the sheet, `crop` narrows the field before tiling, and `individual` remains available | OCR of a generated sheet must recover the burned-in labels and the synthetic fixture text; frame geometry is asserted, not eyeballed. |
| A cancelled or timed-out call leaves a child process running | Process-group termination on cancel and timeout | Start a long job, cancel it, and assert no descendant ffmpeg or whisper process remains. |
| Moving the transcription pipeline regresses a verified path | Mechanical move, no logic edits | Segment-level diff of old versus new output on one sample. |
| 4 GB VRAM contention with other work | Single-flight lock plus free-VRAM precheck, CPU override | Unit test with injected `nvidia-smi` output, plus one forced-CPU run. |
| Cache growth | LRU eviction with a 20 GiB default cap, enforced after each job | Observe reported eviction counts during the integration test. |
| yt-dlp breakage on YouTube changes | Surface stderr verbatim; upgrade path documented | Out of scope to prevent; the failure is diagnosable. |
| Scope creep across clients and seven tools | Non-goals are explicit; any further consumer feature stays deferred | Spec review. |

## 14. Verification plan

1. Static: `python -m py_compile` on every module, `ruff check .` and
   `ruff format --check .` clean.
2. Unit tests: identity stability and change detection, LRU eviction order and
   cap enforcement, timestamp parsing and formatting, frame naming, dedup
   distance decisions, search scoring and `k` capping, profile-to-work
   mapping, VRAM precheck with injected input, crop bounds rejection, tile
   mode selection from the request shape, auto-crop box derivation from
   synthetic difference frames, the audio-video offset warning threshold, and
   value-change detection on synthetic OCR sequences including the
   no-numbers case.
3. Stdio contract tests with synthetic fixtures: drive the server over a real
   MCP stdio pipe as a client would, asserting that every tool advertises its
   schema, that a malformed parameter produces an MCP error rather than a
   crash, and that a cancelled request terminates cleanly.
4. Integration on a synthetic sample: generate a 60-second clip with ffmpeg
   (test pattern, burned-in text, a tone with a deliberate loud spike, and a
   burned-in counter that jumps once). Then run probe, manifest, frames, OCR,
   and search. Assert: scene events found, the loud spike appears as an audio
   event, OCR recovers the burned-in text, the counter jump appears as an
   `ocr-change` candidate at the right timestamp, the contact sheet is one
   file whose dimensions respect `max_edge`, `crop="auto"` returns a box
   smaller than the full frame that contains the counter region, and the
   response status block matches the work actually performed. A second
   fixture generated with a one-second audio offset must make `media_manifest`
   report the offset and warn.
5. Filmstrip check: on a fixture with a two-frame flicker inside a narrow
   window, `tile="filmstrip"` returns one strip and the flicker frame is
   distinguishable from its neighbours.
6. Transcription regression: run old and new pipeline on one short sample,
   diff the `segments` arrays, require identical output.
7. Forced CPU run: `MEDIA_DEVICE=cpu` completes and reports the downgrade
   reason.
8. Cache behaviour: run the cap down to a small value on synthetic entries and
   confirm eviction order, the report, and that source files are untouched.
9. Confinement check: with `MEDIA_ALLOWED_READ_ROOTS` set, a source inside the
   allowlist succeeds and a symlink pointing outside it is refused with the
   resolved path named.
10. opencode smoke test: the server appears connected in `opencode mcp list`,
    and `media_probe` succeeds on the sample.
11. Antigravity smoke test: config snippet in place, server reachable. The
    mechanical half passed over a real stdio session; the in-UI half failed,
    and the user then dropped the client (section 1). The evidence stays in
    `docs/antigravity-smoke.md`.
12. Inline image check: with a model reported to support media in tool
    results, confirm the model receives the image; otherwise confirm the
    path-plus-`read` route, and record which route applied.

Evidence for each item is recorded in the implementation plan's verification
table. No item is marked complete without observed output.

## 15. Migration steps

1. Create the repo and both virtualenvs; commit `.gitignore` before any
   artifact appears.
2. Move the pipeline files from
   `~/.config/opencode/skills/transcribe-diarize/scripts/` into
   `transcribe/`, keeping flags and output contract byte-identical. Move the
   virtualenv or rebuild it from `requirements.txt`.
3. Run the regression diff described in section 14 and record the result.
4. Rewrite `~/.config/opencode/skills/transcribe-diarize/scripts/transcribe.sh`
   as the forwarding wrapper, and `scripts/setup.sh` the same way, since the
   virtualenv and `requirements.txt` move with the pipeline. Leave `SKILL.md`
   and `.env` in place, correcting the paths and environment names inside
   `SKILL.md` so they describe where the pipeline actually lives.
5. Verify the skill still works standalone through the wrapper, including one
   flag the server does not use, such as `--denoise`.
6. Implement the server layers in the order `artifacts.py`, `media.py`,
   `server.py`, with tests alongside each. The synthetic fixture generator
   comes first, because every integration assertion depends on it.
7. Wire opencode and run its smoke test. Antigravity was wired next and then
   dropped after its UI check failed (section 1), and its entry was removed
   again, so opencode is the only wired client.
8. Add the `media-context` skill and the AGENTS.md pointer line. The GEMINI.md
   section was added with Task 15 and removed again with the drop.
9. Run the image passthrough probe and record the outcome; leave
   `image_inline` off unless the result justifies enabling it.
