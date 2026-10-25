# media-mcp

A portable local MCP server that turns video and audio into text and image artifacts a model can read.

Every stage runs on this machine: no cloud transcription, vision, or embedding
service is called. Sources may be absolute local paths, YouTube URLs, or direct
`http(s)` media URLs; artifacts are cached under `~/.cache/media-mcp` and shared
by every client.

## Quickstart

Both virtualenvs are local and untracked. The server environment was created with
`uv` on Python 3.14 and does not carry `pip`, so `uv` is the shortest rebuild
path. `media/requirements.txt` pins the whole closure, and was captured from the
working venv with `uv pip freeze`; the three direct dependencies are `mcp`,
`pytest`, and `ruff`:

```bash
uv venv media/.venv --python 3.14
uv pip install --python media/.venv/bin/python -r media/requirements.txt
media/.venv/bin/python -m pytest -q
```

The transcription pipeline is a separate, much larger environment (torch,
faster-whisper, pyannote) and is built on demand:

```bash
bash transcribe/setup.sh      # creates transcribe/.venv and installs its requirements
```

Register the server with a client, pointing it at
`media/.venv/bin/python media/server.py`:

- [opencode snippet](docs/client-config.md)
- [Server design, tool surface, and safety rules](docs/design.md)
- [Observed opencode run](docs/opencode-smoke.md)

## A first call

```text
media_probe(source="/path/to/clip.mp4")
media_manifest(source="/path/to/clip.mp4")                    # cheap default: scenes, audio events
media_manifest(source="/path/to/clip.mp4", profile="read")     # adds transcription
media_frames(source="/path/to/clip.mp4", at="scenes")          # writes a contact sheet, returns its path
```

Frames are returned as paths by default: reading a frame means reading the PNG
the tool returned. `media_frames(image_inline=true)` also attaches up to eight of
those PNGs as image content, which a vision model sees directly, as recorded in
the [inline image finding](docs/image-inline-finding.md). Every response ends
with what it `computed`, reused from `cache`, `capped`, and `skipped`, so a
truncated stage is visible instead of silent.

## Notes

- Seven tools, all prefixed `media_`: `media_probe`, `media_manifest`,
  `media_frames`, `media_ocr`, `media_transcript`, `media_search`, `media_cache`.
- The default `manifest` profile costs 1 to 3 KB; transcription is the heavy
  stage and always opt-in. The first transcribed call is also the slow one: the
  Hugging Face cache holds only `faster-whisper-tiny` (75 MB), and the pipeline's
  default `large-v3-turbo` downloads roughly 1.6 GB, unless that model is
  already cached or `model` is passed explicitly.
- Sources are read-only. The server writes only inside its cache entry and never
  modifies, moves, or deletes a source file.
- With `MEDIA_ALLOWED_READ_ROOTS` unset, any readable local path is accepted
  after symlink resolution. That is the deliberate default on a single-user
  workstation; set the variable to confine reads to named roots.
- The private-address URL guard is a pre-flight check, not a network sandbox;
  `MEDIA_ALLOW_PRIVATE_URLS=1` overrides it for local streaming.
