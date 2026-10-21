# Client configuration

Authoritative copy of the two client snippets, so a reader of the repository does
not have to extract them from `docs/design.md`. Both clients substitute
environment variables inside their own configuration file, which keeps the
snippets machine-independent; JSON itself never expands `~`.

- opencode uses `{env:VAR}` and a command array.
- Antigravity uses `${VAR}` and a command plus args.

If a future client version stops substituting, replace the marker with the
literal output of `echo $HOME` and record the change in
`docs/opencode-smoke.md` or `docs/antigravity-smoke.md`.

## opencode

In `~/.config/opencode/opencode.jsonc`, inside the existing `mcp.servers`
object:

```jsonc
      "media": {
        "type": "local",
        "command": [
          "{env:HOME}/.clone/Github/media-mcp/media/.venv/bin/python",
          "{env:HOME}/.clone/Github/media-mcp/media/server.py"
        ],
        "enabled": true
      }
```

`{env:HOME}` substitution was observed working: `opencode mcp list` shows
`media connected`. See `docs/opencode-smoke.md`.

Note that the `mcp-profiles` plugin in this setup keeps only `memory` and
`codegraph` always on and marks every other server disabled per project until the
user approves it, so a freshly added entry can read `disabled` in
`opencode mcp list` until that project approves it. That is the plugin's gate,
not a broken entry.

## Antigravity

In `~/.gemini/antigravity/mcp_config.json`, inside the existing `mcpServers`
object:

```json
    "media": {
      "command": "${HOME}/.clone/Github/media-mcp/media/.venv/bin/python",
      "args": ["${HOME}/.clone/Github/media-mcp/media/server.py"],
      "env": {
        "MEDIA_CACHE_DIR": "${HOME}/.cache/media-mcp"
      }
    }
```

The entry as written depends on the client expanding `${HOME}`: launching it
with the marker intact fails with `No such file or directory`, and the same
command with the marker expanded serves all seven tools over stdio. See
`docs/antigravity-smoke.md`, which also records the in-UI confirmation as a
separate, explicitly pending step because a chat panel cannot be driven headlessly.

## One cache, one pipeline

Both clients start the same server, so both read the same artifact cache at
`~/.cache/media-mcp` and the same transcription pipeline at
`<repo>/transcribe/transcribe.sh`. Nothing is duplicated per client.

## Settings a client may want to set

Every variable has a default that works on this machine, so both configurations
stay minimal. The ones worth knowing:

| Variable | Default | Effect |
| --- | --- | --- |
| `MEDIA_CACHE_DIR` | `~/.cache/media-mcp` | Cache root |
| `MEDIA_CACHE_CAP` | `20 GiB` | LRU cap, enforced after each job |
| `MEDIA_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `MEDIA_VRAM_FLOOR` | `2800 MiB` | Free VRAM below which the model tier drops |
| `MEDIA_TRANSCRIBE_BIN` | `<repo>/transcribe/transcribe.sh` | Delegated pipeline entry point |
| `MEDIA_TRANSCRIBE_ENV` | `~/.config/opencode/skills/transcribe-diarize/.env` | Token file the pipeline reads |
| `MEDIA_ALLOWED_READ_ROOTS` | unset | Colon-separated allowlist; sources outside it are refused after symlink resolution |
| `MEDIA_ALLOW_PRIVATE_URLS` | unset | Set to `1` to permit loopback and RFC 1918 targets |
| `MEDIA_MAX_SOURCE_BYTES` | `4 GiB` | Hard refusal threshold for downloads |
| `MEDIA_MAX_SOURCE_SECONDS` | `14400` | Hard refusal threshold for duration |
| `MEDIA_CALL_TIMEOUT_SECONDS` | `900` | Per-call ceiling for child processes |
| `MEDIA_FRAME_MAX_EDGE` | `1600` | Longest frame edge in pixels |
| `MEDIA_FONT_FILE` | unset (`fc-match monospace`) | TTF for tile labels |

`MEDIA_ALLOW_PRIVATE_URLS` deserves a caution: the private-range guard is a
pre-flight check on the URL's resolved address, not a network sandbox. It exists
to stop accidental metadata-endpoint probing, and the override is documented
rather than hidden, so a local streaming case can still be served deliberately.

## Transcription delegation

The server never transcribes itself. `media_transcript` (and the `read` and
`full` profiles) delegate to `<repo>/transcribe/transcribe.sh`, which runs the
repository's own virtualenv at `transcribe/.venv` and reads its token file
through `MEDIA_TRANSCRIBE_ENV`.

- The pipeline was moved out of the skill directory; the skill's
  `scripts/transcribe.sh` is now a forwarder with an anti-recursion guard.
- Rebuild the virtualenv with `bash transcribe/setup.sh`; it creates the venv and
  installs `transcribe/requirements.txt`.
- The Hugging Face cache currently holds only the small
  `Systran/faster-whisper-tiny` model (75 MB). The pipeline's default
  `large-v3-turbo` downloads roughly 1.6 GB on first use, so expect one slow
  first call unless that model is already cached or `model` is passed explicitly.

## Tool cost ladder

`media_manifest`'s `profile` parameter is the only preset; every other tool takes
explicit parameters and never runs heavy work the caller did not ask for.

| Profile | Work | Context cost |
| --- | --- | --- |
| `manifest` (default) | probe, scenes, audio events | 1 to 3 KB |
| `read` | adds transcription | 13k to 15k tokens per hour of speech |
| `visual` | adds OCR and frame paths | OCR text plus at most `count` frame paths |
| `full` | `read` plus `visual`, both capped | the sum, stated |

Frames are written to disk and returned as paths; no profile inlines an image.
Reading a frame means reading the PNG the tool returned.

## See also

- `docs/design.md` sections 5, 10, and 11 for the tool table, the client
  integration rules, and the security posture.
- `docs/opencode-smoke.md` and `docs/antigravity-smoke.md` for the observed
  evidence behind the two snippets.
