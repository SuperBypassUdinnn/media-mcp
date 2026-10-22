# Antigravity smoke test

Status: dropped. The mechanical proof below passed, the in-UI check failed, and
the user retired the client on 2026-10-21 (see the end of this file). The record
is kept as evidence and as a starting point for a future `${VAR}` client.

Task 14 wires the `media` server into Antigravity and records the strongest
evidence a headless agent can produce. The in-UI step is a separate, explicitly
marked question, because Antigravity's server list and chat panel cannot be
driven from a shell.

## Client entry

Inserted at the end of `mcpServers` in `~/.gemini/antigravity/mcp_config.json`,
outside this repository, as a text edit so the five existing objects keep their
exact bytes:

```json
    "media": {
      "command": "${HOME}/.clone/Github/media-mcp/media/.venv/bin/python",
      "args": ["${HOME}/.clone/Github/media-mcp/media/server.py"],
      "env": {
        "MEDIA_CACHE_DIR": "${HOME}/.cache/media-mcp"
      }
    }
```

Pre-edit backup: `/tmp/opencode/backups/20261021-214633/mcp_config.json`
(sha256 `556db59fa8d76154…`, verified identical to the file before the edit).

Validation and the untouched-servers comparison, both against that backup:

```
$ python3 -m json.tool ~/.gemini/antigravity/mcp_config.json > /dev/null
json.tool: valid

top-level keys equal: True
supabase: present=True identical=True
github-mcp-server: present=True identical=True
dart-mcp-server: present=True identical=True
gopls-mcp-server: present=True identical=True
alloydb-postgresql: present=True identical=True
added: ['media']
removed: []
```

The block reads back with `${HOME}` intact, which is what the client is expected
to expand.

## Mechanical proof over real stdio

A throwaway harness (`/tmp/opencode/antigravity-stdio-proof.py`, not committed)
loaded the `media` block verbatim from the config, expanded `${VAR}` the way a
client would, launched exactly that command, and spoke MCP.

First, the negative control — the same block with `${HOME}` left unexpanded, to
show the entry depends on the client substituting:

```
argv: ['${HOME}/.clone/Github/media-mcp/media/.venv/bin/python',
       '${HOME}/.clone/Github/media-mcp/media/server.py']
OSError: [Errno 2] No such file or directory:
'${HOME}/.clone/Github/media-mcp/media/.venv/bin/python'
```

Then the real run:

```
expanded argv: ['/home/superbypassudin/.clone/Github/media-mcp/media/.venv/bin/python',
                '/home/superbypassudin/.clone/Github/media-mcp/media/server.py']
expanded env: {'MEDIA_CACHE_DIR': '/home/superbypassudin/.cache/media-mcp'}

initialize -> {"jsonrpc": "2.0", "id": 1, "result": {
  "capabilities": {"prompts": {"listChanged": false},
                   "resources": {"listChanged": false, "subscribe": false},
                   "tools": {"listChanged": false}},
  "protocolVersion": "2025-06-18",
  "serverInfo": {"name": "media", "version": ""}}}

tools/list -> 7 tools: media_cache, media_frames, media_manifest, media_ocr,
               media_probe, media_search, media_transcript

media_probe -> {"jsonrpc": "2.0", "id": 3, "result": {"content": [{"text":
"# media_probe

Source: /tmp/opencode/smoke/smoke.mp4
Identity: file-sampled ff8b9aaf051f1ce8

Resolution: 1280x720
Duration: 00:00:06.000 (6.000 s)
Frame rate: 25.000 fps
Container: mov,mp4,m4a,3gp,3g2,mj2
Video codec: h264
Audio codec: aac
Audio layout: mono
Streams: video=yes, audio=yes
Stream start_time: video=0.000 s, audio=0.000 s (audio-video offset +0.000 s)
Source size: 1212619 bytes

Estimated processing cost per stage:
- probe: light
- scenes: light
- audio: light
- transcript: heavy
- ocr: medium
- frames: medium

computed: (none)
cache: probe.json
capped: (none)
skipped: (none)", "type": "text"}], "isError": false}}
```

The fixture is the same six-second, two-stream clip Task 13 generated under
`/tmp/opencode/smoke/`, and the echoed numbers match ffprobe's independent
reading. The run also proves the documented environment variable reaches the
child: `MEDIA_CACHE_DIR` was passed exactly as configured and the probe wrote its
entry there.

## What is not proven, and the outcome

The harness proves the configured command starts, speaks MCP, advertises seven
tools, and answers a real call. It does **not** prove that Antigravity's own
process substitutes `${HOME}`, that its server list shows `media`, or that a chat
call in the UI succeeds: those require the Antigravity GUI, which cannot be
driven from a shell.

Status: **failed, and the client is dropped.** The user ran the in-UI check on
2026-10-21 and reported two problems in the same panel (screenshot
`Screenshot_20261021_222035.png`):

- the customization budget was exceeded, at 22,548 tokens for skills (112.7
  percent) with 11 entries excluded, which left the agent almost no room;
- the server list did not show `media`.

The user therefore dropped Antigravity support on 2026-10-21. On 2026-10-22 the
`media` entry was removed from `~/.gemini/antigravity/mcp_config.json` again, so
that file holds only its five original servers (`supabase`, `github-mcp-server`,
`dart-mcp-server`, `gopls-mcp-server`, `alloydb-postgresql`), and the
`~/.gemini/GEMINI.md` section that Task 15 added was removed with it, leaving
that file at its pre-Task-15 length of 53 lines.

The entry above is kept as reference for a future `${VAR}` client, not as a
supported path. Snapshots of the four Antigravity-side files as they stand after
the removal are under
`~/.local/state/opencode-probe/antigravity-snapshots/20261022-213006/`.
