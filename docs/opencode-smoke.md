# opencode smoke test

Task 13 wires the `media` server into opencode and records one real call, so the
integration is evidenced by an observed run rather than by configuration alone.

## Client entry

The block below was inserted at the end of `mcp.servers` in
`~/.config/opencode/opencode.jsonc`, which is outside this repository. It is the
block from design section 10 verbatim: `type` is `"local"`, the command is the
repository's own virtualenv running `media/server.py`, and the paths use the
`{env:HOME}` form.

```jsonc
      // media-mcp: local video/audio context server (transcription, keyframes,
      // OCR, search). Reads the repository's own venv; all state stays under
      // ~/.cache/media-mcp. See ~/.clone/Github/media-mcp/docs/design.md.
      "media": {
        "type": "local",
        "command": [
          "{env:HOME}/.clone/Github/media-mcp/media/.venv/bin/python",
          "{env:HOME}/.clone/Github/media-mcp/media/server.py"
        ],
        "enabled": true
      }
```

No other entry, comment, or indentation changed. Before editing, the
pre-edit state was written to
`/tmp/opencode/backups/20261021-214455/opencode.jsonc`; because the edit
happened in the same turn, that file was rebuilt immediately afterwards by
removing exactly the inserted block, and a comparison confirmed the difference is
that block alone.

Nothing was lost: extracting the keys directly inside the `mcp.servers` object
gives 12 servers before and 13 after, the only addition being `media`.

```
before: 12 ['codegraph', 'memory', 'docker', 'genoffice', 'stocky', 'github',
             'context7', 'gns3-routeros', 'zte-readonly', 'tplink-readonly',
             'fetcher', 'tavily']
after:  13 [..., 'media']
```

## `{env:HOME}` substitution works

`opencode mcp list` printed `media` in its table, so the client substituted
`{env:HOME}` rather than failing on the literal token. No fallback to a literal
path was needed.

```
$ opencode mcp list
✓ codegraph        connected
✓ context7         connected
○ docker           disabled
○ fetcher          disabled
○ genoffice        disabled
✓ github           connected
○ gns3-routeros    disabled
✓ media            connected
✓ memory           connected
○ stocky           disabled
○ tavily           disabled
○ tplink-readonly  disabled
○ zte-readonly     disabled
```

`media` is connected, not awaiting authorization.

### Operational note: the project profile gates non-core servers

The `mcp-profiles` plugin keeps `memory` and `codegraph` always on and marks
every other server disabled for a project until the user approves it, so the
first `opencode mcp list` after the config edit showed `media` as
`disabled` in the profile even though the entry was valid. One approval
(the agent's `mcp_connect` request, persisted in
`~/.local/state/opencode/mcp-profiles/<sha256(directory)[:16]>.json`) turned it
into `connected`. A different project needs its own approval.

## Fixture

Generated with the suite's own generator, never committed, and probed with
ffprobe so the expected numbers are independent of the server:

```
$ media/.venv/bin/python -c "import sys; from pathlib import Path
  sys.path.insert(0, 'tests'); import fixtures
  fixtures.make_sample(Path('/tmp/opencode/smoke/smoke.mp4'),
                       duration=6, text='SMOKE 7', spike_at=3, counter_jump_at=2)"
/tmp/opencode/smoke/smoke.mp4

$ ffprobe -v error -show_entries format=duration -show_entries stream=codec_type \
    -of default=noprint_wrappers=1 /tmp/opencode/smoke/smoke.mp4
codec_type=video
codec_type=audio
duration=6.000000
```

Expected answer: six seconds, two streams.

## Smoke run

```
$ opencode run --model opencode/deepseek-v4.1-flash-free --auto \
    "Call media_probe on /tmp/opencode/smoke/smoke.mp4 and report its duration
     and stream count. Do not call any other media tool."

media_probe({"source": "/tmp/opencode/smoke/smoke.mp4"})

Probe complete for `/tmp/opencode/smoke/smoke.mp4`.

| Property | Value |
|---|---|
| Duration | 00:00:06.000 (6.000 s) |
| Streams | 2 total (video=yes, audio=yes) |
| Resolution | 1280x720 @ 25.000 fps |
| Codecs | h264 video, aac audio (mono) |
| Container | mp4 (mov,mp4,m4a,3gp,3g2,mj2) |
| A/V offset | +0.000 s (both streams start at 0.000 s) |
| Size | 1,212,619 bytes |

Stream count detail: one video stream and one audio stream; no subtitle or data
streams were detected. Only `media_probe` was called, as instructed.
```

The run itself is the evidence: the model resolved `media_probe` through the
MCP catalog and returned the fixture's real duration and stream count, matching
ffprobe's independent reading. Nothing here is transcribed from a cached value:
`--auto` was passed because a non-interactive run cannot answer a permission
prompt, and the prompt restricts the call to the read-only probe.

## Notes

- The commit for this task carries `docs/opencode-smoke.md` only. The
  `opencode.jsonc` edit lives outside the repository and stays uncommitted, as
  does the profile entry the approval wrote.
- `--model opencode/deepseek-v4.1-flash-free` was passed explicitly; the CLI
  does not need a model flag, but naming it makes the run reproducible.
