# Inline image finding

Task 16 asks one question: when a tool result carries MCP image content, does a
model actually receive the pixels, or only the path? The design states the
mechanism in its decisions table (section 3, the frame-delivery row) and restates
it in the tool table (section 5): paths by default, inline behind
`image_inline`. This document records the measurement that decides whether the
flag is implemented.

Date: 2026-10-22. Host: opencode. Result: positive for a vision model, so
`image_inline` now exists on `media_frames`, default off.

## Method

The probe is a throwaway stdio MCP server, never committed, kept at
`~/.local/state/opencode-probe/image-inline-probe.py` (the Task 16 resolutions
place the harness under a temporary path rather than the plan's
`tests/probe_image_inline.py`). It exposes one tool, `image_probe`, whose result
is a text block naming the PNG path — the path-first route — followed by an
`ImageContent` block for that same PNG, the inline route.

The PNG holds three coloured squares (red, green, blue) whose left-to-right
order is randomised on every call and written to `image-inline-probe.order`. The
text block never states the order, so a model that sees only the path can do no
better than a one-in-six guess, and the answer can be compared with the recorded
ground truth.

Shape checks before the client run:

```text
$ ~/.clone/Github/media-mcp/media/.venv/bin/python image-inline-probe.py --selftest
order: ['Blue', 'Red', 'Green']
png: /home/superbypassudin/.local/state/opencode-probe/image-inline-probe.png bytes=650
image block: type=image mime=image/png b64=868
selftest: PASS

$ ffprobe -v error -select_streams v -show_entries stream=width,height,codec_name -of default=nw=1 image-inline-probe.png
codec_name=png
width=360
height=120
```

## Result: opencode, vision model

`image-probe` was registered in `~/.config/opencode/opencode.jsonc` and approved
for this project through the `mcp-profiles` gate, then driven through the real
client with the same prompt used for every attempt: call `image_probe` and name
the coloured squares left to right, using no other tool.

```text
$ opencode run --model nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b --auto "<prompt>"
> build · nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b
execute {"code":"return await tools[\"image-probe\"][\"image_probe\"]();"}
The image shows three coloured squares arranged side by side. In left-to-right order:

1. Blue
2. Red
3. Green

$ cat ~/.local/state/opencode-probe/image-inline-probe.order
Blue Red Green
```

The answer matches the ground truth exactly, so the model received the pixels:
the client passed the MCP `ImageContent` block through to the provider. The
running ANSI colour codes and the leading tool glyph were stripped from that
transcript; the words are verbatim.

Two attempts are recorded for completeness:

- 2026-10-21, model `opencode/nemotron-3.5-super-vl-free`: answered
  `Red, Green, Blue` for a randomised image. That run predates the ground-truth
  sidecar, so a one-in-six guess cannot be excluded; it is supportive, not
  conclusive.
- 2026-10-22, the same free model again: the client reported
  `Error: Free usage limit reached. Connect a provider or subscribe to OpenCode
  Go to keep going, or switch models.` The conclusive run above therefore used
  the provider variant of the same model family, which is the same client path.

## Decision

Positive for a host and model means the flag is implemented (Task 16
resolutions), with these properties:

- `media_frames(image_inline=true)` attaches the PNGs it just wrote, one
  `ImageContent` block per frame, in addition to the unchanged text result.
- Default is off: the response stays path-first and text-only unless the caller
  asks otherwise, because support depends on the model, not only on the client.
  A text-only model produces a false negative, so the caller owns the choice.
- At most 8 PNGs are attached per call (`INLINE_IMAGE_CAP`). The frame budget
  allows 60, and inlining 60 images is not a defensible default; when more
  frames were written than the cap allows, the status block reports
  `capped: inline: attached N of M PNGs (inline cap 8)`.
- A PNG that is missing from disk, or cannot be read, degrades to its path in
  the text with the reason named in the `skipped` line, and never raises.
- Profiles are unaffected: `visual` and `full` still return paths. The parameter
  is read by `media_frames` alone (`_frames_work` has one call site and
  `_visual_stage` takes no inline argument), confirmed over a live stdio
  session where `media_manifest(profile="visual")` returned `blocks=['text']`
  and `media_frames(image_inline=true)` returned `blocks=['text', 'image']`.

## Antigravity

Not measured, and no longer in scope. The Task 14 UI check failed on
2026-10-21 — the client reported its customization budget exceeded and did not
list the `media` server — and the user dropped Antigravity support the same day,
so the per-client record here covers opencode only. The Antigravity
configuration snippet is retained in `docs/client-config.md` as a starting point
for anyone who wants to retry it, not as a supported path.
