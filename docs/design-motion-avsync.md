# Design: Motion Analysis and Audio-Visual Sync for media-mcp

Date: 2026-11-06
Status: accepted — grill-me session 2026-11-06 settled the open decisions;
the verdicts are recorded in section 9 and implementation follows them
Scope: three new analysis layers (motion, rhythm, alignment) on top of the
existing probe/scene/audio-event/frames machinery, their tool surface, and the
artifact files they write. No new external dependencies.

## 1. Problem

Two workflows are out of reach today:

- **Motion-graphics video** (the visual is the content): animated text,
  shapes, and gradual transitions. The agent can OCR individual frames and
  see hard cuts, but cannot tell what moves, when motion starts and stops, or
  that a 1.5-second crossfade between two scenes happened at all —
  `scdet` fires on hard cuts and stays silent through a slow blend.
- **AMV-style edits** (visual cut to an audio track): the agent can list
  scene events and loudness peaks separately, but cannot answer "are the cuts
  on the beat?", "which frame shows at beat 32?", or "extract the frames at
  every onset". There is no onset/beat timeline and no cross-modal alignment.

Both reduce to the same missing primitive: **time-aligned curves** for visual
motion and audio energy, plus a correlation step between them.

## 2. Current capability inventory (measured, from code)

What exists and what it measures:

| Layer | Source | Measures | Gap for the target workflows |
|---|---|---|---|
| `probe` | ffprobe | resolution, fps, duration, codecs, `av_offset_seconds` from stream `start_time` | offset known but unused downstream |
| `detect_scenes` | ffmpeg `scdet` | hard-cut times + scores (threshold 10.0) | blind to gradual transitions, blends, morphs |
| `detect_audio_events` | ffmpeg `silencedetect` + `astats` (0.2 s frames) | silence segments; loud peaks = frames above median + margin | no onsets, no tempo, no beat grid |
| `sample_candidates` / `rank_candidates` | phash | interval + scene-key frames, near-duplicate collapse, budget ranking | cannot be driven by motion or beats |
| `media_frames` | ffmpeg | stills at timestamps / scenes | `at` accepts only explicit times or `scenes` |
| `media_ocr`, `media_transcript` | tesseract, whisper | per-frame text; speech transcript | none — orthogonal |

Everything else this design needs is already on the machine: `ffmpeg` decode
pipes, `numpy` (via the opencv/brisque chain), and `cv2` (optional optical
flow). The transcription venv is untouched.

## 3. Layer M — motion analysis (`motion_curve`)

One ffmpeg decode pass at a reduced analysis rate, read as raw grayscale
through a pipe — no per-frame PNG cost, nothing large stored:

```
ffmpeg -i <src> -vf fps=<analysis_fps>,scale=<width>:-2,format=gray \
       -f rawvideo -   # read with numpy.frombuffer
```

Defaults (overridable per call): `analysis_fps=10`, `width=320`.

Per analysis frame, computed in-process with numpy/cv2:

- **frame-diff energy**: mean absolute difference against the previous
  analysis frame (the cheap, robust motion signal);
- **histogram earth mover's distance**: the mean absolute shift of the two
  frames' cumulative 64-bin grayscale histograms. Measured amendment
  (2026-11-06): the drafted Pearson correlation was rejected — flat-color
  fades keep consecutive histograms shape-correlated through the blend, so
  the signal alternated between "nothing changed" and "total
  displacement". EMD reports the honest few-bins-per-frame shift, and
  motion graphics are full of exactly such flat-color scenes; single-frame
  holes from frame-rate conversion are bridged (2 analysis frames).
- **optical flow magnitude** (optional, off by default): Farneback via the
  already-present `cv2`, computed only every Nth frame pair to bound cost.

Derived events from the energy curve:

- **motion bursts**: contiguous spans where energy clears `k` times the
  rolling median (window ~5 s) — "the animation happens here";
- **static holds**: spans near zero energy — safe keyframe points;
- **gradual transitions**: sustained moderate frame-diff (at least 0.5
  levels — content actually changes) with sustained histogram travel over
  >= 0.4 s — the crossfade/wipe signal `scdet` cannot see;
- **keyframes**: local energy minima inside high-motion segments.

Artifacts: `motion.json` (events + curve summary) in the source's cache
entry; the full curve as a compact tsv. Pure functions live in
`media/media.py`; the server layer renders text and writes artifacts.

## 4. Layer A — rhythm analysis (`onset_curve`)

**Verified 2026-11-06 on this machine**: librosa 1.0.0 installs and runs on
Python 3.14.9 — numba 0.68.1 now ships official cp314 manylinux wheels, so
the early-2026 incompatibility (numba 0.62/0.63 lacked cp314) is gone. A
functional probe on a synthetic 120 BPM click track measured tempo 117.5 BPM
and exactly 19 onsets with the first at 0.511 s. The "librosa needs a Python
downgrade" premise is obsolete.

That reopens the implementation choice for rhythm:

- **A1 — librosa in an auxiliary venv** (recommended): a small pinned env
  next to the source (the `transcribe/.venv` precedent, Python 3.12.16
  there, called through a thin wrapper) or even directly in the main venv.
  Gives `onset_detect`, DP-based `beat_track`, and `tempo` estimates of
  production quality, and multi-band onset envelopes.
- **A2 — pure numpy in-process baseline**: ffmpeg decodes mono PCM
  (`-ac 1 -ar 22050 -f s16le -`), numpy computes an STFT spectral-flux
  onset envelope, onsets as median + k*MAD peaks, tempo by autocorrelation,
  beat grid by phase-locked comb. No new dependency, but the beat grid is
  materially weaker than librosa's DP tracker.
- **A1 with A2 fallback** (the degradation pattern BRISQUE already uses):
  rhythm runs through librosa when its environment is present, degrades to
  A2 with a stated `degraded: true` field when it is not.

Adding librosa to the main pinned venv costs the manifest roughly
numba + llvmlite + scipy + scikit-learn + soundfile + soxr + pooch
(hundreds of MB) and a CI rebuild; an auxiliary venv keeps the 33-pin
manifest and CI untouched. This is open decision D2.

Artifacts: `beats.json` (onsets, tempo, beat grid, confidence,
`degraded` flag).

## 5. Layer S — audio-visual alignment (`av_alignment`)

Pure post-processing over artifacts the previous layers already wrote, plus
the A/V offset from `probe`:

- **cut-onset alignment**: for each `scdet` cut (and each motion-burst
  start), the distance to the nearest audio onset; report median, 90th
  percentile, and the share of cuts within a ±0.1 s tolerance. Compare
  against a shuffled-onsets baseline so a verdict of "synced" is measured,
  not vibes. (Implementation: burst starts join the cut list, deduplicated
  against scdet cuts within 0.25 s; a missing motion.json skips them with
  a stated note. The offset algebra moves each onset to container time —
  `onset + offset` — which is the spec's "subtract from cuts" under the
  measured `audio.start_time - video.start_time` sign convention.)
- **beat-phase of cuts**: circular mean of each cut's position within the
  beat grid (fraction of beat interval); near 0.0/1.0 means cuts land on
  beats.
- **A/V offset correction**: all comparisons subtract the measured
  `av_offset_seconds` first, so a container-level offset never masquerades
  as bad editing.

Artifacts: `sync.json` (metrics + verdict lines).

## 6. Tool surface

Three tools, one job each, all writing file artifacts (the durability
pattern `media_image_sort` now uses):

| Tool | Reads | Writes | Returns |
|---|---|---|---|
| `media_motion` | source | `motion.json` + curve tsv | events, burst/hold/transition summary, artifact path |
| `media_beats` | source | `beats.json` | onsets, tempo, beat grid, artifact path |
| `media_avsync` | cached artifacts + offset | `sync.json` | alignment metrics + verdict |

Plus one convenience hook: `media_frames(at="beats")` reuses the existing
timestamp-list machinery to rasterize frames on the beat grid (requires a
cached `beats.json`; a missing one returns an error that names
`media_beats`, not a silent fallback).

`media_manifest` integration (visual profile gains motion events; full gains
beats + sync) is deliberately deferred until the three tools settle —
option C in open decision D1.

## 7. Cost and testing

Cost: one decimated decode pass for motion (~10 fps at 320 px wide ≈
0.6 MB/s streamed, never stored), one mono PCM decode for rhythm, zero
decodes for alignment. No layer touches the transcription venv or the GPU.

Testing mirrors the repo's existing style — real ffmpeg over synthetic
sources with known ground truth: `testsrc` animations for burst/transition
timing, click tracks at known BPM for onset/beat assertions, a deliberately
desynced cut list to assert the alignment metrics detect the desync, and a
degradation test proving A2 fallback reports `degraded: true` when the
librosa environment is absent.

## 8. Open decisions for grill-me

- **D1 — tool surface**: three tools (recommended), one mode-parameter tool,
  or manifest-profile integration instead.
- **D2 — rhythm engine**: auxiliary librosa venv with pure-numpy fallback
  (recommended), librosa in the main pinned venv, or pure numpy only.
- **D3 — optical flow in v1**: include (better direction/magnitude story)
  or defer (cost) — default off in either case.
- **D4 — motion curve storage**: full curve as tsv sidecar (recommended)
  vs JSON-embedded vs npz.
- **D5 — defaults**: `analysis_fps=10`, `width=320`, burst `k=3`,
  transition >= 0.4 s, alignment tolerance ±0.1 s — confirm or adjust.
- **D6 — manifest integration timing**: defer (recommended) or fold into
  profiles now.

## 9. Decisions (grill-me, 2026-11-06)

| Decision | Verdict |
|---|---|
| D1 tool surface | Three separate tools: `media_motion`, `media_beats`, `media_avsync` |
| D2 rhythm engine | Auxiliary librosa venv + pure-numpy fallback with `degraded: true` |
| D2a aux venv interpreter | Python 3.12 (matches the `transcribe/.venv` precedent) |
| D2b aux venv location | In-repo `rhythm/` — pinned `rhythm/requirements.txt` + wrapper, the `transcribe/` pattern; no secrets, no `.env` |
| D3 optical flow | Deferred to v2; frame-diff + histogram carry v1 |
| D4 curve storage | `motion.json` (events + summary) plus a `motion-curve.tsv` sidecar in the same entry |
| D5 defaults | Accepted as proposed: `analysis_fps=10`, `width=320`, burst `k=3`, transition >= 0.4 s, sync tolerance ±0.1 s |
| D6 manifest integration | Deferred until the three tools are stable |
| `media_frames(at="beats")` without cached beats | Error naming `media_beats`; no implicit decode |
| Push 3 pending commits | Pushed to origin/main on 2026-11-06 |

Implementation notes from the 2026-11-06 review pass:

- **Confidence** (section 4): beat-onset agreement — the share of beats
  with an onset within 70 ms — computed identically in both engines, so
  the two environments stay self-contained.
- **`MEDIA_RHYTHM_BIN`**: a server-side override seam (test + operator),
  mirroring the transcription pipeline's `MEDIA_TRANSCRIBE_BIN`; the
  wrapper itself deliberately ignores it.
- **Burst floor 1.0 level** (section 3): keeps a nearly static clip's
  encoding noise from flagging itself; the rolling median alone is too
  generous on silent sources.
- **`rhythm/setup.sh` interpreter fallback** (D2a): python3.11/python3
  mirrors `transcribe/setup.sh`; CI pins 3.12, the binding path.


