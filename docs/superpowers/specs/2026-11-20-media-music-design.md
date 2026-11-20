# Design: `media_music` — Musicality, Harmonization & Vocal Analysis

Date: 2026-11-20
Status: accepted — brainstorming consensus 2026-11-20 (opt-in vocal ladder; dedicated
`rhythm/music_analyze.py` script); handoff
`~/Downloads/2026-11-20-media-mcp-media-music-handoff.md` is the origin document
Target: media-mcp main repo (new MCP tool `media_music` alongside `media_beats`)

---

## 1. Problem

`media_beats` measures *when* energy happens (tempo, beats, onsets) but nothing
about *musical content*. During the NovaPulse Promo v2 audio review the final mix
"sounded like stacked sound effects, not music", and no tool could detect or
quantify why:

- The BGM itself is real music: `cad36ebe-bgm-online-6.wav` measures harmonic
  ratio 0.902 (HPSS) with key center A (chroma peak 0.987).
- The SFX layer is the problem: 43 cues in 13.65 s (3.15 cues/s), unpitched
  impact/crystal samples whose tonal tails ignore the music's key, plus
  `sidechaincompress` pumping the bed on every transient.
- Character, not just tonality, matters: NovaPulse's picked bed measured onset
  density 2.70/s and flux 0.92 — the most drifting/ambient track in the library
  — while the Supahub reference bed measures density 4.82/s, flux 1.143, and
  low-band 0.507. Harmonic ratio alone cannot distinguish an ambient pad wash
  from an upbeat song.

## 2. Public Contract

New MCP tool, additive and backward-compatible:

```
media_music(source, sfx_manifest=None, with_vocals=False, identity="sampled") -> str
```

Writes `music.json` to the source's cache entry (same durability pattern as
`beats.json`); the response names the artifact path. `sfx_manifest` is a path
to a promo-render-meta / cues JSON (cue list with file paths + optional gains);
without it the tool scores the source track only.

```jsonc
{
  "engine": "librosa",           // "librosa" via rhythm/music.sh, "numpy" fallback
  "degraded": false,             // true when the numpy baseline ran
  "duration": 56.7,
  "key": "A minor",              // Krumhansl-Schmuckler over mean chroma (HPSS harmonic part)
  "key_confidence": 0.71,
  "mode": "minor",               // "major" | "minor" | null
  "harmonic_ratio": 0.902,       // sum(hp^2)/(sum(hp^2)+sum(lp^2)); <0.4 percussion/SFX loop, >0.55 real music
  "transient_density": 2.73,     // onsets per second
  "onset_density": 2.7,          // alias measured over the full decode (character profile)
  "flux": 0.92,                  // mean spectral flux (movement)
  "low_band_ratio": 0.51,        // 40-160 Hz energy share (groove/kick presence)
  "mood_tags": ["upbeat"],       // rule-based from density/flux/low/mode thresholds
  "arrangement": [               // energy sections across the FULL track
    {"start": 0.0, "end": 8.2, "level": "low", "label": "intro"},
    {"start": 8.2, "end": 21.0, "level": "high", "label": "body"}
  ],
  "loopiness": 0.18,             // chroma self-similarity over time; high = one-bar loop
  // mix-scoring block, present only when sfx_manifest given:
  "sfx_density": 3.15,           // cues per second; gate <= 1.2
  "duck_pump_rate": 3.1,         // audible sidechain pumps per second; gate <= 2 per scene
  "key_clash_score": 0.34,       // SFX tonal content vs music key; gate <= 0.25
  // vocal block, present only when with_vocals=true:
  "vocal_presence": 0.62,        // 0..1 energy share of the vocal stem
  "vocal_type": "vocalization",  // "instrumental" | "vocalization" | "lyrics"
  "vocal_timeline": [[0.0, 3.2], [9.1, 12.0]],  // vocal-active seconds (VO mixing)
  "transcript_sample": "wow, wow, wow",
  "language": null,              // detected language when lyrics exist
  "vocal_instrumental_possible": false,
  "source": "/path/to/media.wav"
}
```

Engine decision (brainstorming 2026-11-20): the vocal ladder is **opt-in** via
`with_vocals=True` — the default call runs the fast music core only, so survey
calls never pay 610 MB Roformer inference plus Whisper. Critique consumers pass
`with_vocals=True` when they need the vocal verdict.

## 3. Music Core Analysis (librosa engine: `rhythm/music_analyze.py`)

New headless script in the existing librosa virtualenv, invoked through a new
wrapper `rhythm/music.sh` and the env override `MEDIA_MUSIC_BIN` (exact mirror
of the `rhythm.sh` / `MEDIA_RHYTHM_BIN` pattern). `rhythm/onsets.py` stays
rhythm-focused. The script reads the media path (and optional flags) from argv,
prints one JSON object to stdout, exits non-zero on any failure so the server
falls back to the numpy baseline.

Mono decode through one ffmpeg pass to 22050 Hz s16le wav (same as onsets.py),
then in librosa:

1. **HPSS**: `librosa.decompose.hpss(stft)` → harmonic / percussive components.
   `harmonic_ratio = sum(hp^2) / (sum(hp^2) + sum(lp^2))` over the full track.
2. **Key detection**: `librosa.feature.chroma_cqt(y=harmonic)` → mean chroma
   vector → correlate against the 12 Krumhansl-Kessler major and 12 minor
   profiles → best key (e.g. "A minor"), `key_confidence` = normalized peak
   correlation, `mode` = "major" | "minor".
3. **Transient density**: `librosa.onset.onset_detect(units="time")` count /
   duration (proven in probes).
4. **Character profile**: `onset_density` (onsets/s), `flux` (mean spectral
   flux), `low_band_ratio` (energy share 40-160 Hz of the STFT magnitude).
5. **Mood tags** (rule-based, ordered): density >= 4.0 and flux >= 1.0 and
   low_band_ratio in [0.3, 0.8] → "upbeat"; mode "major" → "cheerful"; mode
   "minor" → "moody"; density < 2.0 → "ambient"; empty list when nothing fires.
6. **Arrangement**: RMS energy over 1-second windows across the full track;
   k-means-free threshold segmentation into "low" / "mid" / "high" levels
   (below 60% / between / above 130% of the track median), labeled "intro" for
   the first low section, "body" for high sections, "breakdown" for low
   sections after a high one.
7. **Loopiness**: cosine similarity of successive 1-second chroma vectors
   against the 1-bar-lagged vector; `loopiness` = mean similarity. High (~1)
   means one-bar loop, low means arranged song.

## 4. Music Core Analysis (numpy fallback in `media/media.py`)

When `MEDIA_MUSIC_BIN` is missing or the script fails, the pure-numpy baseline
runs with `degraded=true` (same contract as `detect_beats`):

- Reuses the existing STFT machinery (`_BEAT_SAMPLE_RATE`, frame 1024, hop 256).
- `harmonic_ratio` via median-filter HPSS approximation on the magnitude
  spectrogram (harmonic = median-filtered along time, percussive = along
  frequency, as in Fitzgerald's original HPSS paper).
- Key detection: magnitude spectrogram folded to 12 pitch classes (bin center
  frequency → nearest semitone), correlated against the same KS profiles —
  coarser than CQT, flagged degraded.
- `transient_density` / `onset_density` from the existing onset picker;
  `flux` from the existing spectral flux; `low_band_ratio` from bin slices;
  arrangement + mood tags identical rules; `loopiness` from the same pitch-class
  vectors. Mix-scoring and vocal blocks follow the same numpy paths.

## 5. Mix Scoring (requires `sfx_manifest`)

1. **`sfx_density`**: cue count / span covered by the manifest cues. Gate <= 1.2.
2. **`duck_pump_rate`**: band-pass 200-2000 Hz (STFT slice), `rms` envelope
   (0.1 s windows), count local minima below 60% of the adjacent local maxima
   at >= 0.5 Hz → pumps per second. Gate <= 2 per scene.
3. **`key_clash_score`**: for each cue file in the manifest, HPSS the cue, take
   its chroma peak pitch-class; distance from the music key's triad
   (root, +3/4, +7 semitones) normalized to dissonance weight; average across
   cues weighted by cue gain (default 1.0 when the manifest omits gains).
   Gate <= 0.25. Cue decode errors count as clash-neutral and are reported in
   a `skipped` note.

## 6. Vocal Detection Ladder (opt-in: `with_vocals=True`)

Three rungs, graceful degradation, each state honestly reported:

### Rung 1 — Roformer stem separation (preferred)
- The user runs Ultimate Vocal Remover with SOTA Roformer checkpoints staged at
  `~/.clone/Github/SuperBypassUdinnn/ultimatevocalremovergui/`:
  - `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (610 MB) — community-reference
    vocals extractor, the default separation model.
  - `dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt` (196 MB) — optional dereverb
    pass on the vocals stem (improves Whisper transcript quality).
  - `melband_roformer_inst_v2.ckpt` (1.5 GB) and
    `model_BandSplit-Roformer_SW_by-jarredou.ckpt` (6 stems) — out of scope for
    v1; the runner is model-agnostic so later passes are additive.
- Model definitions live in the same repo at
  `lib_v5/roformer_native/{bs_roformer,mel_band_roformer}.py` (standalone MIT
  code); configs at `models/MDX_Net_Models/model_data/mdx_c_configs/*.yaml`
  (`chunk_size: 588800`, `dim_t: 801`); a proven working venv with
  torch 2.12.0+cu130 / CUDA available at `ultimatevocalremovergui/.venv/`.
- media-mcp ships a thin headless runner `separators/roformer_runner.py` — it
  imports the `roformer_native` model defs (UVR repo root injected via
  `sys.path`), loads the config yaml + checkpoint, runs chunked inference with
  the UVR venv interpreter, and writes stems to WAV. media-mcp invokes it
  through the subprocess env var `MEDIA_SEPARATE_BIN` (mirroring
  `MEDIA_RHYTHM_BIN`); the runner checks free VRAM before GPU use and falls
  back to CPU. Do NOT import UVR's `separate.py` (GUI-coupled).
- `vocal_presence = rms(vocals_stem)^2 / rms(mix)^2`.
- `vocal_timeline`: active-vocal spans where the vocals-stem RMS stays above
  30% of its own 90th percentile for >= 0.3 s (bridged).

### Rung 2 — pyin fallback (no torch / no MEDIA_SEPARATE_BIN)
- Band-pass 150-1000 Hz on the mix, `librosa.pyin` for F0 contours.
- Vocals show continuous confident F0 with 4-7 Hz vibrato-like modulation and
  formant-band energy; score contour coverage + modulation against pure
  instrument references. `vocal_presence` derived from confident-coverage
  share; `vocal_timeline` from covered spans.

### Rung 3 — vocal_type decision (Whisper on the vocals stem)
When `vocal_presence >= 0.15`, run the existing `media_transcript` whisper
backend **on the vocals stem, not the mix** (much better SNR). Hallucination
guards for instrumentals: drop segments with `no_speech_prob > 0.6` or
`avg_logprob < -1.0`; require `condition_on_previous_text=False`. Then:
- Unique confident real-word tokens >= 5 and not interjection-dominated
  -> `lyrics` (fill `language`, `transcript_sample` with top confident tokens).
- Tokens present but dominated by interjections (wow, oh, yeah, ah, la, na,
  hey, ooh, uh) or high repetition ratio -> `vocalization`.
- Nothing confident -> `instrumental`; when `vocal_presence` was high
  (>= 0.3) also set `vocal_instrumental_possible: true` for review.

## 7. Server Layer

Registration mirrors `media_beats`: `_guarded`, `_entry_for`, identity notes,
artifact path named in the response. `_music_work` orchestrates: music core
(engines per sections 3-4) -> mix scoring when `sfx_manifest` given -> vocal
ladder when `with_vocals=true` -> `_write_json(entry / "music.json", ...)`.
Tool description documents the gates (sfx_density <= 1.2, duck_pump <= 2/scene,
key_clash <= 0.25) and the vocal ladder. `tests/test_stdio_contract.py`
`TOOL_NAMES` gains `media_music`.

## 8. Verification

1. Unit tests offline in `tests/test_music.py` (patterns of
   `test_beats.py`/`test_motion.py`): synthetic click track (assert transient
   density + harmonic ratio only — synthetic clicks have no key), a synthetic
   tonal bed (assert key detection on a synthesized major chord progression),
   a mocked SFX manifest for clash scoring, monochromatic/percussion-only
   degenerate cases, and degraded-fallback coverage.
2. Reference validation:
   - Supahub / SberBank (`Video Testing/Promotional ⁄ Ad/`) — expected PASS.
   - NovaPulse deliverable — expected FAIL `sfx_density 3.15 > 1.2`.
   - Supahub bed character: density 4.82/s, flux 1.143, low-band 0.507,
     harmonic 0.820, key G major.
3. Gates: `pytest -q` full suite, `ruff check .`, `ruff format --check .`,
   `codegraph sync`.

## 9. Implementation Stages

1. Stage 1 — `rhythm/music_analyze.py` + `rhythm/music.sh` (music core, librosa
   engine) + numpy fallback in `media/media.py` + `MusicResult` dataclass +
   `detect_music()` + server tool `media_music` (music block only) + tests.
2. Stage 2 — mix-scoring block (manifest parsing, sfx_density, duck_pump_rate,
   key_clash_score) + tests.
3. Stage 3 — vocal ladder: `separators/roformer_runner.py` + `MEDIA_SEPARATE_BIN`
   wiring + pyin fallback + Whisper-on-stem vocal_type + tests (mocked runner
   for CI; real-model smoke verification locally).
4. Stage 4 — full-suite gate, reference validation, submodule note for
   media-studio.

