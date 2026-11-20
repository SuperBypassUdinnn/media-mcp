# media_music Implementation Plan (Musicality, Mix Scoring & Vocal Analysis)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `media_music` to media-mcp: key/mode detection (Krumhansl-Schmuckler), HPSS harmonic ratio, transient density, character profile (density/flux/low-band/mood tags), arrangement, loopiness, mix scoring against an SFX manifest (sfx_density, duck_pump_rate, key_clash_score), and an opt-in vocal ladder (Roformer stem separation -> pyin fallback -> Whisper-on-stem vocal_type).

**Architecture:** New headless librosa script `rhythm/music_analyze.py` invoked via `rhythm/music.sh` + `MEDIA_MUSIC_BIN` (exact mirror of the rhythm pattern); pure-numpy degraded fallback in `media/media.py` reusing the existing STFT machinery; server tool `media_music` orchestrating music core -> mix scoring (when `sfx_manifest`) -> vocal ladder (when `with_vocals=True`), writing `music.json` to the source's cache entry. Roformer separation ships as `separators/roformer_runner.py` executed with the UVR venv interpreter via `MEDIA_SEPARATE_BIN`.

**Tech Stack:** Python 3.14 (media/.venv), Python 3.12 (rhythm/.venv, librosa 1.0.0), UVR venv (torch 2.12.0+cu130) for separation, FFmpeg, NumPy, Pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-11-20-media-music-design.md`

## Global Constraints

- Strictly additive: `beats.json`, `motion.json` consumers unaffected; new tool only.
- Gates pinned verbatim: sfx_density <= 1.2; duck_pump_rate <= 2 per scene; key_clash_score <= 0.25; harmonic_ratio < 0.4 suggests percussion/SFX, > 0.55 real music.
- Character thresholds pinned: density >= 4.0 and flux >= 1.0 and low_band_ratio in [0.3, 0.8] -> "upbeat"; mode "major" -> "cheerful"; "minor" -> "moody"; density < 2.0 -> "ambient".
- Vocal ladder is opt-in (`with_vocals=False` default); opt-in runs Roformer -> pyin -> Whisper-on-stem with hallucination guards (`no_speech_prob > 0.6`, `avg_logprob < -1.0`, `condition_on_previous_text=False`).
- Offline unit tests only under `tests/`; external engines exercised via mock/fake wrapper scripts (pattern: `tests/fake_transcribe.sh`).
- Gates: 0 errors in `ruff check .` and `ruff format --check .`; full `pytest -q` green; `codegraph sync` after symbol changes.

## Review Focus

1. **Silence / digital-zero audio**: `music_analyze.py` and the numpy fallback must return a valid payload (key null / confidence null, harmonic_ratio 0.0, empty arrangement) without NaN in JSON.
2. **Percussion-only input (no tonal content)**: key detection must return null rather than a confident wrong key; `key_clash_score` skips null-key cues.
3. **Malformed SFX manifest** (missing file, bad JSON, cue without path): tool must raise a named MediaError, never crash with a traceback.
4. **MEDIA_SEPARATE_BIN points to missing/broken runner**: `with_vocals=True` degrades to the pyin rung with `degraded` note, never a silent wrong vocal_type.
5. **Degraded numpy key vs librosa key**: both engines must fill the same contract fields (null-vs-value allowed, field set identical).

---

### Task 1: Librosa music-core engine (`rhythm/music_analyze.py` + `rhythm/music.sh`)

**Files:**
- Create: `rhythm/music_analyze.py`
- Create: `rhythm/music.sh` (executable, mirrors `rhythm/rhythm.sh` incl. one-time setup bootstrap)
- Test: `tests/test_music.py`

**Interfaces:**
- Produces: `music.sh <media-path>` printing one JSON object on stdout:
  `{engine:"librosa", degraded:false, duration, key, key_confidence, mode, harmonic_ratio, transient_density, onset_density, flux, low_band_ratio, mood_tags:[], arrangement:[{start,end,level,label}], loopiness}`.
  Non-zero exit / bad stdout = engine unavailable.

- [ ] **Step 1: Write failing tests** in `tests/test_music.py`

  - `test_music_engine_tonal_bed_key_detection`: synthesize a 6 s C-major triad pad (sines C4 261.63 Hz + E4 329.63 Hz + G4 392.00 Hz at equal amplitude, slow attack, sr 22050 wav); run `rhythm/music.sh`; assert `key == "C major"` (or at minimum `mode == "major"` and key root in {"C"}), `key_confidence >= 0.3`, `harmonic_ratio >= 0.55`.
  - `test_music_engine_transient_density`: 120 BPM click track (reuse a click fixture builder); assert `transient_density` and `onset_density` within [1.6, 2.4] onsets/s.
  - `test_music_engine_silence_payload_valid`: 3 s digital-silence wav; assert valid JSON, `key is None`, `harmonic_ratio == 0.0`, `mood_tags == []`, no NaN anywhere in the payload.
  - `test_music_engine_arrangement_low_high`: click track followed by 4 s of silence appended; assert `arrangement` has >= 2 sections and the first low section starts at 0.0.

- [ ] **Step 2: Run tests to verify failure**

  Run: `media/.venv/bin/python -m pytest tests/test_music.py -v`
  Expected: FAIL (`FileNotFoundError` on `rhythm/music.sh`)

- [ ] **Step 3: Implement `rhythm/music_analyze.py` and `rhythm/music.sh`**

  - `music_analyze.py`: argv = media path; ffmpeg mono 22050 s16le wav to a temp dir (same decode block as `onsets.py`); `librosa.decompose.hpss` on the STFT for harmonic/percussive parts; `harmonic_ratio` over squared magnitudes; `chroma_cqt(y=harmonic)` -> mean chroma -> correlate against the 12 Krumhansl-Kessler major + 12 minor profiles (correlation, not dot product) -> key/mode/confidence (null key when no chroma energy); onsets via `onset_detect(units="time")` -> `transient_density` and `onset_density`; `flux` = mean positive spectral flux; `low_band_ratio` = 40-160 Hz magnitude energy share; mood tags per the pinned rules; arrangement from 1 s RMS windows thresholded at 60%/130% of the track median (labels: first low = "intro", high = "body", later low = "breakdown"); `loopiness` = mean cosine similarity between each 1 s chroma vector and the 1-bar-lagged vector.
  - `music.sh`: copy `rhythm/rhythm.sh` verbatim, rename pipeline paths to `music_analyze.py`.

- [ ] **Step 4: Run tests to verify they pass**

  Run: `media/.venv/bin/python -m pytest tests/test_music.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**

  ```bash
  git add rhythm/music_analyze.py rhythm/music.sh tests/test_music.py
  git commit -m "feat(music): add librosa music-core engine with key, HPSS, character, and arrangement"
  ```

---

### Task 2: `MusicResult` dataclass + `detect_music` orchestration with numpy fallback

**Files:**
- Modify: `media/media.py` (rhythm section neighbourhood, after `detect_beats`)
- Test: `tests/test_music.py`

**Interfaces:**
- Consumes: `music.sh` JSON contract from Task 1.
- Produces: `MusicResult(engine: str, degraded: bool, duration: float, key: str|None, key_confidence: float|None, mode: str|None, harmonic_ratio: float, transient_density: float, onset_density: float, flux: float, low_band_ratio: float, mood_tags: tuple[str,...], arrangement: tuple[dict,...], loopiness: float, mix: dict|None, vocals: dict|None)` and
  `detect_music(path: Path, *, wrapper: Path|None = None, scratch_dir: Path|None = None, cancel: threading.Event|None = None) -> MusicResult`
  (wrapper pattern identical to `detect_beats`; fallback engine `"numpy"`, `degraded=True`).

- [ ] **Step 1: Write failing tests**

  - `test_detect_music_librosa_engine`: call `detect_music` on the C-major bed with `wrapper=<repo>/rhythm/music.sh`; assert `engine == "librosa"`, `degraded is False`, `key == "C major"`, all numeric fields finite.
  - `test_detect_music_numpy_fallback`: call with `wrapper=Path("/nonexistent")` on the click track; assert `engine == "numpy"`, `degraded is True`, contract fields all present (key may be None), `harmonic_ratio` in [0,1], `mood_tags` a tuple.

- [ ] **Step 2: Run tests to verify failure**

  Run: `media/.venv/bin/python -m pytest tests/test_music.py -k detect_music -v`
  Expected: FAIL (`AttributeError: module 'media.media' has no attribute 'detect_music'`)

- [ ] **Step 3: Implement in `media/media.py`**

  - `MusicResult` frozen dataclass (fields above; `mix`/`vocals` default None, filled by later stages).
  - `detect_music`: wrapper parse identical to `detect_beats` (json.loads, guard `ValueError/TypeError/AttributeError`, fall through to baseline); numpy baseline computes harmonic_ratio via median-filter HPSS approximation on the magnitude spectrogram (harmonic = median along time, percussive = median along frequency, Fitzgerald), key via 12-pitch-class fold of bin-center frequencies correlated against the same KS profiles (null when no chroma energy), the rest of the contract from the existing STFT/onset machinery.

- [ ] **Step 4: Run tests to verify they pass**

  Run: `media/.venv/bin/python -m pytest tests/test_music.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**

  ```bash
  git add media/media.py tests/test_music.py
  git commit -m "feat(music): add MusicResult and detect_music with numpy degraded fallback"
  ```

---

### Task 3: MCP tool `media_music` + `music.json` artifact

**Files:**
- Modify: `media/server.py` (new `media_music`/`_music_work` near `_beats_work`; `_TOOL_DESCRIPTIONS`; `build_server` tuple)
- Modify: `tests/test_stdio_contract.py` (`TOOL_NAMES` gains `"media_music"`)
- Test: `tests/test_music.py`

**Interfaces:**
- Consumes: `detect_music` from Task 2.
- Produces: `media_music(source, sfx_manifest=None, with_vocals=False, identity="sampled") -> str`; artifact `entry/music.json` with the section-2 contract fields (mix/vocal blocks absent in this task); response names the artifact path.

- [ ] **Step 1: Write failing tests**

  - `test_media_music_tool_artifact`: monkeypatch `MEDIA_CACHE_DIR` + `MEDIA_MUSIC_BIN=<repo>/rhythm/music.sh`; call `media.server.media_music(click_track)`; assert `# media_music` header, `music.json` exists with engine/key/harmonic_ratio fields, response names the artifact path.
  - `test_media_music_missing_manifest_file`: `sfx_manifest=<nonexistent>.json` raises ToolError naming the manifest path (assert via `pytest.raises(ToolError, match="manifest")` — the mix block is Task 4, but path validation lands here).

- [ ] **Step 2: Run tests to verify failure**

  Run: `media/.venv/bin/python -m pytest tests/test_music.py -k test_media_music -v`
  Expected: FAIL (`AttributeError: module 'media.server' has no attribute 'media_music'`)

- [ ] **Step 3: Implement in `media/server.py`**

  - `_MUSIC_WRAPPER = Path(__file__).resolve().parent.parent / "rhythm" / "music.sh"`; env override `MEDIA_MUSIC_BIN` (mirror `MEDIA_RHYTHM_BIN`).
  - `_music_work(settings, source, sfx_manifest, with_vocals, identity, cancel)`: `_entry_for` + `_record_identity_note`; validate `sfx_manifest` path exists and parses as JSON (named `MediaError` otherwise; full scoring in Task 4); `media.detect_music(path, wrapper=..., scratch_dir=entry, cancel=cancel)`; `_write_json(entry / "music.json", payload)`; `_record_entry`; text summary (engine line, key/mode + confidence, harmonic ratio with the <0.4 / >0.55 interpretation, transient density, character line, mood tags, arrangement table, artifact path, report.status()).
  - `_TOOL_DESCRIPTIONS["media_music"]`: musicality + character + mix-scoring + opt-in vocal ladder description with the pinned gates.
  - `build_server` tuple gains `media_music`; `TOOL_NAMES` gains `"media_music"`.

- [ ] **Step 4: Run tests to verify they pass**

  Run: `media/.venv/bin/python -m pytest tests/test_music.py tests/test_stdio_contract.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**

  ```bash
  git add media/server.py tests/test_music.py tests/test_stdio_contract.py
  git commit -m "feat(server): register media_music with music.json artifact"
  ```

---

### Task 4: Mix scoring block (sfx_density, duck_pump_rate, key_clash_score)

**Files:**
- Modify: `media/media.py` (mix-scoring pure functions near `detect_music`)
- Modify: `media/server.py` (`_music_work` wire-up)
- Test: `tests/test_music.py`

**Interfaces:**
- Consumes: `MusicResult` + `sfx_manifest` JSON (cues list with `path` and optional `gain`).
- Produces: `mix_score_music(result, manifest, pcm_path, scratch_dir, cancel) -> dict` with keys `sfx_density` (cues/s; gate <= 1.2), `duck_pump_rate` (pumps/s; gate <= 2/scene), `key_clash_score` (0..1; gate <= 0.25); `_music_work` writes them into `music.json["mix"]`-level top-level fields and the text summary when a manifest was given.

- [ ] **Step 1: Write failing tests**

  - `test_sfx_density_computation`: manifest with 4 cues spanning 2.0 s -> `sfx_density == 2.0`.
  - `test_duck_pump_rate_detection`: synthesize a wav whose mid-band (400 Hz sine bed) amplitude oscillates 0 dB / -12 dB at 2 Hz (sidechain-style); assert `duck_pump_rate` in [1.5, 2.5].
  - `test_key_clash_score_triad_vs_dissonant`: music key "C major" (reuse the C-major bed via wrapper); manifest cue = synthesized C-major-triad cue -> `key_clash_score < 0.25`; second manifest cue = tritone-heavy cue (C + F#) -> higher score than the triad cue; assert cue decode-error counts as clash-neutral and is reported.
  - `test_mix_block_in_music_json`: tool-level test with manifest; assert `music.json` carries `sfx_density`, `duck_pump_rate`, `key_clash_score` and the response shows gate verdict lines.

- [ ] **Step 2: Run tests to verify failure**

  Run: `media/.venv/bin/python -m pytest tests/test_music.py -k "sfx_density or duck or clash" -v`
  Expected: FAIL (functions missing)

- [ ] **Step 3: Implement mix scoring in `media/media.py` + wire in `media/server.py`**

  - Manifest parse: `{"cues": [{"path": str, "start": float, "gain": float?}, ...]}` or a bare list of cue dicts; each cue `path` must exist (named MediaError otherwise).
  - `sfx_density = len(cues) / max(span, 0.1)` where span = max(cue.end/start+dur) - min(cue.start), falling back to duration.
  - Duck-pump: decode mix PCM (reuse `_decode_mono_pcm`); band-pass 200-2000 Hz via STFT slice + istft (or rfft mask); 0.1 s RMS envelope; count local minima < 60% of adjacent local maxima at >= 0.5 Hz -> pumps/s.
  - Key clash: per cue, HPSS the cue's STFT, mean chroma -> peak pitch-class; distance to the key triad (root, +3/4, +7) with circle-of-fifths dissonance weights; average weighted by `gain` (default 1.0); null-key music -> `key_clash_score = None` with a note; cue decode errors are clash-neutral + reported.

- [ ] **Step 4: Run tests to verify they pass**

  Run: `media/.venv/bin/python -m pytest tests/test_music.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**

  ```bash
  git add media/media.py media/server.py tests/test_music.py
  git commit -m "feat(music): add mix scoring with sfx_density, duck_pump_rate, and key_clash_score"
  ```

---

### Task 5: Vocal ladder (Roformer runner + pyin fallback + Whisper-on-stem)

**Files:**
- Create: `separators/roformer_runner.py`
- Modify: `media/media.py` (vocal ladder functions after the mix-scoring block)
- Modify: `media/server.py` (`_music_work` wire-up)
- Create: `tests/fake_separate.sh`
- Test: `tests/test_music.py`

**Interfaces:**
- Consumes: `with_vocals=True`, `MEDIA_SEPARATE_BIN` env var (UVR-venv runner).
- Produces: `vocal_analysis(pcm_path, *, separate_bin, scratch_dir, cancel, transcript_backend) -> dict` with keys `vocal_presence` (0..1), `vocal_type` ("instrumental"|"vocalization"|"lyrics"), `vocal_timeline` ([[start,end],...]), `transcript_sample` (str|None), `language` (str|None), `vocal_instrumental_possible` (bool), `degraded_rung` ("roformer"|"pyin"); `_music_work` writes them top-level into `music.json` and the text summary when `with_vocals=True`.

- [ ] **Step 1: Write failing tests**

  - `test_vocal_ladder_roformer_rung_via_fake`: `tests/fake_separate.sh` writes a vocals-stem wav (synthesized 300 Hz warbling tone = "vocal") + instrumental wav to the dir the runner contract names; set `MEDIA_SEPARATE_BIN` to it; assert `vocal_presence > 0.15`, `vocal_timeline` non-empty, `degraded_rung == "roformer"`; vocal_type via the mocked transcript backend returning interjections -> `"vocalization"`.
  - `test_vocal_ladder_pyin_fallback`: `MEDIA_SEPARATE_BIN=/nonexistent`; input wav with a continuous 220 Hz confident-F0 tone; assert rung degrades to `"pyin"`, `vocal_presence` computed, no crash.
  - `test_vocal_type_lyrics_vs_instrumental`: mocked transcript backend returning >= 5 unique real-word tokens -> `"lyrics"` + language filled; returning nothing confident on high vocal_presence -> `"instrumental"` + `vocal_instrumental_possible=True`.
  - `test_roformer_runner_smoke_skipped_when_no_venv`: real-runner test auto-skips when the UVR venv or the vocals checkpoint is absent (CI-safe).

- [ ] **Step 2: Run tests to verify failure**

  Run: `media/.venv/bin/python -m pytest tests/test_music.py -k vocal -v`
  Expected: FAIL (functions missing)

- [ ] **Step 3: Implement**

  - `separators/roformer_runner.py`: argv = `<media.wav> <out_dir> [--model <name>] [--stems vocals,instrumental]`; imports `roformer_native.bs_roformer` (UVR repo root from `UVR_ROOT` env or default `~/.clone/Github/SuperBypassUdinnn/ultimatevocalremovergui`), loads `models/MDX_Net_Models/model_data/mdx_c_configs/*.yaml` matching the checkpoint, chunked inference (`chunk_size: 588800`, `dim_t: 801`), free-VRAM check (`torch.cuda.mem_get_info`) with CPU fallback, writes `<out_dir>/vocals.wav` + `<out_dir>/instrumental.wav`; exits non-zero with stderr on failure. Executed with the UVR venv interpreter via `MEDIA_SEPARATE_BIN` (the env var points at a wrapper script, mirroring `MEDIA_RHYTHM_BIN`).
  - `media/media.py` vocal ladder: rung 1 invokes `MEDIA_SEPARATE_BIN <wav> <scratch_dir>` (missing/broken -> rung 2); `vocal_presence = rms(vocals)^2/rms(mix)^2`; `vocal_timeline` from vocals-RMS > 30% of its own 90th percentile, spans >= 0.3 s bridged; rung 2 = band-pass 150-1000 Hz + `librosa.pyin` is NOT available in media/.venv -> implement pyin via autocorrelation F0 on the band-passed PCM (confidence from peak prominence; coverage + 4-7 Hz modulation score); rung 3 = `vocal_type` decision tree with the pinned hallucination guards and interjection list (wow, oh, yeah, ah, la, na, hey, ooh, uh), transcript via the existing transcribe backend on the vocals stem (`condition_on_previous_text=False`).
  - `media/server.py`: wire `with_vocals` through `_music_work`; text summary gains `Vocals: <presence> <type> (rung)` + timeline preview + transcript sample.

- [ ] **Step 4: Run tests to verify they pass**

  Run: `media/.venv/bin/python -m pytest tests/test_music.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**

  ```bash
  git add separators/roformer_runner.py media/media.py media/server.py tests/fake_separate.sh tests/test_music.py
  git commit -m "feat(music): add opt-in vocal ladder with roformer separation and whisper-on-stem"
  ```

---

### Task 6: Full suite gate, reference validation & sync

**Files:**
- Verify: reference videos + NovaPulse deliverable
- Test: full suite

- [ ] **Step 1: Reference validation script** (workspace script, not repo tree)

  - Supahub + SberBank: `media_music` via `MEDIA_MUSIC_BIN` -> key/mode/harmonic/character fields valid; Supahub bed matches handoff character (density ~4.8/s, flux ~1.14, low-band ~0.51, key G major within tolerance).
  - NovaPulse deliverable (`~/Projects/novapulse-promo-v2` mastered mix + cue manifest): expect FAIL `sfx_density 3.15 > 1.2` and `duck_pump_rate` > gate; record measured numbers.
  - Vocal smoke (local only, auto-skip if UVR venv missing): `with_vocals=True` on the Supahub bed -> `vocal_type` in the three-state set; report measured `vocal_presence`.

- [ ] **Step 2: Full gates**

  Run: `media/.venv/bin/python -m pytest -q && media/.venv/bin/python -m ruff check . && media/.venv/bin/python -m ruff format --check .`
  Expected: all green (300+ tests).

- [ ] **Step 3: CodeGraph sync**

  Run: `codegraph sync /home/superbypassudin/.clone/Github/SuperBypassUdinnn/media-mcp`
  Expected: synchronized.

- [ ] **Step 4: Record results in the task report** (verification only; commit only if clean deliverable files were produced)

