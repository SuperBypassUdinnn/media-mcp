# Beat Character & Color Motion Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `media-mcp` with frequency-band beat classification & downbeat tracking (`media_beats`) and chromatic delta / color-sweep motion detection (`media_motion`), resolving analytical gaps for automated editing agents.

**Architecture:** Additive updates to `media/media.py`, `rhythm/onsets.py`, and `media/server.py`. Audio rhythm decomposes spectral flux into low (<150Hz), mid (150-2500Hz), and high (>2500Hz) bands with 4/4 downbeat inference in both librosa and pure-numpy engines. Visual motion upgrades to decimated RGB24 decoding to measure chromatic delta and p95 localized color shifts, classifying `"color_sweep"` events and preventing false holds.

**Tech Stack:** Python 3.12 / 3.14, FFmpeg (`libx264`, `rawvideo`, `s16le`), NumPy, Librosa 1.0.0, Pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-11-19-beat-character-and-color-motion-design.md`

## Global Constraints

- Python 3.14 for `media/.venv`, Python 3.12 for `rhythm/.venv`.
- All modifications are strictly additive and backward-compatible with existing consumers of `beats.json` and `motion.json`.
- Offline unit tests only under `tests/`; synthetic media fixtures generated on-the-fly with known ground truth.
- Code quality gates: 0 errors in `ruff check .` and `ruff format --check .`.
- CodeGraph sync after modifying codebase symbols.

## Review Focus

1. **Audio with no detectable beats or silence**: `beat_details` must be empty tuple `()`, not raise an error.
2. **Audio shorter than a single bar (e.g. < 4 beats)**: `is_downbeat` and `bar_index` must safely degrade without index errors.
3. **Monochromatic or flat-color video**: `p95_chroma` and `chroma_diff` must stay near 0.0 without divide-by-zero or NaN in hue shifts.
4. **Video with intense luma bursts AND color sweeps**: both `burst` and `color_sweep` should coexist cleanly or burst takes precedence for keyframing.
5. **Decoded raster memory bounded**: scratch file stream pattern must be preserved; memory never holds more than two frames concurrently.

---

### Task 1: Beat Character & Multi-Band Rhythm Analysis in Core Engines

**Files:**
- Modify: `media/media.py:1220-1500`
- Modify: `rhythm/onsets.py:1-105`
- Test: `tests/test_beats.py`

**Interfaces:**
- Produces: `BeatDetail(t: float, strength: float, band: str, is_downbeat: bool, bar_index: int, beat_in_bar: int, phrase_energy: float)`, `BeatResult.beat_details: tuple[BeatDetail, ...]`

- [ ] **Step 1: Write failing tests for beat character, bands, and downbeats**

Add tests to `tests/test_beats.py`:
- `test_beat_details_frequency_bands`: Synthetic WAV with alternating 80 Hz and 5000 Hz clicks; asserts `band == "low"` on even beats and `"high"` on odd beats.
- `test_downbeat_detection_4_4`: 16-beat click track with accentuated beat 1; asserts `is_downbeat == True` on beats 0, 4, 8, 12, and `beat_in_bar` in `1..4`.
- `test_numpy_fallback_emits_beat_details`: Fallback engine produces valid `beat_details` with `degraded=True`.

- [ ] **Step 2: Run tests to verify failure**

Run: `media/.venv/bin/python -m pytest tests/test_beats.py -k "test_beat_details" -v`
Expected: FAIL (`AttributeError: 'BeatResult' object has no attribute 'beat_details'`)

- [ ] **Step 3: Implement `BeatDetail` dataclass and multi-band analysis in `rhythm/onsets.py` and `media/media.py`**

- In `media/media.py`: define `BeatDetail`, update `BeatResult` dataclass to include `beat_details: tuple[BeatDetail, ...] = ()`.
- In `rhythm/onsets.py`: compute multi-band onset strength envelopes using `librosa.onset.onset_strength_multi` for `<150 Hz` (low), `150-2500 Hz` (mid), and `>2500 Hz` (high). Compute per-beat `strength`, dominant `band`, 4/4 downbeat alignment phase, `bar_index`, `beat_in_bar`, `is_downbeat`, and `phrase_energy` (8-beat moving average). Return in payload `beat_details`.
- In `media/media.py:detect_beats`: parse `beat_details` from librosa output. In pure-numpy fallback, split STFT spectrum into 3 frequency bands (`0..7`, `7..117`, `117..513`), compute band flux, downbeat phase, and `beat_details`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `media/.venv/bin/python -m pytest tests/test_beats.py -v`
Expected: PASS (all tests pass)

- [ ] **Step 5: Commit**

```bash
git add media/media.py rhythm/onsets.py tests/test_beats.py
git commit -m "feat(beats): add multi-band character, downbeat tracking, and beat_details"
```

---

### Task 2: Server Layer & Artifact Serialization for `media_beats`

**Files:**
- Modify: `media/server.py:2750-2825`
- Test: `tests/test_beats.py`
- Test: `tests/test_stdio_contract.py`

**Interfaces:**
- Consumes: `BeatResult.beat_details` from Task 1.
- Produces: `beats.json` artifact containing `"beat_details"`, updated `media_beats` text summary.

- [ ] **Step 1: Write failing test for `beats.json` artifact structure**

In `tests/test_beats.py:test_media_beats_tool`:
Assert `payload["beat_details"]` is a list of dicts with keys `{"t", "strength", "band", "is_downbeat", "bar_index", "beat_in_bar", "phrase_energy"}`.

- [ ] **Step 2: Run test to verify failure**

Run: `media/.venv/bin/python -m pytest tests/test_beats.py::test_media_beats_tool -v`
Expected: FAIL (`KeyError: 'beat_details'`)

- [ ] **Step 3: Update `_beats_work` in `media/server.py`**

- In `_beats_work`: serialize `result.beat_details` into `entry / "beats.json"`.
- In output summary: add downbeat count and band breakdown (e.g. `Character: 8 low (downbeat), 12 mid, 16 high`).
- Update `_TOOL_DESCRIPTIONS["media_beats"]` to document `beat_details`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `media/.venv/bin/python -m pytest tests/test_beats.py tests/test_stdio_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add media/server.py tests/test_beats.py
git commit -m "feat(server): serialize beat_details and spectral breakdown in media_beats"
```

---

### Task 3: Decimated RGB24 Frame Decoding & Chromatic Metrics in `motion_curve`

**Files:**
- Modify: `media/media.py:845-1035`
- Test: `tests/test_motion.py`

**Interfaces:**
- Produces: `MotionCurvePoint(t, diff, hist_dist, chroma_diff: float, p95_chroma: float, hue_shift: float)`

- [ ] **Step 1: Write failing tests for chromatic signals in `motion_curve`**

Add tests to `tests/test_motion.py`:
- `test_motion_curve_chromatic_delta`: Synthetic video with blue background transitioning to red with identical luma; asserts `point.chroma_diff > 20.0` while `point.diff < 2.0`.
- `test_motion_curve_p95_chroma_local_sweep`: Synthetic video with static dark frame and a small 5% text patch changing color; asserts `p95_chroma > 15.0` while global `chroma_diff < 2.0`.
- `test_motion_curve_monochromatic_zero`: Synthetic black/white/gray video; asserts `chroma_diff == 0.0` and `p95_chroma == 0.0`.

- [ ] **Step 2: Run tests to verify failure**

Run: `media/.venv/bin/python -m pytest tests/test_motion.py -k "test_motion_curve_chromatic" -v`
Expected: FAIL (`TypeError: MotionCurvePoint.__init__() missing required argument 'chroma_diff'` or similar)

- [ ] **Step 3: Update `MotionCurvePoint` and `motion_curve` implementation in `media/media.py`**

- Update `MotionCurvePoint` with `chroma_diff: float`, `p95_chroma: float`, `hue_shift: float`.
- Update `motion_curve`: change ffmpeg `-vf` from `format=gray` to `format=rgb24`.
- Frame buffer read size: `frame_bytes = width * scaled_height * 3`.
- Compute luma diff via Rec.601 integer/float arithmetic ($Y = 0.299R + 0.587G + 0.114B$).
- Compute per-pixel Euclidean chroma delta $\Delta C = \sqrt{\Delta U^2 + \Delta V^2}$ (where $U, V$ are derived from RGB).
- Compute `chroma_diff = float(np.mean(delta_C))`, `p95_chroma = float(np.percentile(delta_C, 95))`.
- Compute `hue_shift` circular difference on saturated pixels ($S > 30$).
- Calculate 1D EMD `_motion_hist_dist` on the luma channel to preserve transition detection.

- [ ] **Step 4: Run tests to verify they pass**

Run: `media/.venv/bin/python -m pytest tests/test_motion.py -k "test_motion_curve" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add media/media.py tests/test_motion.py
git commit -m "feat(motion): add RGB24 decoding, chroma delta, and p95_chroma to motion_curve"
```

---

### Task 4: Color Sweep Event Classification & Hold Disambiguation in `motion_events`

**Files:**
- Modify: `media/media.py:1050-1200`
- Test: `tests/test_motion.py`

**Interfaces:**
- Consumes: `MotionCurvePoint(t, diff, hist_dist, chroma_diff, p95_chroma, hue_shift)` from Task 3.
- Produces: `MotionEvent(kind="color_sweep", ...)`

- [ ] **Step 1: Write failing test for `color_sweep` event and hold disambiguation**

Add tests to `tests/test_motion.py`:
- `test_motion_events_color_sweep_detected`: Curve with luma `diff = 0.2` (below hold threshold) but `p95_chroma = 25.0` for 1.0s; asserts event `color_sweep` is emitted with detail containing peak chroma delta.
- `test_motion_events_hold_suppressed_by_color_sweep`: Asserts that spans overlapping active `color_sweep` do NOT emit `hold` events.

- [ ] **Step 2: Run test to verify failure**

Run: `media/.venv/bin/python -m pytest tests/test_motion.py -k "color_sweep" -v`
Expected: FAIL (`AssertionError: 'color_sweep' not in kinds` and false hold detected)

- [ ] **Step 3: Update `motion_events` in `media/media.py`**

- Define `color_sweep` run mask: `(p95_chroma >= 8.0) | (chroma_diff >= 1.5)`.
- Extract `color_sweep` runs with min duration `0.3` seconds and 2-frame gap bridge.
- In hold extraction: filter out any hold whose temporal span overlaps any `color_sweep` span by $> 0.1$ seconds.

- [ ] **Step 4: Run tests to verify they pass**

Run: `media/.venv/bin/python -m pytest tests/test_motion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add media/media.py tests/test_motion.py
git commit -m "feat(motion): classify color_sweep events and suppress false holds"
```

---

### Task 5: Server Layer Artifacts & TSV Serialization for `media_motion`

**Files:**
- Modify: `media/server.py:2650-2750`
- Test: `tests/test_motion.py`

**Interfaces:**
- Consumes: `MotionCurvePoint` with color fields from Task 3, `MotionEvent(kind="color_sweep")` from Task 4.
- Produces: `motion.json` with color summary, `motion-curve.tsv` with `chroma_diff` and `p95_chroma` columns.

- [ ] **Step 1: Write failing test for `motion.json` color summary and TSV columns**

In `tests/test_motion.py:test_media_motion_tool`:
- Assert `motion.json` payload contains `summary["color_peak"]`, `summary["color_mean"]`, `summary["p95_chroma_peak"]`.
- Assert `motion-curve.tsv` header is `t\tdiff\thist_dist\tchroma_diff\tp95_chroma`.

- [ ] **Step 2: Run test to verify failure**

Run: `media/.venv/bin/python -m pytest tests/test_motion.py::test_media_motion_tool -v`
Expected: FAIL (`KeyError: 'color_peak'` or header mismatch)

- [ ] **Step 3: Update `_motion_work` in `media/server.py`**

- In `_motion_work`: compute `color_diffs = [p.chroma_diff for p in points]` and `p95_diffs = [p.p95_chroma for p in points]`.
- Add `"color_peak"`, `"color_mean"`, `"p95_chroma_peak"` to `summary` dictionary in `motion.json`.
- Update `motion-curve.tsv` header and row format to:
  `f"{p.t:.3f}\t{p.diff}\t{p.hist_dist}\t{p.chroma_diff}\t{p.p95_chroma}\n"`
- Update `_TOOL_DESCRIPTIONS["media_motion"]` to document chromatic delta and color sweeps.

- [ ] **Step 4: Run tests to verify they pass**

Run: `media/.venv/bin/python -m pytest tests/test_motion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add media/server.py tests/test_motion.py
git commit -m "feat(server): serialize color motion summary and TSV chroma columns"
```

---

### Task 6: Real-World Ground-Truth Verification, Full Suite & CodeGraph Sync

**Files:**
- Verify: `/home/superbypassudin/Downloads/Video Testing/Promotional ⁄ Ad/Best SaaS Marketing Video | Promotional Brand Video | Supahub [aAvDI1qae-U].mp4`
- Verify: `/home/superbypassudin/Downloads/Video Testing/AMV/The Introverts - Amv Edit | Talia (Slowed) [80plX-fV9xI].mp4`
- Test: `tests/test_beats.py`, `tests/test_motion.py`, `tests/test_avsync.py`, full suite

- [ ] **Step 1: Execute validation against Supahub promo reference video**

Run script to verify:
1. `media_motion` on Supahub detects `color_sweep` events in the first 4 seconds (where kinetic typography changes color) and does NOT emit false `hold` events.
2. `media_beats` on Supahub outputs 144 beats with `beat_details` having non-zero `strength`, valid `band` in `{"low", "mid", "high"}`, downbeats on beat 1 of bars, and dynamic `phrase_energy`.

- [ ] **Step 2: Execute validation against Talia AMV reference video**

Run script to verify:
1. `media_beats` on Talia detects tempo and downbeat impacts aligned with bass drops.
2. `media_avsync` successfully correlates visual cuts and burst starts against downbeats.

- [ ] **Step 3: Run full pytest suite, ruff linter, and format checks**

Run:
```bash
media/.venv/bin/python -m pytest -q
media/.venv/bin/python -m ruff check .
media/.venv/bin/python -m ruff format --check .
```
Expected: All 290+ tests PASS, 0 linter errors, format clean.

- [ ] **Step 4: Synchronize CodeGraph index**

Run:
```bash
codegraph sync ~/.clone/Github/SuperBypassUdinnn/media-mcp
```
Expected: CodeGraph successfully synchronized.

- [ ] **Step 5: Commit any validation notes or fixtures if applicable**

```bash
git status --porcelain
# Commit only if clean deliverable changes were added
```

