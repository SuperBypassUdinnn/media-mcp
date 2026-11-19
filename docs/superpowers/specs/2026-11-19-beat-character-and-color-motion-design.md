# Design: Beat Character & Color Motion Analysis in media-mcp

Date: 2026-11-19
Status: accepted (brainstorming consensus 2026-11-19)
Target Repository: `~/.clone/Github/SuperBypassUdinnn/media-mcp` (submodule `submodules/media-mcp`)
Scope: Audio beat character and frequency band decomposition (`media_beats`), chromatic delta and gradient motion tracking (`media_motion`), public data contract updates, and ground-truth validation.

---

## 1. Context and Problem Statement

`media-mcp` provides local multimodal context for video and audio analysis via MCP tools. During evaluation on promotional and AMV references (*Supahub*, *SberBank*, *Talia*, *Solo Leveling*), two structural analytical gaps were identified:

1. **Beat Character & Energy Deficiency (`media_beats`)**:
   - `detect_beats` returns a uniform timestamp grid (`beats[]`), raw onsets (`onsets[]`), `tempo`, and overall `confidence`.
   - It lacks per-beat strength, frequency-band character (low-end kick/sub-bass vs mid-range snare/vocal vs high-frequency shimmer/hat), downbeat bar alignment (`bar_index`, `beat_in_bar`), and phrase-level musical energy curves.
   - Consequently, automated editing agents cannot determine whether a given beat is a heavy impact warranting a scene transition or camera shake, or a gentle hi-hat subdivision suitable for secondary micro-animations.

2. **Color & Gradient Motion Blindspot (`media_motion`)**:
   - `motion_curve` decodes frames into 1-byte grayscale rasters (`format=gray`), calculating mean luma difference and 1D EMD histogram shifts.
   - In modern motion design, kinetic typography often animates text color word-by-word, sweeps multi-stop linear gradients, or modulates glow colors over dark or monochromatic backgrounds while global luma shifts remain negligible.
   - `motion_events` misclassifies these active chromatic animations as static `hold` spans because luma variation stays below threshold, blinding downstream planning agents to critical visual pacing.

## 2. Public Contract Updates (Additive and Backward-Compatible)

All modifications are strictly additive. Existing consumers expecting `beats`, `onsets`, `tempo`, and `confidence` in `beats.json`, or `diff` and `hist_dist` in `motion.json` and `motion-curve.tsv`, continue to operate without breaking.

### 2.1 Rhythm Contracts (`media_beats`)

#### `BeatDetail` and `BeatResult` Dataclasses
```python
@dataclass(frozen=True)
class BeatDetail:
    """Analytical breakdown of a single beat in the audio timeline."""
    t: float              # Beat timestamp in media seconds (e.g. 0.512)
    strength: float       # Normalized local onset energy (0.0 to 1.0)
    band: str             # Dominant spectral band: "low", "mid", or "high"
    is_downbeat: bool     # True if inferred as the first beat of a bar
    bar_index: int        # Zero-based bar index along the timeline
    beat_in_bar: int      # 1-based beat number within the bar (1..4 for 4/4 meter)
    phrase_energy: float  # Moving average musical energy around this beat (0.0 to 1.0)

@dataclass(frozen=True)
class BeatResult:
    engine: str
    degraded: bool
    duration: float
    tempo: float | None
    beats: tuple[float, ...]
    onsets: tuple[float, ...]
    confidence: float | None
    # Additive field:
    beat_details: tuple[BeatDetail, ...] = ()
```

#### `beats.json` Artifact Schema
```json
{
  "engine": "librosa",
  "degraded": false,
  "duration": 56.694,
  "tempo": 161.5,
  "confidence": 0.924,
  "beats": [0.139, 0.511, 0.882],
  "onsets": [0.139, 0.325, 0.511],
  "beat_details": [
    {
      "t": 0.139,
      "strength": 0.85,
      "band": "low",
      "is_downbeat": true,
      "bar_index": 0,
      "beat_in_bar": 1,
      "phrase_energy": 0.42
    },
    {
      "t": 0.511,
      "strength": 0.38,
      "band": "high",
      "is_downbeat": false,
      "bar_index": 0,
      "beat_in_bar": 2,
      "phrase_energy": 0.45
    }
  ],
  "source": "/path/to/media.mp4"
}
```

### 2.2 Visual Motion Contracts (`media_motion`)

#### `MotionCurvePoint` and `MotionEvent` Dataclasses
```python
@dataclass(frozen=True)
class MotionCurvePoint:
    t: float
    diff: float           # Mean absolute grayscale luma difference (0.0 to 255.0)
    hist_dist: float      # Grayscale 1D EMD histogram distance (0.0 to 1.0)
    chroma_diff: float    # Mean chromatic Euclidean distance across frame (0.0 to 255.0)
    p95_chroma: float     # 95th percentile chromatic delta (local element color sweep)
    hue_shift: float      # Mean circular hue shift on saturated pixels (0.0 to 180.0 deg)

@dataclass(frozen=True)
class MotionEvent:
    kind: str             # "burst", "hold", "transition", "keyframe", "color_sweep"
    start: float
    end: float
    detail: str
```

#### `motion.json` and `motion-curve.tsv` Artifact Schema
- `motion.json` summary object adds:
  `"color_peak"` (max `chroma_diff`), `"color_mean"` (average `chroma_diff`), and `"p95_chroma_peak"`.
- `motion-curve.tsv` columns:
  `t\tdiff\thist_dist\tchroma_diff\tp95_chroma\n`

## 3. Rhythm Analysis Architecture (`media_beats`)

Both the librosa engine in `rhythm/onsets.py` and the pure-numpy baseline in `media/media.py` are extended in parallel to preserve the degradation contract.

### 3.1 Spectral Band Definitions
Standard musical frequency bands:
- **Low Band (`< 150 Hz`)**: Sub-bass, bass guitar, and bass drum / kick impacts.
- **Mid Band (`150 Hz - 2500 Hz`)**: Snares, claps, vocal fundamental, and synth body.
- **High Band (`> 2500 Hz`)**: Hi-hats, cymbals, shimmer, and transient percussion.

### 3.2 Multi-Band Envelope Extraction

#### Librosa Engine (`rhythm/onsets.py`)
1. Compute multi-band onset strength envelopes via `librosa.onset.onset_strength_multi`:
   - FFT frequencies: `librosa.fft_frequencies(sr=sr, n_fft=2048)`.
   - Frequency slices: `low = 0..idx_150`, `mid = idx_150..idx_2500`, `high = idx_2500..len(freqs)`.
   - Result: 3-channel envelope array of shape `(3, time_frames)`.
2. For each detected beat timestamp $t$:
   - Sample envelope values across the three bands around time $t$.
   - Compute total strength: sum of band energies, scaled to $[0.0, 1.0]$ relative to the 98th percentile envelope magnitude.
   - Classify dominant `band`: the channel yielding the highest ratio above its respective median noise floor.

#### Pure-Numpy Baseline (`media/media.py`)
1. In `_beat_envelope`, the STFT `spectrum` computed with frame 1024 and hop 256 is split into three frequency bin slices:
   - Bin width $= 22050 / 1024 \approx 21.53$ Hz.
   - `low_slice = 0:7` ($0 - 150.7$ Hz).
   - `mid_slice = 7:117` ($150.7 - 2519.4$ Hz).
   - `high_slice = 117:513` ($2519.4 - 11025.0$ Hz).
2. Spectral flux is calculated per band by positive half-wave rectification of consecutive frames.
3. Beat strength and dominant band assignment mirror the librosa logic.

### 3.3 Downbeat and Musical Phrase Inference
Modern promotional videos and AMVs predominantly utilize 4/4 meter.
1. **Downbeat Phase Alignment**:
   - For a sequence of beats $b_0, b_1, \dots, b_{N-1}$ and hypothetical bar phase $p \in \{0, 1, 2, 3\}$:
   - Candidate downbeats are beats where $(i - p) \pmod 4 == 0$.
   - Calculate the downbeat score for each phase candidate:
     $$S(p) = \frac{\sum_{i \equiv p \pmod 4} \text{low\_strength}(b_i)}{\sum_{i \not\equiv p \pmod 4} \text{low\_strength}(b_i) + \epsilon}$$
   - The phase maximizing $S(p)$ is selected as the downbeat phase $p^*$.
2. **Bar Indexing**:
   - `bar_index = (i - p^*) // 4`.
   - `beat_in_bar = ((i - p^*) % 4) + 1` (values 1, 2, 3, 4).
   - `is_downbeat = (beat_in_bar == 1)`.
3. **Phrase Energy**:
   - A standard musical phrase spans 8 or 16 beats (2 to 4 bars).
   - `phrase_energy` is the moving window average of beat strengths over an 8-beat symmetric window, normalized to $[0.0, 1.0]$. This differentiates calm verse/intro sections from high-density chorus/drop sections.

---

## 4. Color & Chromatic Motion Architecture (`media_motion`)

### 4.1 Decimated RGB Frame Pipeline
1. `motion_curve` replaces `format=gray` with `format=rgb24`:
   `ffmpeg -i <src> -vf fps=<fps>,scale=<width>:-2,format=rgb24 -f rawvideo <scratch>`
2. Frame dimensions: `width x scaled_height x 3` bytes. For a 320x180 resolution at 10 fps, stream throughput is $\approx 1.73$ MB/s. Scratch files are deleted immediately upon curve completion.

### 4.2 Per-Frame Color Metrics
For consecutive RGB frames $F_{t}$ and $F_{t-1}$:
1. **Luma Difference (`diff`)**:
   Standard Rec.601 luma: $Y = 0.299 R + 0.587 G + 0.114 B$.
   $$\text{diff} = \frac{1}{HW} \sum |Y_{t} - Y_{t-1}|$$
2. **Grayscale EMD Distance (`hist_dist`)**:
   Preserved 64-bin cumulative distribution shift for transition detection.
3. **Chroma Delta (`chroma_diff`)**:
   Compute color differences in YUV / Lab color space:
   $$U = -0.147 R - 0.289 G + 0.436 B, \quad V = 0.615 R - 0.515 G - 0.100 B$$
   $$\Delta C(x, y) = \sqrt{(U_t - U_{t-1})^2 + (V_t - V_{t-1})^2}$$
   - `chroma_diff`: Mean value of $\Delta C$ across all pixels.
   - `p95_chroma`: 95th percentile value of $\Delta C$. This metric isolates localized color changes (such as colored kinetic typography or animated buttons covering $5\%$ of the frame) even when the remainder of the background is static.
4. **Dominant Hue Shift (`hue_shift`)**:
   On pixels with color saturation $S > 30$, compute circular hue difference $\Delta H \in [0^\circ, 180^\circ]$. Mean hue shift indicates overall palette rotation or lighting color grading sweeps.

### 4.3 Color Sweep Event Classification (`motion_events`)
1. **Event Trigger**:
   A candidate `color_sweep` span occurs when:
   - `p95_chroma >= 8.0` OR `chroma_diff >= 1.5`, lasting at least `0.3` seconds (using the existing 2-frame gap bridge).
2. **Hold Disambiguation**:
   A temporal span where `color_sweep` is active MUST NOT be emitted as a `hold`, even if luma `diff <= hold_max_diff`. This fixes the *Supahub* false-hold failure mode where animated typography was previously classified as static holding.

## 5. Testing and Verification Strategy

### 5.1 Offline Unit Tests (TDD)
1. **`tests/test_beats.py`**:
   - `test_beat_details_frequency_bands`: Synthesize test WAV audio with alternating 80 Hz sine tones (low kick) and 5000 Hz bursts (high hi-hat); assert `band == "low"` on even beats and `"high"` on odd beats.
   - `test_downbeat_detection_4_4`: Assert `is_downbeat` is true every 4th beat and `beat_in_bar` cycles `1, 2, 3, 4`.
   - `test_phrase_energy_dynamics`: Audio with quiet section followed by loud section verifies that `phrase_energy` increases dynamically.
   - `test_numpy_fallback_emits_beat_details`: Verify fallback engine produces valid `beat_details` with `degraded=True`.
2. **`tests/test_motion.py`**:
   - `test_color_sweep_detection`: Synthesize an MP4 where white text against a black background transitions from pure blue to pure yellow with identical average luma; assert `color_sweep` event is detected and NO `hold` is generated during the sweep.
   - `test_motion_curve_rgb_signals`: Verify `chroma_diff`, `p95_chroma`, and `hue_shift` values on synthetic color rotations.

### 5.2 Real-World Reference Validation
Verify on reference assets in `/home/superbypassudin/Downloads/Video Testing/`:
- **Supahub** (`Promotional ⁄ Ad/Best SaaS Marketing Video...`): Verify that the kinetic typography color animation during the opening 4 seconds is classified as `color_sweep` instead of 3 false `hold` events.
- **SberBank** (`Promotional ⁄ Ad/Video Markting...`): Verify UI color changes and button animations surface in `motion.json`.
- **AMV Talia & Solo Leveling** (`AMV/`): Verify downbeats coincide with major bass drops and musical cut points in `media_avsync`.

---

## 6. Implementation Stages

1. **Stage 1: Rhythm Engine Extension (`media/media.py` & `rhythm/onsets.py`)**
   - Add `BeatDetail` dataclass and update `BeatResult`.
   - Implement multi-band spectral decomposition, downbeat estimation, and phrase energy in `rhythm/onsets.py` and `media/media.py`.
   - Update `media/server.py` `_beats_work` to write `beat_details` to `beats.json` and tool output.
   - Add unit tests in `tests/test_beats.py`.
2. **Stage 2: Color Motion Engine Extension (`media/media.py` & `media/server.py`)**
   - Update `motion_curve` to decode RGB24 and calculate `chroma_diff`, `p95_chroma`, `hue_shift`.
   - Update `MotionCurvePoint` and `MotionEvent` dataclasses.
   - Implement `color_sweep` detection and hold-suppression in `motion_events`.
   - Update `media/server.py` `_motion_work` to write color summaries and expanded `motion-curve.tsv`.
   - Add unit tests in `tests/test_motion.py`.
3. **Stage 3: Integration, Real-World Verification & Submodule Sync**
   - Run full test suite offline (`pytest -q`).
   - Run linter gates (`ruff check .`, `ruff format --check .`).
   - Run CLI validation against reference videos in `Downloads/Video Testing/`.
   - Update submodule reference in `media-studio` if needed.

