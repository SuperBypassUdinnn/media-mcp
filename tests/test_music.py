"""Tests for the librosa music-core engine: ``rhythm/music.sh``.

The audio fixtures are synthesized numpy audio written through the stdlib
``wave`` module so they encode exactly once and decode identically everywhere:
- a 6 s C-major triad pad (sines at 261.63 / 329.63 / 392.00 Hz),
- a 10 s 120 BPM click track,
- 3 s of digital silence,
- a click track followed by 4 s of silence.

The librosa engine test runs the repository's ``rhythm/music.sh`` when its
virtualenv exists and skips the librosa-specific assertions when it does not.
"""

from __future__ import annotations

import asyncio
import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

import media.server as server_mod
from media import media as media_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
RATE = 22050
CLICK_SECONDS = 10.0


def _write_wav(path: Path, samples: list[float], rate: int = RATE) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(
            b"".join(
                int(max(-1.0, min(1.0, s)) * 32767).to_bytes(2, "little", signed=True)
                for s in samples
            )
        )
    return path


def _write_tonal_bed(path: Path) -> Path:
    """6 s C-major triad pad: C4/E4/G4 sines at equal amplitude, slow attack."""
    total = int(RATE * 6.0)
    samples = [0.0] * total
    for freq in (261.63, 329.63, 392.00):
        for index in range(total):
            t = index / RATE
            amp = min(1.0, t / 0.5)
            samples[index] += (1.0 / 3.0) * amp * math.sin(2 * math.pi * freq * t)
    return _write_wav(path, samples)


def _write_click_track(path: Path, bpm: float = 120.0) -> Path:
    """A mono 16-bit WAV: one decaying 1 kHz click per beat at `bpm`."""
    interval = 60.0 / bpm
    total = int(RATE * CLICK_SECONDS)
    samples = [0.0] * total
    click_len = 400
    for index in range(click_len):
        envelope = math.exp(-8.0 * index / click_len)
        samples[index] = 0.8 * envelope * math.sin(2 * math.pi * 1000 * index / RATE)
    beats = [
        int(beat * interval * RATE) for beat in range(int(CLICK_SECONDS / interval))
    ]
    pcm = [0.0] * total
    for start in beats:
        for index in range(click_len):
            if start + index < total:
                pcm[start + index] += samples[index]
    return _write_wav(path, pcm)


def _write_silence(path: Path, seconds: float = 3.0) -> Path:
    return _write_wav(path, [0.0] * int(RATE * seconds))


def _write_click_plus_silence(path: Path, bpm: float = 120.0) -> Path:
    """Click track for 5 s then 4 s of silence (9 s total)."""
    interval = 60.0 / bpm
    click_seconds = 5.0
    total = int(RATE * (click_seconds + 4.0))
    samples = [0.0] * total
    click_len = 400
    for index in range(click_len):
        envelope = math.exp(-8.0 * index / click_len)
        samples[index] = 0.8 * envelope * math.sin(2 * math.pi * 1000 * index / RATE)
    beats = [
        int(beat * interval * RATE) for beat in range(int(click_seconds / interval))
    ]
    pcm = [0.0] * total
    for start in beats:
        for index in range(click_len):
            if start + index < total:
                pcm[start + index] += samples[index]
    return _write_wav(path, pcm)


def _run_music_sh(wrapper: Path, media_path: Path) -> dict:
    import subprocess

    done = subprocess.run(
        [str(wrapper), str(media_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _no_na(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_no_na(v) for v in value.values())
    if isinstance(value, list):
        return all(_no_na(v) for v in value)
    return True


wrapper = REPO_ROOT / "rhythm" / "music.sh"
venv = REPO_ROOT / "rhythm" / ".venv" / "bin" / "python"


@pytest.mark.skipif(not venv.exists(), reason="rhythm/.venv not built")
def test_music_engine_tonal_bed_key_detection(tmp_path):
    path = _write_tonal_bed(tmp_path / "pad.wav")
    payload = _run_music_sh(wrapper, path)
    assert payload["engine"] == "librosa"
    assert payload["degraded"] is False
    assert payload["key"] == "C major" or (
        payload["mode"] == "major" and payload["key"] and payload["key"].startswith("C")
    ), payload
    assert 0.3 <= payload["key_confidence"] <= 1.0, payload
    assert payload["harmonic_ratio"] >= 0.55, payload


@pytest.mark.skipif(not venv.exists(), reason="rhythm/.venv not built")
def test_music_engine_transient_density(tmp_path):
    path = _write_click_track(tmp_path / "clicks.wav")
    payload = _run_music_sh(wrapper, path)
    assert 1.6 <= payload["transient_density"] <= 2.4, payload
    assert 1.6 <= payload["onset_density"] <= 2.4, payload


@pytest.mark.skipif(not venv.exists(), reason="rhythm/.venv not built")
def test_music_engine_silence_payload_valid(tmp_path):
    path = _write_silence(tmp_path / "silence.wav")
    payload = _run_music_sh(wrapper, path)
    assert payload["engine"] == "librosa"
    assert payload["degraded"] is False
    assert payload["key"] is None, payload
    assert payload["harmonic_ratio"] == 0.0, payload
    assert payload["mood_tags"] == [], payload
    assert _no_na(payload), payload


def _write_white_noise(path: Path, seconds: float = 2.0) -> Path:
    """2 s of band-limited white noise (no tonal content)."""
    rng = np.random.default_rng(0)
    total = int(RATE * seconds)
    samples = [float(s) for s in rng.standard_normal(total)]
    return _write_wav(path, samples)


@pytest.mark.skipif(not venv.exists(), reason="rhythm/.venv not built")
def test_music_engine_noise_key_confidence_low(tmp_path):
    path = _write_white_noise(tmp_path / "noise.wav")
    payload = _run_music_sh(wrapper, path)
    assert payload["engine"] == "librosa"
    assert payload["degraded"] is False
    # White noise has no tonal center: either no key, or a low confidence.
    assert payload["key"] is None or payload["key_confidence"] < 0.5, payload


@pytest.mark.skipif(not venv.exists(), reason="rhythm/.venv not built")
def test_music_engine_arrangement_low_high(tmp_path):
    path = _write_click_plus_silence(tmp_path / "clicks_then_silence.wav")
    payload = _run_music_sh(wrapper, path)
    arrangement = payload["arrangement"]
    assert len(arrangement) >= 2, payload
    assert arrangement[0]["start"] == 0.0, payload
    # The appended silence must be detected as a low section after the clicks.
    low_sections = [s for s in arrangement if s["level"] == "low"]
    assert low_sections, payload
    assert low_sections[-1]["end"] == round(payload["duration"], 3), payload


def test_detect_music_librosa_engine(tmp_path):
    path = _write_tonal_bed(tmp_path / "pad.wav")
    result = media_mod.detect_music(path, wrapper=wrapper)
    assert result.engine == "librosa"
    assert result.degraded is False
    assert result.key == "C major"
    for name in (
        "duration",
        "key_confidence",
        "harmonic_ratio",
        "transient_density",
        "onset_density",
        "flux",
        "low_band_ratio",
        "loopiness",
    ):
        value = getattr(result, name)
        assert value is not None and math.isfinite(value), (name, value)


def test_detect_music_numpy_fallback(tmp_path):
    path = _write_tonal_bed(tmp_path / "pad.wav")
    result = media_mod.detect_music(path, wrapper=Path("/nonexistent"))
    assert result.engine == "numpy"
    assert result.degraded is True
    # Later stages fill mix/vocals; the baseline leaves them unset.
    assert result.mix is None and result.vocals is None
    # A sustained sine pad is harmonic-dominated: pins the HPSS axis
    # orientation (harmonic = median along the time axis).
    assert result.harmonic_ratio > 0.5, result
    # The C-major triad fold must land on pitch class 0 = C: pins the
    # pitch-class mapping (A440 -> pitch class 9).
    assert result.key == "C major", result
    assert result.mode == "major", result
    assert result.key_confidence is not None and 0.0 <= result.key_confidence <= 1.0
    assert isinstance(result.mood_tags, tuple)
    assert isinstance(result.arrangement, tuple)
    for name in (
        "duration",
        "transient_density",
        "onset_density",
        "flux",
        "low_band_ratio",
        "loopiness",
    ):
        value = getattr(result, name)
        assert value is not None and math.isfinite(value), (name, value)


def test_media_music_tool_artifact(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    monkeypatch.setenv("MEDIA_MUSIC_BIN", str(wrapper))
    click_path = _write_click_track(tmp_path / "clicks.wav")
    text = asyncio.run(server_mod.media_music(str(click_path)))
    assert text.startswith("# media_music\n"), text
    music_jsons = list(cache.rglob("music.json"))
    assert len(music_jsons) == 1, music_jsons
    assert str(music_jsons[0]) in text, text
    assert "Artifact:" in text, text
    payload = json.loads(music_jsons[0].read_text(encoding="utf-8"))
    for field in (
        "engine",
        "key",
        "harmonic_ratio",
        "duration",
        "mood_tags",
        "arrangement",
    ):
        assert field in payload, payload


def test_media_music_missing_manifest_file(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    monkeypatch.setenv("MEDIA_MUSIC_BIN", str(wrapper))
    click_path = _write_click_track(tmp_path / "clicks.wav")
    missing = tmp_path / "does-not-exist.json"
    with pytest.raises(ToolError, match="manifest"):
        asyncio.run(server_mod.media_music(str(click_path), sfx_manifest=str(missing)))


def _write_triad_cue(path: Path) -> Path:
    """0.5 s C-major triad cue (C4/E4/G4)."""
    total = int(RATE * 0.5)
    samples = [0.0] * total
    for freq in (261.63, 329.63, 392.00):
        for index in range(total):
            t = index / RATE
            samples[index] += (1.0 / 3.0) * math.sin(2 * math.pi * freq * t)
    return _write_wav(path, samples)


def _write_tritone_cue(path: Path) -> Path:
    """0.5 s tritone cue (C4 + F#4)."""
    total = int(RATE * 0.5)
    samples = [0.0] * total
    for freq in (261.63, 369.99):
        for index in range(total):
            t = index / RATE
            samples[index] += 0.5 * math.sin(2 * math.pi * freq * t)
    return _write_wav(path, samples)


def _write_duck_bed(path: Path, seconds: float = 2.0) -> Path:
    """400 Hz sine bed amplitude-gated 0 dB / -12 dB at 2 Hz."""
    total = int(RATE * seconds)
    samples = [0.0] * total
    for index in range(total):
        t = index / RATE
        amp = 1.0 if (math.floor(2 * 2 * t) % 2 == 0) else 10 ** (-12 / 20)
        samples[index] = amp * 0.5 * math.sin(2 * math.pi * 400 * t)
    return _write_wav(path, samples)


def _write_music_bed(path: Path, seconds: float = 3.0) -> Path:
    """3 s C-major pad used as the music source for key-clash tests."""
    total = int(RATE * seconds)
    samples = [0.0] * total
    for freq in (261.63, 329.63, 392.00):
        for index in range(total):
            t = index / RATE
            samples[index] += (1.0 / 3.0) * 0.4 * math.sin(2 * math.pi * freq * t)
    return _write_wav(path, samples)


def test_sfx_density_computation(tmp_path):
    cue_paths = [_write_triad_cue(tmp_path / f"cue{i}.wav") for i in range(4)]
    manifest = {
        "cues": [
            {"path": str(p), "start": i * 0.5, "gain": 1.0}
            for i, p in enumerate(cue_paths)
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    music_bed = _write_music_bed(tmp_path / "music.wav")
    result = media_mod.detect_music(music_bed, wrapper=Path("/nonexistent"))
    assert result.key == "C major"
    scores = media_mod.mix_score_music(result, manifest, music_bed)
    assert scores["sfx_density"] == 2.0, scores
    assert scores["key_clash_score"] < 0.25, scores
    assert scores["skipped_cues"] == [], scores


def test_duck_pump_rate_detection(tmp_path):
    music_bed = _write_music_bed(tmp_path / "music.wav", seconds=2.0)
    duck_bed = _write_duck_bed(tmp_path / "duck.wav", seconds=2.0)
    result = media_mod.detect_music(music_bed, wrapper=Path("/nonexistent"))
    scores = media_mod.mix_score_music(result, [], duck_bed)
    assert 1.5 <= scores["duck_pump_rate"] <= 2.5, scores


def test_key_clash_score_triad_vs_dissonant(tmp_path):
    music_bed = _write_music_bed(tmp_path / "music.wav")
    triad_cue = _write_triad_cue(tmp_path / "triad.wav")
    tritone_cue = _write_tritone_cue(tmp_path / "tritone.wav")
    bad_cue = tmp_path / "not-audio.wav"
    bad_cue.write_text("this is not audio", encoding="utf-8")
    result = media_mod.detect_music(music_bed, wrapper=Path("/nonexistent"))
    assert result.key == "C major", result

    triad_manifest = {"cues": [{"path": str(triad_cue), "start": 0.0}]}
    triad_score = media_mod.mix_score_music(result, triad_manifest, music_bed)
    assert triad_score["key_clash_score"] < 0.25, triad_score

    tritone_manifest = {"cues": [{"path": str(tritone_cue), "start": 0.0}]}
    tritone_score = media_mod.mix_score_music(result, tritone_manifest, music_bed)
    assert tritone_score["key_clash_score"] > triad_score["key_clash_score"], (
        triad_score,
        tritone_score,
    )

    mixed_manifest = {
        "cues": [
            {"path": str(triad_cue), "start": 0.0},
            {"path": str(bad_cue), "start": 0.5},
        ]
    }
    mixed_score = media_mod.mix_score_music(result, mixed_manifest, music_bed)
    assert str(bad_cue) in mixed_score["skipped_cues"], mixed_score
    assert mixed_score["key_clash_score"] == triad_score["key_clash_score"], (
        mixed_score,
        triad_score,
    )


def test_mix_block_in_music_json(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    monkeypatch.setenv("MEDIA_MUSIC_BIN", str(Path("/nonexistent")))
    music_bed = _write_music_bed(tmp_path / "music.wav")
    triad_cue = _write_triad_cue(tmp_path / "triad.wav")
    manifest = {"cues": [{"path": str(triad_cue), "start": 0.0}]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    text = asyncio.run(
        server_mod.media_music(str(music_bed), sfx_manifest=str(manifest_path))
    )
    music_jsons = list(cache.rglob("music.json"))
    assert len(music_jsons) == 1, music_jsons
    payload = json.loads(music_jsons[0].read_text(encoding="utf-8"))
    for field in ("sfx_density", "duck_pump_rate", "key_clash_score", "skipped_cues"):
        assert field in payload, payload
    assert "sfx_density:" in text, text
    assert "duck_pump_rate:" in text, text
    assert "key_clash_score:" in text, text
    assert "PASS" in text or "FAIL" in text, text


def _write_tone_wav(path: Path, freq: float, seconds: float, amp: float = 0.5) -> Path:
    """Write a mono sine tone of `freq` Hz for `seconds` at `amp` amplitude."""
    total = int(RATE * seconds)
    samples = [
        amp * math.sin(2 * math.pi * freq * index / RATE) for index in range(total)
    ]
    return _write_wav(path, samples)


FAKE_SEPARATE = REPO_ROOT / "tests" / "fake_separate.sh"


def _run_vocal_analysis(
    tmp_path: Path,
    mix_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    separate_bin: str | Path | None = FAKE_SEPARATE,
    transcript_backend=None,
) -> dict:
    """Run vocal_analysis with `separate_bin` and a scratch dir under tmp_path."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    if separate_bin is not None:
        monkeypatch.setenv("MEDIA_SEPARATE_BIN", str(separate_bin))
    return media_mod.vocal_analysis(
        mix_path,
        separate_bin=separate_bin,
        scratch_dir=scratch,
        cancel=None,
        transcript_backend=transcript_backend,
    )


def _interjection_backend(_stem: Path) -> list[dict[str, object]]:
    """Mock transcript backend returning interjection-dominated text."""
    return [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "wow",
            "no_speech_prob": 0.1,
            "avg_logprob": -0.2,
        },
        {
            "start": 1.0,
            "end": 2.0,
            "text": "oh wow",
            "no_speech_prob": 0.1,
            "avg_logprob": -0.2,
        },
    ]


def _lyrics_backend(_stem: Path) -> list[dict[str, object]]:
    """Mock transcript backend returning confident real-word lyrics."""
    return [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world this is my song",
            "language": "en",
            "no_speech_prob": 0.1,
            "avg_logprob": -0.2,
        }
    ]


def _silent_backend(_stem: Path) -> list[dict[str, object]]:
    """Mock transcript backend returning nothing confident (all dropped)."""
    return [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "music",
            "no_speech_prob": 0.9,
            "avg_logprob": -2.0,
        }
    ]


def test_vocal_ladder_roformer_rung_via_fake(tmp_path, monkeypatch):
    """Rung 1 (roformer via fake) -> vocal_type vocalization from interjections."""
    mix_path = _write_tone_wav(tmp_path / "mix.wav", 300.0, 4.0, amp=0.05)
    result = _run_vocal_analysis(
        tmp_path, mix_path, monkeypatch, transcript_backend=_interjection_backend
    )
    assert result["degraded_rung"] == "roformer", result
    assert result["vocal_presence"] > 0.15, result
    assert result["vocal_timeline"], result
    assert result["vocal_type"] == "vocalization", result


def test_vocal_ladder_pyin_fallback(tmp_path, monkeypatch):
    """Missing separate_bin degrades to rung 2 (pyin) without crashing."""
    mix_path = _write_tone_wav(tmp_path / "mix.wav", 220.0, 4.0, amp=0.5)
    result = _run_vocal_analysis(
        tmp_path, mix_path, monkeypatch, separate_bin=Path("/nonexistent")
    )
    assert result["degraded_rung"] == "pyin", result
    assert result["vocal_presence"] >= 0.0, result
    assert "vocal_type" in result, result


def test_vocal_type_lyrics_vs_instrumental(tmp_path, monkeypatch):
    """Real-word lyrics -> lyrics+language; nothing confident -> instrumental+possible."""
    mix_path = _write_tone_wav(tmp_path / "mix.wav", 300.0, 4.0, amp=0.05)

    lyrics = _run_vocal_analysis(
        tmp_path, mix_path, monkeypatch, transcript_backend=_lyrics_backend
    )
    assert lyrics["vocal_type"] == "lyrics", lyrics
    assert lyrics["language"] == "en", lyrics
    assert lyrics["transcript_sample"] is not None, lyrics

    instrumental = _run_vocal_analysis(
        tmp_path, mix_path, monkeypatch, transcript_backend=_silent_backend
    )
    assert instrumental["vocal_type"] == "instrumental", instrumental
    assert instrumental["vocal_instrumental_possible"] is True, instrumental


UVR_ROOT = (
    Path.home() / ".clone" / "Github" / "SuperBypassUdinnn" / "ultimatevocalremovergui"
)
UVR_VENV = UVR_ROOT / ".venv" / "bin" / "python"
VOCALS_CKPT = UVR_ROOT / "checkpoints" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


@pytest.mark.skipif(
    not UVR_VENV.exists() or not VOCALS_CKPT.exists(),
    reason="UVR venv or vocals checkpoint absent (CI-safe skip)",
)
def test_roformer_runner_smoke_skipped_when_no_venv(tmp_path):
    """Real-runner smoke test; skipped when UVR venv/checkpoint absent."""
    import subprocess

    mix_path = _write_tone_wav(tmp_path / "mix.wav", 300.0, 3.0, amp=0.5)
    out_dir = tmp_path / "stems"
    done = subprocess.run(
        [
            str(UVR_VENV),
            str(REPO_ROOT / "separators" / "roformer_runner.py"),
            str(mix_path),
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert (out_dir / "vocals.wav").is_file()
    assert (out_dir / "instrumental.wav").is_file()
