#!/usr/bin/env python3
"""Transcribe audio/video with speaker labels (SPEAKER_00, SPEAKER_01, ...).

Input is a local media file or a YouTube URL (extracted via yt-dlp).
Default model is large-v3-turbo with automatic fallback to medium then CPU.
Diarization uses pyannote speaker-diarization-3.1 when an HF token exists;
a missing token skips diarization with a warning instead of failing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

# PyTorch 2.6+ defaults torch.load to weights_only=True, which refuses to
# unpickle the official PyAnnote Lightning checkpoints. The weights come from
# the gated official pyannote repos (trusted source), so opt out back to the
# classic loader. setdefault keeps an explicit user choice intact.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video with speaker diarization."
    )
    parser.add_argument("input", help="Local media file or YouTube URL.")
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="faster-whisper model (default: large-v3-turbo).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Expected speaker count for diarization (default: 2). "
        "Use 1 for a single speaker: diarization is skipped and speaker "
        "labels are omitted from all outputs.",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip diarization even when a token is available.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable the Silero VAD pre-filter (VAD is on by default; it "
        "prevents hallucinated repeats on noisy audio).",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enhance audio with DeepFilterNet before transcription "
        "(recommended for noisy recordings; downloads an 8 MB model once).",
    )
    parser.add_argument(
        "--out",
        default="./transcripts",
        help="Root output directory (default: ./transcripts).",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Language code (id, en) or auto (default: auto).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--compute",
        default="auto",
        help="CTranslate2 compute type (default: auto -> int8/float32).",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        default=True,
        help="Condition the Whisper decode on previous text (default: on). "
        "Pass --no-condition-on-previous-text to disable it (hallucination "
        "guard on instrumental stems).",
    )
    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_false",
        dest="condition_on_previous_text",
        help="Disable conditioning on previous text.",
    )
    return parser.parse_args(argv)


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def warn_large_input(path: Path) -> None:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 500:
        print(
            f"WARNING: input is {size_mb:.0f} MB (soft limit 500 MB); "
            "processing continues.",
            file=sys.stderr,
        )


def prepare_wav(source: str, workdir: Path) -> Path:
    """Return a 16 kHz mono WAV path for transcription."""
    if is_url(source):
        if shutil.which("yt-dlp") is None:
            raise RuntimeError("yt-dlp is required for URL input.")
        target = workdir / "source.wav"
        subprocess.run(
            [
                "yt-dlp",
                "--extract-audio",
                "--audio-format",
                "wav",
                "--postprocessor-args",
                "-ar 16000 -ac 1",
                "-o",
                str(target),
                source,
            ],
            check=True,
        )
        resolved = target.with_suffix(".wav")
        if not resolved.exists():
            candidates = list(workdir.glob("*.wav"))
            if not candidates:
                raise RuntimeError("yt-dlp audio extraction produced no WAV.")
            resolved = candidates[0]
        return resolved
    local = Path(source)
    if not local.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    warn_large_input(local)
    if local.suffix.lower() == ".wav":
        return local
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for non-WAV input.")
    target = workdir / "converted.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(local), "-ar", "16000", "-ac", "1", str(target)],
        check=True,
        capture_output=True,
    )
    return target


def denoise_wav(wav: Path, workdir: Path) -> Path:
    try:
        from df import enhance, init_df
        from df.io import load_audio, save_audio
    except ImportError as exc:
        raise RuntimeError(
            "DeepFilterNet is not installed. Run: "
            "scripts/setup.sh (installs deepfilternet into the skill venv)."
        ) from exc
    print("Enhancing audio with DeepFilterNet ...")
    model, df_state, _ = init_df()
    audio, _ = load_audio(str(wav), sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    target = workdir / "denoised.wav"
    save_audio(str(target), enhanced, df_state.sr())
    return target


def transcribe_with_fallback(
    wav: Path,
    model: str,
    language: str,
    device: str,
    compute: str,
    use_vad: bool = True,
    condition_on_previous_text: bool = True,
) -> tuple[list[dict], str | None]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc
    tiers = [model]
    if model != "medium":
        tiers.append("medium")
    last_error: Exception | None = None
    for tier in tiers:
        for dev in [device] if device != "auto" else ["cuda", "cpu"]:
            ctype = compute
            if ctype == "auto":
                ctype = "int8" if dev == "cuda" else "int8_float32"
            try:
                print(f"Loading model {tier} on {dev} ({ctype}) ...")
                whisper = WhisperModel(tier, device=dev, compute_type=ctype)
                lang = None if language == "auto" else language
                vad_params = {"min_silence_duration_ms": 500} if use_vad else None
                segments, info = whisper.transcribe(
                    str(wav),
                    language=lang,
                    vad_filter=use_vad,
                    vad_parameters=vad_params,
                    word_timestamps=True,
                    condition_on_previous_text=condition_on_previous_text,
                )
                detected_language = getattr(info, "language", None)
                return [
                    {
                        "start": float(s.start),
                        "end": float(s.end),
                        "text": s.text.strip(),
                        "words": [
                            {
                                "start": float(w.start),
                                "end": float(w.end),
                                "text": w.word.strip(),
                            }
                            for w in (s.words or [])
                            if w.word.strip()
                        ],
                        "no_speech_prob": (
                            float(s.no_speech_prob)
                            if getattr(s, "no_speech_prob", None) is not None
                            else None
                        ),
                        "avg_logprob": (
                            float(s.avg_logprob)
                            if getattr(s, "avg_logprob", None) is not None
                            else None
                        ),
                    }
                    for s in segments
                ], detected_language
            except Exception as exc:
                message = str(exc).lower()
                if "memory" in message or "cuda" in message or "cublas" in message:
                    print(
                        f"Tier {tier} on {dev} failed ({exc}); falling back.",
                        file=sys.stderr,
                    )
                    last_error = exc
                    continue
                raise
    raise RuntimeError(f"All transcription tiers failed: {last_error}")


def diarize(wav: Path, num_speakers: int) -> list[dict]:
    token = hf_token()
    if token is None:
        raise RuntimeError("HF_TOKEN is missing.")
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Run: pip install -r requirements.txt"
        ) from exc
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token
    )
    result = pipeline(str(wav), num_speakers=num_speakers)
    turns = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    return turns


def speaker_for_span(start: float, end: float, turns: list[dict]) -> str:
    best = "SPEAKER_00"
    best_overlap = 0.0
    for turn in turns:
        overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best = turn["speaker"]
    return best


def normalize_speakers(segments: list[dict]) -> list[dict]:
    # Normalize labels to SPEAKER_00 ordering by first appearance.
    order: dict[str, str] = {}
    for seg in segments:
        if seg["speaker"] not in order:
            order[seg["speaker"]] = f"SPEAKER_{len(order):02d}"
    for seg in segments:
        seg["speaker"] = order[seg["speaker"]]
    return segments


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    # Word-level assignment: each word gets the speaker whose turn overlaps
    # it most, then consecutive same-speaker words merge. This keeps the
    # minority speaker on long mixed segments instead of letting the
    # majority speaker swallow them whole. Segments without word timings
    # fall back to whole-segment overlap.
    labeled: list[dict] = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            labeled.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "speaker": speaker_for_span(seg["start"], seg["end"], turns),
                }
            )
            continue
        current: dict | None = None
        for word in words:
            speaker = speaker_for_span(word["start"], word["end"], turns)
            gap = word["start"] - current["end"] if current else 0.0
            if current is None or speaker != current["speaker"] or gap > 1.5:
                if current is not None:
                    labeled.append(current)
                current = {
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["text"],
                    "speaker": speaker,
                }
            else:
                current["end"] = word["end"]
                current["text"] += " " + word["text"]
        if current is not None:
            labeled.append(current)
    return normalize_speakers(labeled)


def format_timestamp(seconds: float, vtt: bool = False) -> str:
    delta = timedelta(seconds=max(0.0, seconds))
    total_ms = int(delta.total_seconds() * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    sep = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def write_srt(segments: list[dict], path: Path, show_labels: bool = True) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, seg in enumerate(segments, start=1):
            handle.write(f"{index}\n")
            handle.write(
                f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n"
            )
            text = seg["text"]
            if show_labels:
                text = f"[{seg.get('speaker', 'SPEAKER_00')}] {text}"
            handle.write(f"{text}\n\n")


def write_vtt(segments: list[dict], path: Path, show_labels: bool = True) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("WEBVTT\n\n")
        for seg in segments:
            handle.write(
                f"{format_timestamp(seg['start'], vtt=True)} --> "
                f"{format_timestamp(seg['end'], vtt=True)}\n"
            )
            text = seg["text"]
            if show_labels:
                text = f"<v {seg.get('speaker', 'SPEAKER_00')}>{text}</v>"
            handle.write(f"{text}\n\n")


def write_json(
    segments: list[dict], path: Path, meta: dict, show_labels: bool = True
) -> None:
    rows = segments
    if not show_labels:
        rows = [{k: v for k, v in seg.items() if k != "speaker"} for seg in segments]
    path.write_text(
        json.dumps({"meta": meta, "segments": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown(
    segments: list[dict], path: Path, meta: dict, show_labels: bool = True
) -> None:
    lines = [
        f"# Transcript: {meta.get('title', 'recording')}",
        "",
        f"Source: {meta.get('source', '')}",
        f"Model: {meta.get('model', '')}",
        "",
    ]
    current: str | None = None
    for seg in segments:
        if show_labels:
            speaker = seg.get("speaker", "SPEAKER_00")
            if speaker != current:
                lines.append(f"## {speaker}")
                lines.append("")
                current = speaker
        lines.append(f"[{format_timestamp(seg['start'])}] {seg['text']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out)
    with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp:
        workdir = Path(tmp)
        # CLI boundary: any input-prep failure (file, yt-dlp, ffmpeg) is fatal.
        try:
            wav = prepare_wav(args.input, workdir)
            if args.denoise:
                wav = denoise_wav(wav, workdir)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        stem = Path(args.input).stem if not is_url(args.input) else "youtube-audio"
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        device = resolve_device(args.device)
        # CLI boundary: any transcription failure across fallback tiers is fatal.
        try:
            segments, detected_language = transcribe_with_fallback(
                wav,
                args.model,
                args.language,
                device,
                args.compute,
                use_vad=not args.no_vad,
                condition_on_previous_text=args.condition_on_previous_text,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        for seg in segments:
            seg.setdefault("speaker", "SPEAKER_00")
        # A declared single speaker needs no separation: skip diarization
        # and omit speaker labels from every output.
        show_labels = args.num_speakers != 1
        if not show_labels:
            print("Single speaker declared; skipping diarization and labels.")
        elif args.no_diarize:
            print("Diarization skipped by --no-diarize.")
        elif hf_token() is None:
            print(
                "WARNING: HF_TOKEN is missing; continuing without speaker labels.",
                file=sys.stderr,
            )
        else:
            # Graceful degradation: any diarization backend failure keeps transcription.
            try:
                turns = diarize(wav, args.num_speakers)
                segments = assign_speakers(segments, turns)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARNING: diarization failed ({exc}); continuing "
                    "without speaker labels.",
                    file=sys.stderr,
                )
        meta = {
            "source": args.input,
            "model": args.model,
            "title": stem,
            "language": detected_language,
        }
        write_srt(segments, out_dir / f"{stem}.srt", show_labels)
        write_vtt(segments, out_dir / f"{stem}.vtt", show_labels)
        write_json(segments, out_dir / f"{stem}.json", meta, show_labels)
        write_markdown(segments, out_dir / f"{stem}.md", meta, show_labels)
        print(f"Saved 4 files to {out_dir}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
