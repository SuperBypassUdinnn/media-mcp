"""Synthetic media fixtures for the test suite.

Every ffmpeg invocation uses an argument list, never a shell string. Burned-in
text uses the `ass` filter from a generated ASS file because this ffmpeg build
has no `drawtext` filter; the ASS file is written inside a scratch directory
under the caller's own output directory, so no fixture artifact lands in the
repository.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

VIDEO_SIZE = (1280, 720)
VIDEO_RATE = 25
TEXT_PANEL_HEIGHT = 240
FONT_NAME = "monospace"
FONT_SIZE = 64
COUNTER_BEFORE = "80"
COUNTER_AFTER = "48"
SILENT_VIDEO_SECONDS = 4
AUDIO_SECONDS = 4
AUDIO_RATE = 48000
BASE_VOLUME = 0.1
SPIKE_VOLUME = 5.0
SPIKE_SECONDS = 1.0
# The motion-patch clip: a static dark background with one opaque light patch
# whose exact on-screen rectangle and switch-on time the tests assert against.
MOTION_PATCH = (700, 420, 300, 180)
MOTION_PATCH_APPEARS_AT = 4.2

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class FixtureError(RuntimeError):
    """Raised when ffmpeg or ffprobe refuses to build a fixture."""


def make_sample(
    path: Path,
    duration: int = 60,
    text: str = "LANJUT 42",
    spike_at: int = 30,
    counter_jump_at: int = 20,
    audio_offset: float = 0.0,
) -> Path:
    """Write a probe-ready mp4 with a pattern, burned-in text, and a tone.

    The clip is a `testsrc` segment followed by a `testsrc2` segment, so a real
    hard cut sits at `counter_jump_at` seconds for `scdet` to find. A light
    panel is composited over the top of both segments and carries the text and a
    counter that holds one value before the cut and another after it. The audio
    is a sine tone at a low base level with a one-second loud spike whose
    position `spike_at` is measured in audio content time, so `audio_offset`
    delays the tone and its spike together by starting the audio stream later.
    """
    if duration < 1:
        raise ValueError("duration must be at least one second")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jump = counter_jump_at if 0 < counter_jump_at < duration else None

    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        ass_path = Path(scratch) / "fixture.ass"
        ass_path.write_text(_ass_document(duration, text, jump), encoding="utf-8")
        _run(_sample_argv(path, ass_path, duration, spike_at, audio_offset, jump))
    return path


def make_silent_video(path: Path) -> Path:
    """Write a motionless video carrying a digitally silent audio track."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = VIDEO_SIZE
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            (
                f"color=c=0x808080:size={width}x{height}:rate={VIDEO_RATE}"
                f":duration={SILENT_VIDEO_SECONDS}"
            ),
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout=mono:sample_rate={AUDIO_RATE}",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(path),
        ]
    )
    return path


def make_audio_only(path: Path) -> Path:
    """Write an audio-only file with no video stream."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate={AUDIO_RATE}:duration={AUDIO_SECONDS}",
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(path),
        ]
    )
    return path


def make_motion_patch(
    path: Path,
    *,
    duration: int = 6,
    size: tuple[int, int] = (1280, 720),
    rate: int = 25,
    patch: tuple[int, int, int, int] = MOTION_PATCH,
    appears_at: float = MOTION_PATCH_APPEARS_AT,
) -> Path:
    """Write a static dark clip with one light patch that turns on and stays on.

    `testsrc` motion fills the entire frame, so auto crop over it can never
    shrink tighter than the frame edges and the crop tests would be unfalsifiable.
    This builder instead leaves the background motionless and restricts all real
    motion to one opaque `drawbox` rectangle that is off before `appears_at` and
    on afterwards, so the auto-crop pass over a window that straddles
    `appears_at` (4.0 to 5.0 at fps=10) measures exactly the `patch` rectangle.
    The clip carries no audio; frame tests do not need a stream.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    px, py, pw, ph = patch
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            (f"color=c=0x202020:size={width}x{height}:rate={rate}:duration={duration}"),
            "-vf",
            (
                f"drawbox=x={px}:y={py}:w={pw}:h={ph}"
                f":color=white@1:t=fill:enable={_filter_quote(f'between(t,{appears_at},{duration + 1})')}"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )
    return path


def make_value_flicker(
    path: Path,
    *,
    duration: int = 8,
    base: str = "80",
    flick: str = "48",
    flick_at: float = 4.0,
    flick_seconds: float = 1.0,
    size: tuple[int, int] = VIDEO_SIZE,
    rate: int = VIDEO_RATE,
) -> Path:
    """Write a static light-panel clip carrying exactly one flickering number.

    The panel shows `Nilai <base>` for the whole clip except a
    `flick_seconds` window starting at `flick_at`, which shows `Nilai
    <flick>` and then returns to the base. A `testsrc` background would put
    motion in every frame and `make_sample`-style text plus counter would
    put two numbers in frame, making the first numeric token ambiguous; so
    the background is a motionless color and the one value line is the only
    text on the panel. The clip exists so value tracking can be tested
    against real OCR with a known ground truth: exactly one glitch, at
    `flick_at`, back to the base after `flick_seconds`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    panel_height = height // 3

    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        ass_path = Path(scratch) / "flicker.ass"
        ass_path.write_text(
            _flicker_ass_document(size, duration, base, flick, flick_at, flick_seconds),
            encoding="utf-8",
        )
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                (
                    f"color=c=0x202020:size={width}x{height}"
                    f":rate={rate}:duration={duration}"
                ),
                "-vf",
                (
                    f"drawbox=x=0:y=0:w=iw:h={panel_height}"
                    f":color=white@1:t=fill,"
                    f"ass=filename={_filter_quote(ass_path)}"
                ),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ]
        )
    return path


def png_size(png_bytes: bytes) -> tuple[int, int]:
    """Return the (width, height) of a PNG stream, read from its IHDR chunk."""
    if png_bytes[:8] != _PNG_SIGNATURE:
        raise ValueError("not a PNG stream")
    if png_bytes[12:16] != b"IHDR":
        raise ValueError("PNG stream without an IHDR chunk")
    width, height = struct.unpack(">II", png_bytes[16:24])
    return width, height


def _sample_argv(
    path: Path,
    ass_path: Path,
    duration: int,
    spike_at: int,
    audio_offset: float,
    jump: int | None,
) -> list[str]:
    _width, height = VIDEO_SIZE
    argv = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    argv += ["-f", "lavfi", "-i", _video_source(height, jump or duration, "testsrc")]
    if jump is not None:
        argv += [
            "-f",
            "lavfi",
            "-i",
            _video_source(height, duration - jump, "testsrc2"),
        ]
    audio_index = 1 if jump is None else 2
    if audio_offset:
        argv += ["-itsoffset", f"{audio_offset}"]
    argv += [
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate={AUDIO_RATE}:duration={duration}",
    ]

    background = "[0:v]"
    parts: list[str] = []
    if jump is not None:
        parts.append("[0:v][1:v]concat=n=2:v=1:a=0[background]")
        background = "[background]"
    parts.append(
        f"{background}drawbox=x=0:y=0:w=iw:h={TEXT_PANEL_HEIGHT}"
        ":color=white@1:t=fill[panel]"
    )
    parts.append(f"[panel]ass=filename={_filter_quote(ass_path)}[v]")
    parts.append(
        f"[{audio_index}:a]volume=volume='{_spike_expression(spike_at, audio_offset)}'"
        ":eval=frame[a]"
    )

    argv += [
        "-filter_complex",
        ";".join(parts),
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(path),
    ]
    return argv


def _video_source(height: int, duration: int, name: str) -> str:
    width, _ = VIDEO_SIZE
    return f"{name}=size={width}x{height}:rate={VIDEO_RATE}:duration={duration}"


def _spike_expression(spike_at: int, audio_offset: float) -> str:
    start = spike_at + audio_offset
    stop = start + SPIKE_SECONDS
    return f"if(between(t,{start},{stop}),{SPIKE_VOLUME},{BASE_VOLUME})"


def _filter_quote(path: Path) -> str:
    escaped = str(path).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _ass_document(duration: int, text: str, jump: int | None) -> str:
    width, height = VIDEO_SIZE
    events = [f"Dialogue: 0,{_ass_time(0)},{_ass_time(duration)},Text,,0,0,0,,{text}"]
    for second in range(int(duration) + 1):
        value = COUNTER_BEFORE if jump is None or second < jump else COUNTER_AFTER
        events.append(
            f"Dialogue: 0,{_ass_time(second)},{_ass_time(second + 1)}"
            f",Counter,,0,0,0,,{value}"
        )
    styles = (
        f"Style: Text,{FONT_NAME},{FONT_SIZE},&H00000000,&H00000000,&H00FFFFFF,&H00000000"
        ",0,0,0,0,100,100,0,0,1,0,0,7,60,60,40,1"
        "\n"
        f"Style: Counter,{FONT_NAME},{FONT_SIZE},&H00000000,&H00000000,&H00FFFFFF,&H00000000"
        ",0,0,0,0,100,100,0,0,1,0,0,9,60,60,40,1"
    )
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour"
        ", OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX"
        ", ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment"
        ", MarginL, MarginR, MarginV, Encoding\n"
        f"{styles}\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV"
        ", Effect, Text\n" + "\n".join(events) + "\n"
    )


def _flicker_ass_document(
    size: tuple[int, int],
    duration: int,
    base: str,
    flick: str,
    flick_at: float,
    flick_seconds: float,
) -> str:
    """The ASS document for `make_value_flicker`: one line, three windows.

    The same header, style, and black-on-white proven by `_ass_document`,
    but exactly one Dialogue line at a time: the base value before
    `flick_at`, the flick value across `[flick_at, flick_at +
    flick_seconds)`, and the base value again until the clip ends.
    """
    width, height = size
    stop = min(flick_at + flick_seconds, float(duration))
    events = [
        f"Dialogue: 0,{_ass_time(0)},{_ass_time(flick_at)},Text,,0,0,0,,Nilai {base}",
        f"Dialogue: 0,{_ass_time(flick_at)},{_ass_time(stop)},Text,,0,0,0,,Nilai {flick}",
        f"Dialogue: 0,{_ass_time(stop)},{_ass_time(duration)},Text,,0,0,0,,Nilai {base}",
    ]
    style = (
        f"Style: Text,{FONT_NAME},{FONT_SIZE},&H00000000,&H00000000,&H00FFFFFF,&H00000000"
        ",0,0,0,0,100,100,0,0,1,0,0,7,60,60,40,1"
    )
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour"
        ", OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX"
        ", ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment"
        ", MarginL, MarginR, MarginV, Encoding\n"
        f"{style}\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV"
        ", Effect, Text\n" + "\n".join(events) + "\n"
    )


def _ass_time(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    hours, rest = divmod(centiseconds, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, centis = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(argv, check=False, capture_output=True, text=True)
    if done.returncode != 0:
        raise FixtureError(f"{argv[0]} exited {done.returncode}: {done.stderr.strip()}")
    return done
