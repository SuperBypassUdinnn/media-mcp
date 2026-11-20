"""Tests for the motion layer: ``motion_curve``, ``motion_events``, and
``media_motion``.

The event extraction is tested purely, with hand-built curves whose bursts,
holds, transitions, and keyframes sit at known indices. The ffmpeg path is
tested against synthetic clips with known ground truth: a static color
source must produce holds and no bursts, a continuously animated ``testsrc``
must produce bursts, and a one-second crossfade between two colors must
produce a transition event where the blend happens.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import media.server
from media import media as media_mod

SIZE = "320x240"
RATE = 25


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(argv, check=False, capture_output=True, text=True)
    if done.returncode != 0:
        raise AssertionError(f"fixture ffmpeg failed: {done.stderr}")
    return done


def _write_static_video(path: Path, seconds: float = 3.0) -> Path:
    """A motionless gray clip: every frame-diff should sit at the floor."""
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
            f"color=c=0x808080:size={SIZE}:rate={RATE}:duration={seconds}",
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


def _write_lossless_color_video(path: Path, color: str, seconds: float = 1.0) -> Path:
    """A lossless RGB24 solid-color clip (exact pixel preservation)."""
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
            f"color=c={color}:size={SIZE}:rate={RATE}:duration={seconds}",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "rgb24",
            str(path),
        ]
    )
    return path


def _write_color_pair_video(
    path: Path, color_a: str, color_b: str, seconds: float = 1.0
) -> Path:
    """Solid color A for `seconds` then solid color B, lossless RGB24."""
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
            f"color=c={color_a}:size={SIZE}:rate={RATE}:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color_b}:size={SIZE}:rate={RATE}:duration={seconds}",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "rgb24",
            str(path),
        ]
    )
    return path


def _write_patch_pair_video(
    path: Path, patch_color_a: str, patch_color_b: str, seconds: float = 1.0
) -> Path:
    """Dark background with a small color patch that is A then B, lossless RGB24.

    The patch covers roughly 8% of the frame so a moderate local chroma change
    lifts p95_chroma while the global mean chroma_diff stays under the floor.
    """
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
            f"color=c=0x141414:size={SIZE}:rate={RATE}:duration={seconds * 2}",
            "-f",
            "lavfi",
            "-i",
            f"color=c={patch_color_a}:size=80x80:rate={RATE}:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"color=c={patch_color_b}:size=80x80:rate={RATE}:duration={seconds}",
            "-filter_complex",
            (
                "[0:v]format=rgb24[bg];"
                "[1:v]format=rgb24[pa];"
                "[2:v]format=rgb24[pb];"
                "[bg][pa]overlay=x=0:y=0:enable='lte(t,1)'[v1];"
                "[v1][pb]overlay=x=0:y=0:enable='gt(t,1)'[v]"
            ),
            "-map",
            "[v]",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "rgb24",
            str(path),
        ]
    )
    return path


def _write_motion_burst_video(path: Path) -> Path:
    """Static 1 s, then a white patch slides right for 2 s, then static 1 s.

    The patch (80x80 white on black at 320x240) changes ~8% of the frame by
    ~255 levels per analysis step, so the slide's frame difference towers
    over the static spans' encoding noise — a bimodal curve with known
    ground truth: motion between t=1 and t=3.
    """
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
            f"color=c=black:size={SIZE}:rate={RATE}:d=4",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:size=80x80",
            "-filter_complex",
            (
                "[0:v][1:v]overlay=x='(t-1)*100':y=80:shortest=1:"
                "enable='between(t,1,3)'[v]"
            ),
            "-map",
            "[v]",
            "-an",
            "-t",
            "4",
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


def _write_xfade_video(path: Path) -> Path:
    """Two color segments joined by a 1 s crossfade centered at t=2.0.

    The red segment lasts 2.5 s, the fade runs 1.5-2.5 s (offset 1.5,
    duration 1.0), and blue holds for 2.5 s after its input start.
    """
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
            f"color=c=red:size={SIZE}:rate={RATE}:d=2.5",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:size={SIZE}:rate={RATE}:d=2.5",
            "-filter_complex",
            "[0:v][1:v]xfade=transition=fade:duration=1.0:offset=1.5[v]",
            "-map",
            "[v]",
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


def _points(
    diffs: list[float],
    dists: list[float] | None = None,
    chroma_diffs: list[float] | None = None,
    p95_chromas: list[float] | None = None,
    hue_shifts: list[float] | None = None,
    fps: float = 10.0,
) -> list[media_mod.MotionCurvePoint]:
    """Build a curve from per-index values, defaulting distances to 0.0."""
    n = len(diffs)
    if dists is None:
        dists = [0.0] * n
    if chroma_diffs is None:
        chroma_diffs = [0.0] * n
    if p95_chromas is None:
        p95_chromas = [0.0] * n
    if hue_shifts is None:
        hue_shifts = [0.0] * n
    return [
        media_mod.MotionCurvePoint(
            t=i / fps,
            diff=d,
            hist_dist=c,
            chroma_diff=cd,
            p95_chroma=p,
            hue_shift=h,
        )
        for i, (d, c, cd, p, h) in enumerate(
            zip(diffs, dists, chroma_diffs, p95_chromas, hue_shifts, strict=True)
        )
    ]


def test_motion_events_burst_and_hold():
    # 1 s flat, 1 s busy (diff 50), 1 s flat at 10 fps.
    curve = _points([0.0] * 10 + [50.0] * 10 + [0.0] * 10)
    events = media_mod.motion_events(curve, analysis_fps=10.0)
    kinds = [e.kind for e in events]
    assert "burst" in kinds, kinds
    burst = next(e for e in events if e.kind == "burst")
    assert burst.start == 1.0
    assert abs(burst.end - 2.0) < 0.15
    assert "peak diff 50.00" in burst.detail
    hold = next(e for e in events if e.kind == "hold")
    assert hold.start == 0.0
    assert abs(hold.end - 1.0) < 0.15


def test_motion_events_transition_uses_hist_distance():
    # Diff stays moderate the whole time; hist shift appears only 1.0-2.5 s.
    curve = _points([20.0] * 30, dists=[0.03] * 10 + [0.0] * 5 + [0.03] * 15, fps=10.0)
    events = media_mod.motion_events(curve, analysis_fps=10.0)
    transitions = [e for e in events if e.kind == "transition"]
    assert len(transitions) == 2, transitions
    assert transitions[0].start == 0.0
    assert abs(transitions[0].end - 1.0) < 0.15
    assert transitions[1].start == 1.5
    assert "max hist shift 0.030" in transitions[0].detail


def test_motion_events_bridge_closes_small_gaps():
    # A sustained transition with two single-frame holes stays ONE event:
    # the 2-frame bridge covers frame-rate-conversion duplicates.
    dists = [0.0, 0.03, 0.03, 0.0, 0.03, 0.03, 0.03, 0.0, 0.0, 0.03]
    curve = _points([20.0] * 10, dists=dists, fps=10.0)
    events = media_mod.motion_events(curve, analysis_fps=10.0)
    transitions = [e for e in events if e.kind == "transition"]
    assert len(transitions) == 1, transitions
    assert transitions[0].start == 0.1
    assert abs(transitions[0].end - 1.0) < 0.15


def test_motion_events_bridge_does_not_join_far_runs():
    # A 3-frame gap (above bridge_frames=2) keeps two transitions separate.
    dists = [0.03] * 5 + [0.0] * 3 + [0.03] * 5
    curve = _points([20.0] * 13, dists=dists, fps=10.0)
    events = media_mod.motion_events(curve, analysis_fps=10.0)
    transitions = [e for e in events if e.kind == "transition"]
    assert len(transitions) == 2, transitions


def test_motion_events_short_runs_are_dropped():
    # A 2-frame diff spike (0.2 s) is below burst_min_s=0.3: no burst.
    curve = _points([0.0] * 10 + [80.0] * 2 + [0.0] * 10)
    events = media_mod.motion_events(curve, analysis_fps=10.0)
    assert [e for e in events if e.kind == "burst"] == []


def test_motion_events_keyframes_inside_burst():
    # Busy span with a dip in the middle at index 15: one keyframe at 1.5 s.
    diffs = [0.0] * 10 + [50.0] * 5 + [10.0] + [50.0] * 4 + [0.0] * 10
    curve = _points(diffs)
    events = media_mod.motion_events(curve, analysis_fps=10.0)
    keyframes = [e for e in events if e.kind == "keyframe"]
    assert len(keyframes) == 1, keyframes
    assert keyframes[0].start == keyframes[0].end == 1.5
    assert "inside burst" in keyframes[0].detail


def test_motion_events_color_sweep_detected():
    # 1 s flat, 1 s color sweep (luma diff 0.2 below the hold floor,
    # p95_chroma 25), 1 s flat at 10 fps.
    curve = _points(
        [0.0] * 10 + [0.2] * 10 + [0.0] * 10,
        p95_chromas=[0.0] * 10 + [25.0] * 10 + [0.0] * 10,
        fps=10.0,
    )
    events = media_mod.motion_events(curve, analysis_fps=10.0)
    sweeps = [e for e in events if e.kind == "color_sweep"]
    assert len(sweeps) == 1, events
    assert sweeps[0].start == 1.0
    assert abs(sweeps[0].end - 2.0) < 0.15
    assert "25.00" in sweeps[0].detail, sweeps[0].detail


def test_motion_events_hold_suppressed_by_color_sweep():
    # Hold at 0-2 s, color sweep at 1-2 s (luma diff 0.2, p95_chroma 25),
    # burst at 2-2.3 s, hold at 2.3-3.3 s at 10 fps. The 0-2 s hold overlaps
    # the sweep by 1.0 s (> 0.1 s) and is suppressed; the 2.3-3.3 s hold does
    # not overlap the sweep and survives.
    curve = _points(
        [0.0] * 10 + [0.2] * 10 + [80.0] * 3 + [0.0] * 10,
        p95_chromas=[0.0] * 10 + [25.0] * 10 + [0.0] * 13,
        fps=10.0,
    )
    events = media_mod.motion_events(curve, analysis_fps=10.0)
    sweeps = [e for e in events if e.kind == "color_sweep"]
    assert len(sweeps) == 1, events
    assert sweeps[0].start == 1.0 and abs(sweeps[0].end - 2.0) < 0.15
    holds = [e for e in events if e.kind == "hold"]
    assert [(round(h.start, 3), round(h.end, 3)) for h in holds] == [
        (2.3, 3.3)
    ], holds


def test_motion_events_empty_curve():
    assert media_mod.motion_events([], analysis_fps=10.0) == []


def test_motion_curve_static_video(tmp_path):
    path = _write_static_video(tmp_path / "static.mp4")
    points = media_mod.motion_curve(path, scratch_dir=tmp_path)
    assert len(points) >= 20  # 3 s at 10 fps
    assert points[0].diff == 0.0 and points[0].hist_dist == 0.0
    # A motionless clip stays under the burst floor everywhere.
    events = media_mod.motion_events(points, analysis_fps=10.0)
    assert [e for e in events if e.kind == "burst"] == []
    assert [e for e in events if e.kind == "hold"], events
    # The scratch raster is gone after the call.
    assert not (tmp_path / "motion-raster.gray").exists()


def test_motion_curve_motion_burst_video(tmp_path):
    """The sliding patch produces exactly one burst overlapping t=1..3."""
    path = _write_motion_burst_video(tmp_path / "burst.mp4")
    points = media_mod.motion_curve(path, scratch_dir=tmp_path)
    events = media_mod.motion_events(points, analysis_fps=10.0)
    bursts = [e for e in events if e.kind == "burst"]
    assert len(bursts) == 1, events
    assert 0.8 <= bursts[0].start <= 1.3, bursts
    assert 2.7 <= bursts[0].end <= 3.4, bursts


def test_motion_curve_xfade_video_has_transition(tmp_path):
    path = _write_xfade_video(tmp_path / "xfade.mp4")
    points = media_mod.motion_curve(path, scratch_dir=tmp_path)
    events = media_mod.motion_events(points, analysis_fps=10.0)
    transitions = [e for e in events if e.kind == "transition"]
    assert transitions, events
    # The blend runs 1.5-2.5 s; the detected transition must overlap it.
    assert transitions[0].start < 2.5 and transitions[0].end > 1.5


def test_motion_curve_chromatic_delta(tmp_path):
    # Gray (luma ~128, zero chroma) then a saturated orange (same luma ~128,
    # high chroma): luma diff stays near zero while chroma delta is large.
    path = _write_color_pair_video(tmp_path / "chroma.mp4", "0x808080", "0xFF5800")
    points = media_mod.motion_curve(path, scratch_dir=tmp_path, analysis_fps=10.0)
    assert len(points) >= 15, points
    cut = points[10]  # first frame of the second color (t = 1.0 s)
    assert cut.diff < 2.0, cut
    assert cut.chroma_diff > 20.0, cut


def test_motion_curve_p95_chroma_local_sweep(tmp_path):
    # Dark frame with a small color patch that changes hue mid-clip: a strong
    # local chroma delta on a minority of pixels lifts p95_chroma while the
    # global mean chroma_diff stays under the floor.
    path = _write_patch_pair_video(tmp_path / "patch.mp4", "0x3C3C3C", "0x462C2C")
    points = media_mod.motion_curve(path, scratch_dir=tmp_path, analysis_fps=10.0)
    assert len(points) >= 15, points
    cut = max(points, key=lambda p: p.p95_chroma)  # frame where the patch flips
    assert cut.chroma_diff < 2.0, cut
    assert cut.p95_chroma > 15.0, cut


def test_motion_curve_monochromatic_zero(tmp_path):
    # Purely gray content: no chroma anywhere, so all chromatic signals are 0.
    path = _write_lossless_color_video(tmp_path / "mono.mp4", "0x808080")
    points = media_mod.motion_curve(path, scratch_dir=tmp_path, analysis_fps=10.0)
    assert points, points
    for point in points:
        assert point.chroma_diff == 0.0, point
        assert point.p95_chroma == 0.0, point
        assert point.hue_shift == 0.0, point


def test_motion_curve_rejects_params_and_audio_only(tmp_path):
    audio_only = tmp_path / "audio.m4a"
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
            "sine=frequency=440:duration=1",
            "-c:a",
            "aac",
            str(audio_only),
        ]
    )
    try:
        media_mod.motion_curve(audio_only, scratch_dir=tmp_path)
    except media_mod.MediaError as error:
        assert "video stream" in str(error)
    else:
        raise AssertionError("audio-only source must raise MediaError")
    video = _write_static_video(tmp_path / "v.mp4", seconds=0.5)
    try:
        media_mod.motion_curve(video, analysis_fps=0, scratch_dir=tmp_path)
    except media_mod.MediaError as error:
        assert "analysis_fps" in str(error)
    else:
        raise AssertionError("non-positive analysis_fps must raise MediaError")


def test_media_motion_tool(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(cache))
    path = _write_xfade_video(tmp_path / "clip.mp4")
    result = asyncio.run(media.server.media_motion(str(path)))
    assert "# media_motion" in result
    assert f"Source: {path}" in result
    assert "transition" in result
    motion_json = next(cache.rglob("motion.json"))
    payload = json.loads(motion_json.read_text())
    assert payload["params"] == {"analysis_fps": 10.0, "width": 320}
    assert payload["points"] > 0
    assert any(e["kind"] == "transition" for e in payload["events"])
    # Spec section 3: motion.json carries a curve summary beside the events.
    summary = payload["summary"]
    assert summary["span_s"] > 0
    assert summary["peak_diff"] >= summary["mean_diff"] >= 0.0
    assert 0.0 <= summary["quiet_share"] <= 1.0
    curve_tsv = next(cache.rglob("motion-curve.tsv"))
    header, *rows = curve_tsv.read_text().splitlines()
    assert header == "t\tdiff\thist_dist"
    assert len(rows) == payload["points"]
    assert "Artifacts:" in result and "motion-curve.tsv" in result
