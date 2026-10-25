"""Tests for frame extraction, crop resolution, tiling, and burned labels."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from media import media
from tests import fixtures

# Exactly as the brief writes them ------------------------------------------------


def test_extract_frame_respects_max_edge(tmp_path):
    png = media.extract_frame(
        make_sample(tmp_path / "s.mp4", duration=4), 1.0, None, 320
    )
    width, height = png_size(png)
    assert max(width, height) <= 320


def test_sheet_is_a_single_image_under_max_edge(tmp_path):
    src = fixtures.make_motion_patch(tmp_path / "src.mp4")
    frames = [
        (float(i), media.extract_frame(src, float(i), None, 320)) for i in range(6)
    ]
    sheet = media.render_sheet(frames, labels=True, max_edge=800)
    assert png_size(sheet)[0] <= 800


def test_filmstrip_of_a_window_is_one_image(tmp_path):
    src = fixtures.make_motion_patch(tmp_path / "src.mp4")
    frames = [(i / 10, media.extract_frame(src, i / 10, None, 240)) for i in range(8)]
    strip = media.render_filmstrip(frames, labels=True, max_edge=600)
    assert png_size(strip)[1] > png_size(strip)[0]


def test_auto_crop_is_tighter_than_the_full_frame(tmp_path):
    src = fixtures.make_motion_patch(tmp_path / "src.mp4")
    box = media.resolve_crop("auto", probe_of(src), src, 4.0, 5.0, fps=10)
    assert box is not None and box.width < probe_of(src).width


def test_auto_crop_on_a_static_window_returns_none(tmp_path):
    silent = fixtures.make_silent_video(tmp_path / "still.mp4")
    assert (
        media.resolve_crop("auto", probe_of(silent), silent, 0.0, 1.0, fps=10) is None
    )


def test_out_of_bounds_crop_is_rejected_with_measured_size(tmp_path):
    src = fixtures.make_motion_patch(tmp_path / "src.mp4")
    with pytest.raises(media.MediaError) as err:
        media.resolve_crop(
            {"x": 0, "y": 0, "width": 99999, "height": 10}, probe_of(src), src, 0, 1, 1
        )
    assert str(probe_of(src).width) in str(err.value)


def test_render_stages_under_the_callers_scratch_dir(tmp_path, monkeypatch):
    # The caller owns a cache entry directory on the same filesystem as its
    # output, so the staging root must be selectable: with `scratch_dir` set,
    # neither the tile files nor the ASS documents land in the OS temp dir.
    src = fixtures.make_motion_patch(tmp_path / "src.mp4")
    frames = [
        (float(i), media.extract_frame(src, float(i), None, 320)) for i in range(3)
    ]
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    system_temp = Path(tempfile.gettempdir())
    leaked: list[str] = []
    real_run_child = media.run_child

    def spying_run_child(argv, **kwargs):
        for prefix in ("media-tiles-*", "media-ass-*"):
            leaked.extend(str(path) for path in system_temp.glob(prefix))
        return real_run_child(argv, **kwargs)

    monkeypatch.setattr(media, "run_child", spying_run_child)
    sheet = media.render_sheet(frames, labels=True, max_edge=400, scratch_dir=scratch)
    assert png_size(sheet)[0] <= 400
    assert leaked == [], f"staging leaked into the system temp dir: {leaked}"


def test_one_tile_label_failure_degrades_that_tile_only(tmp_path, monkeypatch):
    # A label is decoration: one tile's burn failing must not fail the whole
    # sheet. The failed tile falls back to unlabelled, the others keep their
    # labels, and the reason is reported rather than swallowed.
    src = fixtures.make_motion_patch(tmp_path / "src.mp4")
    frames = [
        (float(i), media.extract_frame(src, float(i), None, 320)) for i in range(3)
    ]
    doomed = 1.0
    real_label_tile = media._label_tile
    labelled: list[float] = []

    def flaky(frame, timestamp, dimensions, family, font, scratch_dir=None):
        if timestamp == doomed:
            raise media.MediaError("label burn simulated failure")
        labelled.append(timestamp)
        return real_label_tile(frame, timestamp, dimensions, family, font, scratch_dir)

    monkeypatch.setattr(media, "_label_tile", flaky)
    notes: list[str] = []
    sheet = media.render_sheet(frames, labels=True, max_edge=400, notes=notes)

    assert png_size(sheet)[0] <= 400, "one failed tile must not fail the sheet"
    assert labelled == [0.0, 2.0], "the other tiles stay labelled"
    assert len(notes) == 1, notes
    assert "1.0" in notes[0] and "simulated failure" in notes[0]


def test_missing_font_yields_unlabelled_tiles(tmp_path, monkeypatch):
    src = fixtures.make_motion_patch(tmp_path / "src.mp4")
    frames = [
        (float(i), media.extract_frame(src, float(i), None, 320)) for i in range(3)
    ]
    monkeypatch.setenv("MEDIA_FONT_FILE", "/nonexistent/font.ttf")
    assert media.resolve_font() is None
    assert media.render_sheet(frames, labels=True, max_edge=400) is not None


def test_audio_only_source_has_no_video_stream(tmp_path):
    with pytest.raises(media.MediaError) as err:
        media.extract_frame(make_audio_only(tmp_path / "a.m4a"), 1.0, None, 320)
    assert "no video stream" in str(err.value).lower()


# Resolutions of the brief --------------------------------------------------------


def test_motion_box_pure_function_finds_one_known_differing_cell():
    tiles, tw, th = 3, 4, 4
    stride = tw * tiles
    raw = bytearray(tiles * tw * th)
    # One cell, tile 2, tile-space (dy=1, dx=2): index math follows the
    # `tile=<n>x1` layout the ffmpeg side emits, rows of tw*tiles bytes.
    raw[1 * stride + 2 * tw + 2] = 200
    assert media._motion_box(bytes(raw), tw, th, tiles) == media.Box(2, 1, 1, 1)


def test_motion_box_pure_function_rejects_a_motionless_raster():
    raw = bytes([50]) * (2 * 4 * 4)
    assert media._motion_box(raw, 4, 4, 2) is None


def test_motion_box_pure_function_rejects_an_all_different_raster():
    tiles, tw, th = 2, 4, 4
    stride = tw * tiles
    raw = bytearray(tiles * tw * th)
    for dy in range(th):
        for dx in range(tw):
            raw[dy * stride + 0 * tw + dx] = 0
            raw[dy * stride + 1 * tw + dx] = 255
    assert media._motion_box(bytes(raw), tw, th, tiles) is None


def test_auto_crop_covers_the_known_moving_patch(tmp_path):
    src = fixtures.make_motion_patch(tmp_path / "src.mp4")
    box = media.resolve_crop("auto", probe_of(src), src, 4.0, 5.0, fps=10)
    assert box is not None
    px, py, pw, ph = fixtures.MOTION_PATCH
    tolerance = 8 + max(media.probe(src).width // 64, media.probe(src).height // 36)
    assert box.x < px + pw + tolerance and box.x + box.width > px - tolerance
    assert box.y < py + ph + tolerance and box.y + box.height > py - tolerance


def test_window_past_the_clip_end_is_clamped_never_garbage(tmp_path):
    # A 6-second clip asked for 5.8-6.4: the window runs past the end, and an
    # unclamped read would under-fill the tile grid and poison the mask.
    src = fixtures.make_motion_patch(tmp_path / "src.mp4")
    result = media.probe(src)
    box = media.resolve_crop("auto", result, src, 5.8, 6.4, fps=10)
    assert box is None or (
        box.x >= 0
        and box.y >= 0
        and box.x + box.width <= result.width
        and box.y + box.height <= result.height
    )


def test_crop_dict_missing_a_key_names_that_key(tmp_path):
    src = fixtures.make_motion_patch(tmp_path / "src.mp4")
    with pytest.raises(media.MediaError) as err:
        media.resolve_crop({"x": 0, "y": 0, "width": 100}, probe_of(src), src, 0, 1, 1)
    assert "height" in str(err.value).lower()


def test_jpeg_fallback_reaches_mjpeg_only_through_a_shrunk_threshold(
    tmp_path, monkeypatch
):
    # On flat synthetic frames JPEG is larger than PNG, so the real
    # 2 MiB threshold is never crossed here; shrinking the module constant
    # makes the fallback reachable and proves the re-encode path.
    src = fixtures.make_motion_patch(tmp_path / "src.mp4")
    png = media.extract_frame(src, 5.0, None, 320)
    assert media.sniff_image_format(png) == "png"
    monkeypatch.setattr(media, "JPEG_FALLBACK_BYTES", 128)
    shrunk = media.extract_frame(src, 5.0, None, 320)
    assert media.sniff_image_format(shrunk) == "jpeg"
    assert shrunk[:3] == b"\xff\xd8\xff"


def test_burned_labels_are_ocr_readable(tmp_path):
    # Tiles big enough that reading the timestamps is honest, not lucky:
    # max_edge 1000 over four frames gives 500-pixel-wide tiles and a
    # proportionally large font.
    src = fixtures.make_motion_patch(tmp_path / "src.mp4")
    timestamps = (1.0, 2.0, 3.0, 4.0)
    frames = [(t, media.extract_frame(src, t, None, 640)) for t in timestamps]
    sheet = media.render_sheet(frames, labels=True, max_edge=1000)
    destination = tmp_path / "sheet.png"
    destination.write_bytes(sheet)
    done = subprocess.run(
        ["tesseract", str(destination), "stdout"],
        capture_output=True,
        check=False,
        text=True,
    )
    found = sum(1 for t in timestamps if _label_text(t) in done.stdout)
    assert found >= 2, f"tesseract read: {done.stdout!r} (stderr: {done.stderr!r})"


def test_resolve_font_prefers_media_font_file(tmp_path, monkeypatch):
    font = tmp_path / "custom.ttf"
    font.write_bytes(b"placeholder file that only needs to exist")
    monkeypatch.setenv("MEDIA_FONT_FILE", str(font))
    assert media.resolve_font() == font


def test_resolve_font_unset_falls_back_to_fontconfig(monkeypatch):
    monkeypatch.delenv("MEDIA_FONT_FILE", raising=False)
    font = media.resolve_font()
    # On this host fc-match answers with a real Noto Sans Mono file; a host
    # without fontconfig may legitimately see None here instead.
    assert font is None or font.is_file()


# Helpers the brief's tests need --------------------------------------------------


def make_sample(path: Path, **kwargs) -> Path:
    """The brief calls an unqualified `make_sample`; bind the real fixture."""
    return fixtures.make_sample(path, **kwargs)


def make_audio_only(path: Path) -> Path:
    """The brief calls an unqualified `make_audio_only`; bind the fixture."""
    return fixtures.make_audio_only(path)


def png_size(png_bytes: bytes) -> tuple[int, int]:
    """The brief calls an unqualified `png_size`; bind the fixture helper."""
    return fixtures.png_size(png_bytes)


def probe_of(path: Path) -> media.ProbeResult:
    """The brief calls `probe_of`; bind it to `media.probe`."""
    return media.probe(path)


def _label_text(seconds: float) -> str:
    """The label format the design fixes: `HH:MM:SS.mmm` burned per tile."""
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
