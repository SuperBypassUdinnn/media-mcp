"""Tests for the probe layer: source validation, ffprobe, and source limits."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from media import media
from tests import fixtures

# Exactly as the brief writes them ------------------------------------------------


def test_probe_reports_duration_and_streams(tmp_path):
    result = media.probe(make_sample(tmp_path / "s.mp4", duration=6))
    assert result.has_video and result.has_audio
    assert 5.0 <= result.duration <= 7.0


def test_offset_is_detected(tmp_path):
    result = media.probe(
        make_sample(tmp_path / "off.mp4", duration=6, audio_offset=1.0)
    )
    assert abs(media.av_offset_seconds(result)) > 0.5


def test_directory_source_is_refused_with_the_path_named(tmp_path):
    with pytest.raises(media.MediaError) as err:
        media.validate_source(str(tmp_path), None)
    assert str(tmp_path) in str(err.value)


def test_zero_byte_file_surfaces_ffprobe_stderr(tmp_path):
    bad = tmp_path / "empty.mp4"
    bad.write_bytes(b"")
    with pytest.raises(media.MediaError) as err:
        media.probe(bad)
    assert "ffprobe" in str(err.value).lower()


def test_unicode_and_space_paths_work(tmp_path):
    f = make_sample(tmp_path / "Rekaman Uji Coba 1.mp4", duration=4)
    assert media.probe(f).has_video


def test_duration_beyond_warning_threshold_is_reported_not_refused(tmp_path):
    result = media.probe(make_sample(tmp_path / "long.mp4", duration=4))
    long = media.with_duration(result, 7300.0)  # pure helper, no 2h fixture
    assert media.source_limits(long)["warn"] is True
    assert media.source_limits(long)["refuse"] is False
    assert media.source_limits(media.with_duration(result, 20000.0))["refuse"] is True


# Helpers the brief's tests need --------------------------------------------------


def make_sample(path: Path, **kwargs) -> Path:
    """The brief calls an unqualified `make_sample`; bind the real fixture."""
    return fixtures.make_sample(path, **kwargs)


# Additional coverage the brief's produced interfaces promise ---------------------


def test_missing_probe_target_is_refused_with_the_path_named(tmp_path):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(media.MediaError) as err:
        media.probe(missing)
    assert str(missing) in str(err.value)


def test_corrupt_file_is_a_media_error_not_a_traceback(tmp_path):
    bad = tmp_path / "junk.mp4"
    bad.write_bytes(b"not an mp4 container at all" * 64)
    with pytest.raises(media.MediaError):
        media.probe(bad)


def test_probe_reports_dimensions_codecs_and_fps(tmp_path):
    result = media.probe(fixtures.make_sample(tmp_path / "d.mp4", duration=2))
    assert (result.width, result.height) == fixtures.VIDEO_SIZE
    assert result.fps == pytest.approx(25.0, abs=0.1)
    assert result.video_codec == "h264"
    assert result.audio_codec == "aac"
    assert result.size_bytes > 0


def test_audio_only_probe_has_no_video(tmp_path):
    result = media.probe(fixtures.make_audio_only(tmp_path / "a.m4a"))
    assert result.has_audio and not result.has_video
    assert result.video_codec == ""
    assert result.width == 0 and result.height == 0


def test_plain_sample_offset_is_near_zero(tmp_path):
    result = media.probe(fixtures.make_sample(tmp_path / "p.mp4", duration=4))
    assert abs(media.av_offset_seconds(result)) < 0.3


# Probe metadata: container, audio layout, and per-stream start_time (fix M4) ------


def test_probe_reports_container_layout_and_stream_start_times(tmp_path):
    result = media.probe(fixtures.make_sample(tmp_path / "meta.mp4", duration=4))
    assert "mp4" in result.container
    assert result.audio_layout == "mono"
    assert isinstance(result.video_start_time, float)
    assert isinstance(result.audio_start_time, float)
    assert result.audio_offset == pytest.approx(
        result.audio_start_time - result.video_start_time, abs=1e-6
    )


def test_probe_reports_the_raw_start_times_of_a_delayed_audio_stream(tmp_path):
    result = media.probe(
        fixtures.make_sample(tmp_path / "delayed.mp4", duration=4, audio_offset=1.0)
    )
    assert result.video_start_time is not None
    assert result.audio_start_time is not None
    raw_difference = result.audio_start_time - result.video_start_time
    assert raw_difference == pytest.approx(1.0, abs=0.05)


def test_audio_only_probe_has_no_video_start_time_and_still_reports_the_container(
    tmp_path,
):
    result = media.probe(fixtures.make_audio_only(tmp_path / "a.m4a"))
    assert result.video_start_time is None
    assert result.audio_start_time is not None
    assert result.container
    assert result.audio_layout
    assert result.audio_offset == 0.0


def test_validate_source_returns_the_file_inside_roots(tmp_path):
    root = tmp_path / "roots"
    f = fixtures.make_sample(root / "in.mp4", duration=2)
    assert media.validate_source(str(f), [root]) == f
    assert media.validate_source(str(f), None) == f


def test_validate_source_refuses_missing_path(tmp_path):
    with pytest.raises(media.MediaError) as err:
        media.validate_source(str(tmp_path / "gone.mp4"), None)
    assert "gone.mp4" in str(err.value)


def test_validate_source_refuses_symlink_escaping_the_roots(tmp_path):
    root = tmp_path / "roots"
    root.mkdir()
    outside = fixtures.make_sample(tmp_path / "out.mp4", duration=2)
    link = root / "link.mp4"
    link.symlink_to(outside)
    with pytest.raises(media.MediaError) as err:
        media.validate_source(str(link), [root])
    assert str(outside.resolve()) in str(err.value)


def test_validate_source_refuses_unreadable_file(tmp_path):
    f = fixtures.make_sample(tmp_path / "locked.mp4", duration=2)
    f.chmod(0)
    if os.access(f, os.R_OK):
        pytest.skip("file permissions are not enforced for this user")
    with pytest.raises(media.MediaError):
        media.validate_source(str(f), None)


def test_with_duration_is_a_pure_copy(tmp_path):
    result = media.probe(fixtures.make_sample(tmp_path / "w.mp4", duration=2))
    copy = media.with_duration(result, 99.0)
    assert copy.duration == 99.0
    assert result.duration < 4.0  # the original is untouched
    assert copy == dataclasses.replace(result, duration=99.0)


def test_default_limit_constants_match_the_design():
    assert media.MEDIA_MAX_SOURCE_SECONDS == 14400
    assert media.MEDIA_MAX_SOURCE_BYTES == 4 * 1024**3


def _result(duration: float) -> media.ProbeResult:
    """A minimal hand-built result for limit boundary checks."""
    return media.ProbeResult(
        duration=duration,
        width=0,
        height=0,
        fps=0.0,
        video_codec="",
        audio_codec="",
        has_video=False,
        has_audio=False,
        audio_offset=0.0,
        size_bytes=0,
    )


def test_source_limits_border_warnings_at_half_the_cap():
    assert media.source_limits(_result(7200.0))["warn"] is False
    assert media.source_limits(_result(7200.5))["warn"] is True
    assert media.source_limits(_result(7200.5))["refuse"] is False


def test_source_limits_refuse_beyond_the_hard_cap():
    assert media.source_limits(_result(14400.0))["refuse"] is False
    assert media.source_limits(_result(14400.5))["refuse"] is True


def test_source_limits_use_bytes_as_well():
    big = dataclasses.replace(_result(1.0), size_bytes=media.MEDIA_MAX_SOURCE_BYTES + 1)
    limits = media.source_limits(big)
    assert limits["refuse"] is True and limits["warn"] is True
    small = dataclasses.replace(
        _result(1.0), size_bytes=media.MEDIA_MAX_SOURCE_BYTES // 2 + 1
    )
    assert media.source_limits(small) == {"warn": True, "refuse": False}
