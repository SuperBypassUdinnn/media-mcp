"""Tests for the profile presets and the shared parameter validators.

These are unit tests: the profiles, the timestamp contract, the status block
and the parameter validators are all pure, so they are exercised without a
server process. The stdio contract tests cover the same validators where they
have to reject a real call.
"""

from __future__ import annotations

import pytest

from media import artifacts, server

# The brief's profile test --------------------------------------------------------


def test_profiles_map_to_declared_work():
    assert server.PROFILES["manifest"]["transcript"] is False
    assert server.PROFILES["read"]["transcript"] is True
    assert server.PROFILES["visual"]["ocr"] is True


# Resolutions of the brief --------------------------------------------------------


def test_profiles_are_four_presets_over_six_stages():
    stages = {"probe", "scenes", "audio", "transcript", "ocr", "frames"}
    assert set(server.PROFILES) == {"manifest", "read", "visual", "full"}
    for name, profile in server.PROFILES.items():
        assert set(profile) == stages, name
    assert server.PROFILES["full"] == dict.fromkeys(stages, True)
    assert server.PROFILES["manifest"] == {
        "probe": True,
        "scenes": True,
        "audio": True,
        "transcript": False,
        "ocr": False,
        "frames": False,
    }
    assert server.PROFILES["read"]["frames"] is False
    assert server.PROFILES["read"]["ocr"] is False
    assert server.PROFILES["visual"]["transcript"] is False
    assert server.PROFILES["visual"]["frames"] is True


def test_parse_timestamp_accepts_the_three_forms():
    assert server.parse_timestamp("00:00:05.500") == 5.5
    assert server.parse_timestamp("01:30") == 90.0
    assert server.parse_timestamp("1:02:03") == 3723.0
    assert server.parse_timestamp(12) == 12.0
    assert server.parse_timestamp("12.25") == 12.25
    assert server.parse_timestamp(0) == 0.0


def test_parse_timestamp_rejects_negative_and_junk_naming_the_value():
    for bad in ("-1", -2.5, "junk", "1:2:3:4", "", "12s"):
        with pytest.raises(server.MediaError) as err:
            server.parse_timestamp(bad)
        assert repr(bad) in str(err.value) or str(bad) in str(err.value)


def test_timestamps_render_through_the_project_formatter():
    assert artifacts.format_timestamp(server.parse_timestamp("01:30")) == (
        "00:01:30.000"
    )
    assert artifacts.format_timestamp(server.parse_timestamp(3661.25)) == (
        "01:01:01.250"
    )


def test_status_block_is_four_named_lines_in_order():
    block = server.status_block(
        ["probe", "scenes"],
        ["probe.json"],
        ["transcript: large-v3-turbo (cap 60 candidates)"],
        {"ocr": "the manifest profile does not include OCR"},
    )
    lines = block.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("computed:")
    assert lines[1].startswith("cache:")
    assert lines[2].startswith("capped:")
    assert lines[3].startswith("skipped:")
    assert "probe" in lines[0] and "scenes" in lines[0]
    assert "probe.json" in lines[1]
    assert "transcript" in lines[2] and "cap 60" in lines[2]
    assert "ocr" in lines[3] and "manifest profile" in lines[3]


def test_status_block_says_none_for_every_empty_group():
    block = server.status_block([], [], [], {})
    for line in block.splitlines():
        assert line.endswith("(none)")


def test_window_with_a_timestamp_list_is_an_error_naming_window():
    with pytest.raises(server.MediaError) as err:
        server.validate_window(at=[1, 2], window=5)
    assert "window" in str(err.value).lower()


def test_window_with_scenes_is_an_error_naming_window():
    for at in ("scenes", None):
        with pytest.raises(server.MediaError) as err:
            server.validate_window(at=at, window=5)
        assert "window" in str(err.value).lower()


def test_window_needs_a_positive_span():
    with pytest.raises(server.MediaError) as err:
        server.validate_window(at=1.0, window=0)
    assert "window" in str(err.value).lower()
    server.validate_window(at="00:00:05", window=2)


def test_count_fps_and_k_name_the_value_and_the_cap():
    assert server.validate_count(None) == server.COUNT_DEFAULT
    assert server.validate_count(60) == 60
    for bad in (0, 61, -1):
        with pytest.raises(server.MediaError) as err:
            server.validate_count(bad)
        assert str(bad) in str(err.value) and "60" in str(err.value)

    assert server.validate_fps(30) == 30
    for bad in (-1, 31):
        with pytest.raises(server.MediaError) as err:
            server.validate_fps(bad)
        assert str(bad) in str(err.value) and "30" in str(err.value)

    assert server.validate_k(None) == server.K_DEFAULT
    assert server.validate_k(20) == 20
    for bad in (0, 21):
        with pytest.raises(server.MediaError) as err:
            server.validate_k(bad)
        assert str(bad) in str(err.value) and "20" in str(err.value)


def test_crop_mapping_must_be_complete_integral_and_positive():
    with pytest.raises(server.MediaError) as err:
        server.validate_crop({"x": 0, "y": 0, "width": 10})
    assert "height" in str(err.value)

    with pytest.raises(server.MediaError) as err:
        server.validate_crop({"x": 0, "y": 0, "width": 10.5, "height": 10})
    assert "width" in str(err.value)

    for width in (0, -5):
        with pytest.raises(server.MediaError) as err:
            server.validate_crop({"x": 0, "y": 0, "width": width, "height": 10})
        assert "width" in str(err.value)

    with pytest.raises(server.MediaError) as err:
        server.validate_crop({"x": -1, "y": 0, "width": 10, "height": 10})
    assert "x" in str(err.value)


def test_a_named_whole_number_is_refused_under_its_own_name():
    """The `psm` range reuses the count check but must refuse as `psm`."""
    with pytest.raises(server.MediaError) as err:
        server.validate_count(14, 6, 13, name="psm")
    assert "psm" in str(err.value) and "13" in str(err.value)
    assert server.validate_count(None, 6, 13, name="psm") == 6
    assert server.validate_count(13, 6, 13, name="psm") == 13


def test_crop_unknown_keys_are_errors_and_auto_is_allowed():
    with pytest.raises(server.MediaError) as err:
        server.validate_crop({"x": 0, "y": 0, "width": 10, "height": 10, "pad": 4})
    assert "pad" in str(err.value)

    assert server.validate_crop("auto") == "auto"
    assert server.validate_crop(None) is None
    assert server.validate_crop({"x": 1, "y": 2, "width": 3, "height": 4}) == {
        "x": 1,
        "y": 2,
        "width": 3,
        "height": 4,
    }
    with pytest.raises(server.MediaError) as err:
        server.validate_crop("middle")
    assert "middle" in str(err.value)


def test_settings_read_the_environment_with_the_design_defaults(monkeypatch):
    for name in (
        "MEDIA_CACHE_DIR",
        "MEDIA_CACHE_CAP",
        "MEDIA_DEVICE",
        "MEDIA_VRAM_FLOOR",
        "MEDIA_FRAME_MAX_EDGE",
        "MEDIA_MAX_SOURCE_BYTES",
        "MEDIA_MAX_SOURCE_SECONDS",
        "MEDIA_ALLOWED_READ_ROOTS",
        "MEDIA_CALL_TIMEOUT_SECONDS",
        "MEDIA_ALLOW_PRIVATE_URLS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = server.load_settings()
    assert settings.cache_cap_bytes == 20 * 1024**3
    assert settings.device == "auto"
    assert settings.vram_floor_mib == 2800
    assert settings.frame_max_edge == 1600
    assert settings.max_source_bytes == 4 * 1024**3
    assert settings.max_source_seconds == 14400.0
    assert settings.allowed_read_roots is None
    assert settings.call_timeout_seconds == 900
    assert settings.allow_private_urls is False

    monkeypatch.setenv("MEDIA_CACHE_CAP", "1 GiB")
    monkeypatch.setenv("MEDIA_ALLOWED_READ_ROOTS", "/a:/b")
    monkeypatch.setenv("MEDIA_CALL_TIMEOUT_SECONDS", "99999")
    monkeypatch.setenv("MEDIA_ALLOW_PRIVATE_URLS", "1")
    changed = server.load_settings()
    assert changed.cache_cap_bytes == 1024**3
    assert [str(root) for root in changed.allowed_read_roots] == ["/a", "/b"]
    # The ceiling wins over an over-eager environment value.
    assert changed.call_timeout_seconds == server.TIMEOUT_CEILING
    assert changed.allow_private_urls is True
