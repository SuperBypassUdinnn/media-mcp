"""Tests for audio event detection: silence segments and loudness peaks."""

from __future__ import annotations

from media import media
from tests import fixtures

_KINDS = {"silence_start", "silence_end", "loud"}


def test_loud_spike_yields_a_loud_event_and_the_quiet_stretch_stays_quiet(tmp_path):
    sample = fixtures.make_sample(tmp_path / "spike.mp4", duration=8, spike_at=4)
    events = media.detect_audio_events(sample)
    loud = [event for event in events if event.kind == "loud"]
    assert loud, f"a deliberate spike must yield a loud event, got {events}"
    assert any(abs(event.timestamp - 4.0) <= 0.5 for event in loud), (
        f"no loud event within half a second of the spike: {loud}"
    )
    assert all(abs(event.timestamp - 4.0) <= 0.5 for event in loud), (
        f"the quiet tone must not be reported as a peak: {loud}"
    )
    _assert_ordered_and_bounded(events, duration=8.0)


def test_bounded_silent_stretch_yields_a_bracketing_pair(tmp_path):
    clip = fixtures.make_audio_gap(
        tmp_path / "gap.m4a", duration=6, gap_at=2.0, gap_seconds=1.0
    )
    events = media.detect_audio_events(clip)
    starts = [event for event in events if event.kind == "silence_start"]
    ends = [event for event in events if event.kind == "silence_end"]
    assert len(starts) == 1, f"one bounded gap must yield one start: {events}"
    assert len(ends) == 1, f"one bounded gap must yield one end: {events}"
    assert abs(starts[0].timestamp - 2.0) <= 0.5, starts[0].timestamp
    assert abs(ends[0].timestamp - 3.0) <= 0.5, ends[0].timestamp
    assert starts[0].timestamp < ends[0].timestamp
    assert starts[0].timestamp <= 2.5 <= ends[0].timestamp
    assert starts[0].detail == "00:00:02.005", starts[0].detail
    assert ends[0].detail == "00:00:01.003", ends[0].detail
    _assert_ordered_and_bounded(events, duration=6.0)


def test_audio_only_source_is_accepted_and_produces_events(tmp_path):
    tone = fixtures.make_audio_only(tmp_path / "tone.m4a")
    result = media.probe(tone)
    assert result.has_audio and not result.has_video
    # A steady tone has no silence segment and no peak above its own baseline;
    # having nothing to report is a result, never an error.
    assert media.detect_audio_events(tone) == []

    gap = fixtures.make_audio_gap(tmp_path / "gap.m4a")
    events = media.detect_audio_events(gap)
    assert {event.kind for event in events} == {"silence_start", "silence_end"}


def test_source_without_audio_returns_no_events(tmp_path):
    clip = fixtures.make_motion_patch(tmp_path / "mute.mp4", duration=2)
    assert media.probe(clip).has_audio is False
    assert media.detect_audio_events(clip) == []


def test_noise_db_above_the_file_level_removes_the_loud_event(tmp_path):
    sample = fixtures.make_sample(tmp_path / "spike.mp4", duration=8, spike_at=4)
    default = media.detect_audio_events(sample)
    assert any(event.kind == "loud" for event in default), default

    # The fixture's loudest peak measures about -4 dB, so 0 dB sits above every
    # level in the file and the peak rule's absolute gate drops the spike. The
    # silence side of a raised gate is asserted on an audio-only tone in
    # test_noise_db_turns_an_audible_source_into_one_cover_all_pair, because
    # this fixture already carries silence segments at the default thresholds.
    raised = media.detect_audio_events(sample, noise_db=0.0)
    assert not any(event.kind == "loud" for event in raised), raised
    assert {event.kind for event in raised} <= {"silence_start", "silence_end"}


def test_silent_audio_only_source_reports_one_cover_all_pair(tmp_path):
    clip = fixtures.make_silent_audio(tmp_path / "silent.m4a", duration=4)
    result = media.probe(clip)
    events = media.detect_audio_events(clip)
    assert [(event.kind, event.timestamp) for event in events] == [
        ("silence_start", 0.0),
        ("silence_end", result.duration),
    ]
    # The decoded silent tail runs past the container (4.010667 s for a 4.0 s
    # file); the pair must still close at the probed duration rather than lose
    # its end and leave the start dangling.
    assert events[1].timestamp == result.duration


def test_trailing_silence_pair_closes_inside_the_probed_duration(tmp_path):
    clip = fixtures.make_audio_gap(
        tmp_path / "tail.m4a", duration=6, gap_at=4.5, gap_seconds=1.5
    )
    result = media.probe(clip)
    events = media.detect_audio_events(clip)
    assert [event.kind for event in events] == ["silence_start", "silence_end"]
    start, end = events
    assert 0.0 <= start.timestamp < end.timestamp <= result.duration
    assert start.timestamp >= 4.0
    assert end.timestamp == result.duration


def test_noise_db_turns_an_audible_source_into_one_cover_all_pair(tmp_path):
    clip = fixtures.make_audio_gap(tmp_path / "gap.m4a", duration=6)
    result = media.probe(clip)
    default = media.detect_audio_events(clip)
    assert [event.kind for event in default] == ["silence_start", "silence_end"]
    assert default[0].timestamp < 2.5 < default[1].timestamp

    # The tone peaks around -12 dB, so the raised gate makes the whole clip
    # silent. If noise_db were not passed to the filter, the pair would still
    # bracket only the gap; if the pair were filtered per event, the end past
    # the container duration would vanish.
    raised = media.detect_audio_events(clip, noise_db=0.0)
    assert [(event.kind, event.timestamp) for event in raised] == [
        ("silence_start", 0.0),
        ("silence_end", result.duration),
    ]


def test_min_silence_longer_than_the_gap_removes_the_pair(tmp_path):
    clip = fixtures.make_audio_gap(tmp_path / "gap.m4a", duration=6)
    events = media.detect_audio_events(clip, min_silence=2.0)
    assert not any(event.kind.startswith("silence") for event in events), events


def _assert_ordered_and_bounded(events, duration):
    stamps = [event.timestamp for event in events]
    assert stamps == sorted(stamps), f"events are not ordered: {stamps}"
    assert all(0.0 <= stamp <= duration for stamp in stamps), stamps
    assert all(event.kind in _KINDS for event in events)
    assert all(isinstance(event.detail, str) and event.detail for event in events)
