"""Tests for the delegated transcription contract.

Every test here stands in for the real pipeline: the committed
`fake_transcribe.sh` is copied into `tmp_path`, so no test needs the
transcription toolchain installed and nothing is written outside the
temporary directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from media import contracts, media

FAKE_SOURCE = Path(__file__).with_name("fake_transcribe.sh")


def _install_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, executable: bool = True
) -> Path:
    """Copy the fake pipeline into `tmp_path` and point the variable at it."""
    fake = tmp_path / "fake_transcribe.sh"
    shutil.copy(FAKE_SOURCE, fake)
    fake.chmod(0o755 if executable else 0o644)
    monkeypatch.setenv("MEDIA_TRANSCRIBE_BIN", str(fake))
    return fake


def _write_script(path: Path, body: str, executable: bool = True) -> Path:
    """Write a stand-in script under `tmp_path` for a single failure shape."""
    path.write_text(body, encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


# Exactly as the brief writes them ------------------------------------------------


def test_missing_binary_is_reported_with_the_variable_and_path(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDIA_TRANSCRIBE_BIN", str(tmp_path / "nope.sh"))
    status = contracts.self_check()
    assert status.ok is False
    assert "MEDIA_TRANSCRIBE_BIN" in status.detail and "nope.sh" in status.detail


def test_argv_uses_an_out_dir_inside_the_cache_entry(tmp_path):
    argv = contracts.build_argv(
        "in.mp4", tmp_path / "entry", "large-v3-turbo", 2, "auto", "auto", False, 900
    )
    assert argv[-2] == "--out" and str(tmp_path / "entry") == argv[-1]


def test_fake_transcribe_produces_the_four_artifacts(tmp_path, monkeypatch):
    _install_fake(tmp_path, monkeypatch)
    result = contracts.run_transcript(
        contracts.build_argv("src.mp4", tmp_path, "tiny", 1, "en", "cpu", False, 60),
        60,
        None,
    )
    assert set(result.artifacts) == {"srt", "vtt", "json", "md"}
    assert result.segments[0]["speaker"] == "SPEAKER_00"


def test_parse_rejects_a_json_without_segments(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text('{"meta": {}}')
    with pytest.raises(contracts.ContractError):
        contracts.parse_transcript_json(bad)


# Resolution and self-check -------------------------------------------------------


def test_resolve_transcribe_bin_defaults_to_the_repo_path(monkeypatch):
    monkeypatch.delenv("MEDIA_TRANSCRIBE_BIN", raising=False)
    expected = Path(__file__).resolve().parents[1] / "transcribe" / "transcribe.sh"
    assert contracts.resolve_transcribe_bin() == expected


def test_resolve_transcribe_bin_honours_the_override(monkeypatch, tmp_path):
    target = tmp_path / "custom.sh"
    monkeypatch.setenv("MEDIA_TRANSCRIBE_BIN", str(target))
    assert contracts.resolve_transcribe_bin() == target


def test_self_check_accepts_an_executable_binary(tmp_path, monkeypatch):
    fake = _install_fake(tmp_path, monkeypatch)
    status = contracts.self_check()
    assert status.ok is True
    assert str(fake) in status.detail


def test_self_check_rejects_a_non_executable_copy(tmp_path, monkeypatch):
    fake = _install_fake(tmp_path, monkeypatch, executable=False)
    status = contracts.self_check()
    assert status.ok is False
    assert str(fake) in status.detail
    assert "MEDIA_TRANSCRIBE_BIN" in status.detail


def test_self_check_reports_a_failing_help_probe(tmp_path, monkeypatch):
    script = _write_script(
        tmp_path / "broken.sh", "#!/bin/sh\necho 'no help here' >&2\nexit 2\n"
    )
    monkeypatch.setenv("MEDIA_TRANSCRIBE_BIN", str(script))
    status = contracts.self_check()
    assert status.ok is False
    assert "2" in status.detail and "no help here" in status.detail


# Argument building ---------------------------------------------------------------


def test_build_argv_emits_only_flags_the_pipeline_accepts(tmp_path):
    entry = tmp_path / "entry"
    argv = contracts.build_argv(
        "in.mp4", entry, "medium", 2, "en", "cuda", queue_flag=True, timeout_seconds=600
    )
    assert argv[-2:] == ["--out", str(entry)]
    body = argv[2:-2]
    flags = {token for token in body if token.startswith("--")}
    assert flags <= {
        "--model",
        "--num-speakers",
        "--no-diarize",
        "--language",
        "--device",
        "--condition-on-previous-text",
    }
    assert body == [
        "--model",
        "medium",
        "--num-speakers",
        "2",
        "--language",
        "en",
        "--device",
        "cuda",
        "--condition-on-previous-text",
    ]


def test_build_argv_skips_auto_model_and_uses_no_diarize_at_zero(tmp_path):
    argv = contracts.build_argv("in.mp4", tmp_path, "auto", 0, "", "", False, 60)
    assert argv[2:-2] == ["--no-diarize", "--condition-on-previous-text"]


# Running the delegated pipeline --------------------------------------------------


def test_run_transcript_reports_the_exit_code_and_stderr(tmp_path, monkeypatch):
    _install_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_TRANSCRIBE_FAIL", "1")
    with pytest.raises(contracts.ContractError) as err:
        contracts.run_transcript(
            contracts.build_argv(
                "src.mp4", tmp_path / "out", "tiny", 1, "en", "cpu", False, 60
            ),
            60,
            None,
        )
    assert "exited 3" in str(err.value)
    assert "deliberate failure" in str(err.value)


def test_run_transcript_names_the_missing_artifact(tmp_path, monkeypatch):
    _install_fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_TRANSCRIBE_PARTIAL", "1")
    out_dir = tmp_path / "out"
    with pytest.raises(contracts.ContractError) as err:
        contracts.run_transcript(
            contracts.build_argv("src.mp4", out_dir, "tiny", 1, "en", "cpu", False, 60),
            60,
            None,
        )
    message = str(err.value)
    assert "md" in message
    assert str(out_dir / "src") in message


def test_tier_prefers_the_confirmed_meta_tier(tmp_path, monkeypatch):
    _install_fake(tmp_path, monkeypatch)
    result = contracts.run_transcript(
        contracts.build_argv("src.mp4", tmp_path, "small", 1, "en", "cpu", False, 60),
        60,
        None,
    )
    assert result.tier == "tiny"


def test_tier_is_unknown_when_nothing_confirmed_one(tmp_path, monkeypatch):
    script = _write_script(
        tmp_path / "bare.sh",
        "#!/bin/sh\n"
        "set -eu\n"
        "out=./transcripts\n"
        'while [ "$#" -gt 0 ]; do\n'
        '    case "$1" in\n'
        "        --out) out=$2; shift 2 ;;\n"
        "        *) shift ;;\n"
        "    esac\n"
        "done\n"
        'mkdir -p "$out/src"\n'
        ': > "$out/src/src.srt"\n'
        ': > "$out/src/src.vtt"\n'
        ': > "$out/src/src.md"\n'
        """printf '%s\\n' '{"segments": []}' > "$out/src/src.json"\n""",
    )
    monkeypatch.setenv("MEDIA_TRANSCRIBE_BIN", str(script))
    result = contracts.run_transcript(
        contracts.build_argv("src.mp4", tmp_path / "out", "", 0, "", "", False, 60),
        60,
        None,
    )
    assert result.tier == "unknown"


# JSON parsing --------------------------------------------------------------------


def test_parse_accepts_a_segment_without_a_speaker(tmp_path):
    payload = {
        "meta": {"model": "tiny"},
        "segments": [{"start": 0.0, "end": 1.0, "text": "halo"}],
    }
    path = tmp_path / "x.json"
    path.write_text(json.dumps(payload))
    segments, meta = contracts.parse_transcript_json(path)
    assert segments == payload["segments"]
    assert meta == {"model": "tiny"}


@pytest.mark.parametrize(
    "body",
    [
        "[]",
        "not json at all",
        '{"segments": {}}',
        '{"segments": [{"start": 0.0}]}',
    ],
)
def test_parse_rejects_wrong_shapes_naming_the_path(tmp_path, body):
    bad = tmp_path / "x.json"
    bad.write_text(body)
    with pytest.raises(contracts.ContractError) as err:
        contracts.parse_transcript_json(bad)
    assert str(bad) in str(err.value)


def test_parse_rejects_a_missing_file(tmp_path):
    missing = tmp_path / "absent.json"
    with pytest.raises(contracts.ContractError) as err:
        contracts.parse_transcript_json(missing)
    assert str(missing) in str(err.value)


def test_contract_error_is_a_media_error():
    assert issubclass(contracts.ContractError, media.MediaError)
