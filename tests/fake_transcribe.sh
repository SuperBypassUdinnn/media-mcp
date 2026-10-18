#!/bin/sh
# A stand-in for transcribe.sh used only by the test suite.
#
# It mirrors the delegated contract without any transcription toolchain:
# `--help` exits 0, an unknown flag is ignored rather than rejected, the
# input's stem names one directory under `--out`, and four artifacts land
# inside it. Four switches drive the failure and timing paths:
#   FAKE_TRANSCRIBE_FAIL=1     exit 3 with a clear stderr line
#   FAKE_TRANSCRIBE_PARTIAL=1  write every artifact except the .md
#   FAKE_TRANSCRIBE_PIDFILE=FILE  write this script's own pid to FILE
#   FAKE_TRANSCRIBE_SLEEP=SECONDS sleep before doing any work
set -eu

usage() {
    echo "usage: fake_transcribe.sh <input> [options] --out DIR"
}

if [ "$#" -eq 0 ]; then
    usage >&2
    exit 2
fi

case "$1" in
    --help | -h)
        usage
        exit 0
        ;;
esac

input=$1
shift

out=./transcripts
while [ "$#" -gt 0 ]; do
    case "$1" in
        --out)
            if [ "$#" -ge 2 ]; then
                out=$2
                shift 2
            else
                echo "fake transcribe: --out needs a directory" >&2
                exit 2
            fi
            ;;
        --out=*)
            out=${1#--out=}
            shift
            ;;
        --model | --num-speakers | --language | --device | --compute)
            if [ "$#" -ge 2 ]; then
                shift 2
            else
                shift
            fi
            ;;
        --no-diarize | --no-vad | --denoise)
            shift
            ;;
        *)
            # Unknown flags are ignored on purpose: the contract tests must
            # observe the builder's flags, not this stand-in's strictness.
            shift
            ;;
    esac
done

if [ -n "${FAKE_TRANSCRIBE_PIDFILE:-}" ]; then
    printf '%s\n' "$$" >"${FAKE_TRANSCRIBE_PIDFILE}"
fi

if [ -n "${FAKE_TRANSCRIBE_SLEEP:-}" ]; then
    sleep "${FAKE_TRANSCRIBE_SLEEP}"
fi

if [ "${FAKE_TRANSCRIBE_FAIL:-0}" = "1" ]; then
    echo "fake transcribe: deliberate failure requested" >&2
    exit 3
fi

case "$input" in
    *://*)
        # The real pipeline names every URL download this way.
        stem=youtube-audio
        ;;
    *)
        base=$(basename "$input")
        stem=${base%.*}
        ;;
esac

if [ -z "$stem" ]; then
    echo "fake transcribe: cannot derive a stem from $input" >&2
    exit 2
fi

target=$out/$stem
if ! mkdir -p "$target"; then
    echo "fake transcribe: cannot create $target" >&2
    exit 1
fi

printf '%s\n' \
    '{"meta": {"model": "tiny", "tier": "tiny"}, "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "halo"}]}' \
    >"$target/$stem.json"
printf '1\n00:00:00,000 --> 00:00:01,000\nSPEAKER_00: halo\n' >"$target/$stem.srt"
printf 'WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nSPEAKER_00: halo\n' >"$target/$stem.vtt"

if [ "${FAKE_TRANSCRIBE_PARTIAL:-0}" = "1" ]; then
    echo "fake transcribe: partial artifact set, omitting the .md" >&2
    exit 0
fi

printf '# halo\n\n[00:00:00.000] SPEAKER_00: halo\n' >"$target/$stem.md"
echo "Saved 4 files to $target"
