#!/bin/sh
# A stand-in for yt-dlp used only by the test suite.
#
# It never touches the network: the prepared fixture named by
# FAKE_YTDLP_SOURCE is copied to the `-o` target, with the two template
# fields the ingest layer uses (`%(id)s` and `%(ext)s`) resolved the way the
# real binary resolves them for a direct URL. The script is committed without
# the executable bit; each test copies it to a directory on PATH, chmods it,
# and names the copy `yt-dlp`.
#
# Switches:
#   FAKE_YTDLP_SOURCE=FILE       the file copied to the -o target (required)
#   FAKE_YTDLP_EXTRA=FILE        an extra file copied beside the target, to
#                                simulate a run that leaves several files
#   FAKE_YTDLP_MARKFILE=FILE     append one line per run, before doing any work
#   FAKE_YTDLP_FAIL=1            write FAKE_YTDLP_STDERR to stderr, exit 7
#   FAKE_YTDLP_STDERR=TEXT       the stderr text of that failure, verbatim
#   FAKE_YTDLP_SLEEP=SECONDS     sleep before doing any work
#
# A successful run writes one line of its own to stderr, so a test can prove
# that the ingest layer carries the child's final stderr lines even on success.
set -eu

if [ -n "${FAKE_YTDLP_MARKFILE:-}" ]; then
    printf 'invoked\n' >>"${FAKE_YTDLP_MARKFILE}"
fi

if [ -n "${FAKE_YTDLP_SLEEP:-}" ]; then
    sleep "${FAKE_YTDLP_SLEEP}"
fi

if [ "${FAKE_YTDLP_FAIL:-0}" = "1" ]; then
    printf '%s\n' "${FAKE_YTDLP_STDERR:-fake yt-dlp: deliberate failure}" >&2
    exit 7
fi

source_file=${FAKE_YTDLP_SOURCE:-}
if [ -z "$source_file" ] || [ ! -f "$source_file" ]; then
    echo "fake yt-dlp: FAKE_YTDLP_SOURCE must name a file" >&2
    exit 2
fi

target=
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o | --output)
            if [ "$#" -ge 2 ]; then
                target=$2
                shift 2
            else
                shift
            fi
            ;;
        --output=*)
            target=${1#--output=}
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$target" ]; then
    echo "fake yt-dlp: no -o target was given" >&2
    exit 2
fi

base=$(basename "$source_file")
extension=${base##*.}
stem=${base%.*}
target=$(printf '%s' "$target" |
    sed -e "s/%(ext)s/${extension}/g" -e "s/%(id)s/${stem}/g")

mkdir -p "$(dirname "$target")"
cp "$source_file" "$target"
if [ -n "${FAKE_YTDLP_EXTRA:-}" ] && [ -f "${FAKE_YTDLP_EXTRA}" ]; then
    cp "${FAKE_YTDLP_EXTRA}" "$(dirname "$target")/$(basename "${FAKE_YTDLP_EXTRA}")"
fi
echo "fake yt-dlp: wrote $target" >&2
