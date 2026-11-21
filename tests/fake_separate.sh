#!/usr/bin/env bash
# A stand-in for the Roformer separation runner, used only by the test suite.
#
# Mirrors the delegated runner contract without any torch/UVR toolchain:
#   argv[1] = input media wav   argv[2] = output directory
# Writes vocals.wav (synthesized 300 Hz warbling tone with tremolo = "vocal")
# and instrumental.wav (a quiet 440 Hz bed) into the output directory.
#
# Switches:
#   FAKE_SEPARATE_FAIL=1   exit 3 with a clear stderr line (broken runner path)
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: fake_separate.sh <media.wav> <out_dir>" >&2
    exit 2
fi

input=$1
out=$2

if [ "${FAKE_SEPARATE_FAIL:-0}" = "1" ]; then
    echo "fake separate: deliberate failure requested" >&2
    exit 3
fi

mkdir -p "$out"

# Duration of the input, measured so the stems match the mix length.
duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input" 2>/dev/null || echo 6)
duration=${duration:-6}

# Vocals: sustained 300 Hz tone -> "vocal" stem (constant RMS => full-span timeline).
ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i "sine=frequency=300:duration=${duration}" \
    -ac 1 -ar 22050 "$out/vocals.wav" >/dev/null 2>&1

# Instrumental: quiet 440 Hz bed -> "instrumental" stem.
ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i "sine=frequency=440:duration=${duration}" \
    -af "volume=0.1" \
    -ac 1 -ar 22050 "$out/instrumental.wav" >/dev/null 2>&1

echo "fake separate: wrote vocals.wav + instrumental.wav to $out"
