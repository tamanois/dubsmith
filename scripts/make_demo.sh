#!/usr/bin/env bash
# Regenerate the README before/after GIF and the GitHub social-preview poster.
#
#   scripts/make_demo.sh path/to/original.mp4 path/to/dubbed.mp4
#
# The curated outputs are written to docs/media/. Keep the committed total small:
# the GIF is intentionally 720 px wide and 8 fps.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ORIGINAL.mp4 DUBBED.mp4" >&2
  exit 2
fi

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg is required" >&2
  exit 1
}

original=$1
dubbed=$2
out_dir=$(cd "$(dirname "$0")/.." && pwd)/docs/media
font=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf

mkdir -p "$out_dir"

# Copy the full excerpts without container metadata.
ffmpeg -y -v error -i "$original" -map 0 -c copy -map_metadata -1 \
  "$out_dir/demo-before.mp4"
ffmpeg -y -v error -i "$dubbed" -map 0 -c copy -map_metadata -1 \
  "$out_dir/demo-after.mp4"

# 6 s side-by-side loop, small enough for a README.
ffmpeg -y -v error -ss 1 -t 6 \
  -i "$out_dir/demo-before.mp4" -ss 1 -t 6 -i "$out_dir/demo-after.mp4" \
  -filter_complex "[0:v]scale=360:-2,fps=8,setpts=PTS-STARTPTS[l];\
[1:v]scale=360:-2,fps=8,setpts=PTS-STARTPTS[r];\
[l][r]hstack=inputs=2,split[s0][s1];\
[s0]palettegen=max_colors=64[p];[s1][p]paletteuse" \
  -loop 0 "$out_dir/demo-before-after.gif"

# 1280x640 poster with labels; also suitable for GitHub's social preview.
ffmpeg -y -v error -ss 2 -i "$out_dir/demo-before.mp4" \
  -ss 2 -i "$out_dir/demo-after.mp4" \
  -filter_complex "[0:v]scale=640:640:force_original_aspect_ratio=increase,\
crop=640:640,drawtext=fontfile=${font}:text='ORIGINAL':fontcolor=white:\
fontsize=28:box=1:boxcolor=black@0.65:boxborderw=12:x=24:y=24[l];\
[1:v]scale=640:640:force_original_aspect_ratio=increase,crop=640:640,\
drawtext=fontfile=${font}:text='DUBBED — FRENCH':fontcolor=white:fontsize=28:\
box=1:boxcolor=black@0.65:boxborderw=12:x=24:y=24[r];\
[l][r]hstack=inputs=2" \
  -frames:v 1 "$out_dir/demo-poster.png"

echo "wrote $out_dir/demo-before-after.gif and $out_dir/demo-poster.png"
