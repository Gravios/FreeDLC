#!/usr/bin/env bash
# Downscale every video in SRC into DST as <stem>_<W>x<H>.mp4: the whole field of
# view (scaled, never cropped), the same frame count, plain colour H.264. These
# are the processed videos a workspace trains on; register the folder with
#
#   fdlc videos <project> --register DST
#   fdlc extract --all --project <project> --match-original
#
# Usage:
#   ./reduce_videos.sh [--size WxH] [--crf N] [--preset P] [SRC [DST]]
#
#   SRC  folder with the full-resolution videos   (default: ./original)
#   DST  folder for the reduced videos            (default: ./reduced)
#   --size    output size, even numbers           (default: 640x360)
#   --crf     x264 quality, lower is better       (default: 14)
#   --preset  x264 speed/size trade-off           (default: slow)
#
# Re-run safe: existing outputs are skipped and a failed encode leaves nothing
# behind. Labeled outputs (*.fdlc.mp4) and triplet videos in SRC are ignored.
set -u

size="640x360" crf=14 preset="slow"
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --size|--crf|--preset)
      if [ $# -lt 2 ]; then echo "$1 needs a value" >&2; exit 2; fi ;;&
    --size)     size="$2"; shift 2 ;;
    --crf)      crf="$2"; shift 2 ;;
    --preset)   preset="$2"; shift 2 ;;
    -h|--help)  sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)         echo "unknown option: $1" >&2; exit 2 ;;
    *)          args+=("$1"); shift ;;
  esac
done
src="${args[0]:-original}"
dst="${args[1]:-reduced}"

if ! [[ "$size" =~ ^([0-9]+)x([0-9]+)$ ]]; then
  echo "--size must look like 640x360, got: $size" >&2; exit 2
fi
w="${BASH_REMATCH[1]}" h="${BASH_REMATCH[2]}"
if [ $((w % 2)) -ne 0 ] || [ $((h % 2)) -ne 0 ]; then
  echo "--size needs even width and height, got: $size" >&2; exit 2
fi
if [ ! -d "$src" ]; then echo "source folder not found: $src" >&2; exit 1; fi
mkdir -p "$dst"

shopt -s nullglob nocaseglob
ok=0 skip=0 fail=0
for f in "$src"/*.{mp4,mov,avi,mkv,m4v}; do
  case "${f,,}" in *.fdlc.*|*-triplet.*|*.part.mp4) continue ;; esac
  stem="$(basename "${f%.*}")"
  out="$dst/${stem}_${size}.mp4"
  part="$dst/${stem}_${size}.part.mp4"
  if [ -f "$out" ]; then
    echo "skip (exists): $out"
    skip=$((skip + 1))
    continue
  fi
  echo "encoding: $f -> $out"
  # -fps_mode passthrough: never duplicate or drop a frame, so frame N of the
  # reduced video is frame N of the original. </dev/null keeps ffmpeg from
  # reading the terminal.
  if ffmpeg -hide_banner -loglevel warning -stats -y -i "$f" \
      -map 0:v:0 -an -fps_mode passthrough \
      -vf "scale=${w}:${h}:flags=area,setsar=1" \
      -c:v libx264 -preset "$preset" -crf "$crf" -pix_fmt yuv420p \
      -movflags +faststart "$part" </dev/null; then
    mv "$part" "$out"
    ok=$((ok + 1))
  else
    echo "FAILED: $f" >&2
    rm -f "$part"
    fail=$((fail + 1))
  fi
done

echo "done: $ok encoded, $skip skipped, $fail failed"
[ "$fail" -eq 0 ]
