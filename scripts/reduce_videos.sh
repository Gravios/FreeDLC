#!/usr/bin/env bash
#
# Reduce every video in ./original to 192x108 in ./reduced, applying an identical
# filter chain to each file. Re-run safe: outputs that already exist are skipped, so
# an interrupted batch resumes where it left off. A file that fails to encode has its
# partial output removed so the next run retries it rather than skipping it.
#
# Usage:   ./reduce_videos.sh                 # uses ./original -> ./reduced
#          ./reduce_videos.sh SRC DST         # or point at other folders

set -u

src="${1:-original}"
dst="${2:-reduced}"

if [ ! -d "$src" ]; then
  echo "source folder not found: $src" >&2
  exit 1
fi
mkdir -p "$dst"

# match common video extensions, case-insensitively; expand to nothing if none match
shopt -s nullglob nocaseglob

ok=0 skip=0 fail=0
for f in "$src"/*.{mp4,mov,avi,mkv,m4v}; do
  stem="$(basename "${f%.*}")"
  out="$dst/${stem}_192x108.mp4"

  if [ -f "$out" ]; then
    echo "skip (exists): $out"
    skip=$((skip + 1))
    continue
  fi

  echo "encoding: $f -> $out"
  if ffmpeg -hide_banner -loglevel warning -stats -i "$f" \
      -vf "scale=192:108:flags=bicubic" \
      -c:v libx264 -preset veryslow -crf 12 -pix_fmt yuv420p \
      "$out"; then
    ok=$((ok + 1))
  else
    echo "FAILED: $f" >&2
    rm -f "$out"          # don't leave a truncated file a later run would treat as done
    fail=$((fail + 1))
  fi
done

echo "done: $ok encoded, $skip skipped, $fail failed"
[ "$fail" -eq 0 ]
