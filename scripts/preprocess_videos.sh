#!/usr/bin/env bash
# Prepare every video under a folder tree for FreeDLC: downscale it to the size
# the models train on, as plain H.264 MP4, with every frame kept.
#
#   <dir>/K3_cam.h264  ->  <dir>/K3_cam_640x360.mp4
#
# Raw H.264/H.265 streams (.h264 .264 .h265 .hevc, e.g. from Raspberry Pi
# cameras) carry no usable timestamps, so ffmpeg would otherwise put every frame
# at time 0 and keep only a few. Their frames are numbered at --fps instead:
#
#   ffmpeg -fflags +genpts -r FPS -i IN -vf "setpts=N/(FPS*TB),scale=W:H" \
#          -fps_mode passthrough -c:v libx264 -crf CRF -preset PRESET OUT
#
# Videos already in a container (.mp4 .mov .avi .mkv .m4v) keep their own
# timing and are only scaled, never retimed.
#
# Usage:
#   ./preprocess_videos.sh [options] ROOT
#
#   ROOT          folder searched recursively
#   --out DIR     write into DIR, mirroring ROOT's subfolders (default: beside each video)
#   --size WxH    output size, even numbers         (default: 640x360)
#   --fps N       frame rate of raw streams          (default: 25)
#   --crf N       x264 quality, lower is better      (default: 18)
#   --preset P    x264 speed/size trade-off          (default: fast)
#   --raw-only    only raw streams; leave container videos alone
#   -n            list what would be done, encode nothing
#
# Re-run safe: an existing output is skipped, and a failed encode leaves nothing
# behind (it is written as .part.mp4 first). Outputs (*_<W>x<H>.mp4), labeled
# videos (*.fdlc.*) and triplet videos (*-triplet.*) are never taken as input.
# Every output's frame count is checked against its source's (which reads both
# files once more); an output that differs is kept and reported as CHECK.
set -u

size="640x360" fps=25 crf=18 preset="fast" out="" raw_only=0 dry=0
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --size|--fps|--crf|--preset|--out)
      if [ $# -lt 2 ]; then echo "$1 needs a value" >&2; exit 2; fi ;;&
    --size)     size="$2"; shift 2 ;;
    --fps)      fps="$2"; shift 2 ;;
    --crf)      crf="$2"; shift 2 ;;
    --preset)   preset="$2"; shift 2 ;;
    --out)      out="$2"; shift 2 ;;
    --raw-only) raw_only=1; shift ;;
    -n)         dry=1; shift ;;
    -h|--help)  sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)         echo "unknown option: $1" >&2; exit 2 ;;
    *)          args+=("$1"); shift ;;
  esac
done
if [ ${#args[@]} -ne 1 ]; then echo "usage: $0 [options] ROOT   (--help for more)" >&2; exit 2; fi
root="${args[0]%/}"
[ -d "$root" ] || { echo "not a folder: $root" >&2; exit 1; }

if ! [[ "$size" =~ ^([0-9]+)x([0-9]+)$ ]]; then echo "--size must look like 640x360, got: $size" >&2; exit 2; fi
w="${BASH_REMATCH[1]}" h="${BASH_REMATCH[2]}"
if [ $((w % 2)) -ne 0 ] || [ $((h % 2)) -ne 0 ]; then echo "--size needs even width and height, got: $size" >&2; exit 2; fi
if ! [[ "$fps" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "--fps must be a number, got: $fps" >&2; exit 2; fi

count_frames() {  # frames ffmpeg decodes from a file (reads it all)
  ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames \
    -of csv=p=0 "$1" 2>/dev/null | head -1
}

ok=0 skip=0 fail=0 check=0
while IFS= read -r -d '' f; do
  name="$(basename "$f")"
  lower="${name,,}"
  case "$lower" in
    *.fdlc.*|*-triplet.*|*.part.mp4|*_[0-9]*x[0-9]*.mp4) continue ;;
  esac
  case "$lower" in
    *.h264|*.264|*.h265|*.hevc) raw=1 ;;
    *.mp4|*.mov|*.avi|*.mkv|*.m4v) raw=0; [ "$raw_only" -eq 1 ] && continue ;;
    *) continue ;;
  esac
  rel="${f#"$root"/}"
  dir="$(dirname "$rel")"
  dest_dir="${out:-$root}/$dir"
  stem="${name%.*}"
  dest="$dest_dir/${stem}_${size}.mp4"
  part="$dest_dir/${stem}_${size}.part.mp4"
  if [ -f "$dest" ]; then
    echo "skip (exists): $dest"; skip=$((skip + 1)); continue
  fi
  if [ "$raw" -eq 1 ]; then
    inopts=(-fflags +genpts -r "$fps")
    vf="setpts=N/(${fps}*TB),scale=${w}:${h}"
  else
    inopts=()
    vf="scale=${w}:${h}"
  fi
  if [ "$dry" -eq 1 ]; then
    echo "would encode ($([ "$raw" -eq 1 ] && echo "raw, ${fps} fps" || echo container)): $f -> $dest"
    ok=$((ok + 1)); continue
  fi
  mkdir -p "$dest_dir"
  echo "encoding: $f -> $dest"
  # -fps_mode passthrough: never duplicate or drop a frame, so frame N of the
  # output is frame N of the source. </dev/null keeps ffmpeg off the terminal.
  if ffmpeg -hide_banner -loglevel error -stats -y "${inopts[@]}" -i "$f" \
      -map 0:v:0 -an -vf "$vf" -fps_mode passthrough \
      -c:v libx264 -crf "$crf" -preset "$preset" -pix_fmt yuv420p \
      -movflags +faststart "$part" </dev/null; then
    n_in="$(count_frames "$f")" n_out="$(count_frames "$part")"
    mv "$part" "$dest"
    if [ -n "$n_in" ] && [ "$n_in" = "$n_out" ]; then
      echo "  ok: $n_out frames"
      ok=$((ok + 1))
    else  # kept, so a long encode is not lost to a counting quirk; look at it
      echo "CHECK: $dest has ${n_out:-?} frames, its source ${n_in:-?}" >&2
      check=$((check + 1))
    fi
  else
    echo "FAILED: $f" >&2
    rm -f "$part"; fail=$((fail + 1))
  fi
done < <(find "$root" -type f -print0 | sort -z)

if [ "$dry" -eq 1 ]; then
  echo "dry run: $ok to encode, $skip already done"
else
  echo "done: $ok encoded, $skip skipped, $fail failed, $check with a different frame count"
fi
[ "$fail" -eq 0 ] && [ "$check" -eq 0 ]
