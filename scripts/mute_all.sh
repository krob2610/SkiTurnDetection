#!/usr/bin/env bash
# mute_all.sh
#
# Description:
#   Recursively strip every *.MP4 file under the given directory of
#   all audio tracks. Video is copied bit-for-bit, no re-encoding.
#   Each file is replaced in place. Nothing with sound is left behind.
#
# Usage:
#   ./mute_all.sh /absolute/path/to/root_mute_folder
#
# Prerequisites:
#   brew install ffmpeg            # install ffmpeg on macOS
#   chmod +x mute_all.sh           # make the script executable
#
# Caveat:
#   The operation is destructive. Test on a single file first or back up.
#   You must pass an **absolute** path. The script aborts otherwise.

set -euo pipefail

### sanity checks -------------------------------------------------------------
[[ $# -eq 1 ]] || { echo "Usage: $0 /absolute/path" >&2; exit 1; }
ROOT=$1
[[ -d "$ROOT" ]] || { echo "'$ROOT' is not a directory" >&2; exit 1; }
[[ "$ROOT" = /* ]] || { echo "Path must be absolute"    >&2; exit 1; }

tmpfile() { mktemp -t muteXXXXXX.mp4; }   # proper .mp4 extension for ffmpeg

### helpers -------------------------------------------------------------------
strip_ctl() { tr -d '[:cntrl:]'; }

### main ----------------------------------------------------------------------
find "$ROOT" -type f -iname '*.MP4' -print0 |
while IFS= read -r -d '' raw; do
  pretty=$(printf '%s' "$raw" | strip_ctl)      # bez ukrytych znaków

  # Check if path is missing leading slash and fix it
  if [[ "$raw" != /* ]]; then
    echo "⚠️  Found path without leading slash, fixing: \"$raw\""
    raw="/$raw"
    pretty="/$pretty"
  fi

  # Debug info - show the exact path being processed
  echo "DEBUG: Processing file: \"$raw\""
  
  # Check if file exists, otherwise try to find it from ROOT
  if [ ! -f "$raw" ]; then
    echo "⚠️  Cannot access file that find command found: \"$pretty\""
    
    # Try to find the file by its basename within ROOT
    filename=$(basename "$raw")
    echo "    Looking for \"$filename\" within $ROOT..."
    possible_path=$(find "$ROOT" -type f -name "$filename" -print -quit)
    
    if [ -n "$possible_path" ] && [ -f "$possible_path" ]; then
      echo "    ✅ Found at: $possible_path"
      raw="$possible_path"
      pretty="$possible_path"
    else
      echo "    ❌ Not found anywhere in $ROOT"
      echo "    Directory content of supposed parent:"
      ls -la "$(dirname "$raw")" 2>/dev/null || echo "    (Cannot list directory)"
      echo
    fi
  fi

  printf 'Muting: %s\n' "$pretty"

  tmp=$(tmpfile)
  if ffmpeg -loglevel error -y -i "$raw" -c copy -an -f mp4 "$tmp"; then
      mv "$tmp" "$raw"
  else
      echo "ERROR muting: $pretty" >&2
      rm -f "$tmp"
  fi
done

echo "Done."