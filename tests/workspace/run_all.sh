#!/usr/bin/env bash
#
# Run every workspace smoke suite torch-free.
#
# The workspace modules are import-light (torch/cv2/pyarrow are lazy), so we stub
# the `deeplabcut` package parent and run each suite as a standalone script. This
# works without a GPU/torch install. On a full install, `pytest tests/workspace/`
# runs the same files as normal pytest tests.
#
# Usage:  bash tests/workspace/run_all.sh
set -u

repo="$(cd "$(dirname "$0")/../.." && pwd)"
stub="$(mktemp -d)"
mkdir -p "$stub/deeplabcut"
: > "$stub/deeplabcut/__init__.py"
ln -s "$repo/deeplabcut/workspace" "$stub/deeplabcut/workspace"

fail=0
for suite in "$repo"/tests/workspace/test_*.py; do
  if ! PYTHONPATH="$stub" python3 "$suite"; then
    echo "FAIL $(basename "$suite")"
    fail=1
  fi
done

rm -rf "$stub"
[ "$fail" -eq 0 ] && echo "all workspace suites passed"
exit "$fail"
