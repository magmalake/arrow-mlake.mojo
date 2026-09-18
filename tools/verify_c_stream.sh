#!/usr/bin/env bash
# Have Mojo produce an ArrowArrayStream and pyarrow drain it.
# Needs `uv` on PATH; shares the venv with verify_c_import.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
LIB="build/libamcstream${SHLIB_EXT:-.so}"
[ -f "$LIB" ] || LIB="build/libamcstream.dylib"
[ -f "$LIB" ] || LIB="build/libamcstream.so"
VENV="${TMPDIR:-/tmp}/arrow-mlake-mojo-venv"
uv venv --quiet --allow-existing "$VENV" 2>/dev/null || uv venv --quiet "$VENV"
VIRTUAL_ENV="$VENV" uv pip install --quiet 'pyarrow>=21,<26'
"$VENV/bin/python" tools/verify_c_stream.py "$LIB"
