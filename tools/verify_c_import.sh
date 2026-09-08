#!/usr/bin/env bash
# Have pyarrow produce Arrow data over the C Data Interface and import it.
# Needs `uv` on PATH; builds a throwaway venv.
set -euo pipefail
cd "$(dirname "$0")/.."
LIB="build/libamcimport${SHLIB_EXT:-.so}"
[ -f "$LIB" ] || LIB="build/libamcimport.dylib"
[ -f "$LIB" ] || LIB="build/libamcimport.so"
VENV="${TMPDIR:-/tmp}/arrow-mlake-mojo-venv"
uv venv --quiet --allow-existing "$VENV" 2>/dev/null || uv venv --quiet "$VENV"
VIRTUAL_ENV="$VENV" uv pip install --quiet 'pyarrow>=21,<26'
"$VENV/bin/python" tools/produce_c_data.py "$LIB"
