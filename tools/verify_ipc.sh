#!/usr/bin/env bash
# pyarrow and arrow-mlake read each other's IPC streams. Needs `uv` on PATH;
# builds a throwaway venv, the same one verify_c_import.sh uses.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${TMPDIR:-/tmp}/arrow-mlake-mojo-venv"
uv venv --quiet --allow-existing "$VENV" 2>/dev/null || uv venv --quiet "$VENV"
VIRTUAL_ENV="$VENV" uv pip install --quiet 'pyarrow>=21,<26'
"$VENV/bin/python" tools/verify_ipc.py build/ipc-roundtrip
