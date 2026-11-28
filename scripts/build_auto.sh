#!/usr/bin/env bash
# Build the bundled AUTO-07p for THIS machine (Linux, macOS, or Windows Git Bash).
#
# A wrapper around scripts/build_auto.py, which does the work on every OS:
# it compiles AUTO's library into vendor/auto-07p/lib/<platform>/ (one folder
# per OS / CPU, so objects compiled on another machine are never linked) and
# then runs a small continuation to check that AUTO works. Arguments are
# passed on (--force rebuilds, --check only checks).
#
# Needs gfortran on PATH.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$(command -v python || command -v python3)}"
exec "$PY" "$ROOT/scripts/build_auto.py" "$@"
