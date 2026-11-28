# Source this in a shell before working with the project:   source scripts/activate.sh
# (Replaces agent/activate_project.sh. Python code does not need it -- the
# regime pipeline sets up AUTO itself via dynalyst.auto_env -- but it puts
# AUTO's own command-line tools on PATH and the project on PYTHONPATH.)
_DYNALYST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
if command -v conda >/dev/null 2>&1; then
  conda activate "${ENV_NAME:-dynalyst}" 2>/dev/null || true
fi
export AUTO_DIR="${AUTO_DIR:-$_DYNALYST_ROOT/vendor/auto-07p}"
export PATH="$AUTO_DIR/cmds:$AUTO_DIR/bin:$PATH"
export PYTHONPATH="$_DYNALYST_ROOT:$AUTO_DIR/python:$PYTHONPATH"
unset _DYNALYST_ROOT
