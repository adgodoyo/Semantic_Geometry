#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1

resolve_python() {
  local candidate
  for candidate in "${PYTHON_BIN:-}" "$(command -v python3 2>/dev/null || true)" /opt/anaconda3/bin/python3 /usr/bin/python3; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if "$candidate" - <<'PY' >/dev/null 2>&1
import numpy
PY
    then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "No suitable Python interpreter found. Set PYTHON_BIN to an environment with numpy installed." >&2
  exit 1
}

PYTHON_BIN="$(resolve_python)"
export PYTHON_BIN

cd "$ROOT/src/multilingual"
"$PYTHON_BIN" -B make_additional_requested_figures.py

"$PYTHON_BIN" -B "$ROOT/src/paper/build_paper_figures.py"

cd "$ROOT/paper"
latexmk -g -pdf -interaction=nonstopmode -halt-on-error paper.tex
