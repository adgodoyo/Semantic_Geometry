#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export ROOT

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

copy_result() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
  fi
}

cd "$ROOT/src/multilingual"

"$PYTHON_BIN" -B analyze.py
copy_result "$ROOT/results/multilingual/E_cosine_gaps.png" "$ROOT/results/multilingual/cosine_gap_summary.png"
"$PYTHON_BIN" -B analysis_g_all_families.py
"$PYTHON_BIN" -B analysis_g_pair_metric_breakdown.py
"$PYTHON_BIN" -B analysis_g_pair_metric_breakdown_mahalanobis.py

copy_result "$ROOT/results/multilingual/G2_crosslingual_index_by_family.png" "$ROOT/results/multilingual/crosslingual_index_by_family.png"
copy_result "$ROOT/results/multilingual/G2_pair_metric_breakdown_full_layers.csv" "$ROOT/results/multilingual/pairwise_cosine_by_layer.csv"
copy_result "$ROOT/results/multilingual/G2_pair_metric_breakdown_mahalanobis_full_layers.csv" "$ROOT/results/multilingual/pairwise_distances_by_layer.csv"
copy_result "$ROOT/results/multilingual/G2_pair_metric_breakdown_raw_langresid.png" "$ROOT/results/multilingual/raw_vs_langresid_grid.png"
copy_result "$ROOT/results/multilingual/G2_pair_metric_breakdown_top123.png" "$ROOT/results/multilingual/deflation_top1_top3_grid.png"
copy_result "$ROOT/results/multilingual/G2_pair_metric_breakdown_top456.png" "$ROOT/results/multilingual/deflation_top4_top6_grid.png"
"$PYTHON_BIN" -B make_additional_requested_figures.py

"$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root / "scripts"))

from curate_release import prune_tree, remove_junk
from project_manifest import MULTILINGUAL_RESULT_KEEP

multilingual_results = root / "results" / "multilingual"
prune_tree(multilingual_results, MULTILINGUAL_RESULT_KEEP)
remove_junk(multilingual_results)
PY
